// Package remotestore implements remote model blob storage (client + shared auth/transport).
//
// Auth signs requests with HMAC-SHA256 over a shared LAN secret.
// Why HMAC (not mTLS) in v1: closed fabric, cheap to deploy, secret via env/_FILE.
// Why no TLS by default: LAN/IB isolation; terminate TLS at a proxy when the path
// leaves the trusted fabric. Large blob PUTs sign an empty body so multi-GB
// streams are not buffered for MAC — integrity is the digest path + server hash.
package remotestore

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	// AuthHeader is the HTTP Authorization scheme prefix for storage requests.
	AuthHeader = "Authorization"
	AuthScheme = "HMAC-SHA256"

	// DefaultReplayWindow rejects timestamps older/newer than this skew.
	// Why ~5m: absorbs clock drift on LAN without making replays unbounded.
	DefaultReplayWindow = 5 * time.Minute

	// CapabilityPath is the control-plane endpoint for transport capability probe.
	CapabilityPath = "/v1/capability"
)

var (
	ErrAuthMissing   = errors.New("missing authorization")
	ErrAuthInvalid   = errors.New("invalid authorization")
	ErrAuthExpired   = errors.New("authorization timestamp outside replay window")
	ErrAuthNoSecret  = errors.New("storage secret not configured")
	ErrAuthScheme    = errors.New("unsupported authorization scheme")
)

// Capability describes what bulk transports a storage peer supports.
// Why advertise instead of guessing: IB may be present but unused; TCP always works.
// Payload-agnostic: same shape for blob, KV, or future compute buffers.
type Capability struct {
	Transports []string `json:"transports"` // e.g. "tcp", "rdma"
	RDMA       *RDMACap `json:"rdma,omitempty"`
	UDPPort    int      `json:"udp_port,omitempty"` // roadmap
	L2OK       bool     `json:"l2_ok,omitempty"`    // roadmap
}

// RDMACap advertises InfiniBand verbs endpoint info for QP setup.
// Why require a real GID: fake "local:device" ads flipped Available() true while
// bytes still moved over HTTP — logs claimed via=rdma incorrectly.
type RDMACap struct {
	Device   string `json:"device,omitempty"`
	GID      string `json:"gid,omitempty"`
	GIDIndex int    `json:"gid_index,omitempty"`
	Port     int    `json:"port,omitempty"`
	LID      uint16 `json:"lid,omitempty"`
	// Verbs is true when storaged was built with -tags rdma and opened an HCA
	// for real RDMA READ (not preference-only).
	Verbs bool `json:"verbs,omitempty"`
}

// Auth signs and verifies HMAC-SHA256 request credentials.
// Header format: Authorization: HMAC-SHA256 <unix_ts>.<hex_hmac>
// MAC input: "<ts>\n<METHOD>\n<path>\n<body_sha256_hex>"
type Auth struct {
	Secret       []byte
	ReplayWindow time.Duration
	Now          func() time.Time // injectable for tests
}

// NewAuth loads the shared secret from ZEROLLAMA_STORAGE_SECRET or the file
// named by ZEROLLAMA_STORAGE_SECRET_FILE.
func NewAuth(secret string) (*Auth, error) {
	secret = strings.TrimSpace(secret)
	if secret == "" {
		if p := strings.TrimSpace(os.Getenv("ZEROLLAMA_STORAGE_SECRET_FILE")); p != "" {
			b, err := os.ReadFile(p)
			if err != nil {
				return nil, fmt.Errorf("read storage secret file: %w", err)
			}
			secret = strings.TrimSpace(string(b))
		}
	}
	if secret == "" {
		return nil, ErrAuthNoSecret
	}
	return &Auth{
		Secret:       []byte(secret),
		ReplayWindow: DefaultReplayWindow,
		Now:          time.Now,
	}, nil
}

// SignRequest sets Authorization on req using method, path, and body digest.
func (a *Auth) SignRequest(req *http.Request, body []byte) error {
	if a == nil || len(a.Secret) == 0 {
		return ErrAuthNoSecret
	}
	ts := a.now().UTC().Unix()
	path := req.URL.EscapedPath()
	if q := req.URL.RawQuery; q != "" {
		path = path + "?" + q
	}
	sig := a.sign(ts, req.Method, path, body)
	req.Header.Set(AuthHeader, fmt.Sprintf("%s %d.%s", AuthScheme, ts, sig))
	return nil
}

// VerifyRequest checks Authorization against method, path, and body.
func (a *Auth) VerifyRequest(req *http.Request, body []byte) error {
	if a == nil || len(a.Secret) == 0 {
		return ErrAuthNoSecret
	}
	hdr := strings.TrimSpace(req.Header.Get(AuthHeader))
	if hdr == "" {
		return ErrAuthMissing
	}
	scheme, rest, ok := strings.Cut(hdr, " ")
	if !ok || !strings.EqualFold(scheme, AuthScheme) {
		return ErrAuthScheme
	}
	tsStr, sig, ok := strings.Cut(strings.TrimSpace(rest), ".")
	if !ok || tsStr == "" || sig == "" {
		return ErrAuthInvalid
	}
	ts, err := strconv.ParseInt(tsStr, 10, 64)
	if err != nil {
		return ErrAuthInvalid
	}
	window := a.ReplayWindow
	if window <= 0 {
		window = DefaultReplayWindow
	}
	now := a.now().UTC()
	t := time.Unix(ts, 0).UTC()
	if t.Before(now.Add(-window)) || t.After(now.Add(window)) {
		return ErrAuthExpired
	}
	path := req.URL.EscapedPath()
	if q := req.URL.RawQuery; q != "" {
		path = path + "?" + q
	}
	want := a.sign(ts, req.Method, path, body)
	if !hmac.Equal([]byte(strings.ToLower(sig)), []byte(want)) {
		return ErrAuthInvalid
	}
	return nil
}

// Middleware wraps an http.Handler with HMAC verification.
// Why buffer PUT/POST: Verify needs the body hash; an earlier version skipped
// write methods entirely (footgun if storaged ever relied on Middleware alone).
// Body is restored so handlers can still stream/read it.
func (a *Auth) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body []byte
		if r.Method == http.MethodPut || r.Method == http.MethodPost {
			var err error
			body, err = io.ReadAll(io.LimitReader(r.Body, 64<<20))
			_ = r.Body.Close()
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			r.Body = io.NopCloser(bytes.NewReader(body))
			r.ContentLength = int64(len(body))
		}
		if err := a.VerifyRequest(r, body); err != nil {
			http.Error(w, err.Error(), http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (a *Auth) sign(ts int64, method, path string, body []byte) string {
	sum := sha256.Sum256(body)
	payload := fmt.Sprintf("%d\n%s\n%s\n%s", ts, strings.ToUpper(method), path, hex.EncodeToString(sum[:]))
	mac := hmac.New(sha256.New, a.Secret)
	mac.Write([]byte(payload))
	return hex.EncodeToString(mac.Sum(nil))
}

func (a *Auth) now() time.Time {
	if a.Now != nil {
		return a.Now()
	}
	return time.Now()
}

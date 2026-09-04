package remotestore

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

func jsonDecoder(r io.Reader) *json.Decoder {
	return json.NewDecoder(r)
}

// TCPTransport implements BulkTransport via HTTP Range-GET.
// Why TCP last in the chain but always present: works on every fabric (Ethernet,
// IPoIB) and is the only data plane until verbs QP ships.
type TCPTransport struct {
	Auth   *Auth
	Client *http.Client
}

func NewTCPTransport(auth *Auth) *TCPTransport {
	return &TCPTransport{
		Auth: auth,
		Client: &http.Client{
			Timeout: 30 * time.Minute,
		},
	}
}

func (t *TCPTransport) Name() string { return "tcp" }

func (t *TCPTransport) Available(cap Capability) bool {
	if len(cap.Transports) == 0 {
		return true // assume TCP if peer didn't advertise
	}
	for _, n := range cap.Transports {
		if strings.EqualFold(n, "tcp") || strings.EqualFold(n, "http") {
			return true
		}
	}
	return true // TCP always usable as last resort for HTTP control-plane peers
}

func (t *TCPTransport) FetchChunk(ctx context.Context, baseURL, digest string, offset, length int64) (io.ReadCloser, int64, error) {
	digest = normalizeDigest(digest)
	url := strings.TrimRight(baseURL, "/") + "/v1/blob/" + digest
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, err
	}
	if length > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", offset, offset+length-1))
	} else if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	if t.Auth != nil {
		if err := t.Auth.SignRequest(req, nil); err != nil {
			return nil, 0, err
		}
	}
	resp, err := t.Client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		defer resp.Body.Close()
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, 0, fmt.Errorf("tcp fetch %s: %s: %s", digest, resp.Status, strings.TrimSpace(string(b)))
	}
	n := resp.ContentLength
	if n < 0 {
		if cr := resp.Header.Get("Content-Range"); cr != "" {
			// bytes start-end/total
			if i := strings.LastIndex(cr, "/"); i >= 0 {
				if tot, err := strconv.ParseInt(cr[i+1:], 10, 64); err == nil && length <= 0 {
					n = tot - offset
				}
			}
		}
	}
	return resp.Body, n, nil
}

// HeadBlob checks existence and size via HEAD.
func (t *TCPTransport) HeadBlob(ctx context.Context, baseURL, digest string) (size int64, ok bool, err error) {
	digest = normalizeDigest(digest)
	url := strings.TrimRight(baseURL, "/") + "/v1/blob/" + digest
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, url, nil)
	if err != nil {
		return 0, false, err
	}
	if t.Auth != nil {
		if err := t.Auth.SignRequest(req, nil); err != nil {
			return 0, false, err
		}
	}
	resp, err := t.Client.Do(req)
	if err != nil {
		return 0, false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return 0, false, nil
	}
	if resp.StatusCode != http.StatusOK {
		return 0, false, fmt.Errorf("head %s: %s", digest, resp.Status)
	}
	size = resp.ContentLength
	if resp.Header.Get("X-Zerollama-Partial") == "1" {
		return size, false, nil
	}
	return size, true, nil
}

// GetCapability probes /v1/capability over TCP.
// Why distinguish 401/403 from 404: auth misconfig must not be cached as
// "TCP-only OK"; missing capability endpoint (older peer) may fall back to TCP.
func GetCapability(ctx context.Context, auth *Auth, client *http.Client, baseURL string) (Capability, error) {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	url := strings.TrimRight(baseURL, "/") + CapabilityPath
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return Capability{}, err
	}
	if auth != nil {
		if err := auth.SignRequest(req, nil); err != nil {
			return Capability{}, err
		}
	}
	resp, err := client.Do(req)
	if err != nil {
		return Capability{}, err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
		// continue
	case http.StatusUnauthorized, http.StatusForbidden:
		return Capability{}, fmt.Errorf("capability probe: %s", resp.Status)
	case http.StatusNotFound:
		// Older peers without capability endpoint: assume TCP only.
		return Capability{Transports: []string{"tcp"}}, nil
	default:
		// Non-fatal probe failure: fall back to TCP-only.
		return Capability{Transports: []string{"tcp"}}, nil
	}
	var cap Capability
	dec := jsonDecoder(resp.Body)
	if err := dec.Decode(&cap); err != nil {
		return Capability{Transports: []string{"tcp"}}, nil
	}
	if len(cap.Transports) == 0 {
		cap.Transports = []string{"tcp"}
	}
	return cap, nil
}

func normalizeDigest(d string) string {
	d = strings.TrimSpace(d)
	d = strings.ReplaceAll(d, ":", "-")
	if !strings.HasPrefix(d, "sha256-") && len(d) == 64 {
		d = "sha256-" + d
	}
	return d
}

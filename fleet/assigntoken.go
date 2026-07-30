// Package fleet assignment tokens (F5): short-TTL HMAC holds so two agents
// do not race the same warm queue slot after assign.
package fleet

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	defaultAssignTTL     = 8 * time.Second
	maxAssignTTL         = 5 * time.Minute // raised from 30s: fleet hold windows can span CGo init + model load
	minAssignTTL         = 2 * time.Second
	AssignmentTokenHeader = "X-Zerollama-Assignment-Token"
)

var (
	ErrAssignTokenDisabled = errors.New("assignment tokens disabled")
	ErrAssignTokenInvalid  = errors.New("invalid assignment token")
	ErrAssignTokenExpired  = errors.New("assignment token expired")
)

// AssignTokenClaims is the signed payload inside an assignment token.
type AssignTokenClaims struct {
	NodeID    string    `json:"n"`
	Model     string    `json:"m"`
	ExpiresAt time.Time `json:"-"`
	ExpUnix   int64     `json:"e"`
	JTI       string    `json:"j"`
}

type assignTokenWire struct {
	N string `json:"n"`
	M string `json:"m"`
	E int64  `json:"e"`
	J string `json:"j"`
}

// AssignTokenSecret reads ZEROLLAMA_FLEET_ASSIGN_SECRET (required to mint/verify).
func AssignTokenSecret() string {
	return strings.TrimSpace(os.Getenv("ZEROLLAMA_FLEET_ASSIGN_SECRET"))
}

// AssignTokenEnabled is true when a secret is set and not explicitly disabled.
func AssignTokenEnabled() bool {
	if AssignTokenSecret() == "" {
		return false
	}
	s := strings.TrimSpace(os.Getenv("ZEROLLAMA_FLEET_ASSIGN_TOKEN"))
	if s == "" {
		return true
	}
	return s == "1" || strings.EqualFold(s, "on") || strings.EqualFold(s, "true")
}

// AssignTokenTTL is the hold window (default 8s, clamped 2s–5m).
func AssignTokenTTL() time.Duration {
	raw := strings.TrimSpace(os.Getenv("ZEROLLAMA_FLEET_ASSIGN_TTL"))
	if raw == "" {
		return defaultAssignTTL
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		if sec, err2 := strconv.Atoi(raw); err2 == nil && sec > 0 {
			d = time.Duration(sec) * time.Second
		} else {
			return defaultAssignTTL
		}
	}
	if d < minAssignTTL {
		return minAssignTTL
	}
	if d > maxAssignTTL {
		return maxAssignTTL
	}
	return d
}

// AssignPushHoldEnabled gates fleet→node POST /api/fleet/assign-hold after mint (default on).
func AssignPushHoldEnabled() bool {
	if !AssignTokenEnabled() {
		return false
	}
	s := strings.TrimSpace(os.Getenv("ZEROLLAMA_FLEET_ASSIGN_PUSH"))
	if s == "" {
		return true
	}
	return s == "1" || strings.EqualFold(s, "on") || strings.EqualFold(s, "true")
}

// MintAssignToken creates a short-TTL HMAC token for nodeID + model.
func MintAssignToken(nodeID, model string, now time.Time) (token string, exp time.Time, jti string, err error) {
	secret := AssignTokenSecret()
	if secret == "" || !AssignTokenEnabled() {
		return "", time.Time{}, "", ErrAssignTokenDisabled
	}
	nodeID = strings.TrimSpace(nodeID)
	model = strings.TrimSpace(model)
	if nodeID == "" || model == "" {
		return "", time.Time{}, "", fmt.Errorf("node_id and model required")
	}
	ttl := AssignTokenTTL()
	exp = now.UTC().Add(ttl)
	jti = fmt.Sprintf("%d-%s", now.UnixNano(), shortHash(nodeID+model+secret))
	wire := assignTokenWire{
		N: nodeID,
		M: model,
		E: exp.Unix(),
		J: jti,
	}
	raw, err := json.Marshal(wire)
	if err != nil {
		return "", time.Time{}, "", err
	}
	payload := base64.RawURLEncoding.EncodeToString(raw)
	sig := signAssignPayload(secret, payload)
	return payload + "." + sig, exp, jti, nil
}

// ParseAssignToken verifies HMAC and expiry.
func ParseAssignToken(token string, now time.Time) (AssignTokenClaims, error) {
	secret := AssignTokenSecret()
	if secret == "" || !AssignTokenEnabled() {
		return AssignTokenClaims{}, ErrAssignTokenDisabled
	}
	token = strings.TrimSpace(token)
	payload, sig, ok := strings.Cut(token, ".")
	if !ok || payload == "" || sig == "" {
		return AssignTokenClaims{}, ErrAssignTokenInvalid
	}
	if !hmac.Equal([]byte(signAssignPayload(secret, payload)), []byte(sig)) {
		return AssignTokenClaims{}, ErrAssignTokenInvalid
	}
	raw, err := base64.RawURLEncoding.DecodeString(payload)
	if err != nil {
		return AssignTokenClaims{}, ErrAssignTokenInvalid
	}
	var wire assignTokenWire
	if json.Unmarshal(raw, &wire) != nil {
		return AssignTokenClaims{}, ErrAssignTokenInvalid
	}
	if wire.N == "" || wire.M == "" || wire.J == "" || wire.E <= 0 {
		return AssignTokenClaims{}, ErrAssignTokenInvalid
	}
	exp := time.Unix(wire.E, 0).UTC()
	if !now.UTC().Before(exp) {
		return AssignTokenClaims{}, ErrAssignTokenExpired
	}
	return AssignTokenClaims{
		NodeID:    wire.N,
		Model:     wire.M,
		ExpiresAt: exp,
		ExpUnix:   wire.E,
		JTI:       wire.J,
	}, nil
}

func signAssignPayload(secret, payload string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(payload))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func shortHash(s string) string {
	sum := sha256.Sum256([]byte(s))
	return base64.RawURLEncoding.EncodeToString(sum[:6])
}

package fleet

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

// withAssignmentToken mints F5 fields onto a successful assign response.
func withAssignmentToken(resp AssignResponse, model string, now time.Time) AssignResponse {
	if !AssignTokenEnabled() {
		return resp
	}
	tok, exp, _, err := MintAssignToken(resp.NodeID, model, now)
	if err != nil {
		return resp
	}
	resp.AssignmentToken = tok
	resp.ExpiresAt = &exp
	sec := int(exp.Sub(now.UTC()).Seconds())
	if sec < 1 {
		sec = 1
	}
	resp.ExpiresIn = sec
	return resp
}

// pushAssignHold asks the chosen node to soft-hold one queue slot until TTL.
func (m *Manager) pushAssignHold(ctx context.Context, resp AssignResponse) {
	if m == nil || resp.AssignmentToken == "" || resp.URL == "" {
		return
	}
	body, _ := json.Marshal(map[string]string{"token": resp.AssignmentToken})
	url := strings.TrimRight(resp.URL, "/") + "/api/fleet/assign-hold"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	client := m.client
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Second}
	}
	r, err := client.Do(req)
	if err != nil {
		return
	}
	defer r.Body.Close()
}

// AssignHoldRequest is POST /api/fleet/assign-hold on a zerollama node.
type AssignHoldRequest struct {
	Token string `json:"token"`
}

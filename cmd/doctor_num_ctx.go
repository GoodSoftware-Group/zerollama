package cmd

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// doctorCheckOversizedNumCtx covers minefield trap 79: absurd num_ctx must not
// present as a silent empty/length failure. Zerollama clamps; we assert the
// response is not the trap-79 signature (200 + empty + length, no clamp).
func doctorCheckOversizedNumCtx(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-79 (oversized num_ctx)"
	if m.NumCtx <= 0 {
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: "loaded num_ctx unknown — skip oversized context probe",
		}
	}
	ask := m.NumCtx * 5
	if ask < 200_000 {
		ask = 200_000
	}
	client := &http.Client{Timeout: 60 * time.Second}
	status, body, err := doctorPostJSON(client, strings.TrimSuffix(base, "/")+"/api/chat", map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Say hi."},
		},
		"stream":  false,
		"think":   false,
		"options": map[string]any{"num_ctx": ask, "num_predict": 8, "temperature": 0},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "probe failed: " + err.Error()}
	}
	if status != http.StatusOK {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("oversized num_ctx=%d rejected with HTTP %d (not trap-79 silent accept)", ask, status),
		}
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(body), &out); err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "decode: " + err.Error()}
	}
	content := strings.TrimSpace(doctorMessageString(out, "content"))
	done, _ := out["done_reason"].(string)
	clamped := false
	if g, ok := out["ggml_num_ctx"].(map[string]any); ok {
		if c, _ := g["num_ctx_clamped"].(bool); c {
			clamped = true
		}
	}
	if content == "" && (done == "length" || done == "") && !clamped {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("num_ctx=%d accepted with empty content done_reason=%q and no ggml_num_ctx clamp (minefield trap 79)", ask, done),
			FixHint: "clamp num_ctx to model context_length and surface ggml_num_ctx; do not raise num_predict",
		}
	}
	detail := fmt.Sprintf("oversized num_ctx=%d did not show trap-79 empty/length signature", ask)
	if clamped {
		detail += " (ggml_num_ctx clamp reported)"
	}
	if content != "" {
		detail += fmt.Sprintf("; got content (%d chars)", len(content))
	}
	return doctorCheck{Name: name, Status: "ok", Detail: detail + " on " + m.Name}
}

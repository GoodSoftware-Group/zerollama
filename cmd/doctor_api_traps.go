package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// doctorCheckThinkingGate covers minefield trap 29: server thinking-off is a
// default, not a hard gate, unless ZEROLLAMA_THINKING_GATE=deny|strip.
// When a thinking model is warm and the gate is unset, warn — that is the
// trap-12 / trap-29 interaction the minefield doctor flags.
func doctorCheckThinkingGate(thinkingLoaded bool) doctorCheck {
	const name = "thinking gate (trap 29)"
	gate := envconfig.ThinkingGate()
	switch gate {
	case "deny":
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: "ZEROLLAMA_THINKING_GATE=deny — client enable → HTTP 400",
		}
	case "strip":
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: "ZEROLLAMA_THINKING_GATE=strip — client enable forced off",
		}
	default:
		if thinkingLoaded {
			return doctorCheck{
				Name:    name,
				Status:  "warn",
				Detail:  "thinking model warm and ZEROLLAMA_THINKING_GATE unset — clients can re-enable thinking (minefield trap 29)",
				FixHint: "set ZEROLLAMA_THINKING_GATE=deny|strip on lanes sized for non-thinking max_tokens",
			}
		}
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: "ZEROLLAMA_THINKING_GATE unset (default allow) — not a hard gate; set deny|strip when measuring non-thinking arms",
		}
	}
}

// doctorCheckUnknownFields covers trap 77: invented top-level keys must 400.
func doctorCheckUnknownFields(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-77 (unknown fields)"
	client := &http.Client{Timeout: 10 * time.Second}

	probe := map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "hi"},
		},
		"stream":                                false,
		"__minefield_unvalidated_field_probe__": true,
	}
	status, body, err := doctorPostJSON(client, base+"/api/chat", probe)
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "probe failed: " + err.Error()}
	}
	if status != http.StatusBadRequest {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("/api/chat accepted unknown field with HTTP %d (minefield trap 77)", status),
			FixHint: "unknown top-level keys must 400; see api.CheckUnknownChatFields",
		}
	}
	if !strings.Contains(body, "__minefield_unvalidated_field_probe__") && !strings.Contains(strings.ToLower(body), "unknown") {
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: fmt.Sprintf("/api/chat returned 400 but body lacks unknown-field signal: %s", truncateDoctorDetail(body, 160)),
		}
	}

	// OpenAI surface parity.
	v1 := map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "hi"},
		},
		"stream":                                false,
		"__minefield_unvalidated_field_probe__": true,
	}
	status, body, err = doctorPostJSON(client, base+"/v1/chat/completions", v1)
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "/v1 probe failed: " + err.Error()}
	}
	if status != http.StatusBadRequest {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("/v1/chat/completions accepted unknown field with HTTP %d (minefield trap 77)", status),
			FixHint: "BindChatCompletionRequest must reject invented keys",
		}
	}
	_ = body
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("/api/chat + /v1 reject unknown top-level keys on %s", m.Name),
	}
}

// doctorCheckToolChoiceNone covers trap 78 on the OpenAI path: tool_choice=none
// must not return structured tool_calls (tools are omitted before chat).
func doctorCheckToolChoiceNone(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-78 (tool_choice none)"
	if !m.SupportsTools {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise tools — skipped", m.Name),
		}
	}
	client := &http.Client{Timeout: doctorLiveHTTPTimeout}
	payload := map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Call get_time for timezone UTC."},
		},
		"stream":      false,
		"tool_choice": "none",
		"max_tokens":  64,
		"temperature": 0,
		"tools": []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":        "get_time",
					"description": "Get the current time",
					"parameters": map[string]any{
						"type":       "object",
						"properties": map[string]any{"timezone": map[string]any{"type": "string"}},
					},
				},
			},
		},
	}
	if m.SupportsThinking {
		payload["think"] = false
	}
	status, body, err := doctorPostJSON(client, base+"/v1/chat/completions", payload)
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "probe failed: " + err.Error()}
	}
	if status != http.StatusOK {
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: fmt.Sprintf("HTTP %d: %s", status, truncateDoctorDetail(body, 160)),
		}
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(body), &out); err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "decode: " + err.Error()}
	}
	if doctorOpenAIHasToolCalls(out) {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "tool_choice=none still returned tool_calls (minefield trap 78)",
			FixHint: "openai FromChatRequest must omit tools when tool_choice is none",
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("tool_choice=none produced no tool_calls on %s", m.Name),
	}
}

func doctorOpenAIHasToolCalls(resp map[string]any) bool {
	choices, _ := resp["choices"].([]any)
	for _, c := range choices {
		ch, _ := c.(map[string]any)
		msg, _ := ch["message"].(map[string]any)
		if msg == nil {
			continue
		}
		if calls, ok := msg["tool_calls"].([]any); ok && len(calls) > 0 {
			return true
		}
	}
	return false
}

func doctorPostJSON(client *http.Client, url string, payload map[string]any) (status int, body string, err error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return 0, "", err
	}
	resp, err := client.Post(url, "application/json", bytes.NewReader(raw))
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return resp.StatusCode, "", err
	}
	return resp.StatusCode, string(b), nil
}

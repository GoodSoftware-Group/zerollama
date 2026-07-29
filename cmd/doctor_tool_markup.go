package cmd

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// doctorCheckToolMarkup covers minefield trap 26: tool markup must not remain
// trapped in thinking (or as unparsed prose) when the model intended a call.
// Also flags leftover <tool_call> beside structured tool_calls (partial parse).
func doctorCheckToolMarkup(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-26 (tool markup in think)"
	if !m.SupportsTools {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise tools — skipped", m.Name),
		}
	}

	client := &http.Client{Timeout: 60 * time.Second}
	payload := map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "What time is it in Tokyo? Use the get_time tool."},
		},
		"stream": false,
		"tools": []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":        "get_time",
					"description": "Get the current time for a timezone",
					"parameters": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"timezone": map[string]any{"type": "string"},
						},
						"required": []string{"timezone"},
					},
				},
			},
		},
		"tool_choice": map[string]any{
			"type":     "function",
			"function": map[string]any{"name": "get_time"},
		},
		"options": map[string]any{"temperature": 0, "num_predict": 256},
	}
	if m.SupportsThinking {
		payload["think"] = true
	}

	// Prefer /v1 where tool_choice objects are first-class.
	st, body, err := doctorPostJSON(client, base+"/v1/chat/completions", payload)
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "v1 tools probe failed: " + err.Error()}
	}
	if st != http.StatusOK {
		// Fall back to /api/chat without object tool_choice.
		delete(payload, "tool_choice")
		resp, err := doctorChatOnce(base, payload)
		if err != nil {
			return doctorCheck{
				Name:   name,
				Status: "warn",
				Detail: fmt.Sprintf("v1 HTTP %d; /api/chat fallback failed: %s", st, err.Error()),
			}
		}
		return doctorJudgeToolMarkup(name, m.Name, resp)
	}

	var v1 map[string]any
	if err := json.Unmarshal([]byte(body), &v1); err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "decode v1: " + err.Error()}
	}
	return doctorJudgeToolMarkupV1(name, m.Name, v1)
}

func doctorJudgeToolMarkup(name, model string, resp map[string]any) doctorCheck {
	calls := doctorMessageToolCalls(resp)
	content := doctorMessageString(resp, "content")
	thinking := doctorMessageString(resp, "thinking")
	return doctorJudgeToolMarkupFields(name, model, len(calls), content, thinking)
}

func doctorJudgeToolMarkupV1(name, model string, resp map[string]any) doctorCheck {
	choices, _ := resp["choices"].([]any)
	if len(choices) == 0 {
		return doctorCheck{Name: name, Status: "warn", Detail: "v1 response missing choices"}
	}
	ch0, _ := choices[0].(map[string]any)
	msg, _ := ch0["message"].(map[string]any)
	if msg == nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "v1 choice missing message"}
	}
	content, _ := msg["content"].(string)
	thinking, _ := msg["thinking"].(string)
	if thinking == "" {
		thinking, _ = msg["reasoning"].(string)
	}
	if thinking == "" {
		thinking, _ = msg["reasoning_content"].(string)
	}
	calls, _ := msg["tool_calls"].([]any)
	return doctorJudgeToolMarkupFields(name, model, len(calls), content, thinking)
}

func doctorJudgeToolMarkupFields(name, model string, nCalls int, content, thinking string) doctorCheck {
	blob := content + "\n" + thinking
	hasMarkup := strings.Contains(blob, "<tool_call>")

	switch {
	case nCalls == 0 && hasMarkup:
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("raw <tool_call> in content/thinking but empty tool_calls on %s (minefield trap 26)", model),
			FixHint: "thinking parser should end think at <tool_call>; see thinking.ImplicitThinkEndMarkers",
		}
	case nCalls > 0 && strings.Contains(thinking, "<tool_call>"):
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("tool_calls present but <tool_call> still in thinking on %s (trap 26 partial)", model),
			FixHint: "scan thinking for leftover tool markup before scoring multi-call turns",
		}
	case nCalls > 0:
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("structured tool_calls (%d) without trapped markup on %s", nCalls, model),
		}
	default:
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: fmt.Sprintf("no tool_calls and no <tool_call> markup on %s (inconclusive — model may have refused)", model),
		}
	}
}

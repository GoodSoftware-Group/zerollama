package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// doctorCheckServingTraps probes a live Go API for minefield-style serving traps.
// It only uses already-loaded models (/api/ps) so doctor never triggers a cold load
// on the operator's production stack. Missing coverage is reported as warn, not fail.
func doctorCheckServingTraps() []doctorCheck {
	base, _ := doctorProbeGoAPI()
	if base == "" {
		return []doctorCheck{
			doctorCheckThinkingGate(false),
			{
				Name:    "serving traps",
				Status:  "warn",
				Detail:  "no Go API reachable — skip live minefield probes",
				FixHint: "run zerollama serve, then re-run doctor with a warm model (zerollama run …)",
			},
		}
	}

	loaded, err := doctorFetchLoadedModels(base)
	if err != nil {
		return []doctorCheck{
			doctorCheckThinkingGate(false),
			{
				Name:   "serving traps",
				Status: "warn",
				Detail: "could not read /api/ps: " + err.Error(),
			},
		}
	}
	if len(loaded) == 0 {
		return []doctorCheck{
			doctorCheckThinkingGate(false),
			{
				Name:    "serving traps",
				Status:  "warn",
				Detail:  "no loaded model — live traps need a warm runner (minefield doctor coverage)",
				FixHint: "zerollama run <model> then re-run doctor; checks cover traps 77, 78, 66, 48, 55/61, 01/03, 12/64/65, 19",
			},
		}
	}

	pick := loaded[0]
	var out []doctorCheck
	out = append(out, doctorCheckThinkingGate(pick.SupportsThinking))
	out = append(out, doctorCheckUnknownFields(base, pick))
	out = append(out, doctorCheckKwargDeadness(base, pick))
	out = append(out, doctorCheckToolChoiceNone(base, pick))
	out = append(out, doctorCheckStreamContent(base, pick))
	out = append(out, doctorCheckHistoryAssembly(base, pick))
	out = append(out, doctorCheckReasoningRoundTrip(base, pick))
	out = append(out, doctorCheckOrphanedThinkClose(base, pick))
	out = append(out, doctorCheckThinkToggleInjection(base, pick))
	out = append(out, doctorCheckLatencyReconciliation(base, pick))
	out = append(out, doctorCheckContextCeilings(pick))
	out = append(out, doctorCheckOversizedNumCtx(base, pick))
	out = append(out, doctorCheckReasoningField(base, pick))
	out = append(out, doctorCheckThinkRoundtrip(base, pick))
	out = append(out, doctorCheckTokenCeiling(base, pick))
	out = append(out, doctorCheckToolCallShape(base, pick))
	out = append(out, doctorCheckToolMarkup(base, pick))
	return out
}

type doctorLoadedModel struct {
	Name             string
	SupportsThinking bool
	SupportsTools    bool
	HasChatTemplate  bool
	TrainCtx         int
	NumCtx           int
}

func doctorFetchLoadedModels(base string) ([]doctorLoadedModel, error) {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(base + "/api/ps")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	var body struct {
		Models []struct {
			Name           string `json:"name"`
			Model          string `json:"model"`
			ContextLength  int    `json:"context_length"`
			LoadedMetadata *struct {
				NumCtx             int  `json:"num_ctx"`
				TrainContextLength int  `json:"train_context_length"`
				SupportsThinking   bool `json:"supports_thinking"`
				SupportsTools      bool `json:"supports_tools"`
				HasChatTemplate    bool `json:"has_chat_template"`
			} `json:"loaded_metadata"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, err
	}
	out := make([]doctorLoadedModel, 0, len(body.Models))
	for _, m := range body.Models {
		name := m.Name
		if name == "" {
			name = m.Model
		}
		lm := doctorLoadedModel{Name: name, NumCtx: m.ContextLength}
		if m.LoadedMetadata != nil {
			lm.SupportsThinking = m.LoadedMetadata.SupportsThinking
			lm.SupportsTools = m.LoadedMetadata.SupportsTools
			lm.HasChatTemplate = m.LoadedMetadata.HasChatTemplate
			lm.TrainCtx = m.LoadedMetadata.TrainContextLength
			if m.LoadedMetadata.NumCtx > 0 {
				lm.NumCtx = m.LoadedMetadata.NumCtx
			}
		}
		out = append(out, lm)
	}
	return out, nil
}

// doctorCheckReasoningField covers minefield traps 01/03: reasoning/thinking field present
// when think is on for a thinking-capable model.
func doctorCheckReasoningField(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-01/03 (reasoning field)"
	if !m.SupportsThinking {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise thinking — skipped", m.Name),
		}
	}
	resp, err := doctorChatOnce(base, map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Reply with exactly OK. Think briefly first."},
		},
		"stream":  false,
		"think":   true,
		"options": map[string]any{"temperature": 0, "num_predict": 128},
	})
	if err != nil {
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: "chat probe failed: " + err.Error(),
		}
	}
	thinking := doctorMessageString(resp, "thinking")
	// Some stacks put reasoning under message.reasoning (trap 01 field-name split).
	reasoning := doctorMessageString(resp, "reasoning")
	content := doctorMessageString(resp, "content")
	if thinking == "" && reasoning == "" {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("think=true but neither message.thinking nor message.reasoning populated (content_len=%d) (minefield trap 01/03)", len(content)),
			FixHint: "inspect the assembled prompt and the model's think tags; field name may differ by route",
		}
	}
	field := "thinking"
	if thinking == "" {
		field = "reasoning"
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("think=true populated message.%s on %s", field, m.Name),
	}
}

// doctorCheckThinkRoundtrip covers traps 12/64/65 family: think on/off must not
// silently return empty content with HTTP 200. Also probes default /api/generate
// (omit think) — milkey-class models park answers in thinking only on generate.
// Why generate arm: live doctor historically only hit /api/chat; chat already
// defaults think=false before parser Init, so milkey looked fine while benches
// using /api/generate scored 0. FixHint points at doctor --repair-models.
func doctorCheckThinkRoundtrip(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-12/64/65 (think empty content)"
	if !m.SupportsThinking {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise thinking — skipped", m.Name),
		}
	}
	for _, think := range []any{true, false} {
		resp, err := doctorChatOnce(base, map[string]any{
			"model": m.Name,
			"messages": []map[string]string{
				{"role": "user", "content": "Say the word ping and nothing else."},
			},
			"stream":  false,
			"think":   think,
			"options": map[string]any{"temperature": 0, "num_predict": 64},
		})
		if err != nil {
			return doctorCheck{
				Name:   name,
				Status: "warn",
				Detail: fmt.Sprintf("think=%v probe failed: %v", think, err),
			}
		}
		content := doctorMessageString(resp, "content")
		thinking := doctorMessageString(resp, "thinking")
		if content == "" && thinking == "" {
			return doctorCheck{
				Name:    name,
				Status:  "warn",
				Detail:  fmt.Sprintf("think=%v returned empty content and empty thinking (minefield trap 12/64/65)", think),
				FixHint: "raise num_predict / check answer-in-reasoning routing; see docs/model-serving-minefield.md",
			}
		}
		// Trap 64 shape: content null/empty but thinking holds the whole answer.
		if think == false && content == "" && thinking != "" {
			return doctorCheck{
				Name:    name,
				Status:  "warn",
				Detail:  "think=false but answer landed in thinking with empty content (minefield trap 64)",
				FixHint: "zerollama doctor --repair-models --apply; check think kwarg / template toggles",
			}
		}
	}

	// Unload so prefix-cache from chat arms cannot poison the generate probe
	// (same isolation modelrepair.liveProbes uses).
	// Why: without unload, doctor trap-12 and --repair-models can disagree on
	// the same tag after a prior slash or think generation.
	_ = doctorUnloadModel(base, m.Name)

	gen, err := doctorGenerateOnce(base, map[string]any{
		"model":  m.Name,
		"prompt": "Say the word ping and nothing else.",
		"stream": false,
		"options": map[string]any{
			"temperature": 0,
			"num_predict": 64,
			"num_ctx":     2048,
		},
		"keep_alive": "30s",
	})
	if err != nil {
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: "default /api/generate probe failed: " + err.Error(),
		}
	}
	genResp, _ := gen["response"].(string)
	genThink, _ := gen["thinking"].(string)
	if strings.TrimSpace(genResp) == "" && strings.TrimSpace(genThink) != "" {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("default /api/generate empty response with thinking_len=%d on %s (milkey-class trap 12/64)", len(genThink), m.Name),
			FixHint: "zerollama doctor --repair-models --apply " + m.Name + "; rebuild/restart serve so Think defaults before parser Init",
		}
	}

	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("think on/off + default generate returned non-empty channels on %s", m.Name),
	}
}

// doctorUnloadModel expires a runner (keep_alive:0) so later probes start cold.
// Why: llama-server prefix KV after a bad generation can make a clean follow-up
// still collapse; repair and trap-12 generate arms both need a cold start.
func doctorUnloadModel(base, name string) error {
	_, err := doctorGenerateOnce(base, map[string]any{
		"model":      name,
		"prompt":     "",
		"keep_alive": 0,
	})
	return err
}

// doctorCheckToolCallShape covers trap 19: tool-defined request should return
// structured tool_calls, not prose describing a call.
func doctorCheckToolCallShape(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-19 (tool_calls shape)"
	if !m.SupportsTools {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise tools — skipped", m.Name),
		}
	}
	resp, err := doctorChatOnce(base, map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Call get_time for timezone UTC. Do not answer in prose."},
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
		"options": map[string]any{"temperature": 0, "num_predict": 128},
	})
	if err != nil {
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: "chat probe failed: " + err.Error(),
		}
	}
	calls := doctorMessageToolCalls(resp)
	content := doctorMessageString(resp, "content")
	if len(calls) > 0 {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("structured tool_calls (%d) on %s", len(calls), m.Name),
		}
	}
	// Inconclusive if the model simply refused; only warn when content looks like a prose tool call.
	lower := strings.ToLower(content)
	if strings.Contains(lower, "get_time") || strings.Contains(lower, `"name"`) || strings.Contains(lower, "function") {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "tools defined but response has prose tool description and empty tool_calls (minefield trap 19)",
			FixHint: "check template/parser flags; native schema may be dropped",
		}
	}
	return doctorCheck{
		Name:    name,
		Status:  "warn",
		Detail:  "tools advertised but no tool_calls in response (inconclusive — model may have refused)",
		FixHint: "re-run with a tool-friendly prompt or inspect template tool rendering",
	}
}

func doctorChatOnce(base string, payload map[string]any) (map[string]any, error) {
	return doctorAPIOnce(base, "/api/chat", payload)
}

func doctorGenerateOnce(base string, payload map[string]any) (map[string]any, error) {
	return doctorAPIOnce(base, "/api/generate", payload)
}

func doctorAPIOnce(base, path string, payload map[string]any) (map[string]any, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 45 * time.Second}
	resp, err := client.Post(base+path, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncateDoctorDetail(string(raw), 200))
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func doctorMessageString(resp map[string]any, key string) string {
	msg, _ := resp["message"].(map[string]any)
	if msg == nil {
		return ""
	}
	s, _ := msg[key].(string)
	return s
}

func doctorMessageToolCalls(resp map[string]any) []any {
	msg, _ := resp["message"].(map[string]any)
	if msg == nil {
		return nil
	}
	calls, _ := msg["tool_calls"].([]any)
	return calls
}

func truncateDoctorDetail(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

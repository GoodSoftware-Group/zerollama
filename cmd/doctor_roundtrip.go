package cmd

import (
	"fmt"
	"net/http"
	"strings"
	"time"
)

// doctorCheckReasoningRoundTrip covers minefield trap 63: of the common
// field×gate combinations, exactly one shape preserves prior reasoning in the
// assembled prompt on this stack.
//
// Zerollama's correct shape is native `thinking` with `think` on (preservePriorThinkingForRender).
// `reasoning` / `reasoning_content` on /api/chat are dropped before render.
// Nemotron's `truncate_history_thinking` is an unknown kwarg → HTTP 400 (loud),
// not a silent preserve switch.
func doctorCheckReasoningRoundTrip(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-63 (reasoning round-trip)"
	if !m.SupportsThinking {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise thinking — skipped", m.Name),
		}
	}

	history := func(asstKey, asstVal string) map[string]any {
		asst := map[string]any{"role": "assistant", "content": "A1"}
		asst[asstKey] = asstVal
		return map[string]any{
			"model": m.Name,
			"messages": []map[string]any{
				{"role": "user", "content": "Q1"},
				asst,
				{"role": "user", "content": "Q2"},
			},
			"stream":             false,
			"think":              true,
			"_debug_render_only": true,
			"options":            map[string]any{"num_predict": 1},
		}
	}

	thinkRender, err := doctorDebugRender(base, history("thinking", doctorHistoryMarker))
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "thinking arm failed: " + err.Error()}
	}
	reasonRender, err := doctorDebugRender(base, history("reasoning", doctorHistoryMarker))
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "reasoning arm failed: " + err.Error()}
	}
	rcRender, err := doctorDebugRender(base, history("reasoning_content", doctorHistoryMarker))
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "reasoning_content arm failed: " + err.Error()}
	}

	thinkOK := strings.Contains(thinkRender, doctorHistoryMarker)
	reasonOK := strings.Contains(reasonRender, doctorHistoryMarker)
	rcOK := strings.Contains(rcRender, doctorHistoryMarker)

	// Gate polarity probe: foreign preserve kwarg must not silently no-op as "preserve".
	client := &http.Client{Timeout: 30 * time.Second}
	gatePayload := history("thinking", doctorHistoryMarker)
	gatePayload["chat_template_kwargs"] = map[string]any{"truncate_history_thinking": false}
	status, body, err := doctorPostJSON(client, strings.TrimSuffix(base, "/")+"/api/chat", gatePayload)
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "gate kwarg probe failed: " + err.Error()}
	}
	gateRejected := status == http.StatusBadRequest && strings.Contains(body, "truncate_history_thinking")

	var parts []string
	statusOut := "ok"
	fixHint := ""

	switch {
	case thinkOK && !reasonOK && !rcOK:
		parts = append(parts, "exactly one preserve shape: thinking (trap 63)")
	case !thinkOK && !reasonOK && !rcOK:
		statusOut = "warn"
		parts = append(parts, "no arm preserved prior reasoning (trap 63/04)")
		fixHint = "resend under thinking with think on; see docs/model-serving-minefield.md"
	default:
		statusOut = "warn"
		parts = append(parts, fmt.Sprintf(
			"preserve map thinking=%v reasoning=%v reasoning_content=%v (want thinking-only)",
			thinkOK, reasonOK, rcOK,
		))
		if fixHint == "" {
			fixHint = "clients must use the native write field; do not port reasoning_content from another stack"
		}
	}

	if gateRejected {
		parts = append(parts, "truncate_history_thinking rejected loudly (not a silent Nemotron preserve)")
	} else if status == http.StatusOK {
		statusOut = "warn"
		parts = append(parts, "truncate_history_thinking accepted — risk of unread preserve kwarg (trap 07/63)")
		fixHint = "unknown chat_template_kwargs should 400; do not treat foreign gates as no-ops"
	} else {
		parts = append(parts, fmt.Sprintf("truncate_history_thinking HTTP %d (expected 400)", status))
	}

	return doctorCheck{
		Name:    name,
		Status:  statusOut,
		Detail:  fmt.Sprintf("%s on %s", strings.Join(parts, "; "), m.Name),
		FixHint: fixHint,
	}
}

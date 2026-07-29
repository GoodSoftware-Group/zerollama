package cmd

import (
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"
)

const doctorHistoryMarker = "ZEROLLAMA_MINEFIELD_RMARK_zzq9"

var doctorEmptyThinkRe = regexp.MustCompile(`(?s)<think>\s*</think>`)

// doctorCheckHistoryAssembly covers minefield traps 04 / 20 / 25 using
// /api/chat _debug_render_only (assembled prompt inspection).
func doctorCheckHistoryAssembly(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-04/20/25 (history render)"
	if !m.SupportsThinking {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise thinking — skipped", m.Name),
		}
	}

	// Trap 04: prior-turn thinking resent under the native write field, then a
	// new user turn. Stock Ollama Go templates often drop .Thinking before the
	// last user index — marker absence is the trap.
	stripped, err := doctorDebugRender(base, map[string]any{
		"model": m.Name,
		"messages": []map[string]any{
			{"role": "user", "content": "Step 1: what next?"},
			{"role": "assistant", "content": "Doing step 1.", "thinking": doctorHistoryMarker + " step1"},
			{"role": "user", "content": "Step 2: what next?"},
			{"role": "assistant", "content": "Doing step 2.", "thinking": doctorHistoryMarker + " step2"},
			{"role": "user", "content": "Step 3: what next?"},
		},
		"stream":             false,
		"think":              true,
		"_debug_render_only": true,
		"options":            map[string]any{"num_predict": 1},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "debug render failed: " + err.Error()}
	}

	emptyShells := len(doctorEmptyThinkRe.FindAllStringIndex(stripped, -1))
	hasMarker := strings.Contains(stripped, doctorHistoryMarker)

	// Trap 20 write-field probe: last message is assistant with thinking vs a
	// wrong-field alias. Last-message .Thinking is what stock templates emit.
	thinkArm, err := doctorDebugRender(base, map[string]any{
		"model": m.Name,
		"messages": []map[string]any{
			{"role": "user", "content": "hi"},
			{"role": "assistant", "content": "ok", "thinking": doctorHistoryMarker},
		},
		"stream":             false,
		"think":              true,
		"_debug_render_only": true,
		"options":            map[string]any{"num_predict": 1},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "write-field thinking arm failed: " + err.Error()}
	}
	reasonArm, err := doctorDebugRender(base, map[string]any{
		"model": m.Name,
		"messages": []map[string]any{
			{"role": "user", "content": "hi"},
			// "reasoning" is not api.Message's write field on /api/chat — must not render.
			{"role": "assistant", "content": "ok", "reasoning": doctorHistoryMarker},
		},
		"stream":             false,
		"think":              true,
		"_debug_render_only": true,
		"options":            map[string]any{"num_predict": 1},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "write-field reasoning arm failed: " + err.Error()}
	}
	thinkOK := strings.Contains(thinkArm, doctorHistoryMarker)
	reasonOK := strings.Contains(reasonArm, doctorHistoryMarker)

	var parts []string
	status := "ok"
	fixHint := ""

	if emptyShells > 0 {
		status = "warn"
		parts = append(parts, fmt.Sprintf("%d empty <think></think> shell(s) (trap 25)", emptyShells))
		fixHint = "resend prior reasoning under thinking, or use a template that skips empty wrappers"
	} else {
		parts = append(parts, "no empty think shells (trap 25)")
	}

	if !hasMarker {
		status = "warn"
		parts = append(parts, "prior-turn thinking marker absent from assembled prompt (trap 04 — template strips history reasoning)")
		if fixHint == "" {
			fixHint = "multi-turn thinking studies need a template that emits prior .Thinking, or measure single-turn only; see docs/model-serving-minefield.md"
		}
	} else {
		parts = append(parts, "prior-turn thinking marker preserved (trap 04)")
	}

	switch {
	case thinkOK && !reasonOK:
		parts = append(parts, "write field is thinking not reasoning (trap 20)")
	case !thinkOK && reasonOK:
		status = "warn"
		parts = append(parts, "write field appears to be reasoning not thinking (trap 20)")
	case thinkOK && reasonOK:
		parts = append(parts, "both thinking and reasoning rendered (unusual)")
	default:
		parts = append(parts, "write-field probe inconclusive (neither field rendered on last assistant)")
	}

	return doctorCheck{
		Name:    name,
		Status:  status,
		Detail:  fmt.Sprintf("%s on %s", strings.Join(parts, "; "), m.Name),
		FixHint: fixHint,
	}
}

func doctorDebugRender(base string, payload map[string]any) (string, error) {
	client := &http.Client{Timeout: 45 * time.Second}
	status, body, err := doctorPostJSON(client, strings.TrimSuffix(base, "/")+"/api/chat", payload)
	if err != nil {
		return "", err
	}
	if status != http.StatusOK {
		return "", fmt.Errorf("HTTP %d: %s", status, truncateDoctorDetail(body, 200))
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(body), &out); err != nil {
		return "", err
	}
	dbg, _ := out["debug_info"].(map[string]any)
	if dbg == nil {
		return "", fmt.Errorf("missing debug_info in response")
	}
	s, _ := dbg["rendered_template"].(string)
	if s == "" {
		return "", fmt.Errorf("empty rendered_template")
	}
	return s, nil
}

package cmd

import (
	"fmt"
	"strings"
)

// doctorCheckThinkToggleInjection covers minefield trap 66 (Ollama mirror):
// stock Qwen templates append " /think" or " /no_think" to the last user turn.
// We still detect injection via _debug_render_only, and assert assistant content
// does not echo a trailing toggle (server strips it).
func doctorCheckThinkToggleInjection(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-66 (think toggle inject)"
	if !m.SupportsThinking {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise thinking — skipped", m.Name),
		}
	}

	rendered, err := doctorDebugRender(base, map[string]any{
		"model": m.Name,
		"messages": []map[string]any{
			{"role": "user", "content": "Reply with exactly: OK"},
		},
		"stream":             false,
		"think":              true,
		"_debug_render_only": true,
		"options":            map[string]any{"num_predict": 1},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "debug render failed: " + err.Error()}
	}
	injected := strings.Contains(rendered, "/think") || strings.Contains(rendered, "/no_think")

	resp, err := doctorChatOnce(base, map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Reply with exactly: OK"},
		},
		"stream":  false,
		"think":   true,
		"options": map[string]any{"temperature": 0, "num_predict": 16},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "chat probe failed: " + err.Error()}
	}
	content := strings.TrimSpace(doctorMessageString(resp, "content"))
	leaked := strings.HasSuffix(strings.ToLower(content), "/think") ||
		strings.HasSuffix(strings.ToLower(content), "/no_think")

	switch {
	case leaked:
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("assistant content ends with think toggle %q (minefield trap 66 mirror)", content),
			FixHint: "strip trailing /think|/no_think from assistant output; see server/chat_sanitize.go",
		}
	case injected:
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("template injects think toggle into last user turn; assistant output cleaned on %s", m.Name),
		}
	default:
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("no /think|/no_think injection observed on %s", m.Name),
		}
	}
}

package cmd

import (
	"fmt"
	"net/http"
	"strings"
	"time"
)

// doctorCheckThinkOffSpill covers minefield trap 126: thinking-off must not
// keep generating a trace and dump it into content (often with a stray </think>).
// Ling's chat_template_kwargs.thinking:false is mapped to Think on this stack;
// native think:false is the same contract.
func doctorCheckThinkOffSpill(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-126 (think-off spill)"
	if !m.SupportsThinking {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise thinking — skipped", m.Name),
		}
	}

	client := &http.Client{Timeout: 15 * time.Second}
	status, body, err := doctorPostJSON(client, strings.TrimSuffix(base, "/")+"/api/chat", map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Say ping"},
		},
		"stream":               false,
		"chat_template_kwargs": map[string]any{"thinking": false},
		"options":              map[string]any{"temperature": 0, "num_predict": 1},
		"_debug_render_only":   true,
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "kwargs thinking:false probe failed: " + err.Error()}
	}
	kwargsOK := status == http.StatusOK
	if status == http.StatusBadRequest && strings.Contains(body, "thinking") {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "chat_template_kwargs.thinking rejected — Ling-style off switch is unread (trap 07/126)",
			FixHint: "map kwargs.thinking to Think like enable_thinking; see api/chat_thinking_aliases.go",
		}
	}

	resp, err := doctorChatOnce(base, map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Say the word ping and nothing else."},
		},
		"stream":  false,
		"think":   false,
		"options": map[string]any{"temperature": 0, "num_predict": 96},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "think=false chat failed: " + err.Error()}
	}
	content := doctorMessageString(resp, "content")
	thinking := doctorMessageString(resp, "thinking")
	if strings.Contains(content, "</think>") || strings.Contains(content, "<think>") {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("think=false spilled think tags into content (trap 126) on %s", m.Name),
			FixHint: "do not treat thinking:false as a latency win; sanitize content or keep think on for free-text",
		}
	}

	parts := []string{"think=false content has no think tags"}
	if kwargsOK {
		parts = append(parts, "kwargs.thinking:false accepted (mapped to Think)")
	}
	if thinking != "" {
		parts = append(parts, "thinking field still populated with think=false — inspect parser")
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: strings.Join(parts, "; ") + " on " + m.Name,
	}
}

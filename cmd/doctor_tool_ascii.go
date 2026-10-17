package cmd

import (
	"fmt"
	"strings"
)

const doctorToolUnicodeMarker = "café日本語"

// doctorCheckToolASCIIEscape covers minefield trap 148: tool JSON
// ensure_ascii=True can expand non-ASCII into \uXXXX and change prompt tokens.
func doctorCheckToolASCIIEscape(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-148 (tool JSON ascii escape)"
	if !m.SupportsTools && !m.HasChatTemplate {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise tools/template — skipped", m.Name),
		}
	}
	render, err := doctorDebugRender(base, map[string]any{
		"model": m.Name,
		"messages": []map[string]any{
			{"role": "user", "content": "use the tool"},
			{
				"role":    "assistant",
				"content": "",
				"tool_calls": []map[string]any{
					{
						"function": map[string]any{
							"name":      "note",
							"arguments": `{"city":"` + doctorToolUnicodeMarker + `"}`,
						},
					},
				},
			},
			{"role": "tool", "content": `{"ok":true}`},
		},
		"stream":             false,
		"think":              false,
		"_debug_render_only": true,
		"options":            map[string]any{"num_predict": 1},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "debug render failed: " + err.Error()}
	}
	if strings.Contains(render, `\u00e9`) || strings.Contains(render, `\u65e5`) {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("tool JSON rendered with \\u escapes (trap 148) on %s", m.Name),
			FixHint: "pin tool serialization; request-body equality is not prompt-token equality",
		}
	}
	if strings.Contains(render, "café") || strings.Contains(render, "日本語") {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: "tool JSON kept non-ASCII verbatim (trap 148) on " + m.Name,
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: "tool unicode not visible in render (inconclusive) on " + m.Name,
	}
}

// doctorCheckThinkPrefixSplit covers trap 166: think on/off (and effort)
// render different prefixes, so they do not share a prefix cache.
func doctorCheckThinkPrefixSplit(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-166 (think switch splits prefix cache)"
	if !m.SupportsThinking {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise thinking — skipped", m.Name),
		}
	}
	payload := func(think bool) map[string]any {
		return map[string]any{
			"model": m.Name,
			"messages": []map[string]any{
				{"role": "user", "content": "Step 1"},
			},
			"stream":             false,
			"think":              think,
			"_debug_render_only": true,
			"options":            map[string]any{"num_predict": 1},
		}
	}
	on, err := doctorDebugRender(base, payload(true))
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "think=true render failed: " + err.Error()}
	}
	off, err := doctorDebugRender(base, payload(false))
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "think=false render failed: " + err.Error()}
	}
	if on == off {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: "think on/off assembled the same prefix (unusual) on " + m.Name,
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("think on/off renders differ (%d vs %d bytes) — each setting has its own prefix cache (trap 166) on %s", len(on), len(off), m.Name),
	}
}

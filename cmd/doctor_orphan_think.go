package cmd

import (
	"fmt"
	"strings"
)

// doctorCheckOrphanedThinkClose covers minefield trap 02: content must not
// start with a stray </think> (parser stripped the open but not the close, or
// the template pre-opened thinking and the model only emitted the closer).
func doctorCheckOrphanedThinkClose(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-02 (orphaned </think>)"
	if !m.SupportsThinking {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise thinking — skipped", m.Name),
		}
	}

	arms := []struct {
		label string
		extra map[string]any
	}{
		{"think=true", map[string]any{"think": true}},
		{"think=false", map[string]any{"think": false}},
		{"think absent", nil},
	}

	var bad []string
	var okArms []string
	for _, arm := range arms {
		payload := map[string]any{
			"model": m.Name,
			"messages": []map[string]string{
				{"role": "user", "content": "Reply with exactly: OK"},
			},
			"stream":  false,
			"options": map[string]any{"temperature": 0, "num_predict": 32},
		}
		for k, v := range arm.extra {
			payload[k] = v
		}
		resp, err := doctorChatOnce(base, payload)
		if err != nil {
			return doctorCheck{Name: name, Status: "warn", Detail: arm.label + " failed: " + err.Error()}
		}
		content := strings.TrimLeft(doctorMessageString(resp, "content"), " \t\n\r")
		if strings.HasPrefix(content, "</think>") {
			bad = append(bad, arm.label)
		} else {
			okArms = append(okArms, arm.label)
		}
	}

	if len(bad) > 0 {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("content starts with </think> on %s (minefield trap 02)", strings.Join(bad, ", ")),
			FixHint: "thinking.Parser strips orphaned closers; rebuild/restart; send think explicitly as a workaround",
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("no leading </think> on %s (%s)", m.Name, strings.Join(okArms, ", ")),
	}
}

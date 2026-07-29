package cmd

import (
	"fmt"
	"net/http"
	"strings"
	"time"
)

// doctorCheckKwargDeadness covers minefield trap 07 for this stack's contract:
// unknown chat_template_kwargs must be rejected (loud), not accepted-and-ignored.
// A paired control without the invented key must still succeed so a 400 is about
// the kwargs, not the model/endpoint.
func doctorCheckKwargDeadness(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-07 (kwarg deadness)"
	client := &http.Client{Timeout: 30 * time.Second}

	control := map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Say OK."},
		},
		"stream":  false,
		"options": map[string]any{"temperature": 0, "num_predict": 8},
	}
	if m.SupportsThinking {
		control["think"] = false
	}
	stCtrl, bodyCtrl, err := doctorPostJSON(client, base+"/api/chat", control)
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "control failed: " + err.Error()}
	}
	if stCtrl != http.StatusOK {
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: fmt.Sprintf("control /api/chat HTTP %d — cannot judge kwargs rejection: %s",
				stCtrl, truncateDoctorDetail(bodyCtrl, 120)),
		}
	}

	probe := map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Say OK."},
		},
		"stream": false,
		"chat_template_kwargs": map[string]any{
			"bogus_kwarg_zzq":   true,
			"reasoning_effort": "high",
		},
		"options": map[string]any{"temperature": 0, "num_predict": 8},
	}
	st, body, err := doctorPostJSON(client, base+"/api/chat", probe)
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "kwargs probe failed: " + err.Error()}
	}

	switch {
	case st == http.StatusBadRequest &&
		(strings.Contains(body, "bogus_kwarg_zzq") || strings.Contains(body, "chat_template_kwargs")):
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("unknown chat_template_kwargs rejected (HTTP 400); control OK on %s (trap 07 loud-dead)", m.Name),
		}
	case st == http.StatusOK:
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "invented chat_template_kwargs.bogus_kwarg_zzq accepted with HTTP 200 (minefield trap 07)",
			FixHint: "reject unknown nested kwargs in api.validateChatTemplateKwargs",
		}
	default:
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: fmt.Sprintf("kwargs probe HTTP %d (want 400 naming bogus_kwarg_zzq): %s",
				st, truncateDoctorDetail(body, 160)),
		}
	}
}

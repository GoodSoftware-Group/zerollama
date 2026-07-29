package cmd

import (
	"fmt"
	"os"
	"strings"
)

// doctorDeepProbes enables slow minefield probes (trap 12 ceiling at 512).
// Set ZEROLLAMA_DOCTOR_DEEP=1 when measuring thinking conversion floors.
func doctorDeepProbes() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_DOCTOR_DEEP")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

// doctorCheckTokenCeiling covers minefield trap 12: hard task + think-on +
// modest max_tokens can return HTTP 200 with empty content and a full reasoning
// channel (honest truncation into thinking). Skipped unless ZEROLLAMA_DOCTOR_DEEP=1.
func doctorCheckTokenCeiling(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-12 (token ceiling)"
	if !m.SupportsThinking {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s does not advertise thinking — skipped", m.Name),
		}
	}
	if !doctorDeepProbes() {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: "skipped (set ZEROLLAMA_DOCTOR_DEEP=1 or run scripts/minefield_ceiling_probe.sh)",
		}
	}

	resp, err := doctorChatOnce(base, map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Write a python function that validates RFC3339 timestamps without external libraries, with tests."},
		},
		"stream":  false,
		"think":   true,
		"options": map[string]any{"temperature": 0, "num_predict": 512},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "ceiling probe failed: " + err.Error()}
	}
	content := strings.TrimSpace(doctorMessageString(resp, "content"))
	thinking := strings.TrimSpace(doctorMessageString(resp, "thinking"))
	doneReason, _ := resp["done_reason"].(string)
	if doneReason == "" {
		doneReason, _ = resp["finish_reason"].(string)
	}

	if content == "" && thinking != "" && (doneReason == "length" || doneReason == "") {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("think=true num_predict=512: empty content, %d chars thinking, done_reason=%q (minefield trap 12)", len(thinking), doneReason),
			FixHint: "raise max_tokens for thinking lanes; do not score empty content as capability collapse — see docs/model-serving-minefield.md",
		}
	}
	if content == "" && thinking == "" {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "ceiling probe returned empty content and empty thinking",
			FixHint: "raise num_predict / inspect think routing",
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("ceiling probe produced content (%d chars) thinking (%d chars) done_reason=%q on %s", len(content), len(thinking), doneReason, m.Name),
	}
}

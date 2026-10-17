package cmd

import (
	"encoding/json"
	"fmt"
	"strings"
)

// doctorCheckThinkOnContent covers minefield trap 162: think-on short answers
// can land entirely in thinking with empty content and finish stop.
func doctorCheckThinkOnContent(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-162 (think-on answer in reasoning)"
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
			{"role": "user", "content": "Multiple choice. The capital of France is: A) Madrid B) Paris C) Rome. Reply with only the letter."},
		},
		"stream":  false,
		"think":   true,
		"options": map[string]any{"temperature": 0, "num_predict": 64},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "probe failed: " + err.Error()}
	}
	content := strings.TrimSpace(doctorMessageString(resp, "content"))
	thinking := strings.TrimSpace(doctorMessageString(resp, "thinking"))
	done := doctorDoneReason(resp)
	if content == "" && thinking != "" && (done == "stop" || done == "") {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("think=true short MCQ: empty content, answer in thinking (%d chars) done_reason=%q (trap 162) on %s", len(thinking), done, m.Name),
			FixHint: "score thinking+content; do not treat empty content + stop as capability collapse",
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("think=true short MCQ content=%d thinking=%d done_reason=%q on %s", len(content), len(thinking), done, m.Name),
	}
}

// doctorCheckThinkBudgetCut covers trap 164: a tiny thinking budget that
// actually cuts reasoning can hide the answer and run to length.
func doctorCheckThinkBudgetCut(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-164 (thinking budget hides answer)"
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
			{"role": "user", "content": "What is 17*19? Show the arithmetic, then the number."},
		},
		"stream":  false,
		"think":   true,
		"options": map[string]any{"temperature": 0, "num_predict": 24},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "probe failed: " + err.Error()}
	}
	content := strings.TrimSpace(doctorMessageString(resp, "content"))
	thinking := strings.TrimSpace(doctorMessageString(resp, "thinking"))
	done := doctorDoneReason(resp)
	if content == "" && thinking != "" && done == "length" {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("num_predict=24 think=true: empty content, thinking cut, done_reason=length (trap 164/12) on %s", m.Name),
			FixHint: "raise the thinking budget; a cap that hits mid-thought removes the answer and can take longer",
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("tiny-budget arm content=%d thinking=%d done_reason=%q on %s", len(content), len(thinking), done, m.Name),
	}
}

// doctorCheckJSONThinkOn covers trap 149: format=json can pass with think off
// and run to the cap / fail parse with think on.
func doctorCheckJSONThinkOn(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-149 (json + think-on runaway)"
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
			"messages": []map[string]string{
				{"role": "user", "content": `Reply with JSON only: {"ok":true}`},
			},
			"stream":  false,
			"think":   think,
			"format":  "json",
			"options": map[string]any{"temperature": 0, "num_predict": 96},
		}
	}
	off, err := doctorChatOnce(base, payload(false))
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "think=false json probe failed: " + err.Error()}
	}
	on, err := doctorChatOnce(base, payload(true))
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "think=true json probe failed: " + err.Error()}
	}
	offOK := jsonLooksClosed(doctorMessageString(off, "content"))
	onContent := strings.TrimSpace(doctorMessageString(on, "content"))
	onOK := jsonLooksClosed(onContent)
	onDone := doctorDoneReason(on)
	if offOK && !onOK && (onDone == "length" || onContent == "" || !strings.Contains(onContent, "}")) {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("format=json passed think=false but think=true failed (done=%q) — trap 149 on %s", onDone, m.Name),
			FixHint: "do not certify structured output from a think-off probe; A/B both modes",
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("format=json think-off parse=%v think-on parse=%v done=%q on %s", offOK, onOK, onDone, m.Name),
	}
}

// doctorCheckEmptyThoughtMarkers covers trap 158: empty <think></think>
// shells can count as reasoning tokens while the field looks empty.
func doctorCheckEmptyThoughtMarkers(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-158 (empty thought markers)"
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
			{"role": "user", "content": "Say ping"},
		},
		"stream":  false,
		"think":   false,
		"options": map[string]any{"temperature": 0, "num_predict": 32},
	})
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "probe failed: " + err.Error()}
	}
	content := doctorMessageString(resp, "content")
	thinking := strings.TrimSpace(doctorMessageString(resp, "thinking"))
	if doctorEmptyThinkRe.MatchString(content) || doctorEmptyThinkRe.MatchString(thinking) {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("think=false still has empty <think></think> shell (trap 158) on %s", m.Name),
			FixHint: "verify thinking-off from semantic reasoning text, not a nonzero reasoning_tokens count",
		}
	}
	if thinking != "" {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "think=false populated thinking field (trap 158/126) on " + m.Name,
			FixHint: "agent speed via think-off still needs a reasoning-quality arm (trap 165)",
		}
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: "think=false has no empty think shells and empty thinking field on " + m.Name,
	}
}

func doctorDoneReason(resp map[string]any) string {
	if s, _ := resp["done_reason"].(string); s != "" {
		return s
	}
	if s, _ := resp["finish_reason"].(string); s != "" {
		return s
	}
	return ""
}

func jsonLooksClosed(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	var v any
	return json.Unmarshal([]byte(s), &v) == nil
}

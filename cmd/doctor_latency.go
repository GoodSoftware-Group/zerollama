package cmd

import (
	"fmt"
	"time"
)

// doctorCheckLatencyReconciliation covers minefield trap 48: client-observed
// latency can be dominated by DNS/IPv6/mDNS while the server finished quickly.
// Compare wall clock around /api/chat against response total_duration before
// blaming samplers or thinking (see mining presence_penalty notes).
func doctorCheckLatencyReconciliation(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-48 (latency reconcile)"
	const maxGap = 2 * time.Second

	payload := map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Reply with exactly: ok"},
		},
		"stream":  false,
		"options": map[string]any{"temperature": 0, "num_predict": 8},
	}
	if m.SupportsThinking {
		payload["think"] = false
	}

	t0 := time.Now()
	resp, err := doctorChatOnce(base, payload)
	client := time.Since(t0)
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "chat probe failed: " + err.Error()}
	}

	server, ok := doctorDurationNS(resp["total_duration"])
	if !ok || server <= 0 {
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: "response missing total_duration — cannot reconcile client vs server latency",
		}
	}

	gap := client - server
	if gap < 0 {
		gap = 0
	}
	detail := fmt.Sprintf("client=%s server_total=%s gap=%s on %s",
		client.Round(time.Millisecond), server.Round(time.Millisecond), gap.Round(time.Millisecond), m.Name)

	if gap > maxGap {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  detail + " (minefield trap 48 — large client-only gap; check DNS/IPv6/mDNS before sampler blame)",
			FixHint: "prefer 127.0.0.1 over .local hostnames; curl -4/-6 compare; see checks/latency_reconciliation.py",
		}
	}
	return doctorCheck{Name: name, Status: "ok", Detail: detail}
}

// doctorDurationNS parses Ollama JSON durations (nanoseconds as JSON number).
func doctorDurationNS(v any) (time.Duration, bool) {
	switch t := v.(type) {
	case float64:
		if t <= 0 {
			return 0, false
		}
		return time.Duration(t), true
	case int64:
		if t <= 0 {
			return 0, false
		}
		return time.Duration(t), true
	case int:
		if t <= 0 {
			return 0, false
		}
		return time.Duration(t), true
	default:
		return 0, false
	}
}

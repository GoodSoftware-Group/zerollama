package cmd

import (
	"fmt"
	"os"
	"strings"
)

// doctorCheckNgramStructured covers minefield trap 138: draftless n-gram
// prompt lookup can duplicate tokens inside structured JSON while HTTP 200.
func doctorCheckNgramStructured() doctorCheck {
	const name = "serving trap-138 (ngram structured dup)"
	spec := strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_SPEC_TYPE")))
	if spec == "" {
		spec = strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_LLAMA_SPEC_TYPE")))
	}
	ngramEnv := envconfigTruthy(os.Getenv("ZEROLLAMA_ELIZA_NGRAM"))
	ngram := ngramEnv || strings.Contains(spec, "ngram")

	if !ngram {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: "ngram speculative env off — trap 138 is a structured-output duplication under prompt-lookup",
		}
	}
	return doctorCheck{
		Name:    name,
		Status:  "warn",
		Detail:  fmt.Sprintf("ngram speculative on (spec=%q eliza_ngram=%v) — JSON content can duplicate keys/tokens at temp 0 (trap 138)", spec, ngramEnv),
		FixHint: "A/B structured JSON with n-gram off; parse assistant content strictly; HTTP 200 is not schema-valid",
	}
}

// doctorCheckHTTPConcurrency covers minefield trap 135: C concurrent HTTP
// clients are not C simultaneous model executions. Quote n_parallel / slots.
func doctorCheckHTTPConcurrency() doctorCheck {
	const name = "serving trap-135 (HTTP ≠ model concurrency)"
	slots, _, profileID := doctorAppleProfileSpecDefaults()
	envSlots := doctorEnvInt("ZEROLLAMA_LLAMA_PARALLEL_SLOTS", 0)
	if envSlots > 0 {
		slots = envSlots
	}
	if slots <= 0 {
		slots = 1
	}
	id := profileID
	if id == "" {
		id = "env/default"
	}
	return doctorCheck{
		Name:   name,
		Status: "ok",
		Detail: fmt.Sprintf("engine slots≈%d (%s) — trap 135: C HTTP clients ≠ C in-flight sequences; quote slots and queue, not client count", slots, id),
	}
}

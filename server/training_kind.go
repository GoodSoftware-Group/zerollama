package server

import (
	"fmt"
	"strings"
)

// normalizeTrainingJobKind maps HTTP kind to the embed job command.
//
// WHY reject unknown kinds (instead of coercing to "train"): silent coerce hid typos
// ("grpoo", "sftt") and made T10 look like SFT. Empty / "sft" → train keeps legacy clients.
//
// WHY "rl" aliases "grpo": operators coming from Unsloth/TRL docs say "RL"; one embed cmd.
func normalizeTrainingJobKind(raw string) (string, error) {
	k := strings.ToLower(strings.TrimSpace(raw))
	switch k {
	case "", "train", "sft":
		return "train", nil
	case "run_script":
		return "run_script", nil
	case "grpo", "rl":
		return "grpo", nil
	default:
		return "", fmt.Errorf("unknown kind %q (want train|run_script|grpo|rl)", raw)
	}
}

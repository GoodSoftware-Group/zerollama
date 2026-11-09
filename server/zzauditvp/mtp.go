package voicepipe

import "strings"

// DuplexEnableMTP returns the EnableMTP pointer for duplex prefix warmup and
// reply. Nil keeps the runner default (dense MLX parks MoE). True unparks MoE
// draft. False turns the head off. qwen35 / qwen35moe always get false when a
// draft head exists. No draft head yields nil.
func DuplexEnableMTP(hasDraft bool, architecture string) *bool {
	if !hasDraft {
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(architecture)) {
	case "qwen35", "qwen35moe":
		v := false
		return &v
	default:
		v := true
		return &v
	}
}

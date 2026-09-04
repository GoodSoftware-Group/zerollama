package model

import (
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MTPSpec is optional in-checkpoint multi-token prediction (GGUF `mtp.*`).
type MTPSpec interface {
	HasMTP() bool
	LastHidden() ml.Tensor
	DraftForward(ctx ml.Context, batch input.Batch, hidden ml.Tensor) (ml.Tensor, error)
	// CausalTrunk is true when no GDN/recurrent trunk layers. Then a rejected
	// speculative token can be dropped from causal KV without rewinding SSM state.
	CausalTrunk() bool
}

// AcceptDraftPrefix is how many leading draft ids match the target's next-token
// predictions. Used by the runner once trunk KV can roll back a rejected tail.
func AcceptDraftPrefix(draft, targetNext []int32) int {
	n := min(len(draft), len(targetNext))
	i := 0
	for i < n && draft[i] == targetNext[i] {
		i++
	}
	return i
}

func ArgmaxLogits(logits []float32) int32 {
	if len(logits) == 0 {
		return -1
	}
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return int32(best)
}

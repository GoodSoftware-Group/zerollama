package ollamarunner

import (
	"log/slog"
	"math"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type mtpPairCache interface {
	SetMTPPair(bool)
}

type kvTailRemover interface {
	RemoveKVTail(seq int, beginIndex int32) error
}

// maybeObserveMTP compares the sampled token to last step's MTP proposal, then
// drafts the next id and, when enabled, appends it for a 2-token verify graph.
// Hybrid GDN runs that pair as two AR steps and checkpoints after the first
// token so a reject can Restore SSM state.
func (s *Server) maybeObserveMTP(seq *Sequence, sampled int32, ctx ml.Context) {
	if seq == nil || ctx == nil || !envconfig.GgmlMTPObserveEnabled() {
		return
	}
	spec, ok := s.model.(model.MTPSpec)
	if !ok || !spec.HasMTP() {
		return
	}

	if seq.mtpSkipDraft {
		seq.mtpSkipDraft = false
		seq.mtpDraft = nil
		return
	}

	if len(seq.mtpDraft) > 0 && !seq.mtpVerify {
		seq.mtpSeen++
		if sampled == seq.mtpDraft[0] {
			seq.mtpHits++
		}
		if seq.mtpSeen%16 == 0 {
			mar := 0.0
			if seq.mtpSeen > 0 {
				mar = float64(seq.mtpHits) / float64(seq.mtpSeen)
			}
			slog.Info("ggml mtp observe", "hits", seq.mtpHits, "seen", seq.mtpSeen, "accept", mar)
		}
	}

	hidden := spec.LastHidden()
	if hidden == nil {
		seq.mtpDraft = nil
		return
	}
	if hidden.Dim(1) > 1 {
		last := int32(hidden.Dim(1) - 1)
		idx := ctx.Input().FromInts([]int32{last}, 1)
		hidden = hidden.Rows(ctx, idx)
	}

	pos := int32(0)
	seqID := 0
	if seq.cache != nil {
		pos = int32(len(seq.cache.Inputs) + len(seq.pendingInputs))
		seqID = seq.cache.Id
	}
	batch := input.Batch{
		Positions: []int32{pos},
		Sequences: []int{seqID},
	}
	batch.Inputs = ctx.Input().FromInts([]int32{sampled}, 1)
	batch.Outputs = ctx.Input().FromInts([]int32{0}, 1)

	out, err := spec.DraftForward(ctx, batch, hidden)
	if err != nil {
		slog.Debug("ggml mtp DraftForward", "error", err)
		seq.mtpDraft = nil
		return
	}
	ctx.Compute(out)
	id := model.ArgmaxLogits(out.Floats())
	if id < 0 {
		seq.mtpDraft = nil
		return
	}
	seq.mtpDraft = []int32{id}

	if seq.cache != nil && len(seq.inputs) == 1 && seq.inputs[0] != nil &&
		int32(len(seq.cache.Inputs)+2) <= s.cache.numCtx {
		seq.inputs[0].SameBatch = 1
		seq.inputs = append(seq.inputs, &input.Input{Token: id})
		seq.mtpVerify = true
		seq.mtpIBatch0 = -1
	}
}

func (s *Server) maybeAcceptMTP(seq *Sequence, outputs []float32, vocabSize int, iBatchLast int) (emit, next int32, ok bool) {
	if seq == nil || !seq.mtpVerify {
		return 0, 0, false
	}
	if seq.mtpIBatch0 < 0 || seq.mtpIBatch0 == iBatchLast || vocabSize <= 0 {
		return 0, 0, false
	}
	seq.mtpVerify = false
	seq.mtpDraft = nil
	logits0 := outputs[seq.mtpIBatch0*vocabSize : (seq.mtpIBatch0+1)*vocabSize]
	sampled, err := seq.sampler.Sample(logits0, recentTokenIDs(seq)...)
	if err != nil {
		return 0, 0, false
	}
	if len(seq.cache.Inputs) == 0 {
		return sampled, sampled, true
	}
	draft := seq.cache.Inputs[len(seq.cache.Inputs)-1].Token
	seq.mtpSeen++
	if sampled != draft {
		s.rollbackMTPDraft(seq)
		seq.mtpSkipDraft = true
		return sampled, sampled, true
	}
	seq.mtpHits++
	logits1 := outputs[iBatchLast*vocabSize : (iBatchLast+1)*vocabSize]
	bonus, err := seq.sampler.Sample(logits1, append(recentTokenIDs(seq), draft)...)
	if err != nil {
		return sampled, sampled, true
	}
	if seq.mtpSeen%16 == 0 {
		slog.Info("ggml mtp accept", "hits", seq.mtpHits, "seen", seq.mtpSeen,
			"accept", float64(seq.mtpHits)/float64(seq.mtpSeen))
	}
	return sampled, bonus, true
}

func (s *Server) rollbackMTPDraft(seq *Sequence) {
	if seq == nil || seq.cache == nil || len(seq.cache.Inputs) == 0 {
		return
	}
	begin := int32(len(seq.cache.Inputs) - 1)
	cache := s.model.Config().Cache
	restored := false
	if cc, ok := cache.(kvcache.CheckpointCache); ok {
		if _, ok := cc.PrepareRestore(seq.cache.Id, begin); ok {
			if err := cache.Remove(seq.cache.Id, begin, math.MaxInt32); err == nil {
				restored = true
			}
		}
	}
	if !restored {
		if kv, ok := cache.(kvTailRemover); ok {
			if err := kv.RemoveKVTail(seq.cache.Id, begin); err != nil {
				slog.Debug("ggml mtp RemoveKVTail", "error", err)
			}
		}
	}
	seq.cache.Inputs = seq.cache.Inputs[:len(seq.cache.Inputs)-1]
}

package mlxrunner

import (
	"log/slog"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
)

// pldDraftSession proposes draft tokens by n-gram lookup in the committed
// stream. Deterministic copies use a one-hot draft distribution so Leviathan
// acceptance matches greedy at temp=0 (q=1 ⇒ acceptP = p).
type pldDraftSession struct {
	ids      []int32
	keyLen   int
	maxDraft int
	skip     bool // prompt-gate; may re-enable if the tail echoes the prompt
	frozen   bool // runtime gate; sticky for the rest of the request
}

func newPLDSession(prompt []int32) *pldDraftSession {
	ids := make([]int32, len(prompt))
	copy(ids, prompt)
	return &pldDraftSession{ids: ids, keyLen: pldKeyLen, maxDraft: pldDraftLen}
}

func (d *pldDraftSession) committed(tokens, _ *mlx.Array, position int) {
	if tokens == nil {
		return
	}
	ids := tokens.Ints()
	for i, id := range ids {
		at := position + i
		want := int32(id)
		switch {
		case at < len(d.ids):
			d.ids[at] = want
		case at == len(d.ids):
			d.ids = append(d.ids, want)
		default:
			// Gap (should not happen if open seeded the prompt). Drop rather
			// than invent tokens; the next propose simply misses.
			return
		}
	}
	d.maybeReenable()
}

func (d *pldDraftSession) settle(next *mlx.Array) {
	if next == nil || next.Size() == 0 {
		return
	}
	id := int32(next.Int())
	if len(d.ids) > 0 && d.ids[len(d.ids)-1] == id {
		return
	}
	d.ids = append(d.ids, id)
	d.maybeReenable()
}

func (d *pldDraftSession) close() {}

func (d *pldDraftSession) inactive() bool { return d.skip || d.frozen }

func (d *pldDraftSession) maybeReenable() {
	if d.frozen || !d.skip {
		return
	}
	if tailMatchFraction(d.ids, pldReenableWindow, pldKeyLen) > pldReenableThresh {
		d.skip = false
		slog.Debug("mlx pld re-enabled", "reason", "tail_echo", "tokens", len(d.ids))
	}
}

func (d *pldDraftSession) propose(current *mlx.Array, maxTokens int) *draftCandidates {
	if d.inactive() || maxTokens <= 0 || d.keyLen <= 0 {
		return nil
	}
	hs := d.haystack(current)
	if len(hs) < d.keyLen {
		return nil
	}
	key := hs[len(hs)-d.keyLen:]
	take := min(maxTokens, d.maxDraft)
	draft := pldFindMatch(hs, key, take)
	if len(draft) == 0 {
		return nil
	}
	return pldOneHotCandidates(draft)
}

func (d *pldDraftSession) haystack(current *mlx.Array) []int32 {
	if current == nil || current.Size() == 0 {
		return d.ids
	}
	id := int32(current.Int())
	if n := len(d.ids); n > 0 && d.ids[n-1] == id {
		return d.ids
	}
	out := make([]int32, len(d.ids)+1)
	copy(out, d.ids)
	out[len(d.ids)] = id
	return out
}

// stackedDrafter tries PLD first (echo/code), then the MTP/assistant head.
// skipPLD is sticky for the rest of the request after a low PLD accept rate.
type stackedDrafter struct {
	pld, mtp                           drafter
	skipPLD                            bool
	usedPLD                            bool
	pldRounds, pldDrafted, pldAccepted int
}

func (d *stackedDrafter) propose(current *mlx.Array, maxTokens int) *draftCandidates {
	d.usedPLD = false
	if !d.skipPLD && d.pld != nil {
		if c := d.pld.propose(current, maxTokens); c != nil {
			d.usedPLD = true
			return c
		}
	}
	if d.mtp != nil {
		return d.mtp.propose(current, maxTokens)
	}
	return nil
}

func (d *stackedDrafter) committed(tokens, hiddens *mlx.Array, position int) {
	if d.pld != nil {
		d.pld.committed(tokens, hiddens, position)
	}
	if d.mtp != nil {
		d.mtp.committed(tokens, hiddens, position)
	}
}

func (d *stackedDrafter) settle(next *mlx.Array) {
	if d.pld != nil {
		d.pld.settle(next)
	}
	if d.mtp != nil {
		d.mtp.settle(next)
	}
}

func (d *stackedDrafter) close() {
	if d.pld != nil {
		d.pld.close()
	}
	if d.mtp != nil {
		d.mtp.close()
	}
}

func pldOneHotCandidates(ids []int32) *draftCandidates {
	n := len(ids)
	tokens := mlx.FromValues(ids, 1, n)
	dists := make([]sampler.Distribution, n)
	one := []float32{1}
	for i, id := range ids {
		dists[i] = sampler.Distribution{
			IDs:   mlx.FromValues([]int32{id}, 1, 1),
			Probs: mlx.FromValues(one, 1, 1),
		}
	}
	return &draftCandidates{tokens: tokens, dist: sampler.ConcatenateDistributions(dists)}
}

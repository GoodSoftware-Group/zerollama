package ollamarunner

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/model/mmradix"
	"github.com/ollama/ollama/runner/common"
	"github.com/ollama/ollama/tokenizer"
)

func (s *Server) score(w http.ResponseWriter, r *http.Request) {
	var req llm.ScoreRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	if len(req.Candidates) == 0 {
		http.Error(w, "candidates required", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.model == nil {
		http.Error(w, "model not loaded", http.StatusServiceUnavailable)
		return
	}

	out := make([]llm.CandidateScore, 0, len(req.Candidates))
	for _, cand := range req.Candidates {
		cs, err := s.scoreCandidateLocked(req.Prompt, cand, req.LengthNormalize, req.IncludeTokenLogprobs)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		out = append(out, cs)
	}

	if err := json.NewEncoder(w).Encode(llm.ScoreResponse{Candidates: out}); err != nil {
		http.Error(w, fmt.Sprintf("encode response: %v", err), http.StatusInternalServerError)
	}
}

func (s *Server) scoreCandidateLocked(prompt, candidate string, lengthNormalize, includeTokens bool) (llm.CandidateScore, error) {
	promptInputs, _, _, _, err := s.inputs(prompt, nil, "", false, false)
	if err != nil {
		return llm.CandidateScore{}, err
	}
	for _, inp := range promptInputs {
		if inp.Multimodal != nil {
			return llm.CandidateScore{}, fmt.Errorf("score does not support vision inputs")
		}
	}

	tok := s.model.(tokenizer.Tokenizer)
	candTokens, err := tok.Encode(candidate, false)
	if err != nil {
		return llm.CandidateScore{}, err
	}
	if len(candTokens) == 0 {
		return llm.CandidateScore{Candidate: candidate}, nil
	}
	if len(promptInputs) == 0 {
		return llm.CandidateScore{}, fmt.Errorf("prompt required to score candidates")
	}

	slot := &s.cache.slots[0]
	slot.InUse = true
	defer func() {
		slot.InUse = false
	}()
	if s.cache.cache != nil {
		if err := s.cache.cache.Remove(slot.Id, 0, math.MaxInt32); err != nil {
			return llm.CandidateScore{}, err
		}
	}
	slot.Inputs = nil

	var logits []float32
	for i, inp := range promptInputs {
		wantLogits := i == len(promptInputs)-1
		logits, err = s.scoreDecodeLocked(slot, mmradix.ClampForEmbed(inp.Token, s.embedVocabSize()), wantLogits)
		if err != nil {
			return llm.CandidateScore{}, err
		}
	}

	var joint float64
	var tokenLPs []llm.TokenLogprob
	for _, token := range candTokens {
		lp := common.LogprobForTokenID(logits, int(token))
		if math.IsInf(lp, -1) {
			return llm.CandidateScore{}, fmt.Errorf("invalid token id %d for candidate %q", token, candidate)
		}
		joint += lp
		if includeTokens {
			piece, _ := tok.Decode([]int32{token})
			tokenLPs = append(tokenLPs, llm.TokenLogprob{Token: piece, Logprob: lp, ID: llm.IntPtr(int(token))})
		}
		logits, err = s.scoreDecodeLocked(slot, token, true)
		if err != nil {
			return llm.CandidateScore{}, err
		}
	}

	cs := llm.CandidateScore{
		Candidate: candidate,
		LogProb:   joint,
		NumTokens: len(candTokens),
		Tokens:    tokenLPs,
	}
	if lengthNormalize && len(candTokens) > 0 {
		cs.LengthNormalizedLogProb = joint / float64(len(candTokens))
	}
	return cs, nil
}

func (s *Server) scoreDecodeLocked(slot *InputCacheSlot, token int32, wantLogits bool) ([]float32, error) {
	ctx := s.model.Backend().NewContext()
	defer ctx.Close()

	pos := int32(len(slot.Inputs))
	batch := input.Batch{
		Positions: []int32{pos},
		Sequences: []int{slot.Id},
	}
	batch.Inputs = ctx.Input().FromInts([]int32{token}, 1)
	if wantLogits {
		batch.Outputs = ctx.Input().FromInts([]int32{0}, 1)
	}

	if cache := s.model.Config().Cache; cache != nil {
		if err := cache.StartForward(ctx, batch, true); err != nil {
			return nil, err
		}
	}

	t, err := model.Forward(ctx, s.model, batch)
	if err != nil {
		return nil, err
	}
	ctx.SetBatchSize(1)
	ctx.Compute(t)

	slot.Inputs = append(slot.Inputs, &input.Input{Token: token})

	if !wantLogits {
		return nil, nil
	}

	outputs := t.Floats()
	if len(outputs) == 0 {
		return nil, fmt.Errorf("no logits returned")
	}
	return outputs, nil
}

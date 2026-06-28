package llamarunner

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/runner/common"
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

	if s.lc == nil {
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
	promptInputs, err := s.inputs(prompt, nil, "", false)
	if err != nil {
		return llm.CandidateScore{}, err
	}
	candTokens, err := s.lc.Model().Tokenize(candidate, false, true)
	if err != nil {
		return llm.CandidateScore{}, err
	}
	if len(candTokens) == 0 {
		return llm.CandidateScore{Candidate: candidate}, nil
	}

	for _, inp := range promptInputs {
		if inp.embed != nil {
			return llm.CandidateScore{}, fmt.Errorf("score does not support vision inputs")
		}
	}

	slot := &s.cache.slots[0]
	slot.InUse = true
	defer func() {
		slot.InUse = false
	}()
	s.lc.KvCacheSeqRm(slot.Id, 0, -1)
	slot.Inputs = nil

	if err := s.decodeInputsLocked(slot, promptInputs); err != nil {
		return llm.CandidateScore{}, err
	}

	var joint float64
	var tokenLPs []llm.TokenLogprob
	var logits []float32
	if len(promptInputs) > 0 {
		logits = s.lc.GetLogitsIth(0)
	} else {
		logits = nil
	}

	for _, tok := range candTokens {
		if logits == nil {
			return llm.CandidateScore{}, fmt.Errorf("prompt required to score candidates")
		}
		lp := common.LogprobForTokenID(logits, tok)
		if math.IsInf(lp, -1) {
			return llm.CandidateScore{}, fmt.Errorf("invalid token id %d for candidate %q", tok, candidate)
		}
		joint += lp
		if includeTokens {
			tokenLPs = append(tokenLPs, llm.TokenLogprob{
				Token:   s.model.TokenToPiece(tok),
				Logprob: lp,
			})
		}
		if err := s.decodeTokenLocked(slot, tok, true); err != nil {
			return llm.CandidateScore{}, err
		}
		logits = s.lc.GetLogitsIth(0)
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

func (s *Server) decodeInputsLocked(slot *InputCacheSlot, inputs []input) error {
	for i, inp := range inputs {
		if inp.embed != nil {
			return fmt.Errorf("score does not support vision inputs")
		}
		wantLogits := i == len(inputs)-1
		if err := s.decodeTokenLocked(slot, inp.token, wantLogits); err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) decodeTokenLocked(slot *InputCacheSlot, token int, wantLogits bool) error {
	batch, err := llama.NewBatch(1, 1, 0)
	if err != nil {
		return err
	}
	defer batch.Free()

	pos := len(slot.Inputs)
	batch.Add(token, nil, pos, wantLogits, slot.Id)
	if err := s.lc.Decode(batch); err != nil {
		return err
	}
	if wantLogits {
		s.lc.Synchronize()
	}
	slot.Inputs = append(slot.Inputs, input{token: token})
	return nil
}

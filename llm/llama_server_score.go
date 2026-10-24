package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"

	"github.com/ollama/ollama/api"
)

const scoreNProbs = 128

var _ Scorer = (*llamaServerRunner)(nil)

func (s *llamaServerRunner) Score(ctx context.Context, req ScoreRequest) (ScoreResponse, error) {
	if ScoreUsesSystemOnePath(req) {
		return s.scoreSystemOne(ctx, req)
	}
	if len(req.Candidates) == 0 {
		return ScoreResponse{}, fmt.Errorf("candidates required")
	}
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return ScoreResponse{}, err
	}
	defer s.sem.Release(1)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return ScoreResponse{}, err
	}
	if status != ServerStatusReady {
		return ScoreResponse{}, fmt.Errorf("unexpected server status: %s", status)
	}

	parseSpecial := true
	addBOS := s.tokenizerAddsBOS()
	promptTokens, err := s.tokenize(ctx, req.Prompt, addBOS, &parseSpecial)
	if err != nil {
		return ScoreResponse{}, err
	}

	const slotID = 0
	out := make([]CandidateScore, 0, len(req.Candidates))
	prefix := append([]int(nil), promptTokens...)

	// Every candidate's first token is predicted from the same prompt, so that
	// distribution is fetched once rather than once per candidate — single-token
	// labels (the LA11 router case) cost one llama-server call in total.
	var firstProbs []llamaServerTokenProb

	for _, cand := range req.Candidates {
		candTokens, err := s.tokenize(ctx, cand, false, &parseSpecial)
		if err != nil {
			return ScoreResponse{}, err
		}
		if len(candTokens) == 0 {
			out = append(out, CandidateScore{Candidate: cand})
			continue
		}

		var joint float64
		var tokenLPs []TokenLogprob
		truncated := false
		ctxPrefix := append([]int(nil), prefix...)

		for k, tok := range candTokens {
			var probs []llamaServerTokenProb
			if k == 0 && firstProbs != nil {
				probs = firstProbs
			} else {
				probs, err = s.nextTokenProbs(ctx, ctxPrefix, slotID)
				if err != nil {
					return ScoreResponse{}, fmt.Errorf("score %q: %w", cand, err)
				}
				if k == 0 {
					firstProbs = probs
				}
			}
			lp, piece, ok := llamaServerLogprobForToken(probs, tok)
			if !ok {
				if !req.AllowMissing {
					return ScoreResponse{}, fmt.Errorf("score %q: target token not in top %d logprobs; try a shorter candidate label", cand, scoreNProbs)
				}
				joint, truncated = math.Inf(-1), true
				break
			}
			joint += lp
			if req.IncludeTokenLogprobs {
				tokenLPs = append(tokenLPs, TokenLogprob{Token: piece, Logprob: lp, ID: IntPtr(tok)})
			}
			ctxPrefix = append(ctxPrefix, tok)
		}

		cs := CandidateScore{
			Candidate: cand,
			LogProb:   joint,
			NumTokens: len(candTokens),
			Tokens:    tokenLPs,
			Truncated: truncated,
		}
		if req.LengthNormalize && len(candTokens) > 0 {
			cs.LengthNormalizedLogProb = joint / float64(len(candTokens))
		}
		out = append(out, cs)
	}

	return ScoreResponse{Candidates: out}, nil
}

// nextTokenProbs returns llama-server's top-scoreNProbs next-token log-probabilities
// after prefix.
func (s *llamaServerRunner) nextTokenProbs(ctx context.Context, prefix []int, slot int) ([]llamaServerTokenProb, error) {
	lsReq := llamaServerCompletionRequest{
		Prompt:      prefix,
		Stream:      false,
		CachePrompt: true,
		IDSlot:      slot,
		NPredict:    1,
		NProbs:      scoreNProbs,
		Temperature: 0,
		// WHY explicit: repeat_penalty has no omitempty, and b11351 llama-server rejects 0
		// ("penalty_repeat must be finite and greater than 0"), which failed every /api/score
		// call and silently sent LA11 score routers to their fallback. 1 = no penalty.
		RepeatPenalty: 1,
	}

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(lsReq); err != nil {
		return nil, err
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, &buf)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	body, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}
	if res.StatusCode >= 400 {
		return nil, api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(body)}
	}

	var lsResp llamaServerCompletionResponse
	if err := json.Unmarshal(body, &lsResp); err != nil {
		return nil, fmt.Errorf("decode score response: %w", err)
	}

	return lsResp.CompletionProbabilities, nil
}

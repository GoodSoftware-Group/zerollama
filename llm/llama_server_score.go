package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/ollama/ollama/api"
)

const scoreNProbs = 128

func (s *llamaServerRunner) Score(ctx context.Context, req ScoreRequest) (ScoreResponse, error) {
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
		ctxPrefix := append([]int(nil), prefix...)

		for _, tok := range candTokens {
			lp, piece, err := s.scoreNextToken(ctx, ctxPrefix, tok, slotID)
			if err != nil {
				return ScoreResponse{}, fmt.Errorf("score %q: %w", cand, err)
			}
			joint += lp
			if req.IncludeTokenLogprobs {
				tokenLPs = append(tokenLPs, TokenLogprob{Token: piece, Logprob: lp})
			}
			ctxPrefix = append(ctxPrefix, tok)
		}

		cs := CandidateScore{
			Candidate: cand,
			LogProb:   joint,
			NumTokens: len(candTokens),
			Tokens:    tokenLPs,
		}
		if req.LengthNormalize && len(candTokens) > 0 {
			cs.LengthNormalizedLogProb = joint / float64(len(candTokens))
		}
		out = append(out, cs)
	}

	return ScoreResponse{Candidates: out}, nil
}

func (s *llamaServerRunner) scoreNextToken(ctx context.Context, prefix []int, target int, slot int) (float64, string, error) {
	lsReq := llamaServerCompletionRequest{
		Prompt:      prefix,
		Stream:      false,
		CachePrompt: true,
		IDSlot:      slot,
		NPredict:    1,
		NProbs:      scoreNProbs,
		Temperature: 0,
	}

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(lsReq); err != nil {
		return 0, "", err
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, &buf)
	if err != nil {
		return 0, "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient().Do(httpReq)
	if err != nil {
		return 0, "", err
	}
	defer res.Body.Close()

	body, err := io.ReadAll(res.Body)
	if err != nil {
		return 0, "", err
	}
	if res.StatusCode >= 400 {
		return 0, "", api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(body)}
	}

	var lsResp llamaServerCompletionResponse
	if err := json.Unmarshal(body, &lsResp); err != nil {
		return 0, "", fmt.Errorf("decode score response: %w", err)
	}

	if lp, piece, ok := llamaServerLogprobForToken(lsResp.CompletionProbabilities, target); ok {
		return lp, piece, nil
	}
	return 0, "", errors.New("target token not in top logprobs; try a shorter candidate label")
}

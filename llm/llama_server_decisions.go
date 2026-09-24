package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	"github.com/ollama/ollama/api"
)

// Decisions implements Decider against llama-server POST /v1/decisions.
// WHY Go packs when possible: LA16 split — public Jev wire + calibration stay in
// Go; C++ is tokenized-only. Prefer PackQuestions + /tokenize; otherwise forward
// high-level {state, questions} (server rejects that shape today and returns 501-
// class error — keep path for future/server-side pack).
func (s *llamaServerRunner) Decisions(ctx context.Context, req DecisionsRequest) (DecisionsResponse, error) {
	if len(req.Questions) == 0 {
		return DecisionsResponse{
			Model:   req.Model,
			Answers: map[string]json.RawMessage{},
		}, nil
	}
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return DecisionsResponse{}, err
	}
	defer s.sem.Release(1)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return DecisionsResponse{}, err
	}
	if status != ServerStatusReady {
		return DecisionsResponse{}, fmt.Errorf("unexpected server status: %s", status)
	}

	body, err := s.decisionsRequestBody(req)
	if err != nil {
		return DecisionsResponse{}, err
	}
	data, err := json.Marshal(body)
	if err != nil {
		return DecisionsResponse{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/decisions", s.port), bytes.NewReader(data))
	if err != nil {
		return DecisionsResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(httpReq)
	if err != nil {
		return DecisionsResponse{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return DecisionsResponse{}, err
	}
	if resp.StatusCode >= 400 {
		return DecisionsResponse{}, api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: string(raw)}
	}

	return decodeDecisionsHTTP(raw, req)
}

func (s *llamaServerRunner) decisionsRequestBody(req DecisionsRequest) (map[string]any, error) {
	tok, err := s.layaTokenizer(context.Background())
	if err != nil {
		return nil, fmt.Errorf("laya tokenizer: %w", err)
	}
	inputs, err := PackQuestions(tok, req.State, req.Questions, defaultLayaMaxLen, defaultLayaHeadMaxLen)
	if err != nil {
		return nil, err
	}
	rows := make([]map[string]any, 0, len(inputs))
	for _, in := range inputs {
		rows = append(rows, map[string]any{
			"tokens":      in.Tokens,
			"marker_pos":  in.MarkerPos,
			"qtype":       in.QType,
			"question_id": in.Question,
		})
	}
	return map[string]any{"inputs": rows}, nil
}

// layaServerTokenizer packs via llama-server /tokenize + GGUF special ids.
type layaServerTokenizer struct {
	s            *llamaServerRunner
	cls, sep, mask int
	maskTok        string
}

func (s *llamaServerRunner) layaTokenizer(ctx context.Context) (LayaTokenizer, error) {
	kv := s.launch.ggufKV
	cls := int(kv.Uint("tokenizer.ggml.cls_token_id", kv.Uint("tokenizer.ggml.bos_token_id", 0)))
	sep := int(kv.Uint("tokenizer.ggml.sep_token_id", kv.Uint("tokenizer.ggml.eos_token_id", 0)))
	mask := int(kv.Uint("tokenizer.ggml.mask_token_id", 0))
	maskTok := "[MASK]"
	if mask == 0 {
		// Fall back: tokenize the literal mask token with specials enabled.
		parse := true
		ids, err := s.tokenize(ctx, maskTok, false, &parse)
		if err != nil || len(ids) != 1 {
			return nil, fmt.Errorf("resolve mask token id (got %v): %w", ids, err)
		}
		mask = ids[0]
	}
	return &layaServerTokenizer{s: s, cls: cls, sep: sep, mask: mask, maskTok: maskTok}, nil
}

func (t *layaServerTokenizer) Encode(text string) ([]int, error) {
	return t.s.Tokenize(context.Background(), text)
}

func (t *layaServerTokenizer) MaskToken() string { return t.maskTok }
func (t *layaServerTokenizer) MaskTokenID() int  { return t.mask }
func (t *layaServerTokenizer) ClsTokenID() int   { return t.cls }
func (t *layaServerTokenizer) SepTokenID() int   { return t.sep }


type decisionsHTTPResult struct {
	QuestionID string    `json:"question_id"`
	QType      int       `json:"qtype"`
	Logits     []float64 `json:"logits"`
	Act        []float64 `json:"act"`
}

func decodeDecisionsHTTP(raw []byte, req DecisionsRequest) (DecisionsResponse, error) {
	var parsed struct {
		Model   string                     `json:"model"`
		Answers map[string]json.RawMessage `json:"answers"`
		Results []decisionsHTTPResult      `json:"results"`
		Usage   struct {
			InputTokens  int `json:"input_tokens"`
			PromptTokens int `json:"prompt_tokens"`
			OutputTokens int `json:"output_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return DecisionsResponse{}, fmt.Errorf("decode decisions: %w", err)
	}

	out := DecisionsResponse{
		Model:   parsed.Model,
		Answers: parsed.Answers,
	}
	if out.Answers == nil {
		out.Answers = map[string]json.RawMessage{}
	}
	out.Usage.InputTokens = parsed.Usage.InputTokens
	if out.Usage.InputTokens == 0 {
		out.Usage.InputTokens = parsed.Usage.PromptTokens
	}
	out.Usage.OutputTokens = parsed.Usage.OutputTokens

	// Prefer already-shaped answers; otherwise calibrate from logits (temps=1.0 stub).
	if len(out.Answers) == 0 && len(parsed.Results) > 0 {
		temps := []float64{1.0, 1.0, 1.0}
		qidOrder := questionIDOrder(req.Questions)
		for i, r := range parsed.Results {
			qid := r.QuestionID
			if qid == "" && i < len(qidOrder) {
				qid = qidOrder[i]
			}
			if qid == "" {
				return DecisionsResponse{}, fmt.Errorf("decisions result %d missing question_id", i)
			}
			q, ok := req.Questions[qid]
			if !ok {
				return DecisionsResponse{}, fmt.Errorf("decisions result unknown question_id %q", qid)
			}
			ans, err := DecodeAnswer(q, r.Logits, r.Act, temps, nil)
			if err != nil {
				return DecisionsResponse{}, fmt.Errorf("decode answer %q: %w", qid, err)
			}
			out.Answers[qid] = ans
		}
	}
	if out.Model == "" {
		out.Model = req.Model
	}
	return out, nil
}

// questionIDOrder returns map keys in an arbitrary but stable-enough order for
// aligning results that omit question_id (Go map iteration). Prefer servers that
// echo question_id.
func questionIDOrder(questions map[string]DecisionQuestion) []string {
	ids := make([]string, 0, len(questions))
	for id := range questions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// PackDecisionsInputs builds tokenized inputs when a tokenizer is available.
// Exported for handlers / tests; llamaServerRunner currently forwards high-level JSON.
func PackDecisionsInputs(tok LayaTokenizer, req DecisionsRequest) ([]LayaPackedInput, error) {
	if tok == nil {
		return nil, fmt.Errorf("tokenizer required")
	}
	return PackQuestions(tok, req.State, req.Questions, defaultLayaMaxLen, defaultLayaHeadMaxLen)
}

// NormalizeDecisionType lowercases and trims a question type string.
func NormalizeDecisionType(t string) string {
	return strings.ToLower(strings.TrimSpace(t))
}

package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

var clmEmbedClient = &http.Client{Timeout: 120 * time.Second}

// CLMAnswer runs native Contrastive-LM inference: embed texts via embURL, project with heads GGUF.
// No Torch / clm-serve. temperature mirrors CLM (divides logits before softmax); default 1.
func CLMAnswer(ctx context.Context, headsPath, embURL, embModel string, req DecisionsRequest, temperature float64) (DecisionsResponse, error) {
	if temperature <= 0 || temperature > 100 {
		temperature = 1
	}
	heads, err := LoadCLMHeads(headsPath)
	if err != nil {
		return DecisionsResponse{}, err
	}
	pairs, ids, err := clmBuildPairs(req.State, req.Questions)
	if err != nil {
		return DecisionsResponse{}, err
	}
	stateTexts := make([]string, 0, len(ids))
	candTexts := make([]string, 0)
	for _, id := range ids {
		p := pairs[id]
		stateTexts = append(stateTexts, p.StateText)
		candTexts = append(candTexts, p.Cands...)
	}
	all := append(append([]string{}, stateTexts...), candTexts...)
	embs, tokens, err := clmEmbed(ctx, embURL, embModel, all)
	if err != nil {
		return DecisionsResponse{}, err
	}
	if len(embs) != len(all) {
		return DecisionsResponse{}, fmt.Errorf("embedder returned %d vectors for %d texts", len(embs), len(all))
	}
	for _, e := range embs {
		if len(e) != heads.Hidden {
			return DecisionsResponse{}, fmt.Errorf("embedding dim %d != clm.hidden_size %d (use Qwen3-8B last-token pooling)", len(e), heads.Hidden)
		}
	}
	zq := heads.ProjectStates(embs[:len(stateTexts)])
	za := heads.ProjectActions(embs[len(stateTexts):])

	answers := make(map[string]json.RawMessage, len(ids))
	k := 0
	for i, id := range ids {
		p := pairs[id]
		n := len(p.Cands)
		logits := make([]float64, n)
		for j := 0; j < n; j++ {
			logits[j] = float64(heads.Scale*clmDot(za[k+j], zq[i])) / temperature
		}
		k += n
		raw, err := clmAnswerFromLogits(req.Questions[id], p.Keys, logits)
		if err != nil {
			return DecisionsResponse{}, fmt.Errorf("question %q: %w", id, err)
		}
		answers[id] = raw
	}
	model := req.Model
	if model == "" {
		model = "clm"
	}
	return DecisionsResponse{
		Model:   model,
		Answers: answers,
		Usage: struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		}{InputTokens: tokens, OutputTokens: 0},
	}, nil
}

func clmEmbed(ctx context.Context, embURL, embModel string, inputs []string) ([][]float32, int, error) {
	embURL = strings.TrimSpace(embURL)
	if embURL == "" {
		return nil, 0, fmt.Errorf("empty embeddings URL")
	}
	if !strings.Contains(embURL, "/v1/embeddings") {
		embURL = strings.TrimSuffix(embURL, "/") + "/v1/embeddings"
	}
	if embModel == "" {
		embModel = "qwen3-8b"
	}
	body, _ := json.Marshal(map[string]any{
		"model": embModel,
		"input": inputs,
	})
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, embURL, bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := clmEmbedClient.Do(httpReq)
	if err != nil {
		return nil, 0, fmt.Errorf("CLM embed: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, 0, fmt.Errorf("CLM embed: upstream %s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	var parsed struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, 0, fmt.Errorf("CLM embed decode: %w", err)
	}
	if len(parsed.Data) == 0 {
		return nil, 0, fmt.Errorf("CLM embed: empty data")
	}
	// Order by index when present
	out := make([][]float32, len(inputs))
	for _, d := range parsed.Data {
		idx := d.Index
		if idx < 0 || idx >= len(out) {
			idx = 0
			for i := range out {
				if out[i] == nil {
					idx = i
					break
				}
			}
		}
		row := make([]float32, len(d.Embedding))
		for i, v := range d.Embedding {
			row[i] = float32(v)
		}
		out[idx] = row
	}
	for i, row := range out {
		if row == nil {
			return nil, 0, fmt.Errorf("CLM embed: missing index %d", i)
		}
	}
	tokens := parsed.Usage.PromptTokens
	if tokens == 0 {
		tokens = parsed.Usage.TotalTokens
	}
	return out, tokens, nil
}

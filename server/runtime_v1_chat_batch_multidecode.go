package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
)

// tryBatchMultidecode silently accelerates Hermes batch when prompts share a long
// prefix. WHY no Hermes client change: same chat.completion.batch wire; probe
// zerollama.capabilities.multidecode. Falls through (returns false) on any miss.
func (s *Server) tryBatchMultidecode(c *gin.Context, modelName string, reqs []any, bodyMap map[string]any) bool {
	caps, _ := zerollamaVersionCapabilities()["multidecode"].(bool)
	if !caps {
		return false
	}
	if len(reqs) < 2 {
		return false
	}

	name, err := getExistingName(model.ParseName(modelName))
	if err != nil {
		return false
	}
	m, err := GetModel(name.String())
	if err != nil || m == nil {
		return false
	}

	var keepAlive *api.Duration
	opts := map[string]any{}
	if o, ok := bodyMap["options"].(map[string]any); ok {
		opts = o
	}

	r, _, _, _, releaseQoS, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, opts, keepAlive, nil, nil, nil)
	if err != nil {
		slog.Debug("multidecode batch: scheduleRunner", "error", err)
		return false
	}
	defer releaseQoS()

	md, ok := r.(llm.MultiDecoder)
	if !ok {
		return false
	}
	tok, ok := r.(interface {
		Tokenize(context.Context, string) ([]int, error)
		Detokenize(context.Context, []int) (string, error)
		ApplyChatTemplate(context.Context, llm.ChatRequest) (string, error)
	})
	if !ok {
		return false
	}

	type leafJob struct {
		tokens    []int
		maxTokens int
		stop      []string
	}
	jobs := make([]leafJob, 0, len(reqs))
	for _, raw := range reqs {
		item, ok := raw.(map[string]any)
		if !ok {
			return false
		}
		// Re-marshal nested request into OpenAI chat shape.
		itemBytes, err := json.Marshal(item)
		if err != nil {
			return false
		}
		var oai openai.ChatCompletionRequest
		if err := json.Unmarshal(itemBytes, &oai); err != nil {
			return false
		}
		if oai.Model == "" {
			oai.Model = modelName
		}
		apiReq, err := openai.FromChatRequest(oai)
		if err != nil {
			slog.Debug("multidecode batch: FromChatRequest", "error", err)
			return false
		}
		prompt, err := tok.ApplyChatTemplate(c.Request.Context(), llm.ChatRequest{
			Messages: apiReq.Messages,
			Tools:    apiReq.Tools,
			Think:    apiReq.Think,
		})
		if err != nil {
			slog.Debug("multidecode batch: template", "error", err)
			return false
		}
		ids, err := tok.Tokenize(c.Request.Context(), prompt)
		if err != nil || len(ids) == 0 {
			return false
		}
		maxTok := 16
		if oai.MaxCompletionTokens != nil && *oai.MaxCompletionTokens > 0 {
			maxTok = *oai.MaxCompletionTokens
		} else if oai.MaxTokens != nil && *oai.MaxTokens > 0 {
			maxTok = *oai.MaxTokens
		}
		var stops []string
		switch v := oai.Stop.(type) {
		case string:
			if v != "" {
				stops = []string{v}
			}
		case []any:
			for _, x := range v {
				if s, ok := x.(string); ok && s != "" {
					stops = append(stops, s)
				}
			}
		case []string:
			stops = v
		}
		jobs = append(jobs, leafJob{tokens: ids, maxTokens: maxTok, stop: stops})
	}

	seqs := make([][]int, len(jobs))
	for i := range jobs {
		seqs[i] = jobs[i].tokens
	}
	lcp := llm.LongestCommonPrefixLen(seqs)
	if lcp < llm.DefaultMultiDecodeLCPThreshold {
		return false
	}
	forest, err := llm.PackForestFromTokenLists(seqs)
	if err != nil {
		return false
	}

	clear := true
	resp, err := md.MultiDecode(c.Request.Context(), llm.MultiDecodeRequest{
		Tokens: forest.Tokens,
		Pos:    forest.Pos,
		Parent: forest.Parent,
		Leaves: forest.Leaves,
		Clear:  &clear,
	})
	if err != nil {
		slog.Debug("multidecode batch: first decode", "error", err)
		return false
	}

	// Map leaf index in forest → generated token stream
	nLeaves := len(forest.Leaves)
	gen := make([][]int32, nLeaves)
	active := make([]bool, nLeaves)
	leafNode := make([]int32, nLeaves)
	maxSteps := 0
	for i := range jobs {
		if jobs[i].maxTokens > maxSteps {
			maxSteps = jobs[i].maxTokens
		}
		active[i] = true
	}
	if maxSteps <= 0 {
		maxSteps = 16
	}

	// Seed with first greedy token per leaf. Result node_id is the forest leaf
	// (parent for writing the sampled token on the next sticky step).
	byLeaf := map[int32]llm.MultiDecodeLeafResult{}
	for _, row := range resp.Results {
		byLeaf[row.Leaf] = row
	}
	for i, li := range forest.Leaves {
		row, ok := byLeaf[li]
		if !ok {
			return false
		}
		gen[i] = []int32{row.Token}
		leafNode[i] = row.NodeID
	}

	for step := 1; step < maxSteps; step++ {
		var stepTokens, stepPos, stepParents, stepLeaves []int32
		var leafMap []int // index into jobs for each new batch row
		for i := range jobs {
			if !active[i] {
				continue
			}
			if len(gen[i]) >= jobs[i].maxTokens {
				active[i] = false
				continue
			}
			if len(jobs[i].stop) > 0 {
				text, err := tok.Detokenize(c.Request.Context(), int32SliceToInt(gen[i]))
				if err == nil && stopHit(text, jobs[i].stop) {
					active[i] = false
					continue
				}
			}
			idx := int32(len(stepTokens))
			// Write previously sampled token under leafNode; predict the next.
			stepTokens = append(stepTokens, gen[i][len(gen[i])-1])
			stepPos = append(stepPos, int32(len(seqs[i])+len(gen[i])-1))
			stepParents = append(stepParents, leafNode[i])
			stepLeaves = append(stepLeaves, idx)
			leafMap = append(leafMap, i)
		}
		if len(stepTokens) == 0 {
			break
		}
		clearFalse := false
		stepResp, err := md.MultiDecode(c.Request.Context(), llm.MultiDecodeRequest{
			Tokens:        stepTokens,
			Pos:           stepPos,
			ParentNodeIDs: stepParents,
			Leaves:        stepLeaves,
			Clear:         &clearFalse,
		})
		if err != nil {
			slog.Debug("multidecode batch: step decode", "step", step, "error", err)
			return false
		}
		for j, row := range stepResp.Results {
			i := leafMap[j]
			gen[i] = append(gen[i], row.Token)
			leafNode[i] = row.NodeID
		}
	}

	completions := make([]map[string]any, len(jobs))
	for i := range jobs {
		text, err := tok.Detokenize(c.Request.Context(), int32SliceToInt(gen[i]))
		if err != nil {
			return false
		}
		text = trimStops(text, jobs[i].stop)
		completions[i] = map[string]any{
			"id":      fmt.Sprintf("chatcmpl-md-%d", i),
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   modelName,
			"choices": []map[string]any{{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": text,
				},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{
				"prompt_tokens":     len(seqs[i]),
				"completion_tokens": len(gen[i]),
				"total_tokens":      len(seqs[i]) + len(gen[i]),
			},
		}
	}

	out := map[string]any{
		"object":      "chat.completion.batch",
		"model":       modelName,
		"count":       len(completions),
		"completions": completions,
	}
	c.JSON(http.StatusOK, out)
	c.Abort()
	return true
}

func int32SliceToInt(in []int32) []int {
	out := make([]int, len(in))
	for i, v := range in {
		out[i] = int(v)
	}
	return out
}

func stopHit(text string, stops []string) bool {
	for _, s := range stops {
		if s != "" && strings.Contains(text, s) {
			return true
		}
	}
	return false
}

func trimStops(text string, stops []string) string {
	for _, s := range stops {
		if s == "" {
			continue
		}
		if i := strings.Index(text, s); i >= 0 {
			return text[:i]
		}
	}
	return text
}

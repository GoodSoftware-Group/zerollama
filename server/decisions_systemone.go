package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

func modelUsesSystemOneScore(m *Model) bool {
	if m == nil || !capabilityContains(m.Config.Capabilities, model.CapabilityDecision) {
		return false
	}
	if isLayaDecisionModel(m) {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(m.Config.Renderer)) {
	case "tev1", "clef", "strands", "qwen3.5", "qwen35", "qwen3.8", "qwen38":
		// Nimble/Tev1 create uses qwen3.5 renderer + decision capability.
		return true
	}
	fam := strings.ToLower(m.Config.ModelFamily)
	if strings.Contains(fam, "clef") || strings.Contains(fam, "tev1") ||
		strings.Contains(fam, "nimble") || strings.Contains(fam, "strands") {
		return true
	}
	for _, f := range m.Config.ModelFamilies {
		fl := strings.ToLower(f)
		if strings.Contains(fl, "clef") || strings.Contains(fl, "tev1") ||
			strings.Contains(fl, "nimble") || strings.Contains(fl, "strands") {
			return true
		}
	}
	// Bare CAPABILITY decision without a score-capable renderer/family must not
	// steal the Laya Decider path — schedule Decider (or 501) instead.
	return false
}

func isLayaDecisionModel(m *Model) bool {
	if m == nil {
		return false
	}
	if strings.EqualFold(m.Config.ModelFamily, "laya") {
		return true
	}
	for _, arch := range m.Config.ModelFamilies {
		if strings.EqualFold(arch, "LayaForDecision") || strings.EqualFold(arch, "laya") {
			return true
		}
	}
	return strings.EqualFold(m.Config.Renderer, "laya")
}

func capabilityContains(caps []string, c model.Capability) bool {
	return slices.Contains(caps, string(c))
}

func decisionScoreEncoding(m *Model) string {
	if m == nil {
		return ""
	}
	switch r := strings.ToLower(strings.TrimSpace(m.Config.Renderer)); r {
	case "tev1", "clef", "strands":
		return r
	}
	fam := strings.ToLower(m.Config.ModelFamily)
	if strings.Contains(fam, "clef") {
		return "clef"
	}
	if strings.Contains(fam, "strands") {
		return "strands"
	}
	if strings.Contains(fam, "tev1") || strings.Contains(fam, "nimble") {
		return "tev1"
	}
	return ""
}

func (s *Server) serveDecisionsSystemOneScore(c *gin.Context, m *Model, name string, dreq decision.Request, requestOpts map[string]any, keepAlive *api.Duration) {
	encoding := decisionScoreEncoding(m)
	compiled, err := decision.Compile(dreq, encoding)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	caps := []model.Capability{model.CapabilityDecision}
	if len(dreq.Images) > 0 {
		caps = append(caps, model.CapabilityVision)
	}

	r, _, _, _, releaseQoS, err := s.scheduleRunner(c.Request.Context(), name, caps, requestOpts, keepAlive, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, dreq.Model, err)
		return
	}
	defer releaseQoS()

	scorer, ok := r.(llm.Scorer)
	if !ok {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("model %q does not support System One scoring; use a decision-capable runner (tev1/clef/nimble) or Laya GGUF", dreq.Model)})
		return
	}

	if err := compiled.Render(func(messages []api.Message) (string, error) {
		if m.System != "" {
			messages = append([]api.Message{{Role: "system", Content: m.System}}, messages...)
		}
		think := &api.ThinkValue{Value: false}
		if m.HasChatTemplate && chatModeForModel(m) == chatExecutionModeNative {
			return r.ApplyChatTemplate(c.Request.Context(), llm.ChatRequest{Messages: messages, Think: think})
		}
		return renderPrompt(m, messages, nil, think)
	}); err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	compiled.Request.MaxTokens = r.ContextLength()
	result, err := scorer.Score(c.Request.Context(), compiled.Request)
	if err != nil {
		status := http.StatusInternalServerError
		var statusErr api.StatusError
		if errors.As(err, &statusErr) {
			status = statusErr.StatusCode
		}
		c.AbortWithStatusJSON(status, gin.H{"error": err.Error()})
		return
	}

	response, err := compiled.Answer(dreq.Model, result)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	out, err := decisionResponseToAPI(response)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if out.Model == "" {
		out.Model = name
	}
	c.JSON(http.StatusOK, out)
}

func decisionResponseToAPI(resp decision.Response) (api.DecisionsResponse, error) {
	out := api.DecisionsResponse{
		Model:   resp.Model,
		Answers: map[string]json.RawMessage{},
	}
	out.Usage.InputTokens = resp.Usage.InputTokens
	out.Usage.OutputTokens = resp.Usage.OutputTokens
	if resp.Answers == nil {
		return out, nil
	}
	for id, ans := range resp.Answers.All() {
		raw, err := json.Marshal(ans)
		if err != nil {
			return api.DecisionsResponse{}, fmt.Errorf("answer %q: %w", id, err)
		}
		out.Answers[id] = raw
	}
	return out, nil
}

func unmarshalDecisionRequest(body []byte, apiReq api.DecisionsRequest) (decision.Request, error) {
	req := decision.Request{Questions: &decision.Questions{}}
	if err := json.Unmarshal(body, &req); err != nil {
		return decision.Request{}, err
	}
	if req.Questions == nil || req.Questions.Len() == 0 {
		return decision.Request{}, fmt.Errorf("questions required")
	}
	if strings.TrimSpace(req.Model) == "" {
		req.Model = apiReq.Model
	}
	if req.KeepAlive == nil {
		req.KeepAlive = apiReq.KeepAlive
	}
	if len(req.Images) == 0 {
		req.Images = apiReq.Images
	}
	return req, nil
}

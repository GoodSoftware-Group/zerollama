package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

func modelUsesSystemOneScore(m *Model) bool {
	if m == nil {
		return false
	}
	hasDecision := capabilityContains(m.Config.Capabilities, model.CapabilityDecision)
	if !hasDecision {
		// Inferred from GGUF {arch}.decision.type (see Model.Capabilities).
		for _, c := range m.Capabilities() {
			if c == model.CapabilityDecision {
				hasDecision = true
				break
			}
		}
	}
	if !hasDecision {
		return false
	}
	if isLayaDecisionModel(m) {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(m.Config.Renderer)) {
	case "tev1", "clef", "strands", "d3", "qwen3.5", "qwen35", "qwen3.8", "qwen38":
		// Gallery / create tags often use qwen3.5 or d3 renderer + decision capability.
		return true
	}
	fam := strings.ToLower(m.Config.ModelFamily)
	if systemOneFamilyHint(fam) {
		return true
	}
	for _, f := range m.Config.ModelFamilies {
		if systemOneFamilyHint(strings.ToLower(f)) {
			return true
		}
	}
	// Gallery GGUFs set {arch}.decision.type; RENDERER may be omitted (nimble/kev/…).
	if encoding := systemOneEncodingFromDecisionType(ggufDecisionType(m)); encoding != "" {
		return true
	}
	// Bare CAPABILITY decision without a score-capable renderer/family must not
	// steal the Laya Decider path — schedule Decider (or 501) instead.
	return false
}

// systemOneFamilyHint matches Modelfile / config family strings for score-path models.
func systemOneFamilyHint(fam string) bool {
	for _, needle := range []string{"clef", "tev1", "nimble", "strands", "kev", "pplx", "d3"} {
		if strings.Contains(fam, needle) {
			return true
		}
	}
	return false
}

// systemOneEncodingFromDecisionType maps GGUF {arch}.decision.type → Compile encoding.
// Empty means "not a System One score model" (e.g. laya uses the Decider path).
func systemOneEncodingFromDecisionType(dtype string) string {
	switch strings.ToLower(strings.TrimSpace(dtype)) {
	case "clef":
		return "clef"
	case "strands":
		return "strands"
	case "tev1", "nimble", "kev", "pplx-decider", "vllm-d3", "d3":
		// Rows / shared-prefix score encodings share the tev1 prompt compiler.
		return "tev1"
	default:
		return ""
	}
}

func ggufDecisionType(m *Model) string {
	if m == nil || m.ModelPath == "" {
		return ""
	}
	f, err := gguf.Open(m.ModelPath)
	if err != nil {
		return ""
	}
	defer f.Close()
	return strings.ToLower(strings.TrimSpace(f.KeyValue("decision.type").String()))
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
	case "d3":
		return "tev1"
	}
	fam := strings.ToLower(m.Config.ModelFamily)
	if strings.Contains(fam, "clef") {
		return "clef"
	}
	if strings.Contains(fam, "strands") {
		return "strands"
	}
	if strings.Contains(fam, "tev1") || strings.Contains(fam, "nimble") ||
		strings.Contains(fam, "kev") || strings.Contains(fam, "pplx") || strings.Contains(fam, "d3") {
		return "tev1"
	}
	if enc := systemOneEncodingFromDecisionType(ggufDecisionType(m)); enc != "" {
		return enc
	}
	return ""
}

func (s *Server) decisionsSystemOneScore(ctx context.Context, m *Model, name string, dreq decision.Request, requestOpts map[string]any, keepAlive *api.Duration) (api.DecisionsResponse, error) {
	encoding := decisionScoreEncoding(m)
	compiled, err := decision.Compile(dreq, encoding)
	if err != nil {
		return api.DecisionsResponse{}, decisionsErr(http.StatusBadRequest, err)
	}

	caps := []model.Capability{model.CapabilityDecision}
	if len(dreq.Images) > 0 {
		caps = append(caps, model.CapabilityVision)
	}

	r, _, _, _, releaseQoS, err := s.scheduleRunner(ctx, name, caps, requestOpts, keepAlive, nil, nil, nil)
	if err != nil {
		return api.DecisionsResponse{}, decisionsScheduleErr(dreq.Model, err)
	}
	defer releaseQoS()

	scorer, ok := r.(llm.Scorer)
	if !ok {
		return api.DecisionsResponse{}, decisionsErr(http.StatusBadRequest, fmt.Errorf("model %q does not support System One scoring; use a decision-capable runner (tev1/clef/nimble) or Laya GGUF", dreq.Model))
	}

	if err := compiled.Render(func(messages []api.Message) (string, error) {
		if m.System != "" {
			messages = append([]api.Message{{Role: "system", Content: m.System}}, messages...)
		}
		think := &api.ThinkValue{Value: false}
		if m.HasChatTemplate && chatModeForModel(m) == chatExecutionModeNative {
			return r.ApplyChatTemplate(ctx, llm.ChatRequest{Messages: messages, Think: think})
		}
		return renderPrompt(m, messages, nil, think)
	}); err != nil {
		return api.DecisionsResponse{}, decisionsErr(http.StatusInternalServerError, err)
	}

	compiled.Request.MaxTokens = r.ContextLength()
	result, err := scorer.Score(ctx, compiled.Request)
	if err != nil {
		status := http.StatusInternalServerError
		var statusErr api.StatusError
		if errors.As(err, &statusErr) {
			status = statusErr.StatusCode
		}
		return api.DecisionsResponse{}, decisionsErr(status, err)
	}

	response, err := compiled.Answer(dreq.Model, result)
	if err != nil {
		return api.DecisionsResponse{}, decisionsErr(http.StatusInternalServerError, err)
	}

	out, err := decisionResponseToAPI(response)
	if err != nil {
		return api.DecisionsResponse{}, decisionsErr(http.StatusInternalServerError, err)
	}
	if out.Model == "" {
		out.Model = name
	}
	return out, nil
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

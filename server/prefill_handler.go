package server

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/types/model"
)

// PrefillHandler serves POST /api/prefill and POST /v1/prefill.
//
// The last message is cut before its last word and evaluated with the prompt
// cache on and sampling off. The draft is rendered with the same chat template
// /api/chat uses, then truncated at U+F8FF so the cached text is a prefix of
// the finished request. Sampling stays off (llama-server prefill_only).
//
// WHY chatPrompt, not llama-server /v1/prefill: /api/chat renders here. The
// server's own template would cache a different string, and the next chat
// would miss.
func (s *Server) PrefillHandler(c *gin.Context) {
	var req api.PrefillRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if strings.TrimSpace(req.Model) == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	if len(req.Messages) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "prefill requires a messages array"})
		return
	}
	last := req.Messages[len(req.Messages)-1]
	if strings.TrimSpace(last.Content) == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "prefill needs text in the last message"})
		return
	}
	for _, msg := range req.Messages {
		if len(msg.Images) > 0 || len(msg.AudioClips) > 0 || len(msg.Videos) > 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "prefill does not cache prompts with media"})
			return
		}
	}

	if served, err := applyModelAlias(c, req.Model); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	} else {
		req.Model = served
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.AbortWithStatusJSON(http.StatusNotImplemented, gin.H{"error": "prefill is not available for cloud models"})
		return
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}
	if msg := textRouteMessage(name.String()); msg != "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": msg})
		return
	}

	r, m, opts, _, releaseQoS, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{model.CapabilityCompletion}, req.Options, req.KeepAlive, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	defer releaseQoS()

	if m == nil || m.IsMLX() {
		c.AbortWithStatusJSON(http.StatusNotImplemented, gin.H{"error": "prefill is not available for this model"})
		return
	}
	pref, ok := r.(llm.PromptPrefiller)
	if !ok {
		c.AbortWithStatusJSON(http.StatusNotImplemented, gin.H{"error": "model runner does not support prefill"})
		return
	}

	result, err := prefillRendered(c, r, pref, m, opts, req)
	if err != nil {
		var se api.StatusError
		if errors.As(err, &se) {
			c.AbortWithStatusJSON(se.StatusCode, gin.H{"error": se.ErrorMessage})
			return
		}
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, api.PrefillResponse{
		Object:       "prefill",
		CachedTokens: result.CachedTokens,
		PromptTokens: result.PromptTokens,
	})
}

func prefillRendered(c *gin.Context, runner llm.LlamaServer, pref llm.PromptPrefiller, m *Model, opts *api.Options, req api.PrefillRequest) (llm.PrefillResult, error) {
	msgs := make([]api.Message, 0, len(m.Messages)+len(req.Messages)+1)
	msgs = append(msgs, m.Messages...)
	msgs = append(msgs, req.Messages...)
	if req.Messages[0].Role != "system" && m.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: m.System}}, msgs...)
	}
	msgs = filterThinkTags(msgs, m)
	msgs = preservePriorThinkingForRender(msgs, req.Think)
	if len(msgs) == 0 {
		return llm.PrefillResult{}, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "prefill needs text in the last message"}
	}

	tools := []api.Tool(req.Tools)
	if parserName := resolveParserName(m); parserName != "" {
		if builtin := parsers.ParserForName(parserName); builtin != nil {
			last := msgs[len(msgs)-1]
			tools = builtin.Init(req.Tools, &last, req.Think)
		}
	}

	last := msgs[len(msgs)-1]
	cut := llm.PrefillBeforeLastWord(last.Content)
	if cut == "" {
		return llm.PrefillResult{}, nil
	}
	last.Content = cut + llm.PrefillSentinel
	msgs[len(msgs)-1] = last

	tokenBudget, detok := chatPromptLimits(m, opts, true, runner.ContextLength(), runner.Detokenize)
	prompt, images, _, _, _, err := chatPrompt(c.Request.Context(), m, runner.Tokenize, opts, msgs, tools, req.Think, true, tokenBudget, detok)
	if err != nil {
		return llm.PrefillResult{}, err
	}
	if len(images) > 0 {
		return llm.PrefillResult{}, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "prefill does not cache prompts with media"}
	}
	end := strings.LastIndex(prompt, llm.PrefillSentinel)
	if end < 0 {
		return llm.PrefillResult{}, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "the chat template dropped the draft"}
	}
	prompt = prompt[:end]
	if prompt == "" {
		return llm.PrefillResult{}, nil
	}
	return pref.PrefillPrompt(c.Request.Context(), prompt, opts)
}

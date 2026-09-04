package openai

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// ResponsesContent is a discriminated union for input content types.
// Concrete types: ResponsesTextContent, ResponsesImageContent,
// ResponsesOutputTextContent, ResponsesFileContent.
type ResponsesContent interface {
	responsesContent() // unexported marker method
}

type ResponsesTextContent struct {
	Type string `json:"type"` // always "input_text"
	Text string `json:"text"`
}

func (ResponsesTextContent) responsesContent() {}

type ResponsesImageContent struct {
	Type string `json:"type"` // always "input_image"
	// TODO(drifkin): is this really required? that seems verbose and a default is specified in the docs
	Detail   string `json:"detail"`              // required
	FileID   string `json:"file_id,omitempty"`   // optional
	ImageURL string `json:"image_url,omitempty"` // optional
}

func (ResponsesImageContent) responsesContent() {}

// ResponsesOutputTextContent represents output text from a previous assistant response
// that is being passed back as part of the conversation history.
type ResponsesOutputTextContent struct {
	Type string `json:"type"` // always "output_text"
	Text string `json:"text"`
}

func (ResponsesOutputTextContent) responsesContent() {}

type ResponsesFileContent struct {
	Type     string `json:"type"` // always "input_file"
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
	Filename string `json:"filename,omitempty"`
}

func (ResponsesFileContent) responsesContent() {}

type ResponsesInputMessage struct {
	Type    string             `json:"type"` // always "message"
	Role    string             `json:"role"` // one of `user`, `system`, `developer`
	Content []ResponsesContent `json:"content,omitempty"`
}

func (m *ResponsesInputMessage) UnmarshalJSON(data []byte) error {
	var aux struct {
		Type    string          `json:"type"`
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	m.Type = aux.Type
	m.Role = aux.Role

	if len(aux.Content) == 0 {
		return nil
	}

	// Try to parse content as a string first (shorthand format)
	var contentStr string
	if err := json.Unmarshal(aux.Content, &contentStr); err == nil {
		m.Content = []ResponsesContent{
			ResponsesTextContent{Type: "input_text", Text: contentStr},
		}
		return nil
	}

	// Otherwise, parse as an array of content items
	var rawItems []json.RawMessage
	if err := json.Unmarshal(aux.Content, &rawItems); err != nil {
		return fmt.Errorf("content must be a string or array: %w", err)
	}

	m.Content = make([]ResponsesContent, 0, len(rawItems))
	for i, raw := range rawItems {
		content, err := unmarshalResponsesContent(raw)
		if err != nil {
			return fmt.Errorf("content[%d]: %w", i, err)
		}
		m.Content = append(m.Content, content)
	}

	return nil
}

func unmarshalResponsesContent(data []byte) (ResponsesContent, error) {
	// Peek at the type field to determine which concrete type to use
	var typeField struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &typeField); err != nil {
		return nil, err
	}

	switch typeField.Type {
	case "input_text":
		var content ResponsesTextContent
		if err := json.Unmarshal(data, &content); err != nil {
			return nil, err
		}
		return content, nil
	case "input_image":
		var content ResponsesImageContent
		if err := json.Unmarshal(data, &content); err != nil {
			return nil, err
		}
		return content, nil
	case "output_text":
		var content ResponsesOutputTextContent
		if err := json.Unmarshal(data, &content); err != nil {
			return nil, err
		}
		return content, nil
	case "input_file":
		var content ResponsesFileContent
		if err := json.Unmarshal(data, &content); err != nil {
			return nil, err
		}
		return content, nil
	default:
		return nil, fmt.Errorf("unknown content type: %s", typeField.Type)
	}
}

type ResponsesOutputMessage struct{}

// ResponsesInputItem is a discriminated union for input items.
// Concrete types: ResponsesInputMessage (more to come)
type ResponsesInputItem interface {
	responsesInputItem() // unexported marker method
}

func (ResponsesInputMessage) responsesInputItem() {}

// ResponsesFunctionCall represents an assistant's function call in conversation history.
type ResponsesFunctionCall struct {
	ID        string `json:"id,omitempty"` // item ID
	Type      string `json:"type"`         // always "function_call"
	CallID    string `json:"call_id"`      // the tool call ID
	Name      string `json:"name"`         // function name
	Arguments string `json:"arguments"`    // JSON arguments string
}

func (ResponsesFunctionCall) responsesInputItem() {}

// ResponsesFunctionCallOutput represents a function call result from the client.
type ResponsesFunctionCallOutput struct {
	Type   string `json:"type"`    // always "function_call_output"
	CallID string `json:"call_id"` // links to the original function call
	Output string `json:"output"`  // the function result

	// OutputItems is populated when output is provided as Responses content
	// items instead of the string shorthand.
	OutputItems []ResponsesContent `json:"-"`
}

func (o *ResponsesFunctionCallOutput) UnmarshalJSON(data []byte) error {
	var aux struct {
		Type   string          `json:"type"`
		CallID string          `json:"call_id"`
		Output json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	o.Type = aux.Type
	o.CallID = aux.CallID
	o.Output = ""
	o.OutputItems = nil

	if len(aux.Output) == 0 {
		return nil
	}

	var output string
	if err := json.Unmarshal(aux.Output, &output); err == nil {
		o.Output = output
		return nil
	}

	var rawItems []json.RawMessage
	if err := json.Unmarshal(aux.Output, &rawItems); err != nil {
		return fmt.Errorf("output must be a string or array: %w", err)
	}

	o.OutputItems = make([]ResponsesContent, 0, len(rawItems))
	var outputText strings.Builder
	for i, raw := range rawItems {
		content, err := unmarshalResponsesContent(raw)
		if err != nil {
			return fmt.Errorf("output[%d]: %w", i, err)
		}
		o.OutputItems = append(o.OutputItems, content)

		switch v := content.(type) {
		case ResponsesTextContent:
			outputText.WriteString(v.Text)
		case ResponsesOutputTextContent:
			outputText.WriteString(v.Text)
		}
	}
	o.Output = outputText.String()
	return nil
}

func (ResponsesFunctionCallOutput) responsesInputItem() {}

// ResponsesReasoningInput represents a reasoning item passed back as input.
// This is used when the client sends previous reasoning back for context.
type ResponsesReasoningInput struct {
	ID               string                      `json:"id,omitempty"`
	Type             string                      `json:"type"` // always "reasoning"
	Summary          []ResponsesReasoningSummary `json:"summary,omitempty"`
	EncryptedContent string                      `json:"encrypted_content,omitempty"`
}

func (ResponsesReasoningInput) responsesInputItem() {}

// unmarshalResponsesInputItem unmarshals a single input item from JSON.
func unmarshalResponsesInputItem(data []byte) (ResponsesInputItem, error) {
	var typeField struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if err := json.Unmarshal(data, &typeField); err != nil {
		return nil, err
	}

	// Handle shorthand message format: {"role": "...", "content": "..."}
	// When type is empty but role is present, treat as a message
	itemType := typeField.Type
	if itemType == "" && typeField.Role != "" {
		itemType = "message"
	}

	switch itemType {
	case "message":
		var msg ResponsesInputMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			return nil, err
		}
		return msg, nil
	case "function_call":
		var fc ResponsesFunctionCall
		if err := json.Unmarshal(data, &fc); err != nil {
			return nil, err
		}
		return fc, nil
	case "function_call_output":
		var output ResponsesFunctionCallOutput
		if err := json.Unmarshal(data, &output); err != nil {
			return nil, err
		}
		return output, nil
	case "reasoning":
		var reasoning ResponsesReasoningInput
		if err := json.Unmarshal(data, &reasoning); err != nil {
			return nil, err
		}
		return reasoning, nil
	default:
		if itemType == "" {
			return nil, fmt.Errorf("input item missing required 'type' field")
		}
		return nil, fmt.Errorf("unknown input item type: %q", itemType)
	}
}

// ResponsesInput can be either:
// - a string (equivalent to a text input with the user role)
// - an array of input items (see ResponsesInputItem)
type ResponsesInput struct {
	Text  string               // set if input was a plain string
	Items []ResponsesInputItem // set if input was an array
}

func (r *ResponsesInput) UnmarshalJSON(data []byte) error {
	// Try string first
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		r.Text = s
		return nil
	}

	// Otherwise, try array of input items
	var rawItems []json.RawMessage
	if err := json.Unmarshal(data, &rawItems); err != nil {
		return fmt.Errorf("input must be a string or array: %w", err)
	}

	r.Items = make([]ResponsesInputItem, 0, len(rawItems))
	for i, raw := range rawItems {
		item, err := unmarshalResponsesInputItem(raw)
		if err != nil {
			return fmt.Errorf("input[%d]: %w", i, err)
		}
		r.Items = append(r.Items, item)
	}

	return nil
}

type ResponsesReasoning struct {
	// originally: optional, default is per-model
	Effort string `json:"effort,omitempty"`

	// originally: deprecated, use `summary` instead. One of `auto`, `concise`, `detailed`
	GenerateSummary string `json:"generate_summary,omitempty"`

	// originally: optional, one of `auto`, `concise`, `detailed`
	Summary string `json:"summary,omitempty"`
}

type ResponsesTextFormat struct {
	Type       string          `json:"type"`             // "text", "json_object", "json_schema"
	Name       string          `json:"name,omitempty"`   // for json_schema
	Schema     json.RawMessage `json:"schema,omitempty"` // flat json_schema
	JsonSchema *JsonSchema     `json:"json_schema,omitempty"`
	Strict     *bool           `json:"strict,omitempty"`
}

type ResponsesText struct {
	Format *ResponsesTextFormat `json:"format,omitempty"`
}

// ResponsesTool represents a tool in the Responses API format.
// Note: This differs from api.Tool which nests fields under "function".
type ResponsesTool struct {
	Type        string         `json:"type"` // "function"
	Name        string         `json:"name"`
	Description *string        `json:"description"` // nullable but required
	Strict      *bool          `json:"strict"`      // nullable but required
	Parameters  map[string]any `json:"parameters"`  // nullable but required
}

type ResponsesRequest struct {
	Model string `json:"model"`

	// originally: optional, default is false
	// for us: not supported
	Background bool `json:"background"`

	// Store true would persist the response. We have no ResponseStore — 400.
	Store *bool `json:"store,omitempty"`

	// originally: optional `string | {id: string}`
	// for us: not supported — 400 if set
	Conversation json.RawMessage `json:"conversation"`

	// previous_response_id chains stored turns. We have no ResponseStore — 400 if set.
	PreviousResponseID *string `json:"previous_response_id,omitempty"`

	// originally: string[]
	// for us: any non-empty value is 400 (no file_search results, encrypted
	// reasoning, or output_text logprobs on this surface).
	Include []string `json:"include"`

	// OpenAI caps built-in tool rounds. 1 keeps a single function call
	// (same as parallel_tool_calls:false). 0 is 400. >1 is a no-op on one turn.
	MaxToolCalls *int `json:"max_tool_calls,omitempty"`

	Input ResponsesInput `json:"input"`

	// optional, inserts a system message at the start of the conversation
	Instructions string `json:"instructions,omitempty"`

	// optional, maps to num_predict
	MaxOutputTokens *int `json:"max_output_tokens,omitempty"`

	Reasoning ResponsesReasoning `json:"reasoning"`
	// ReasoningBudgetTokens outranks reasoning.effort (mlx-serve). 0 off, >0 on.
	ReasoningBudgetTokens *int `json:"reasoning_budget_tokens,omitempty"`

	Temperature       *float64           `json:"temperature"`
	TopP              *float64           `json:"top_p"`
	TopK              *int               `json:"top_k"`
	MinP              *float64           `json:"min_p"`
	TypicalP          *float64           `json:"typical_p"`
	Seed              *int               `json:"seed"`
	FrequencyPenalty  *float64           `json:"frequency_penalty"`
	PresencePenalty   *float64           `json:"presence_penalty"`
	RepetitionPenalty *float64           `json:"repetition_penalty"`
	RepeatPenalty     *float64           `json:"repeat_penalty"`
	LogitBias         map[string]float64 `json:"logit_bias,omitempty"`

	// optional, controls output format (e.g. json_schema)
	Text *ResponsesText `json:"text,omitempty"`

	// optional, default is `"disabled"`
	Truncation *string `json:"truncation"`

	Tools []ResponsesTool `json:"tools,omitempty"`

	// ToolChoice: "none" omits tools for this turn (minefield trap 78).
	// "required" / "any" keep the list. Named function objects keep that tool.
	ToolChoice any `json:"tool_choice,omitempty"`

	// optional, default is false
	Stream *bool `json:"stream,omitempty"`

	EnablePLD         *bool `json:"enable_pld,omitempty"`
	EnableMTP         *bool `json:"enable_mtp,omitempty"`
	EnableDrafter     *bool `json:"enable_drafter,omitempty"`
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`
	// Compression is the same object as /v1/chat/completions (also extra_body.compression).
	Compression *api.ChatCompressionConfig `json:"compression,omitempty"`
	// PromptCacheKey pins L3 + server placeholder sticky elide_from (also extra_body).
	PromptCacheKey *string         `json:"prompt_cache_key,omitempty"`
	SessionID      *string         `json:"session_id,omitempty"`
	CacheReset     *bool           `json:"cache_reset,omitempty"`
	ExtraBody      json.RawMessage `json:"extra_body,omitempty"`
	ServiceTier    string          `json:"service_tier,omitempty"`
}

// FromResponsesRequest converts a ResponsesRequest to api.ChatRequest
func FromResponsesRequest(r ResponsesRequest) (*api.ChatRequest, error) {
	foldResponsesCompression(&r)
	foldResponsesSessionCache(&r)
	foldResponsesLogitBias(&r)
	foldResponsesReasoningBudget(&r)
	if err := rejectUnsupportedResponsesFields(r); err != nil {
		return nil, err
	}
	if r.MaxToolCalls != nil && *r.MaxToolCalls == 1 {
		off := false
		r.ParallelToolCalls = &off
	}
	var messages []api.Message

	// Add instructions as system message if present
	if r.Instructions != "" {
		messages = append(messages, api.Message{
			Role:    "system",
			Content: r.Instructions,
		})
	}

	// Handle simple string input
	if r.Input.Text != "" {
		messages = append(messages, api.Message{
			Role:    "user",
			Content: r.Input.Text,
		})
	}

	// Handle array of input items
	// Track pending reasoning to merge with the next assistant message
	var pendingThinking string

	for _, item := range r.Input.Items {
		switch v := item.(type) {
		case ResponsesReasoningInput:
			// Store thinking to merge with the next assistant message
			pendingThinking = v.EncryptedContent
		case ResponsesInputMessage:
			msg, err := convertInputMessage(v)
			if err != nil {
				return nil, err
			}
			// If this is an assistant message, attach pending thinking
			if msg.Role == "assistant" && pendingThinking != "" {
				msg.Thinking = pendingThinking
				pendingThinking = ""
			}
			messages = append(messages, msg)
		case ResponsesFunctionCall:
			// Convert function call to assistant message with tool calls
			var args api.ToolCallFunctionArguments
			if v.Arguments != "" {
				if err := json.Unmarshal([]byte(v.Arguments), &args); err != nil {
					return nil, fmt.Errorf("failed to parse function call arguments: %w", err)
				}
			}
			toolCall := api.ToolCall{
				ID: v.CallID,
				Function: api.ToolCallFunction{
					Name:      v.Name,
					Arguments: args,
				},
			}

			// Merge tool call into existing assistant message if it has content or tool calls
			if len(messages) > 0 && messages[len(messages)-1].Role == "assistant" {
				lastMsg := &messages[len(messages)-1]
				lastMsg.ToolCalls = append(lastMsg.ToolCalls, toolCall)
				if pendingThinking != "" {
					lastMsg.Thinking = pendingThinking
					pendingThinking = ""
				}
			} else {
				msg := api.Message{
					Role:      "assistant",
					ToolCalls: []api.ToolCall{toolCall},
				}
				if pendingThinking != "" {
					msg.Thinking = pendingThinking
					pendingThinking = ""
				}
				messages = append(messages, msg)
			}
		case ResponsesFunctionCallOutput:
			content := v.Output
			var images []api.ImageData
			if len(v.OutputItems) > 0 {
				var err error
				content, images, err = convertResponsesContent(v.OutputItems)
				if err != nil {
					return nil, err
				}
			}
			messages = append(messages, api.Message{
				Role:       "tool",
				Content:    content,
				Images:     images,
				ToolCallID: v.CallID,
			})
		}
	}

	// If there's trailing reasoning without a following message, emit it
	if pendingThinking != "" {
		messages = append(messages, api.Message{
			Role:     "assistant",
			Thinking: pendingThinking,
		})
	}

	options := make(map[string]any)

	samplingOpts{
		Temperature:       r.Temperature,
		TopP:              r.TopP,
		MinP:              r.MinP,
		TypicalP:          r.TypicalP,
		FrequencyPenalty:  r.FrequencyPenalty,
		PresencePenalty:   r.PresencePenalty,
		RepetitionPenalty: r.RepetitionPenalty,
		RepeatPenalty:     r.RepeatPenalty,
		TopK:              r.TopK,
		Seed:              r.Seed,
		MaxTokens:         r.MaxOutputTokens,
	}.apply(options)
	if r.EnablePLD != nil {
		options["enable_pld"] = *r.EnablePLD
	}
	if r.EnableMTP != nil {
		options["enable_mtp"] = *r.EnableMTP
	}
	if r.EnableDrafter != nil {
		options["enable_drafter"] = *r.EnableDrafter
	}
	if err := putLogitBias(options, r.LogitBias); err != nil {
		return nil, err
	}
	if r.PromptCacheKey != nil && strings.TrimSpace(*r.PromptCacheKey) != "" {
		options["prompt_cache_key"] = strings.TrimSpace(*r.PromptCacheKey)
	} else if r.SessionID != nil && strings.TrimSpace(*r.SessionID) != "" {
		options["prompt_cache_key"] = strings.TrimSpace(*r.SessionID)
	}
	if r.CacheReset != nil && *r.CacheReset {
		options["cache_reset"] = true
	}

	// Convert tools from Responses API format to api.Tool format, then apply
	// tool_choice (none omits; named function keeps that tool only).
	var tools []api.Tool
	for _, t := range r.Tools {
		tool, err := convertTool(t)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}
	tools, err := applyToolChoice(tools, r.ToolChoice)
	if err != nil {
		return nil, err
	}
	messages = api.AppendRequiredToolCallHint(messages, tools, toolChoiceRequiresCall(r.ToolChoice))
	messages = api.AppendOutputBudgetGuidance(messages, api.NumPredictFromMap(options))

	var format json.RawMessage
	if r.Text != nil && r.Text.Format != nil {
		format = formatFromStructuredOutput(r.Text.Format.Type, r.Text.Format.JsonSchema, r.Text.Format.Schema)
	}

	thinkFromAlias := false
	var think *api.ThinkValue
	if t, err := thinkFromReasoningBudget(r.ReasoningBudgetTokens); err != nil {
		return nil, err
	} else if t != nil {
		think = t
		thinkFromAlias = true
	}
	if think == nil {
		if t, err := thinkFromReasoningEffort(r.Reasoning.Effort); err != nil {
			return nil, err
		} else if t != nil {
			think = t
			thinkFromAlias = true
		}
	}

	return &api.ChatRequest{
		Model:             r.Model,
		Messages:          messages,
		Options:           options,
		Tools:             tools,
		Format:            format,
		Think:             think,
		ThinkFromAlias:    thinkFromAlias,
		EnablePLD:         r.EnablePLD,
		EnableMTP:         r.EnableMTP,
		EnableDrafter:     r.EnableDrafter,
		Compression:       r.Compression,
		ParallelToolCalls: r.ParallelToolCalls,
	}, nil
}

func foldResponsesCompression(req *ResponsesRequest) {
	if req == nil || req.Compression != nil || len(req.ExtraBody) == 0 {
		return
	}
	var extra struct {
		Compression *api.ChatCompressionConfig `json:"compression"`
	}
	if json.Unmarshal(req.ExtraBody, &extra) == nil {
		req.Compression = extra.Compression
	}
}

func foldResponsesSessionCache(req *ResponsesRequest) {
	if req == nil || len(req.ExtraBody) == 0 {
		return
	}
	var extra struct {
		PromptCacheKey     *string `json:"prompt_cache_key"`
		SessionID          *string `json:"session_id"`
		CacheReset         *bool   `json:"cache_reset"`
		PreviousResponseID *string `json:"previous_response_id"`
		Store              *bool   `json:"store"`
	}
	if json.Unmarshal(req.ExtraBody, &extra) != nil {
		return
	}
	if req.Store == nil {
		req.Store = extra.Store
	}
	if req.PromptCacheKey == nil {
		req.PromptCacheKey = extra.PromptCacheKey
	}
	if req.SessionID == nil {
		req.SessionID = extra.SessionID
	}
	if req.CacheReset == nil {
		req.CacheReset = extra.CacheReset
	}
	if req.PreviousResponseID == nil {
		req.PreviousResponseID = extra.PreviousResponseID
	}
}

func foldResponsesLogitBias(req *ResponsesRequest) {
	if req == nil || req.LogitBias != nil || len(req.ExtraBody) == 0 {
		return
	}
	var extra struct {
		LogitBias map[string]float64 `json:"logit_bias"`
	}
	if json.Unmarshal(req.ExtraBody, &extra) == nil && len(extra.LogitBias) > 0 {
		req.LogitBias = extra.LogitBias
	}
}

func foldResponsesReasoningBudget(req *ResponsesRequest) {
	if req == nil || req.ReasoningBudgetTokens != nil || len(req.ExtraBody) == 0 {
		return
	}
	var extra struct {
		ReasoningBudgetTokens *int `json:"reasoning_budget_tokens"`
	}
	if json.Unmarshal(req.ExtraBody, &extra) == nil {
		req.ReasoningBudgetTokens = extra.ReasoningBudgetTokens
	}
}

func rejectUnsupportedResponsesFields(r ResponsesRequest) error {
	if r.Background {
		return fmt.Errorf("background:true is not supported (no async response store); omit background or set false")
	}
	if err := rejectStore(r.Store); err != nil {
		return err
	}
	if id := strings.TrimSpace(derefString(r.PreviousResponseID)); id != "" {
		return fmt.Errorf("previous_response_id is not supported (no response store); send the full input instead")
	}
	if responsesConversationSet(r.Conversation) {
		return fmt.Errorf("conversation is not supported (no response store); send messages in input")
	}
	if r.Truncation != nil {
		t := strings.ToLower(strings.TrimSpace(*r.Truncation))
		if t != "" && t != "disabled" {
			return fmt.Errorf("truncation %q is not supported; omit it or set disabled (overflow is a named 400)", *r.Truncation)
		}
	}
	for _, raw := range r.Include {
		v := strings.TrimSpace(raw)
		if v == "" {
			continue
		}
		return fmt.Errorf("include %q is not supported", v)
	}
	if r.MaxToolCalls != nil && *r.MaxToolCalls < 1 {
		return fmt.Errorf("max_tool_calls must be at least 1")
	}
	if err := rejectServiceTier(r.ServiceTier); err != nil {
		return err
	}
	return nil
}

func derefString(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

func responsesParallelToolCalls(r ResponsesRequest) bool {
	if r.MaxToolCalls != nil && *r.MaxToolCalls == 1 {
		return false
	}
	if r.ParallelToolCalls != nil {
		return *r.ParallelToolCalls
	}
	return true
}

func responsesConversationSet(raw json.RawMessage) bool {
	s := strings.TrimSpace(string(raw))
	return s != "" && s != "null" && s != `""` && s != "{}"
}

func convertTool(t ResponsesTool) (api.Tool, error) {
	// Convert parameters from map[string]any to api.ToolFunctionParameters
	var params api.ToolFunctionParameters
	if t.Parameters != nil {
		// Marshal and unmarshal to convert
		b, err := json.Marshal(t.Parameters)
		if err != nil {
			return api.Tool{}, fmt.Errorf("failed to marshal tool parameters: %w", err)
		}
		if err := json.Unmarshal(b, &params); err != nil {
			return api.Tool{}, fmt.Errorf("failed to unmarshal tool parameters: %w", err)
		}
	}

	var description string
	if t.Description != nil {
		description = *t.Description
	}

	return api.Tool{
		Type: t.Type,
		Function: api.ToolFunction{
			Name:        t.Name,
			Description: description,
			Parameters:  params,
		},
	}, nil
}

func convertInputMessage(m ResponsesInputMessage) (api.Message, error) {
	content, images, err := convertResponsesContent(m.Content)
	if err != nil {
		return api.Message{}, err
	}

	return api.Message{
		Role:    m.Role,
		Content: content,
		Images:  images,
	}, nil
}

func convertResponsesContent(contents []ResponsesContent) (string, []api.ImageData, error) {
	var content string
	var images []api.ImageData

	for _, c := range contents {
		switch v := c.(type) {
		case ResponsesTextContent:
			content += v.Text
		case ResponsesOutputTextContent:
			content += v.Text
		case ResponsesImageContent:
			if v.ImageURL == "" {
				continue // Skip if no URL (FileID not supported)
			}
			img, err := decodeImageURL(v.ImageURL)
			if err != nil {
				return "", nil, err
			}
			images = append(images, img)
		case ResponsesFileContent:
			// TODO(drifkin): support inlining text-only file_data when it is safe
			// to decode and of a reasonable size
			return "", nil, fmt.Errorf("file inputs are not currently supported")
		}
	}

	return content, images, nil
}

// Response types for the Responses API

// ResponsesTextField represents the text output configuration in the response.
type ResponsesTextField struct {
	Format ResponsesTextFormat `json:"format"`
}

// ResponsesReasoningOutput represents reasoning configuration in the response.
type ResponsesReasoningOutput struct {
	Effort  *string `json:"effort,omitempty"`
	Summary *string `json:"summary,omitempty"`
}

// ResponsesError represents an error in the response.
type ResponsesError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ResponsesIncompleteDetails represents details about why a response was incomplete.
type ResponsesIncompleteDetails struct {
	Reason string `json:"reason"`
}

type ResponsesResponse struct {
	ID                 string                      `json:"id"`
	Object             string                      `json:"object"`
	CreatedAt          int64                       `json:"created_at"`
	CompletedAt        *int64                      `json:"completed_at"`
	Status             string                      `json:"status"`
	IncompleteDetails  *ResponsesIncompleteDetails `json:"incomplete_details"`
	Model              string                      `json:"model"`
	PreviousResponseID *string                     `json:"previous_response_id"`
	Instructions       *string                     `json:"instructions"`
	Output             []ResponsesOutputItem       `json:"output"`
	Error              *ResponsesError             `json:"error"`
	Tools              []ResponsesTool             `json:"tools"`
	ToolChoice         any                         `json:"tool_choice"`
	Truncation         string                      `json:"truncation"`
	ParallelToolCalls  bool                        `json:"parallel_tool_calls"`
	Text               ResponsesTextField          `json:"text"`
	TopP               float64                     `json:"top_p"`
	PresencePenalty    float64                     `json:"presence_penalty"`
	FrequencyPenalty   float64                     `json:"frequency_penalty"`
	TopLogprobs        int                         `json:"top_logprobs"`
	Temperature        float64                     `json:"temperature"`
	Reasoning          *ResponsesReasoningOutput   `json:"reasoning"`
	Usage              *ResponsesUsage             `json:"usage"`
	MaxOutputTokens    *int                        `json:"max_output_tokens"`
	MaxToolCalls       *int                        `json:"max_tool_calls"`
	Store              bool                        `json:"store"`
	Background         bool                        `json:"background"`
	ServiceTier        string                      `json:"service_tier"`
	Metadata           map[string]any              `json:"metadata"`
	SafetyIdentifier   *string                     `json:"safety_identifier"`
	PromptCacheKey     *string                     `json:"prompt_cache_key"`
}

type ResponsesOutputItem struct {
	ID        string                   `json:"id"`
	Type      string                   `json:"type"` // "message", "function_call", or "reasoning"
	Status    string                   `json:"status,omitempty"`
	Role      string                   `json:"role,omitempty"`      // for message
	Content   []ResponsesOutputContent `json:"content,omitempty"`   // for message
	CallID    string                   `json:"call_id,omitempty"`   // for function_call
	Name      string                   `json:"name,omitempty"`      // for function_call
	Arguments string                   `json:"arguments,omitempty"` // for function_call

	// Reasoning fields
	Summary          []ResponsesReasoningSummary `json:"summary,omitempty"`           // for reasoning
	EncryptedContent string                      `json:"encrypted_content,omitempty"` // for reasoning
}

type ResponsesReasoningSummary struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text"`
}

type ResponsesOutputContent struct {
	Type        string `json:"type"` // "output_text"
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
	Logprobs    []any  `json:"logprobs"`
}

type ResponsesInputTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type ResponsesOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type ResponsesUsage struct {
	InputTokens         int                          `json:"input_tokens"`
	OutputTokens        int                          `json:"output_tokens"`
	TotalTokens         int                          `json:"total_tokens"`
	InputTokensDetails  ResponsesInputTokensDetails  `json:"input_tokens_details"`
	OutputTokensDetails ResponsesOutputTokensDetails `json:"output_tokens_details"`
	Compression         *api.ChatCompressionMeta     `json:"compression_meta,omitempty"`
}

// derefFloat64 returns the value of a float64 pointer, or a default if nil.
func derefFloat64(p *float64, def float64) float64 {
	if p != nil {
		return *p
	}
	return def
}

func responsesStatus(doneReason string) (string, *ResponsesIncompleteDetails) {
	if strings.EqualFold(strings.TrimSpace(doneReason), "length") {
		return "incomplete", &ResponsesIncompleteDetails{Reason: "max_output_tokens"}
	}
	return "completed", nil
}

// ToResponse converts an api.ChatResponse to a Responses API response.
// The request is used to echo back request parameters in the response.
func ToResponse(model, responseID, itemID string, chatResponse api.ChatResponse, request ResponsesRequest) ResponsesResponse {
	foldResponsesSessionCache(&request)
	var output []ResponsesOutputItem

	// Add reasoning item if thinking is present
	if chatResponse.Message.Thinking != "" {
		output = append(output, ResponsesOutputItem{
			ID:   fmt.Sprintf("rs_%s", responseID),
			Type: "reasoning",
			Summary: []ResponsesReasoningSummary{
				{
					Type: "summary_text",
					Text: chatResponse.Message.Thinking,
				},
			},
			EncryptedContent: chatResponse.Message.Thinking, // Plain text for now
		})
	}

	if len(chatResponse.Message.ToolCalls) > 0 {
		toolCalls := ToToolCalls(chatResponse.Message.ToolCalls)
		for i, tc := range toolCalls {
			output = append(output, ResponsesOutputItem{
				ID:        fmt.Sprintf("fc_%s_%d", responseID, i),
				Type:      "function_call",
				Status:    "completed",
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	} else {
		output = append(output, ResponsesOutputItem{
			ID:     itemID,
			Type:   "message",
			Status: "completed",
			Role:   "assistant",
			Content: []ResponsesOutputContent{
				{
					Type:        "output_text",
					Text:        chatResponse.Message.Content,
					Annotations: []any{},
					Logprobs:    []any{},
				},
			},
		})
	}

	var instructions *string
	if request.Instructions != "" {
		instructions = &request.Instructions
	}

	// Build truncation with default
	truncation := "disabled"
	if request.Truncation != nil {
		truncation = *request.Truncation
	}

	tools := request.Tools
	if tools == nil {
		tools = []ResponsesTool{}
	}

	text := ResponsesTextField{
		Format: ResponsesTextFormat{Type: "text"},
	}
	if request.Text != nil && request.Text.Format != nil {
		text.Format = *request.Text.Format
	}

	// Build reasoning output from request
	var reasoning *ResponsesReasoningOutput
	if request.Reasoning.Effort != "" || request.Reasoning.Summary != "" {
		reasoning = &ResponsesReasoningOutput{}
		if request.Reasoning.Effort != "" {
			reasoning.Effort = &request.Reasoning.Effort
		}
		if request.Reasoning.Summary != "" {
			reasoning.Summary = &request.Reasoning.Summary
		}
	}

	status, incomplete := responsesStatus(chatResponse.DoneReason)

	return ResponsesResponse{
		ID:                 responseID,
		Object:             "response",
		CreatedAt:          chatResponse.CreatedAt.Unix(),
		CompletedAt:        nil, // Set by middleware when writing final response
		Status:             status,
		IncompleteDetails:  incomplete,
		Model:              model,
		PreviousResponseID: nil, // Not supported
		Instructions:       instructions,
		Output:             output,
		Error:              nil, // Only populated on failure
		Tools:              tools,
		ToolChoice:         "auto", // Default value
		Truncation:         truncation,
		ParallelToolCalls:  responsesParallelToolCalls(request),
		Text:               text,
		TopP:               derefFloat64(request.TopP, 1.0),
		PresencePenalty:    0, // Default value
		FrequencyPenalty:   0, // Default value
		TopLogprobs:        0, // Default value
		Temperature:        derefFloat64(request.Temperature, 1.0),
		Reasoning:          reasoning,
		Usage: &ResponsesUsage{
			InputTokens:  chatResponse.PromptEvalCount,
			OutputTokens: chatResponse.EvalCount,
			TotalTokens:  chatResponse.PromptEvalCount + chatResponse.EvalCount,
			// TODO(drifkin): wire through the actual values
			InputTokensDetails: ResponsesInputTokensDetails{CachedTokens: chatResponse.CachedPromptTokens},
			// TODO(drifkin): wire through the actual values
			OutputTokensDetails: ResponsesOutputTokensDetails{ReasoningTokens: 0},
			Compression:         chatResponse.Compression,
		},
		MaxOutputTokens:  request.MaxOutputTokens,
		MaxToolCalls:     request.MaxToolCalls,
		Store:            false, // We don't store responses
		Background:       request.Background,
		ServiceTier:      "default", // Default value
		Metadata:         map[string]any{},
		SafetyIdentifier: nil, // Not supported
		PromptCacheKey:   request.PromptCacheKey,
	}
}

// Streaming events: <https://platform.openai.com/docs/api-reference/responses-streaming>

// ResponsesStreamEvent represents a single Server-Sent Event for the Responses API.
type ResponsesStreamEvent struct {
	Event string // The event type (e.g., "response.created")
	Data  any    // The event payload (will be JSON-marshaled)
}

// ResponsesStreamConverter converts api.ChatResponse objects to Responses API
// streaming events. It maintains state across multiple calls to handle the
// streaming event sequence correctly.
type ResponsesStreamConverter struct {
	// Configuration (immutable after creation)
	responseID string
	itemID     string
	model      string
	request    ResponsesRequest

	// State tracking (mutated across Process calls)
	firstWrite      bool
	outputIndex     int
	contentIndex    int
	contentStarted  bool
	toolCallsSent   bool
	accumulatedText string
	sequenceNumber  int

	// Reasoning/thinking state
	accumulatedThinking string
	reasoningItemID     string
	reasoningStarted    bool
	reasoningDone       bool

	// Tool calls state (for final output)
	toolCallItems []map[string]any
}

// newEvent creates a ResponsesStreamEvent with the sequence number included in the data.
func (c *ResponsesStreamConverter) newEvent(eventType string, data map[string]any) ResponsesStreamEvent {
	data["type"] = eventType
	data["sequence_number"] = c.sequenceNumber
	c.sequenceNumber++
	return ResponsesStreamEvent{
		Event: eventType,
		Data:  data,
	}
}

// NewResponsesStreamConverter creates a new converter with the given configuration.
func NewResponsesStreamConverter(responseID, itemID, model string, request ResponsesRequest) *ResponsesStreamConverter {
	foldResponsesSessionCache(&request)
	return &ResponsesStreamConverter{
		responseID: responseID,
		itemID:     itemID,
		model:      model,
		request:    request,
		firstWrite: true,
	}
}

// Process takes a ChatResponse and returns the events that should be emitted.
// Events are returned in order. The caller is responsible for serializing
// and sending these events.
func (c *ResponsesStreamConverter) Process(r api.ChatResponse) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	hasToolCalls := len(r.Message.ToolCalls) > 0
	hasThinking := r.Message.Thinking != ""

	// First chunk - emit initial events
	if c.firstWrite {
		c.firstWrite = false
		events = append(events, c.createResponseCreatedEvent())
		events = append(events, c.createResponseInProgressEvent())
	}

	// Handle reasoning/thinking (before other content)
	if hasThinking {
		events = append(events, c.processThinking(r.Message.Thinking)...)
	}

	// Handle tool calls
	if hasToolCalls {
		events = append(events, c.processToolCalls(r.Message.ToolCalls)...)
		c.toolCallsSent = true
	}

	// Handle text content (only if no tool calls)
	if !hasToolCalls && !c.toolCallsSent && r.Message.Content != "" {
		events = append(events, c.processTextContent(r.Message.Content)...)
	}

	// Done - emit closing events
	if r.Done {
		events = append(events, c.processCompletion(r)...)
	}

	return events
}

// buildResponseObject creates a full response object with all required fields for streaming events.
func (c *ResponsesStreamConverter) buildResponseObject(status string, output []any, usage map[string]any) map[string]any {
	var instructions any = nil
	if c.request.Instructions != "" {
		instructions = c.request.Instructions
	}

	truncation := "disabled"
	if c.request.Truncation != nil {
		truncation = *c.request.Truncation
	}

	var tools []any
	if c.request.Tools != nil {
		for _, t := range c.request.Tools {
			tools = append(tools, map[string]any{
				"type":        t.Type,
				"name":        t.Name,
				"description": t.Description,
				"strict":      t.Strict,
				"parameters":  t.Parameters,
			})
		}
	}
	if tools == nil {
		tools = []any{}
	}

	textFormat := map[string]any{"type": "text"}
	if c.request.Text != nil && c.request.Text.Format != nil {
		textFormat = map[string]any{
			"type": c.request.Text.Format.Type,
		}
		if c.request.Text.Format.Name != "" {
			textFormat["name"] = c.request.Text.Format.Name
		}
		if c.request.Text.Format.Schema != nil {
			textFormat["schema"] = c.request.Text.Format.Schema
		}
		if c.request.Text.Format.Strict != nil {
			textFormat["strict"] = *c.request.Text.Format.Strict
		}
	}

	var reasoning any = nil
	if c.request.Reasoning.Effort != "" || c.request.Reasoning.Summary != "" {
		r := map[string]any{}
		if c.request.Reasoning.Effort != "" {
			r["effort"] = c.request.Reasoning.Effort
		} else {
			r["effort"] = nil
		}
		if c.request.Reasoning.Summary != "" {
			r["summary"] = c.request.Reasoning.Summary
		} else {
			r["summary"] = nil
		}
		reasoning = r
	}

	// Build top_p and temperature with defaults
	topP := 1.0
	if c.request.TopP != nil {
		topP = *c.request.TopP
	}
	temperature := 1.0
	if c.request.Temperature != nil {
		temperature = *c.request.Temperature
	}

	return map[string]any{
		"id":                   c.responseID,
		"object":               "response",
		"created_at":           time.Now().Unix(),
		"completed_at":         nil,
		"status":               status,
		"incomplete_details":   nil,
		"model":                c.model,
		"previous_response_id": nil,
		"instructions":         instructions,
		"output":               output,
		"error":                nil,
		"tools":                tools,
		"tool_choice":          "auto",
		"truncation":           truncation,
		"parallel_tool_calls":  responsesParallelToolCalls(c.request),
		"text":                 map[string]any{"format": textFormat},
		"top_p":                topP,
		"presence_penalty":     0,
		"frequency_penalty":    0,
		"top_logprobs":         0,
		"temperature":          temperature,
		"reasoning":            reasoning,
		"usage":                usage,
		"max_output_tokens":    c.request.MaxOutputTokens,
		"max_tool_calls":       c.request.MaxToolCalls,
		"store":                false,
		"background":           c.request.Background,
		"service_tier":         "default",
		"metadata":             map[string]any{},
		"safety_identifier":    nil,
		"prompt_cache_key":     c.request.PromptCacheKey,
	}
}

func (c *ResponsesStreamConverter) createResponseCreatedEvent() ResponsesStreamEvent {
	return c.newEvent("response.created", map[string]any{
		"response": c.buildResponseObject("in_progress", []any{}, nil),
	})
}

func (c *ResponsesStreamConverter) createResponseInProgressEvent() ResponsesStreamEvent {
	return c.newEvent("response.in_progress", map[string]any{
		"response": c.buildResponseObject("in_progress", []any{}, nil),
	})
}

func (c *ResponsesStreamConverter) processThinking(thinking string) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Start reasoning item if not started
	if !c.reasoningStarted {
		c.reasoningStarted = true
		c.reasoningItemID = fmt.Sprintf("rs_%d", rand.Intn(999999))

		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":      c.reasoningItemID,
				"type":    "reasoning",
				"summary": []any{},
			},
		}))
	}

	// Accumulate thinking
	c.accumulatedThinking += thinking

	// Emit delta
	events = append(events, c.newEvent("response.reasoning_summary_text.delta", map[string]any{
		"item_id":       c.reasoningItemID,
		"output_index":  c.outputIndex,
		"summary_index": 0,
		"delta":         thinking,
	}))

	// TODO(drifkin): consider adding
	// [`response.reasoning_text.delta`](https://platform.openai.com/docs/api-reference/responses-streaming/response/reasoning_text/delta),
	// but need to do additional research to understand how it's used and how
	// widely supported it is

	return events
}

func (c *ResponsesStreamConverter) finishReasoning() []ResponsesStreamEvent {
	if !c.reasoningStarted || c.reasoningDone {
		return nil
	}
	c.reasoningDone = true

	events := []ResponsesStreamEvent{
		c.newEvent("response.reasoning_summary_text.done", map[string]any{
			"item_id":       c.reasoningItemID,
			"output_index":  c.outputIndex,
			"summary_index": 0,
			"text":          c.accumulatedThinking,
		}),
		c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":                c.reasoningItemID,
				"type":              "reasoning",
				"summary":           []map[string]any{{"type": "summary_text", "text": c.accumulatedThinking}},
				"encrypted_content": c.accumulatedThinking, // Plain text for now
			},
		}),
	}

	c.outputIndex++
	return events
}

func (c *ResponsesStreamConverter) processToolCalls(toolCalls []api.ToolCall) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Finish reasoning first if it was started
	events = append(events, c.finishReasoning()...)

	converted := ToToolCalls(toolCalls)

	for i, tc := range converted {
		fcItemID := fmt.Sprintf("fc_%d_%d", rand.Intn(999999), i)

		// Store for final output (with status: completed)
		toolCallItem := map[string]any{
			"id":        fcItemID,
			"type":      "function_call",
			"status":    "completed",
			"call_id":   tc.ID,
			"name":      tc.Function.Name,
			"arguments": tc.Function.Arguments,
		}
		c.toolCallItems = append(c.toolCallItems, toolCallItem)

		// response.output_item.added for function call
		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex + i,
			"item": map[string]any{
				"id":        fcItemID,
				"type":      "function_call",
				"status":    "in_progress",
				"call_id":   tc.ID,
				"name":      tc.Function.Name,
				"arguments": "",
			},
		}))

		// response.function_call_arguments.delta
		if tc.Function.Arguments != "" {
			events = append(events, c.newEvent("response.function_call_arguments.delta", map[string]any{
				"item_id":      fcItemID,
				"output_index": c.outputIndex + i,
				"delta":        tc.Function.Arguments,
			}))
		}

		// response.function_call_arguments.done
		events = append(events, c.newEvent("response.function_call_arguments.done", map[string]any{
			"item_id":      fcItemID,
			"output_index": c.outputIndex + i,
			"arguments":    tc.Function.Arguments,
		}))

		// response.output_item.done for function call
		events = append(events, c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex + i,
			"item": map[string]any{
				"id":        fcItemID,
				"type":      "function_call",
				"status":    "completed",
				"call_id":   tc.ID,
				"name":      tc.Function.Name,
				"arguments": tc.Function.Arguments,
			},
		}))
	}

	return events
}

func (c *ResponsesStreamConverter) processTextContent(content string) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Finish reasoning first if it was started
	events = append(events, c.finishReasoning()...)

	// Emit output item and content part for first text content
	if !c.contentStarted {
		c.contentStarted = true

		// response.output_item.added
		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":      c.itemID,
				"type":    "message",
				"status":  "in_progress",
				"role":    "assistant",
				"content": []any{},
			},
		}))

		// response.content_part.added
		events = append(events, c.newEvent("response.content_part.added", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": c.contentIndex,
			"part": map[string]any{
				"type":        "output_text",
				"text":        "",
				"annotations": []any{},
				"logprobs":    []any{},
			},
		}))
	}

	// Accumulate text
	c.accumulatedText += content

	// Emit content delta
	events = append(events, c.newEvent("response.output_text.delta", map[string]any{
		"item_id":       c.itemID,
		"output_index":  c.outputIndex,
		"content_index": 0,
		"delta":         content,
		"logprobs":      []any{},
	}))

	return events
}

func (c *ResponsesStreamConverter) buildFinalOutput() []any {
	var output []any

	// Add reasoning item if present
	if c.reasoningStarted {
		output = append(output, map[string]any{
			"id":                c.reasoningItemID,
			"type":              "reasoning",
			"summary":           []map[string]any{{"type": "summary_text", "text": c.accumulatedThinking}},
			"encrypted_content": c.accumulatedThinking,
		})
	}

	// Add tool calls if present
	if len(c.toolCallItems) > 0 {
		for _, item := range c.toolCallItems {
			output = append(output, item)
		}
	} else if c.contentStarted {
		// Add message item if we had text content
		output = append(output, map[string]any{
			"id":     c.itemID,
			"type":   "message",
			"status": "completed",
			"role":   "assistant",
			"content": []map[string]any{{
				"type":        "output_text",
				"text":        c.accumulatedText,
				"annotations": []any{},
				"logprobs":    []any{},
			}},
		})
	}

	return output
}

func (c *ResponsesStreamConverter) processCompletion(r api.ChatResponse) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Finish reasoning if not done
	events = append(events, c.finishReasoning()...)

	// Emit text completion events if we had text content
	if !c.toolCallsSent && c.contentStarted {
		// response.output_text.done
		events = append(events, c.newEvent("response.output_text.done", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": 0,
			"text":          c.accumulatedText,
			"logprobs":      []any{},
		}))

		// response.content_part.done
		events = append(events, c.newEvent("response.content_part.done", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": 0,
			"part": map[string]any{
				"type":        "output_text",
				"text":        c.accumulatedText,
				"annotations": []any{},
				"logprobs":    []any{},
			},
		}))

		// response.output_item.done
		events = append(events, c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":     c.itemID,
				"type":   "message",
				"status": "completed",
				"role":   "assistant",
				"content": []map[string]any{{
					"type":        "output_text",
					"text":        c.accumulatedText,
					"annotations": []any{},
					"logprobs":    []any{},
				}},
			},
		}))
	}

	// response.completed
	usage := map[string]any{
		"input_tokens":  r.PromptEvalCount,
		"output_tokens": r.EvalCount,
		"total_tokens":  r.PromptEvalCount + r.EvalCount,
		"input_tokens_details": map[string]any{
			"cached_tokens": r.CachedPromptTokens,
		},
		"output_tokens_details": map[string]any{
			"reasoning_tokens": 0,
		},
	}
	if r.Compression != nil {
		usage["compression_meta"] = r.Compression
	}
	status, details := responsesStatus(r.DoneReason)
	response := c.buildResponseObject(status, c.buildFinalOutput(), usage)
	response["completed_at"] = time.Now().Unix()
	if details != nil {
		response["incomplete_details"] = map[string]any{"reason": details.Reason}
	}
	events = append(events, c.newEvent("response.completed", map[string]any{
		"response": response,
	}))

	return events
}

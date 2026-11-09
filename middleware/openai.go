package middleware

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/media"
	"github.com/ollama/ollama/server/modality/piperphrase"
	"github.com/ollama/ollama/server/modality/transcript"
	"github.com/ollama/ollama/server/modality/whispercli"
)

// maxDecompressedBodySize limits the size of a decompressed request body
const maxDecompressedBodySize = 20 << 20

type BaseWriter struct {
	gin.ResponseWriter
}

type ChatWriter struct {
	stream        bool
	streamOptions *openai.StreamOptions
	id            string
	toolCallSent  bool
	lead          streamLeadHold
	BaseWriter
}

type CompleteWriter struct {
	stream        bool
	streamOptions *openai.StreamOptions
	id            string
	echo          bool
	prompt        string
	echoed        bool
	lead          streamLeadHold
	BaseWriter
}

// streamLeadHold is mlx-serve chat.streamContentLead: an all-whitespace first
// stream chunk is held and prepended to the first real text so concatenated
// SSE matches non-stream (FIM indent). A whitespace-only completion is flushed
// on Done.
type streamLeadHold struct {
	held   string
	closed bool
}

func entirelyWhitespace(s string) bool {
	return s != "" && strings.TrimSpace(s) == ""
}

func (h *streamLeadHold) merge(text string, flush bool) (out string, skip bool) {
	if h == nil {
		return text, false
	}
	if h.closed {
		return text, false
	}
	if entirelyWhitespace(text) {
		h.held += text
		if !flush {
			return "", true
		}
	} else if text == "" && !flush {
		return "", true
	} else {
		out = h.held + text
		h.held = ""
		h.closed = true
		return out, false
	}
	h.closed = true
	out = h.held
	h.held = ""
	return out, out == ""
}

func (w *CompleteWriter) applyEcho(r *api.GenerateResponse) {
	if w == nil || r == nil || !w.echo || w.echoed {
		return
	}
	if r.Response == "" && !r.Done {
		return
	}
	r.Response = w.prompt + r.Response
	w.echoed = true
}

type ListWriter struct {
	BaseWriter
}

type RetrieveWriter struct {
	BaseWriter
	model string
}

type EmbedWriter struct {
	BaseWriter
	model          string
	encodingFormat string
}

func writeSSEData(w http.ResponseWriter, payload any) error {
	d, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := w.Write([]byte(fmt.Sprintf("data: %s\n\n", d))); err != nil {
		return err
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

func (w *BaseWriter) writeError(data []byte) (int, error) {
	var serr api.StatusError
	if err := json.Unmarshal(data, &serr); err != nil {
		// If the error response isn't valid JSON, use the raw bytes as the
		// error message rather than surfacing a confusing JSON parse error.
		serr.ErrorMessage = string(data)
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w.ResponseWriter).Encode(openai.NewError(w.ResponseWriter.Status(), serr.Error())); err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ChatWriter) writeResponse(data []byte) (int, error) {
	// NDJSON {"error":"..."} after keepalives (HTTP 200 already written): surface as
	// OpenAI error SSE instead of failing unmarshal / empty completion.
	var errBody struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(data, &errBody); err == nil && strings.TrimSpace(errBody.Error) != "" {
		if w.stream {
			w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
			payload := map[string]any{
				"error": map[string]any{
					"message": errBody.Error,
					"type":    "server_error",
				},
			}
			if err := writeSSEData(w.ResponseWriter, payload); err != nil {
				return 0, err
			}
			if _, err := w.ResponseWriter.Write([]byte("data: [DONE]\n\n")); err != nil {
				return 0, err
			}
			return len(data), nil
		}
		return w.writeError(data)
	}

	var chatResponse api.ChatResponse
	err := json.Unmarshal(data, &chatResponse)
	if err != nil {
		return 0, err
	}

	// Ollama status keepalive chunks (not OpenAI content). Emit valid data: frames so
	// strict clients (Mercury) count activity; SSE comment frames alone are ignored.
	if w.stream && chatResponse.Status != "" && chatResponse.Message.Content == "" && !chatResponse.Done {
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		if err := writeSSEData(w.ResponseWriter, openai.KeepaliveChunk(w.id, chatResponse.Model)); err != nil {
			return 0, err
		}
		return len(data), nil
	}

	// chat chunk
	if w.stream {
		if chatResponse.Message.Thinking != "" || len(chatResponse.Message.ToolCalls) > 0 {
			w.lead.closed = true
		} else {
			merged, skip := w.lead.merge(chatResponse.Message.Content, chatResponse.Done)
			if skip && !chatResponse.Done {
				return len(data), nil
			}
			chatResponse.Message.Content = merged
		}
		chunks := openai.ToChunks(w.id, chatResponse, w.toolCallSent)
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		for _, c := range chunks {
			d, err := json.Marshal(c)
			if err != nil {
				return 0, err
			}
			if !w.toolCallSent && len(c.Choices) > 0 && len(c.Choices[0].Delta.ToolCalls) > 0 {
				w.toolCallSent = true
			}
			_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
			if err != nil {
				return 0, err
			}
		}

		if chatResponse.Done {
			c := openai.ToChunk(w.id, chatResponse, w.toolCallSent)
			if len(chunks) > 0 {
				c = chunks[len(chunks)-1]
			} else {
				slog.Warn("ToChunks returned no chunks; falling back to ToChunk for usage chunk", "id", w.id, "model", chatResponse.Model)
			}
			if w.streamOptions != nil && w.streamOptions.IncludeUsage {
				u := openai.ToUsage(chatResponse)
				c.Usage = &u
				c.Sglext = openai.SglExtFromMetrics(chatResponse.Metrics)
				c.Choices = []openai.ChunkChoice{}
				d, err := json.Marshal(c)
				if err != nil {
					return 0, err
				}
				_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
				if err != nil {
					return 0, err
				}
			}
			_, err = w.ResponseWriter.Write([]byte("data: [DONE]\n\n"))
			if err != nil {
				return 0, err
			}
		}

		return len(data), nil
	}

	// chat completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToChatCompletion(w.id, chatResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ChatWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *CompleteWriter) writeResponse(data []byte) (int, error) {
	var generateResponse api.GenerateResponse
	err := json.Unmarshal(data, &generateResponse)
	if err != nil {
		return 0, err
	}

	// Ollama status keepalive / prefill progress (empty text, not done).
	if w.stream && generateResponse.Status != "" && generateResponse.Response == "" && !generateResponse.Done {
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		if err := writeSSEData(w.ResponseWriter, openai.CompletionKeepaliveChunk(w.id, generateResponse.Model)); err != nil {
			return 0, err
		}
		return len(data), nil
	}

	// completion chunk
	if w.stream {
		w.applyEcho(&generateResponse)
		merged, skip := w.lead.merge(generateResponse.Response, generateResponse.Done)
		if skip && !generateResponse.Done {
			return len(data), nil
		}
		generateResponse.Response = merged
		c := openai.ToCompleteChunk(w.id, generateResponse)
		if w.streamOptions != nil && w.streamOptions.IncludeUsage {
			c.Usage = &openai.Usage{}
		}
		d, err := json.Marshal(c)
		if err != nil {
			return 0, err
		}

		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
		if err != nil {
			return 0, err
		}

		if generateResponse.Done {
			if w.streamOptions != nil && w.streamOptions.IncludeUsage {
				u := openai.ToUsageGenerate(generateResponse)
				c.Usage = &u
				c.Choices = []openai.CompleteChunkChoice{}
				d, err := json.Marshal(c)
				if err != nil {
					return 0, err
				}
				_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
				if err != nil {
					return 0, err
				}
			}
			_, err = w.ResponseWriter.Write([]byte("data: [DONE]\n\n"))
			if err != nil {
				return 0, err
			}
		}

		return len(data), nil
	}

	// completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	w.applyEcho(&generateResponse)
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToCompletion(w.id, generateResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *CompleteWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *ListWriter) writeResponse(data []byte) (int, error) {
	var listResponse api.ListResponse
	err := json.Unmarshal(data, &listResponse)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToListCompletion(listResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ListWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *RetrieveWriter) writeResponse(data []byte) (int, error) {
	var showResponse api.ShowResponse
	err := json.Unmarshal(data, &showResponse)
	if err != nil {
		return 0, err
	}

	// retrieve completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToModel(showResponse, w.model))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *RetrieveWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *EmbedWriter) writeResponse(data []byte) (int, error) {
	var embedResponse api.EmbedResponse
	err := json.Unmarshal(data, &embedResponse)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToEmbeddingList(w.model, embedResponse, w.encodingFormat))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *EmbedWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

// CtxKeyListVirtualRouters asks ListHandler to append router names (e.g. "auto") as
// virtual models. Set only by ListMiddleware (/v1/models): OpenAI pickers should offer
// "auto", but /api/tags feeds bench/launch/ls, which must only see real models.
const CtxKeyListVirtualRouters = "zerollama.list.virtual_routers"

func ListMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Set(CtxKeyListVirtualRouters, true)
		w := &ListWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Writer = w

		c.Next()
	}
}

func RetrieveMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(api.ShowRequest{Name: c.Param("model")}); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &RetrieveWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			model:      c.Param("model"),
		}

		c.Writer = w

		c.Next()
	}
}

func CompletionsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		req, err := openai.BindCompletionRequest(body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var b bytes.Buffer
		genReq, err := openai.FromCompleteRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if err := json.NewEncoder(&b).Encode(genReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &CompleteWriter{
			BaseWriter:    BaseWriter{ResponseWriter: c.Writer},
			stream:        req.Stream,
			id:            fmt.Sprintf("cmpl-%d", rand.Intn(999)),
			streamOptions: req.StreamOptions,
			echo:          req.Echo,
			prompt:        req.Prompt,
		}

		c.Writer = w
		c.Next()
	}
}

func EmbeddingsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.EmbedRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// Validate encoding_format parameter
		if req.EncodingFormat != "" {
			if !strings.EqualFold(req.EncodingFormat, "float") && !strings.EqualFold(req.EncodingFormat, "base64") {
				c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("Invalid value for 'encoding_format' = %s. Supported values: ['float', 'base64'].", req.EncodingFormat)))
				return
			}
		}

		if req.Input == "" {
			req.Input = []string{""}
		}

		if req.Input == nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "invalid input"))
			return
		}

		if v, ok := req.Input.([]any); ok && len(v) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "invalid input"))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(api.EmbedRequest{Model: req.Model, Input: req.Input, Dimensions: req.Dimensions}); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &EmbedWriter{
			BaseWriter:     BaseWriter{ResponseWriter: c.Writer},
			model:          req.Model,
			encodingFormat: req.EncodingFormat,
		}

		c.Writer = w

		c.Next()
	}
}

func ChatMiddleware(thinkingLookup ...ThinkingLookup) gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(body))

		req, err := openai.BindChatCompletionRequest(body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if len(req.Messages) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "[] is too short - 'messages'"))
			return
		}

		var b bytes.Buffer

		thinking := modelThinking(thinkingLookup, req.Model)
		chatReq, err := openai.FromChatRequestWithContext(c.Request.Context(), req, thinking)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if chatReq.ThinkFromAlias {
			// Survives JSON re-encode of ChatRequest (ThinkFromAlias is json:"-").
			c.Set("think_from_alias", true)
		}

		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ChatWriter{
			BaseWriter:    BaseWriter{ResponseWriter: c.Writer},
			stream:        req.Stream,
			id:            fmt.Sprintf("chatcmpl-%d", rand.Intn(999)),
			streamOptions: req.StreamOptions,
		}

		c.Writer = w

		c.Next()
	}
}

type ResponsesWriter struct {
	BaseWriter
	converter  *openai.ResponsesStreamConverter
	model      string
	stream     bool
	responseID string
	itemID     string
	request    openai.ResponsesRequest
	think      *api.ThinkValue
}

// SetThinking keeps web-search follow-ups aligned with the resolved chat request.
func (w *ResponsesWriter) SetThinking(think *api.ThinkValue) {
	w.think = think
}

func (w *ResponsesWriter) writeEvent(eventType string, data any) error {
	d, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, d)))
	if err != nil {
		return err
	}
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

func (w *ResponsesWriter) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		return 0, err
	}

	if w.stream {
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")

		events := w.converter.Process(chatResponse)
		for _, event := range events {
			if err := w.writeEvent(event.Event, event.Data); err != nil {
				return 0, err
			}
		}
		// mlx-serve: Responses SSE ends with data: [DONE] so proxies that
		// key stream-end off the chat/completions sentinel still work.
		if chatResponse.Done {
			if _, err := w.ResponseWriter.Write([]byte("data: [DONE]\n\n")); err != nil {
				return 0, err
			}
			if f, ok := w.ResponseWriter.(http.Flusher); ok {
				f.Flush()
			}
		}
		return len(data), nil
	}

	// Non-streaming response
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	response := openai.ToResponse(w.model, w.responseID, w.itemID, chatResponse, w.request)
	completedAt := time.Now().Unix()
	response.CompletedAt = &completedAt
	return len(data), json.NewEncoder(w.ResponseWriter).Encode(response)
}

func (w *ResponsesWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}
	return w.writeResponse(data)
}

func ResponsesMiddleware(thinkingLookup ...ThinkingLookup) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.GetHeader("Content-Encoding") == "zstd" {
			reader, err := zstd.NewReader(c.Request.Body, zstd.WithDecoderMaxMemory(8<<20))
			if err != nil {
				c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "failed to decompress zstd body"))
				return
			}
			defer reader.Close()
			c.Request.Body = http.MaxBytesReader(c.Writer, io.NopCloser(reader), maxDecompressedBodySize)
			c.Request.Header.Del("Content-Encoding")
		}

		var req openai.ResponsesRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		thinking := modelThinking(thinkingLookup, req.Model)
		chatReq, err := openai.FromResponsesRequest(req, thinking)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// Clients can retain reasoning effort when switching to a non-thinking
		// model. Ignore that effort, but preserve explicit Ollama think overrides.
		if chatReq.Think == nil {
			c.Set("relax_thinking", true)
		}

		// Check if client requested streaming (defaults to false)
		streamRequested := req.Stream != nil && *req.Stream

		// Pass streaming preference to the underlying chat request
		chatReq.Stream = &streamRequested

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		responseID := fmt.Sprintf("resp_%d", rand.Intn(999999))
		itemID := fmt.Sprintf("msg_%d", rand.Intn(999999))

		w := &ResponsesWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			converter:  openai.NewResponsesStreamConverter(responseID, itemID, req.Model, req),
			model:      req.Model,
			stream:     streamRequested,
			responseID: responseID,
			itemID:     itemID,
			request:    req,
		}

		// Set headers based on streaming mode
		if streamRequested {
			c.Writer.Header().Set("Content-Type", "text/event-stream")
			c.Writer.Header().Set("Cache-Control", "no-cache")
			c.Writer.Header().Set("Connection", "keep-alive")
		}

		c.Writer = w
		c.Next()
	}
}

type ImageWriter struct {
	BaseWriter
}

func (w *ImageWriter) writeResponse(data []byte) (int, error) {
	var generateResponse api.GenerateResponse
	if err := json.Unmarshal(data, &generateResponse); err != nil {
		return 0, err
	}

	// Only write response when done with image
	if generateResponse.Done && generateResponse.Image != "" {
		w.ResponseWriter.Header().Set("Content-Type", "application/json")
		return len(data), json.NewEncoder(w.ResponseWriter).Encode(openai.ToImageGenerationResponse(generateResponse))
	}

	// WHY: OpenAI clients expect {"error":{...}} — never an empty 200. Done-without-
	// image used to be swallowed (NDJSON default + silent return), which crashed
	// Discord agents on data.data. Load-only (empty prompt) is DoneReason "load".
	if generateResponse.Done && generateResponse.DoneReason != "load" && generateResponse.Image == "" {
		msg := "image generation finished without image data"
		if strings.HasPrefix(generateResponse.Response, "error:") {
			msg = strings.TrimSpace(strings.TrimPrefix(generateResponse.Response, "error:"))
		} else if strings.TrimSpace(generateResponse.Response) != "" {
			msg = strings.TrimSpace(generateResponse.Response)
		}
		code := w.ResponseWriter.Status()
		if code == 0 || code == http.StatusOK {
			w.ResponseWriter.WriteHeader(http.StatusInternalServerError)
			code = http.StatusInternalServerError
		}
		w.ResponseWriter.Header().Set("Content-Type", "application/json")
		return len(data), json.NewEncoder(w.ResponseWriter).Encode(openai.NewError(code, msg))
	}

	return len(data), nil
}

func (w *ImageWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != 0 && code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func ImageGenerationsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		req, err := openai.BindImageGenerationRequest(body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if req.Prompt == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt is required"))
			return
		}

		if req.Model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
			return
		}

		genReq, err := openai.FromImageGenerationRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(genReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ImageWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Set(CtxKeyRouteSurface, "image")
		c.Writer = w
		c.Next()
	}
}

func ImageEditsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		req, err := openai.BindImageEditRequest(body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if req.Prompt == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt is required"))
			return
		}

		if req.Model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
			return
		}

		if req.Image == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "image is required"))
			return
		}

		genReq, err := openai.FromImageEditRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(genReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ImageWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Set(CtxKeyRouteSurface, "image_edit")
		c.Writer = w
		c.Next()
	}
}

// CtxKeyTranscriptionResponseFormat is the gin context key for OpenAI transcription response_format.
const CtxKeyTranscriptionResponseFormat = "ollama.transcription.response_format"

// CtxKeyTranscriptionOriginalFilename is the uploaded multipart filename (for STT format detection).
const CtxKeyTranscriptionOriginalFilename = "ollama.transcription.original_filename"

// CtxKeyTranscriptionLanguage is the optional language hint from the multipart form ("language" field).
const CtxKeyTranscriptionLanguage = "ollama.transcription.language"

// CtxKeyTranscriptionPrompt is the optional biasing prompt from the multipart form ("prompt" field).
const CtxKeyTranscriptionPrompt = "ollama.transcription.prompt"

// CtxKeyTranscriptionTemperature is the optional sampling temperature ("temperature" field).
// Empty means llama-server is asked for 0, the previous default.
const CtxKeyTranscriptionTemperature = "ollama.transcription.temperature"

// CtxKeyTranscriptionWords is true when timestamp_granularities includes word.
const CtxKeyTranscriptionWords = "ollama.transcription.words"

// CtxKeyTranscriptionStream is true when the form field stream is true.
const CtxKeyTranscriptionStream = "ollama.transcription.stream"

// CtxKeyTranscriptionLogprobs is true when include[] contains logprobs.
const CtxKeyTranscriptionLogprobs = "ollama.transcription.logprobs"

// CtxKeyTranscriptionDiarize is true when the form field diarize is true.
const CtxKeyTranscriptionDiarize = "ollama.transcription.diarize"

// CtxKeyTranscriptionTinyDiarize is true when the form field tinydiarize is true.
const CtxKeyTranscriptionTinyDiarize = "ollama.transcription.tinydiarize"

// CtxKeyTranscriptionSplitOnWord is true when the form field split_on_word is true.
const CtxKeyTranscriptionSplitOnWord = "ollama.transcription.split_on_word"

// CtxKeyTranscriptionSuppressNST is true when the form field suppress_nst is true.
const CtxKeyTranscriptionSuppressNST = "ollama.transcription.suppress_nst"

// CtxKeyTranscriptionNoFallback is true when the form field no_fallback is true.
const CtxKeyTranscriptionNoFallback = "ollama.transcription.no_fallback"

// CtxKeyTranscriptionCarryInitial is true when carry_initial_prompt is true.
const CtxKeyTranscriptionCarryInitial = "ollama.transcription.carry_initial_prompt"

// CtxKeyTranscriptionMaxLen is the optional max_len character cap. Empty means the cap is off.
const CtxKeyTranscriptionMaxLen = "ollama.transcription.max_len"

// CtxKeyTranscriptionWordThold is the optional word_thold probability. Empty means 0.01.
const CtxKeyTranscriptionWordThold = "ollama.transcription.word_thold"

// CtxKeyTranscriptionMaxContext is the optional max_context cap. Empty means the library cap.
const CtxKeyTranscriptionMaxContext = "ollama.transcription.max_context"

// CtxKeyTranscriptionAudioCtx is the optional audio_ctx frame count. Empty means the full context.
const CtxKeyTranscriptionAudioCtx = "ollama.transcription.audio_ctx"

// CtxKeyTranscriptionOffsetT is the optional offset_t start, in milliseconds. Empty starts at the beginning.
const CtxKeyTranscriptionOffsetT = "ollama.transcription.offset_t"

// CtxKeyTranscriptionDurationMS is the optional duration_ms window, in milliseconds. Empty continues to the end.
const CtxKeyTranscriptionDurationMS = "ollama.transcription.duration_ms"

// CtxKeyTranscriptionSuppressRegex is the optional suppress_regex pattern. Empty leaves every token available.
const CtxKeyTranscriptionSuppressRegex = "ollama.transcription.suppress_regex"

// CtxKeyTranscriptionOffsetN is the optional SRT cue-number offset. Zero starts at 1.
const CtxKeyTranscriptionOffsetN = "ollama.transcription.offset_n"

// CtxKeyTranscriptionNoTimestamps is true when the form field no_timestamps is true.
const CtxKeyTranscriptionNoTimestamps = "ollama.transcription.no_timestamps"

// CtxKeyTranscriptionGrammar is the optional GBNF grammar. Empty leaves sampling unguided.
const CtxKeyTranscriptionGrammar = "ollama.transcription.grammar"

// CtxKeyTranscriptionGrammarRule is the grammar rule decoding starts from.
const CtxKeyTranscriptionGrammarRule = "ollama.transcription.grammar_rule"

// CtxKeyTranscriptionGrammarPenalty is the logit points subtracted from tokens the grammar rejects. Empty means 100.
const CtxKeyTranscriptionGrammarPenalty = "ollama.transcription.grammar_penalty"

// CtxKeyTranscriptionPrintSpecial is true when the form field print_special is true.
const CtxKeyTranscriptionPrintSpecial = "ollama.transcription.print_special"

// CtxKeyTranscriptionDetectLanguage is true when the form field detect_language is true.
const CtxKeyTranscriptionDetectLanguage = "ollama.transcription.detect_language"

// CtxKeyTranscriptionDTW is the optional DTW alignment preset. Empty leaves flash attention on.
const CtxKeyTranscriptionDTW = "ollama.transcription.dtw"

// CtxKeyTranscriptionVAD is true when the form field vad is true.
const CtxKeyTranscriptionVAD = "ollama.transcription.vad"

// CtxKeyTranscriptionVADThreshold is the optional Silero speech probability. Empty stays at 0.5.
const CtxKeyTranscriptionVADThreshold = "ollama.transcription.vad_threshold"

// CtxKeyTranscriptionVADMinSpeechMS is the optional minimum speech length in milliseconds. Empty stays at 250.
const CtxKeyTranscriptionVADMinSpeechMS = "ollama.transcription.vad_min_speech_duration_ms"

// CtxKeyTranscriptionVADMinSilenceMS is the optional pause that ends a speech region. Empty stays at 100.
const CtxKeyTranscriptionVADMinSilenceMS = "ollama.transcription.vad_min_silence_duration_ms"

// CtxKeyTranscriptionVADMaxSpeechS is the optional cap on a speech region, in seconds. Empty leaves the cap off.
const CtxKeyTranscriptionVADMaxSpeechS = "ollama.transcription.vad_max_speech_duration_s"

// CtxKeyTranscriptionVADSpeechPadMS is the optional padding around a speech region. Empty stays at 30.
const CtxKeyTranscriptionVADSpeechPadMS = "ollama.transcription.vad_speech_pad_ms"

// CtxKeyTranscriptionVADSamplesOverlap is the extra audio after each speech region except the last. Empty stays at 0.1.
const CtxKeyTranscriptionVADSamplesOverlap = "ollama.transcription.vad_samples_overlap"

// CtxKeyTranscriptionTemperatureInc is the optional temperature_inc step. Empty means 0.2.
const CtxKeyTranscriptionTemperatureInc = "ollama.transcription.temperature_inc"

// CtxKeyTranscriptionEntropyThold is the optional entropy_thold cutoff. Empty means 2.4.
const CtxKeyTranscriptionEntropyThold = "ollama.transcription.entropy_thold"

// CtxKeyTranscriptionLogprobThold is the optional logprob_thold cutoff. Empty means -1.
const CtxKeyTranscriptionLogprobThold = "ollama.transcription.logprob_thold"

// CtxKeyTranscriptionNoSpeechThold is the optional no_speech_thold cutoff. Empty means 0.6.
const CtxKeyTranscriptionNoSpeechThold = "ollama.transcription.no_speech_thold"

// CtxKeyTranscriptionBeamSize is the optional beam_size width. Empty means 5.
const CtxKeyTranscriptionBeamSize = "ollama.transcription.beam_size"

// CtxKeyTranscriptionBestOf is the optional best_of count. Empty means 5.
const CtxKeyTranscriptionBestOf = "ollama.transcription.best_of"

// CtxKeyTranscriptionMaxTokens is the optional max_tokens cap. Unset means no cap.
const CtxKeyTranscriptionMaxTokens = "ollama.transcription.max_tokens"

// CtxKeySpeechRequest holds a validated [openai.SpeechCreateRequest] for SpeechHandler.
const CtxKeySpeechRequest = "ollama.speech.request"

// CtxKeyVideoCreateRequest holds a validated [openai.VideoCreateRequest] for VideoCreateHandler.
const CtxKeyVideoCreateRequest = "ollama.video.create.request"

// CtxKeyRouteSurface tells GenerateHandler which surface a converted request came from
// ("image" | "image_edit"). WHY: image middlewares rewrite the body into a plain
// GenerateRequest, so the auto-router could not otherwise tell an image prompt from text
// and might resolve a router name to a chat model (docs/design/surfaces/auto-router.md §6.3).
const CtxKeyRouteSurface = "zerollama.route.surface"

// TranscriptionWriter collects streamed chat responses and outputs a transcription response.
type TranscriptionWriter struct {
	BaseWriter
	responseFormat string
	language       string
	task           string
	usage          *openai.TranscriptionUsage
	text           strings.Builder
}

func (w *TranscriptionWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != 0 && code != http.StatusOK {
		return w.writeError(data)
	}

	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		// Not a chat stream chunk — pass through (errors, plain text, etc.).
		return w.ResponseWriter.Write(data)
	}

	// Whisper subprocess path writes openai.TranscriptionResponse {"text":"..."} via
	// c.Data(). That JSON unmarshals into an empty ChatResponse (Done=false) and used
	// to be swallowed here. Detect by top-level "text" without a chat "message".
	var raw map[string]json.RawMessage
	if json.Unmarshal(data, &raw) == nil {
		if _, hasText := raw["text"]; hasText {
			if _, hasMessage := raw["message"]; !hasMessage {
				return w.ResponseWriter.Write(data)
			}
		}
	}

	w.text.WriteString(chatResponse.Message.Content)

	if chatResponse.Done {
		text := strings.TrimSpace(w.text.String())

		ct, body, err := openai.TranscriptionResult(text, w.responseFormat, w.language, w.task, w.usage, nil, nil)
		if err != nil {
			return 0, err
		}
		w.ResponseWriter.Header().Set("Content-Type", ct)
		_, err = w.ResponseWriter.Write(body)
		if err != nil {
			return 0, err
		}
		return len(data), nil
	}

	return len(data), nil
}

// TranscriptionMiddleware handles /v1/audio/transcriptions requests.
// It accepts multipart/form-data with an audio file and converts it to a chat request.
func TranscriptionMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Parse multipart form (limit 25MB).
		if err := c.Request.ParseMultipartForm(25 << 20); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "failed to parse multipart form: "+err.Error()))
			return
		}

		model := c.Request.FormValue("model")
		if model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
			return
		}

		file, header, err := c.Request.FormFile("file")
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "file is required: "+err.Error()))
			return
		}
		defer file.Close()

		audioData, err := io.ReadAll(file)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, "failed to read audio file"))
			return
		}

		if len(audioData) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "audio file is empty"))
			return
		}

		temperature, terr := openai.CleanTranscriptionTemperature(c.Request.FormValue("temperature"))
		if terr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, terr.Error()))
			return
		}
		maxTokens, merr := transcript.MaxTokens(c.Request.FormValue("max_tokens"))
		if merr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, merr.Error()))
			return
		}
		stream, serr := transcript.StreamFlag(c.Request.FormValue("stream"))
		if serr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, serr.Error()))
			return
		}
		req := openai.TranscriptionRequest{
			Model:          model,
			AudioData:      audioData,
			ResponseFormat: c.Request.FormValue("response_format"),
			Language:       c.Request.FormValue("language"),
			Prompt:         c.Request.FormValue("prompt"),
			Temperature:    temperature,
			MaxTokens:      maxTokens,
		}
		gran := append([]string{}, c.Request.Form["timestamp_granularities[]"]...)
		gran = append(gran, c.Request.Form["timestamp_granularities"]...)
		words, gerr := whispercli.WordTimestamps(gran)
		if gerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, gerr.Error()))
			return
		}
		if words && req.ResponseFormat != "verbose_json" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "timestamp_granularities=word requires response_format=verbose_json"))
			return
		}
		inc := append([]string{}, c.Request.Form["include[]"]...)
		inc = append(inc, c.Request.Form["include"]...)
		diarize, derr := whispercli.DiarizeFlag(c.Request.FormValue("diarize"))
		if derr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, derr.Error()))
			return
		}
		tinyDiarize, tderr := whispercli.TinyDiarizeFlag(c.Request.FormValue("tinydiarize"))
		if tderr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, tderr.Error()))
			return
		}
		suppressNST, nerr := whispercli.SuppressNSTFlag(c.Request.FormValue("suppress_nst"))
		if nerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, nerr.Error()))
			return
		}
		noFallback, ferr := whispercli.NoFallbackFlag(c.Request.FormValue("no_fallback"))
		if ferr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, ferr.Error()))
			return
		}
		carryInitial, cerr := whispercli.CarryInitialFlag(c.Request.FormValue("carry_initial_prompt"))
		if cerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, cerr.Error()))
			return
		}
		maxLen, mlerr := whispercli.MaxLen(c.Request.FormValue("max_len"))
		if mlerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, mlerr.Error()))
			return
		}
		splitOnWord, sowerr := whispercli.SplitOnWordFlag(c.Request.FormValue("split_on_word"))
		if sowerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, sowerr.Error()))
			return
		}
		wordThold, wterr := whispercli.WordThold(c.Request.FormValue("word_thold"))
		if wterr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, wterr.Error()))
			return
		}
		maxContext, cxerr := whispercli.MaxContext(c.Request.FormValue("max_context"))
		if cxerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, cxerr.Error()))
			return
		}
		audioCtx, acerr := whispercli.AudioCtx(c.Request.FormValue("audio_ctx"))
		if acerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, acerr.Error()))
			return
		}
		offsetT, oterr := whispercli.OffsetT(c.Request.FormValue("offset_t"))
		if oterr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, oterr.Error()))
			return
		}
		durationMS, dmerr := whispercli.DurationMS(c.Request.FormValue("duration_ms"))
		if dmerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, dmerr.Error()))
			return
		}
		suppressRegex, rxerr := whispercli.SuppressRegex(c.Request.FormValue("suppress_regex"))
		if rxerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, rxerr.Error()))
			return
		}
		offsetN, onerr := whispercli.OffsetN(c.Request.FormValue("offset_n"))
		if onerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, onerr.Error()))
			return
		}
		noTimestamps, nterr := whispercli.NoTimestampsFlag(c.Request.FormValue("no_timestamps"))
		if nterr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, nterr.Error()))
			return
		}
		printSpecial, pserr := whispercli.PrintSpecialFlag(c.Request.FormValue("print_special"))
		if pserr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, pserr.Error()))
			return
		}
		detectLanguage, dlerr := whispercli.DetectLanguageFlag(c.Request.FormValue("detect_language"))
		if dlerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, dlerr.Error()))
			return
		}
		dtw, dterr := whispercli.DTWPreset(c.Request.FormValue("dtw"))
		if dterr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, dterr.Error()))
			return
		}
		vad, verr := whispercli.VADFlag(c.Request.FormValue("vad"))
		if verr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, verr.Error()))
			return
		}
		vadThreshold, vterr := whispercli.VADThreshold(c.Request.FormValue("vad_threshold"))
		if vterr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, vterr.Error()))
			return
		}
		vadMinSpeech, vserr := whispercli.VADMinSpeechMS(c.Request.FormValue("vad_min_speech_duration_ms"))
		if vserr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, vserr.Error()))
			return
		}
		vadMinSilence, vsierr := whispercli.VADMinSilenceMS(c.Request.FormValue("vad_min_silence_duration_ms"))
		if vsierr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, vsierr.Error()))
			return
		}
		vadMaxSpeech, vmxerr := whispercli.VADMaxSpeechS(c.Request.FormValue("vad_max_speech_duration_s"))
		if vmxerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, vmxerr.Error()))
			return
		}
		vadSpeechPad, vperr := whispercli.VADSpeechPadMS(c.Request.FormValue("vad_speech_pad_ms"))
		if vperr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, vperr.Error()))
			return
		}
		vadOverlap, voerr := whispercli.VADSamplesOverlap(c.Request.FormValue("vad_samples_overlap"))
		if voerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, voerr.Error()))
			return
		}
		grammar, grammarRule, gerr := whispercli.Grammar(c.Request.FormValue("grammar"), c.Request.FormValue("grammar_rule"))
		if gerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, gerr.Error()))
			return
		}
		grammarPenalty, gperr := whispercli.GrammarPenalty(c.Request.FormValue("grammar_penalty"))
		if gperr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, gperr.Error()))
			return
		}
		temperatureInc, ierrInc := whispercli.TemperatureInc(c.Request.FormValue("temperature_inc"))
		if ierrInc != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, ierrInc.Error()))
			return
		}
		entropyThold, eerr := whispercli.EntropyThold(c.Request.FormValue("entropy_thold"))
		if eerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, eerr.Error()))
			return
		}
		logprobThold, lperr := whispercli.LogprobThold(c.Request.FormValue("logprob_thold"))
		if lperr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, lperr.Error()))
			return
		}
		noSpeechThold, nserr := whispercli.NoSpeechThold(c.Request.FormValue("no_speech_thold"))
		if nserr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, nserr.Error()))
			return
		}
		beamSize, berr := whispercli.BeamSize(c.Request.FormValue("beam_size"))
		if berr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, berr.Error()))
			return
		}
		bestOf, boerr := whispercli.BestOf(c.Request.FormValue("best_of"))
		if boerr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, boerr.Error()))
			return
		}
		logprobs, ierr := whispercli.IncludeLogprobs(inc)
		if ierr != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, ierr.Error()))
			return
		}
		if logprobs && req.ResponseFormat != "" && req.ResponseFormat != "json" && req.ResponseFormat != "verbose_json" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "include=logprobs requires response_format json or verbose_json"))
			return
		}

		translate := strings.HasSuffix(c.Request.URL.Path, "/translations")
		chatReq, err := openai.FromTranscriptionRequest(req, translate)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)
		c.Request.ContentLength = int64(b.Len())
		c.Request.Header.Set("Content-Type", "application/json")

		c.Set(CtxKeyTranscriptionResponseFormat, req.ResponseFormat)
		c.Set(CtxKeyTranscriptionOriginalFilename, header.Filename)
		c.Set(CtxKeyTranscriptionLanguage, req.Language)
		c.Set(CtxKeyTranscriptionPrompt, req.Prompt)
		c.Set(CtxKeyTranscriptionTemperature, req.Temperature)
		c.Set(CtxKeyTranscriptionWords, words)
		c.Set(CtxKeyTranscriptionStream, stream)
		c.Set(CtxKeyTranscriptionLogprobs, logprobs)
		c.Set(CtxKeyTranscriptionDiarize, diarize)
		c.Set(CtxKeyTranscriptionTinyDiarize, tinyDiarize)
		c.Set(CtxKeyTranscriptionSplitOnWord, splitOnWord)
		c.Set(CtxKeyTranscriptionSuppressNST, suppressNST)
		c.Set(CtxKeyTranscriptionNoFallback, noFallback)
		c.Set(CtxKeyTranscriptionCarryInitial, carryInitial)
		if maxLen != "" {
			c.Set(CtxKeyTranscriptionMaxLen, maxLen)
		}
		if wordThold != "" {
			c.Set(CtxKeyTranscriptionWordThold, wordThold)
		}
		if maxContext != "" {
			c.Set(CtxKeyTranscriptionMaxContext, maxContext)
		}
		if audioCtx != "" {
			c.Set(CtxKeyTranscriptionAudioCtx, audioCtx)
		}
		if offsetT != "" {
			c.Set(CtxKeyTranscriptionOffsetT, offsetT)
		}
		if durationMS != "" {
			c.Set(CtxKeyTranscriptionDurationMS, durationMS)
		}
		if suppressRegex != "" {
			c.Set(CtxKeyTranscriptionSuppressRegex, suppressRegex)
		}
		if offsetN != 0 {
			c.Set(CtxKeyTranscriptionOffsetN, offsetN)
		}
		c.Set(CtxKeyTranscriptionNoTimestamps, noTimestamps)
		c.Set(CtxKeyTranscriptionPrintSpecial, printSpecial)
		c.Set(CtxKeyTranscriptionDetectLanguage, detectLanguage)
		if dtw != "" {
			c.Set(CtxKeyTranscriptionDTW, dtw)
		}
		c.Set(CtxKeyTranscriptionVAD, vad)
		if vadThreshold != "" {
			c.Set(CtxKeyTranscriptionVADThreshold, vadThreshold)
		}
		if vadMinSpeech != "" {
			c.Set(CtxKeyTranscriptionVADMinSpeechMS, vadMinSpeech)
		}
		if vadMinSilence != "" {
			c.Set(CtxKeyTranscriptionVADMinSilenceMS, vadMinSilence)
		}
		if vadMaxSpeech != "" {
			c.Set(CtxKeyTranscriptionVADMaxSpeechS, vadMaxSpeech)
		}
		if vadSpeechPad != "" {
			c.Set(CtxKeyTranscriptionVADSpeechPadMS, vadSpeechPad)
		}
		if vadOverlap != "" {
			c.Set(CtxKeyTranscriptionVADSamplesOverlap, vadOverlap)
		}
		if grammar != "" {
			c.Set(CtxKeyTranscriptionGrammar, grammar)
			c.Set(CtxKeyTranscriptionGrammarRule, grammarRule)
		}
		if grammarPenalty != "" {
			c.Set(CtxKeyTranscriptionGrammarPenalty, grammarPenalty)
		}
		if temperatureInc != "" {
			c.Set(CtxKeyTranscriptionTemperatureInc, temperatureInc)
		}
		if entropyThold != "" {
			c.Set(CtxKeyTranscriptionEntropyThold, entropyThold)
		}
		if logprobThold != "" {
			c.Set(CtxKeyTranscriptionLogprobThold, logprobThold)
		}
		if noSpeechThold != "" {
			c.Set(CtxKeyTranscriptionNoSpeechThold, noSpeechThold)
		}
		if beamSize != "" {
			c.Set(CtxKeyTranscriptionBeamSize, beamSize)
		}
		if bestOf != "" {
			c.Set(CtxKeyTranscriptionBestOf, bestOf)
		}
		if maxTokens != nil {
			c.Set(CtxKeyTranscriptionMaxTokens, *maxTokens)
		}

		respLang := req.Language
		task := "transcribe"
		if translate {
			respLang = "en"
			task = "translate"
		}
		var clipUsage *openai.TranscriptionUsage
		if sec, ok := whispercli.WavSeconds(audioData); ok {
			clipUsage = &openai.TranscriptionUsage{Type: "duration", Seconds: sec}
		}
		w := &TranscriptionWriter{
			BaseWriter:     BaseWriter{ResponseWriter: c.Writer},
			responseFormat: req.ResponseFormat,
			language:       respLang,
			task:           task,
			usage:          clipUsage,
		}

		c.Writer = w
		c.Next()
	}
}

// SpeechMiddleware validates OpenAI POST /v1/audio/speech JSON and stores the request on context.
func SpeechMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.SpeechCreateRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if req.Model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
			return
		}
		if req.Input == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "input is required"))
			return
		}
		maxRunes := 4096
		if strings.TrimSpace(req.Instructions) != "" || req.MaxNewTokens != nil || req.Duration != nil {
			maxRunes = 20000
		}
		if utf8RuneCount(req.Input) > maxRunes {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("input exceeds %d characters", maxRunes)))
			return
		}
		if req.Speed != nil && (*req.Speed < 0.25 || *req.Speed > 4.0) {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "speed must be between 0.25 and 4.0"))
			return
		}
		if _, err := piperphrase.SentenceSilence(req.SentenceSilence); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if _, err := piperphrase.NoiseScale(req.NoiseScale); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if _, err := piperphrase.NoiseW(req.NoiseW); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if _, err := piperphrase.SpeakerID(req.SpeakerID); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if _, err := piperphrase.PhonemeSilence(req.PhonemeSilence); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if strings.TrimSpace(req.Voice) != "" && strings.TrimSpace(req.SpeakerRef) != "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "voice and speaker_ref cannot be used together"))
			return
		}
		if _, err := openai.CleanSpeechLang(req.Lang); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if _, err := openai.DecodeSpeakerRef(req.SpeakerRef); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if err := openai.CleanSpeechTopK(req.TopK); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if err := openai.CleanSpeechTopP(req.TopP); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if err := openai.CleanSpeechRepeatPenalty(req.RepeatPenalty); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		c.Set(CtxKeySpeechRequest, req)
		c.Next()
	}
}

// VideoCreateMiddleware validates OpenAI POST /v1/videos JSON and stores the request on context.
// Body is capped so agents use /v1/media/{session}/{label} for large keyframes instead of inline JSON.
// WHY 8 MiB (media.MaxVideoCreateBody): create should carry label refs only; frames stream via PUT
// (docs/design/media/media-uploads.md). A larger cap would invite base64-in-JSON and defeat the media API.
func VideoCreateMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, media.MaxVideoCreateBody)
		var req openai.VideoCreateRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			msg := err.Error()
			if strings.Contains(msg, "request body too large") || errors.As(err, new(*http.MaxBytesError)) {
				c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, openai.NewError(http.StatusRequestEntityTooLarge, "request body too large; upload keyframes via PUT /v1/media/{session}/{label}"))
				return
			}
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if req.Model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
			return
		}
		if strings.TrimSpace(req.Prompt) == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt is required"))
			return
		}
		c.Set(CtxKeyVideoCreateRequest, req)
		c.Next()
	}
}

func utf8RuneCount(s string) int {
	n := 0
	for range s {
		n++
	}
	return n
}

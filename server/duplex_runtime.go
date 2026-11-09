package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/server/voicepipe"
)

// replyRuntimeChat streams one duplex reply through the Python runtime
// /api/chat path. Plain text GGUFs on this host are runtime-only, so
// scheduleRunner returns ErrRuntimeInferenceModel and the ear hears the
// transcript with no spoken answer.
//
// predict is the token cap. Zero uses the duplex reply budget. The think
// field is left unset: the runtime treats any think value as a legacy-runner
// request and answers 501.
func (s *Server) replyRuntimeChat(ctx context.Context, modelName string, m *Model, instructions, user string, prior []voicepipe.Exchange, emit func(string) error, predict int, stages *voicepipe.StageReporter) error {
	if m == nil {
		return fmt.Errorf("runtime chat: model is missing")
	}
	baseURL := effectiveRuntimeURL()
	if baseURL == "" {
		return fmt.Errorf("runtime chat: ZEROLLAMA_RUNTIME_URL is unset")
	}
	merged, err := s.modelOptions(m, nil)
	if err != nil {
		return err
	}
	opt := duplexOptions(&merged)
	if predict > 0 {
		opt.NumPredict = predict
	}
	raw, err := json.Marshal(opt)
	if err != nil {
		return err
	}
	var clientOpts map[string]any
	if err := json.Unmarshal(raw, &clientOpts); err != nil {
		return err
	}
	rtOpts := runtimeProxyOptions(modelName, opt.NumPredict, opt.NumPredict > 0, clientOpts)
	gguf, _ := rtOpts["gguf"].(string)
	if err := s.prepareRuntimeVRAM(ctx, gguf, false); err != nil {
		return err
	}
	msgs := duplexMessages(instructions, user, prior)
	payload := runtimeChatPayload(api.ChatRequest{
		Model:    modelName,
		Messages: msgs,
	}, rtOpts, true)
	// Any think value makes the runtime return 501 and demand the legacy runner.
	delete(payload, "think")
	slog.Info("duplex reply via runtime", "model", modelName, "predict", opt.NumPredict)
	return postRuntimeChat(ctx, baseURL, payload, emit, stages)
}

func postRuntimeChat(ctx context.Context, baseURL string, payload map[string]any, emit func(string) error, stages *voicepipe.StageReporter) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(baseURL, "/")+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := runtimeProxyClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
		return runtimeChatHTTPError(resp.StatusCode, respBody)
	}
	if err := consumeRuntimeChatNDJSON(ctx, resp.Body, emit, stages); err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	return nil
}

func consumeRuntimeChatNDJSON(ctx context.Context, r io.Reader, emit func(string) error, stages *voicepipe.StageReporter) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
	var stream voicepipe.SentenceStream
	first := true
	for sc.Scan() {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var chunk struct {
			Error   string `json:"error"`
			Done    bool   `json:"done"`
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		}
		if err := json.Unmarshal(line, &chunk); err != nil {
			return fmt.Errorf("runtime chat: %w", err)
		}
		if chunk.Error != "" {
			return runtimeChatFailure(chunk.Error)
		}
		if chunk.Message.Content != "" {
			if first {
				first = false
				stages.Report(voicepipe.StageFirstToken, nil)
			}
			if err := stream.Push(chunk.Message.Content, emit); err != nil {
				return err
			}
		}
		if chunk.Done {
			break
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	return stream.Flush(emit)
}

func runtimeChatHTTPError(status int, body []byte) error {
	text := strings.TrimSpace(string(body))
	var obj map[string]any
	if json.Unmarshal(body, &obj) == nil {
		if e, ok := obj["error"].(string); ok && strings.TrimSpace(e) != "" {
			text = e
		} else if d, ok := obj["detail"].(string); ok && strings.TrimSpace(d) != "" {
			text = d
		}
	}
	if text == "" {
		text = http.StatusText(status)
	}
	return runtimeChatFailure(text)
}

// runtimeChatFailure turns a runtime / llama-server error string into a short
// socket message. Load races become model_load_failed; other failures keep the
// inner text without the HTTP JSON wrapper.
func runtimeChatFailure(raw string) error {
	msg := strings.TrimSpace(raw)
	msg = strings.TrimPrefix(msg, "runtime chat: ")
	msg = strings.TrimSpace(msg)
	if msg == "" {
		msg = "runtime chat failed"
	}
	if inner := unwrapLlamaServerMessage(msg); inner != "" {
		msg = inner
	}
	if isRuntimeModelLoadFailure(msg) {
		return &voicepipe.Error{
			Message: "the chat model did not finish loading",
			Code:    "model_load_failed",
		}
	}
	return &voicepipe.Error{Message: msg}
}

func isRuntimeModelLoadFailure(msg string) bool {
	lower := strings.ToLower(msg)
	switch {
	case strings.Contains(lower, "loading model"):
		return true
	case strings.Contains(lower, "not healthy"):
		return true
	case strings.Contains(lower, "did not finish loading"):
		return true
	default:
		return false
	}
}

func unwrapLlamaServerMessage(msg string) string {
	const prefix = "llama-server /completion HTTP "
	if rest, ok := strings.CutPrefix(msg, prefix); ok {
		if i := strings.Index(rest, ": "); i >= 0 {
			msg = strings.TrimSpace(rest[i+2:])
		}
	}
	start := strings.Index(msg, "{")
	if start < 0 {
		return strings.TrimSpace(msg)
	}
	var nested struct {
		Error any `json:"error"`
	}
	if json.Unmarshal([]byte(msg[start:]), &nested) != nil {
		return strings.TrimSpace(msg)
	}
	switch e := nested.Error.(type) {
	case string:
		if strings.TrimSpace(e) != "" {
			return strings.TrimSpace(e)
		}
	case map[string]any:
		if m, ok := e["message"].(string); ok && strings.TrimSpace(m) != "" {
			return strings.TrimSpace(m)
		}
	}
	return strings.TrimSpace(msg)
}

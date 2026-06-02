package server

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// runtimeProxyActive is true when env or header opts in without per-model config.
// Why X-Zerollama-Runtime header: lets smoke/integration force proxy for a model name that
// is not in the Ollama library (avoids 404 from GenerateHandler manifest lookup).
func runtimeProxyActive(c *gin.Context) bool {
	if effectiveRuntimeURL() == "" {
		return false
	}
	if envconfig.RuntimeProxyAll() {
		return true
	}
	if c != nil && c.GetHeader("X-Zerollama-Runtime") == "1" {
		return true
	}
	return false
}

func resolveRuntimeProxy(c *gin.Context, modelName string) bool {
	if effectiveRuntimeURL() == "" {
		return false
	}
	if runtimeProxyActive(c) {
		return true
	}
	modelRef, err := parseAndValidateModelRef(modelName)
	if err != nil {
		return false
	}
	name, err := getExistingName(modelRef.Name)
	if err != nil {
		return false
	}
	m, err := GetModel(name.String())
	if err != nil {
		return false
	}
	return modelUsesRuntimeInference(m)
}

// numPredictFromOptions returns an output token limit when the client set num_predict > 0.
// Ollama's default is NumPredict: -1 (no limit); the runtime proxy must not impose a cap.
func numPredictFromOptions(options map[string]any) (int, bool) {
	if options == nil {
		return 0, false
	}
	opts := api.DefaultOptions()
	if err := opts.FromMap(options); err == nil && opts.NumPredict > 0 {
		return opts.NumPredict, true
	}
	if v, ok := options["num_predict"]; ok {
		switch n := v.(type) {
		case int:
			if n > 0 {
				return n, true
			}
		case int64:
			if n > 0 {
				return int(n), true
			}
		case float64:
			if n > 0 {
				return int(n), true
			}
		}
	}
	return 0, false
}

func ollamaWantsStream(stream *bool) bool {
	return stream == nil || *stream
}

func forwardRuntimeNDJSON(
	c *gin.Context,
	path string,
	payload map[string]any,
) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	target := effectiveRuntimeURL() + path
	outReq, err := http.NewRequestWithContext(
		c.Request.Context(),
		http.MethodPost,
		target,
		bytes.NewReader(body),
	)
	if err != nil {
		return err
	}
	outReq.Header.Set("Content-Type", "application/json")

	resp, err := runtimeProxyClient.Do(outReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		c.Data(resp.StatusCode, "application/json", respBody)
		c.Abort()
		return nil
	}

	c.Header("Content-Type", "application/x-ndjson")
	c.Status(http.StatusOK)
	_, err = io.Copy(c.Writer, resp.Body)
	c.Abort()
	return err
}

func runtimeOptionsWithNumPredict(nPredict int, limited bool) map[string]any {
	if !limited {
		return map[string]any{}
	}
	return map[string]any{"num_predict": nPredict}
}

func forwardRuntimeGenerate(
	c *gin.Context,
	modelName string,
	prompt string,
	options map[string]any,
) (string, int, error) {
	payload, _ := json.Marshal(map[string]any{
		"model":   modelName,
		"prompt":  prompt,
		"stream":  false,
		"options": options,
	})
	target := effectiveRuntimeURL() + "/api/generate"
	outReq, err := http.NewRequestWithContext(
		c.Request.Context(),
		http.MethodPost,
		target,
		bytes.NewReader(payload),
	)
	if err != nil {
		return "", http.StatusInternalServerError, err
	}
	outReq.Header.Set("Content-Type", "application/json")

	resp, err := runtimeProxyClient.Do(outReq)
	if err != nil {
		return "", http.StatusBadGateway, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", http.StatusBadGateway, err
	}
	if resp.StatusCode >= 300 {
		return string(respBody), resp.StatusCode, nil
	}
	return parseRuntimeGenerateBody(respBody), http.StatusOK, nil
}

func parseRuntimeGenerateBody(respBody []byte) string {
	var ollamaResp struct {
		Response string `json:"response"`
	}
	if err := json.Unmarshal(respBody, &ollamaResp); err == nil && ollamaResp.Response != "" {
		return ollamaResp.Response
	}
	var comp runtimeCompletionResp
	if err := json.Unmarshal(respBody, &comp); err == nil && comp.Content != "" {
		return comp.Content
	}
	var generic map[string]any
	if json.Unmarshal(respBody, &generic) == nil {
		if v, ok := generic["content"].(string); ok {
			return v
		}
		if v, ok := generic["response"].(string); ok {
			return v
		}
	}
	return ""
}

func chatMessagesToPrompt(messages []api.Message) string {
	var b bytes.Buffer
	for _, m := range messages {
		switch m.Role {
		case "system":
			b.WriteString("System: ")
			b.WriteString(m.Content)
			b.WriteString("\n")
		case "user":
			b.WriteString("User: ")
			b.WriteString(m.Content)
			b.WriteString("\n")
		case "assistant":
			b.WriteString("Assistant: ")
			b.WriteString(m.Content)
			b.WriteString("\n")
		default:
			b.WriteString(m.Content)
			b.WriteString("\n")
		}
	}
	return strings.TrimSpace(b.String())
}

func chatNeedsLegacyRunner(messages []api.Message, req api.ChatRequest) bool {
	if req.Logprobs || req.Think != nil {
		return true
	}
	for _, m := range messages {
		if len(m.Images) > 0 || len(m.Videos) > 0 {
			return true
		}
		if m.Thinking != "" {
			return true
		}
	}
	return false
}

func writeRuntimeProxyError(c *gin.Context, err error) {
	slog.Warn("runtime proxy: request failed", "error", err)
	c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
}

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
	"github.com/ollama/ollama/openai"
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

func resolveRuntimeProxy(c *gin.Context, modelName string, opts map[string]any) bool {
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
	if modelUsesRuntimeInference(m) {
		return true
	}
	if agentCachePrefersRuntime(opts) && modelEligibleForAgentCacheRuntime(m) {
		return true
	}
	return false
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

func writeRuntimeStreamAccepted(c *gin.Context, model string, chat bool) error {
	c.Header("Content-Type", "application/x-ndjson")
	c.Status(http.StatusOK)
	var chunk any
	if chat {
		chunk = chatStatusChunk(model, "accepted", "request accepted", 0, 0)
	} else {
		chunk = generateStatusChunk(model, "accepted", "request accepted", 0, 0)
	}
	bts, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	bts = append(bts, '\n')
	_, err = c.Writer.Write(bts)
	return err
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
		respBody = injectRuntimeRetryAfter(c, resp.StatusCode, respBody)
		recordRuntimeProxyErrorMetrics(resp.StatusCode, respBody)
		c.Data(resp.StatusCode, "application/json", respBody)
		c.Abort()
		return nil
	}

	c.Header("Content-Type", "application/x-ndjson")
	c.Status(http.StatusOK)
	if err := copyRuntimeResponseBody(c.Writer, resp.Body); err != nil {
		metricsIncRequestResult("error")
		return err
	}
	metricsIncRequestResult("ok")
	c.Abort()
	return nil
}

func runtimeOptionsWithNumPredict(nPredict int, limited bool) map[string]any {
	if !limited {
		return map[string]any{}
	}
	return map[string]any{"num_predict": nPredict}
}

// thinkNeedsLegacyRunner is true when the client enabled thinking (not merely omitted).
func thinkNeedsLegacyRunner(think *api.ThinkValue) bool {
	if think == nil {
		return false
	}
	if think.IsString() {
		return strings.TrimSpace(think.String()) != ""
	}
	return think.Bool()
}

// proxyOptsFromV1Body extracts Ollama-style options from an OpenAI v1 chat body.
// WHY fold here: flat SDK-flattened harness keys (qos_class, project_name, …)
// must become options.zerollama for Go routing (resolveRuntimeProxy /
// runtimeForceUnload) — same contract as openai.BindChatCompletionRequest.
// runtimeV1ProxyOptions reuses this so the Python sidecar sees the fold too.
func proxyOptsFromV1Body(body map[string]any) map[string]any {
	if body == nil {
		return nil
	}
	var out map[string]any
	if opts, ok := body["options"].(map[string]any); ok && len(opts) > 0 {
		out = make(map[string]any, len(opts)+1)
		for k, v := range opts {
			out[k] = v
		}
	}
	// Flat top-level aliases first (weaker than nested options.zerollama).
	out = openai.FoldFlatZerollamaMap(out, body)
	mergePromptCacheKeyIntoProxyOpts(&out, body)
	if eb, ok := body["extra_body"].(map[string]any); ok {
		out = openai.FoldFlatZerollamaMap(out, eb)
		if opts, ok := eb["options"].(map[string]any); ok {
			out = mergeOptionsMaps(out, opts)
		}
		if z, ok := eb["zerollama"].(map[string]any); ok {
			// extra_body.zerollama overlays flat aliases; options.zerollama still wins below.
			out = mergeZerollamaIntoOptions(out, map[string]any{"zerollama": z})
		}
		mergePromptCacheKeyIntoProxyOpts(&out, eb)
	}
	// Top-level zerollama (SDK-flattened extra_body.zerollama) underlays nested
	// options.zerollama — same precedence as openai.BindChatCompletionRequest.
	out = underlayZerollamaIntoOptions(out, body)
	if len(out) == 0 {
		return nil
	}
	return out
}

// underlayZerollamaIntoOptions merges src["zerollama"] beneath any existing
// options.zerollama keys (nested/explicit wins on conflict; underlay fills gaps).
// WHY: top-level zerollama after SDK flatten must not clobber body.options.zerollama.
func underlayZerollamaIntoOptions(opts map[string]any, src map[string]any) map[string]any {
	if src == nil {
		return opts
	}
	z, ok := src["zerollama"].(map[string]any)
	if !ok || len(z) == 0 {
		return opts
	}
	if opts == nil {
		opts = map[string]any{}
	}
	existing, _ := opts["zerollama"].(map[string]any)
	opts["zerollama"] = mergeOptionsMaps(z, existing)
	return opts
}

func mergePromptCacheKeyIntoProxyOpts(opts *map[string]any, src map[string]any) {
	if src == nil {
		return
	}
	v, ok := src["prompt_cache_key"].(string)
	if !ok || strings.TrimSpace(v) == "" {
		return
	}
	key := strings.TrimSpace(v)
	if *opts == nil {
		*opts = map[string]any{"prompt_cache_key": key}
		return
	}
	if _, has := (*opts)["prompt_cache_key"]; !has {
		(*opts)["prompt_cache_key"] = key
	}
}

func forwardRuntimeGenerate(
	c *gin.Context,
	modelName string,
	prompt string,
	options map[string]any,
	format json.RawMessage,
) ([]byte, int, error) {
	body := map[string]any{
		"model":   modelName,
		"prompt":  prompt,
		"stream":  false,
		"options": options,
	}
	if len(format) > 0 {
		body["format"] = json.RawMessage(format)
	}
	payload, _ := json.Marshal(body)
	target := effectiveRuntimeURL() + "/api/generate"
	outReq, err := http.NewRequestWithContext(
		c.Request.Context(),
		http.MethodPost,
		target,
		bytes.NewReader(payload),
	)
	if err != nil {
		return nil, http.StatusInternalServerError, err
	}
	outReq.Header.Set("Content-Type", "application/json")

	resp, err := runtimeProxyClient.Do(outReq)
	if err != nil {
		return nil, http.StatusBadGateway, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, http.StatusBadGateway, err
	}
	return respBody, resp.StatusCode, nil
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
	if req.Logprobs || thinkNeedsLegacyRunner(req.Think) {
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
	if isRequestTimeout(err) {
		var to *api.Duration
		if v, ok := c.Get("request_timeout"); ok {
			to, _ = v.(*api.Duration)
		}
		writeRequestTimeout(c, to)
		c.Abort()
		return
	}
	slog.Warn("runtime proxy: request failed", "error", err)
	msg := err.Error()
	if isHostUnstableError(strings.ToLower(msg)) {
		metricsIncRunnerCrash()
		metricsIncRequestResult("host_unstable")
	} else {
		metricsIncRequestResult("error")
	}
	c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": msg})
}

// copyRuntimeResponseBody streams a runtime response to the client with flush after
// each chunk — why: SSE/NDJSON proxies through Gin otherwise buffer until EOF and
// curl without --max-time can hang for minutes on partial streams.
func copyRuntimeResponseBody(w http.ResponseWriter, r io.Reader) error {
	flusher, canFlush := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		nr, err := r.Read(buf)
		if nr > 0 {
			if _, ew := w.Write(buf[:nr]); ew != nil {
				return ew
			}
			if canFlush {
				flusher.Flush()
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/envconfig"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/version"
)

// Default cloud upstream for Zerollama is Eliza (OpenAI/Anthropic-compatible APIs, API-key auth).
// Legacy ollama.com remains available via OLLAMA_CLOUD_BASE_URL; Ed25519 signing applies only there
// because that host's contract uses signed requests, not X-API-Key. See docs/eliza-cloud.md.
const (
	defaultCloudProxyBaseURL      = "https://www.elizacloud.ai:443"
	cloudProxyBaseURLEnv          = "OLLAMA_CLOUD_BASE_URL"
	legacyCloudAnthropicKey       = "legacy_cloud_anthropic_web_search"
	cloudProxyClientVersionHeader = "X-Ollama-Client-Version"

	// maxDecompressedBodySize limits the size of a decompressed request body
	maxDecompressedBodySize = 20 << 20
)

var (
	cloudProxyBaseURL     = defaultCloudProxyBaseURL
	cloudProxySignRequest = signCloudProxyRequest
	cloudProxySigninURL   = signinURL
)

var elizaAPIKeyMissingWarnOnce sync.Once

var hopByHopHeaders = map[string]struct{}{
	"connection":          {},
	"content-length":      {},
	"proxy-connection":    {},
	"keep-alive":          {},
	"proxy-authenticate":  {},
	"proxy-authorization": {},
	"te":                  {},
	"trailer":             {},
	"transfer-encoding":   {},
	"upgrade":             {},
}

func init() {
	baseURL, overridden, err := resolveCloudProxyBaseURL(envconfig.Var(cloudProxyBaseURLEnv), mode)
	if err != nil {
		slog.Warn("ignoring cloud base URL override", "env", cloudProxyBaseURLEnv, "error", err)
		return
	}

	cloudProxyBaseURL = baseURL

	if overridden {
		slog.Info("cloud base URL override enabled", "env", cloudProxyBaseURLEnv, "url", cloudProxyBaseURL, "mode", mode)
	}
}

func cloudPassthroughMiddleware(_ string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method == http.MethodPost && c.GetHeader("Content-Encoding") == "zstd" {
			zr, err := zstd.NewReader(c.Request.Body, zstd.WithDecoderMaxMemory(8<<20))
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "failed to decompress request body"})
				c.Abort()
				return
			}
			defer zr.Close()
			body, err := io.ReadAll(http.MaxBytesReader(c.Writer, io.NopCloser(zr), maxDecompressedBodySize))
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "failed to decompress request body"})
				c.Abort()
				return
			}
			c.Request.Body = io.NopCloser(bytes.NewReader(body))
			c.Request.Header.Del("Content-Encoding")
		}
		c.Next()
	}
}

func cloudModelPathPassthroughMiddleware(_ string) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Next()
	}
}

// cloudV1InferencePassthrough forwards JSON bodies with top-level "model" to the cloud upstream
// before local conversion when the model uses the :cloud source. Multipart routes (e.g. some image flows)
// may not expose "model" in raw JSON here and fall through to local handling.
func cloudV1InferencePassthrough(disabledOperation string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method != http.MethodPost {
			c.Next()
			return
		}
		body, err := readRequestBody(c.Request)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			c.Abort()
			return
		}
		model, ok := extractModelField(body)
		if !ok {
			c.Next()
			return
		}
		modelRef, err := parseAndValidateModelRef(model)
		if err != nil || modelRef.Source != modelSourceCloud {
			c.Next()
			return
		}
		body, err = replaceJSONModelField(body, modelRef.Base)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			c.Abort()
			return
		}
		proxyCloudRequestWithPath(c, body, c.Request.URL.Path, disabledOperation)
		c.Abort()
	}
}

func proxyCloudJSONRequest(c *gin.Context, payload any, disabledOperation string) {
	// TEMP(drifkin): we currently split out this `WithPath` method because we are
	// mapping `/v1/messages` + web_search to `/api/chat` temporarily. Once we
	// stop doing this, we can inline this method.
	proxyCloudJSONRequestWithPath(c, payload, c.Request.URL.Path, disabledOperation)
}

func proxyCloudJSONRequestWithPath(c *gin.Context, payload any, path string, disabledOperation string) {
	body, err := json.Marshal(payload)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	proxyCloudRequestWithPath(c, body, path, disabledOperation)
}

func proxyCloudRequest(c *gin.Context, body []byte, disabledOperation string) {
	proxyCloudRequestWithPath(c, body, c.Request.URL.Path, disabledOperation)
}

func proxyCloudRequestWithPath(c *gin.Context, body []byte, path string, disabledOperation string) {
	proxyCloudUpstream(c, c.Request.Method, path, body, disabledOperation)
}

// proxyCloudElizaGET forwards a GET to the Eliza upstream path (used for model metadata).
func proxyCloudElizaGET(c *gin.Context, upstreamPath string, disabledOperation string) {
	proxyCloudUpstream(c, http.MethodGet, upstreamPath, nil, disabledOperation)
}

func proxyCloudUpstream(c *gin.Context, method, path string, body []byte, disabledOperation string) {
	if disabled, _ := internalcloud.Status(); disabled {
		c.JSON(http.StatusForbidden, gin.H{"error": internalcloud.DisabledError(disabledOperation)})
		return
	}

	rawPath := path
	path = elizaUpstreamPath(path)

	baseURL, err := url.Parse(cloudProxyBaseURL)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	targetURL := baseURL.ResolveReference(&url.URL{
		Path:     path,
		RawQuery: c.Request.URL.RawQuery,
	})

	var bodyReader io.Reader = http.NoBody
	if len(body) > 0 {
		bodyReader = bytes.NewReader(body)
	}

	outReq, err := http.NewRequestWithContext(c.Request.Context(), method, targetURL.String(), bodyReader)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	copyProxyRequestHeaders(outReq.Header, c.Request.Header)
	if clientVersion := strings.TrimSpace(version.Version); clientVersion != "" {
		outReq.Header.Set(cloudProxyClientVersionHeader, clientVersion)
	}
	if outReq.Header.Get("Content-Type") == "" && len(body) > 0 {
		outReq.Header.Set("Content-Type", "application/json")
	}

	applyElizaOutboundAuth(outReq)

	if err := cloudProxySignRequest(outReq.Context(), outReq); err != nil {
		slog.Warn("cloud proxy signing failed", "error", err)
		writeCloudUnauthorized(c)
		return
	}

	// TODO(drifkin): Add phase-specific proxy timeouts.
	// Connect/TLS/TTFB should have bounded timeouts, but once streaming starts
	// we should not enforce a short total timeout for long-lived responses.
	resp, err := http.DefaultClient.Do(outReq)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()

	copyProxyResponseHeaders(c.Writer.Header(), resp.Header)
	c.Status(resp.StatusCode)

	var bodyWriter http.ResponseWriter = c.Writer
	var framedWriter *jsonlFramingResponseWriter
	// TEMP(drifkin): only needed on the cloud-proxied first leg of Anthropic
	// web_search fallback (which is a path we're removing soon). Local
	// /v1/messages writes one JSON value per streamResponse callback directly
	// into WebSearchAnthropicWriter, but this proxy copy loop may coalesce
	// multiple jsonl records into one Write.  WebSearchAnthropicWriter currently
	// unmarshals one JSON value per Write.
	if rawPath == "/api/chat" && resp.StatusCode == http.StatusOK && c.GetBool(legacyCloudAnthropicKey) {
		framedWriter = &jsonlFramingResponseWriter{ResponseWriter: c.Writer}
		bodyWriter = framedWriter
	}

	err = copyProxyResponseBody(bodyWriter, resp.Body)
	if err == nil && framedWriter != nil {
		err = framedWriter.FlushPending()
	}
	if err != nil {
		ctxErr := c.Request.Context().Err()
		if errors.Is(err, context.Canceled) && errors.Is(ctxErr, context.Canceled) {
			slog.Debug(
				"cloud proxy response stream closed by client",
				"path", c.Request.URL.Path,
				"status", resp.StatusCode,
			)
			return
		}

		slog.Warn(
			"cloud proxy response copy failed",
			"path", c.Request.URL.Path,
			"upstream_path", path,
			"status", resp.StatusCode,
			"request_context_canceled", ctxErr != nil,
			"request_context_err", ctxErr,
			"error", err,
		)
		return
	}
}

func replaceJSONModelField(body []byte, model string) ([]byte, error) {
	if len(body) == 0 {
		return body, nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}

	modelJSON, err := json.Marshal(model)
	if err != nil {
		return nil, err
	}
	payload["model"] = modelJSON

	return json.Marshal(payload)
}

func readRequestBody(r *http.Request) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}

	r.Body = io.NopCloser(bytes.NewReader(body))
	return body, nil
}

// extractModelField returns the top-level JSON "model" string used by OpenAI/Anthropic-compatible
// and native embed bodies (including openai.ResponsesRequest). Multipart image routes are not parsed here.
func extractModelField(body []byte) (string, bool) {
	if len(body) == 0 {
		return "", false
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", false
	}

	raw, ok := payload["model"]
	if !ok {
		return "", false
	}

	var model string
	if err := json.Unmarshal(raw, &model); err != nil {
		return "", false
	}

	model = strings.TrimSpace(model)
	return model, model != ""
}

func hasAnthropicWebSearchTool(body []byte) bool {
	if len(body) == 0 {
		return false
	}

	var payload struct {
		Tools []struct {
			Type string `json:"type"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return false
	}

	for _, tool := range payload.Tools {
		if strings.HasPrefix(strings.TrimSpace(tool.Type), "web_search") {
			return true
		}
	}

	return false
}

func isOllamaComUpstream() bool {
	u, err := url.Parse(cloudProxyBaseURL)
	if err != nil {
		return false
	}
	return strings.EqualFold(u.Hostname(), "ollama.com")
}

// elizaUpstreamPath maps Zerollama paths to Eliza /api/v1/... when not using legacy ollama.com.
// ollama.com keeps paths unchanged so existing signed-proxy behavior stays byte-compatible.
func elizaUpstreamPath(path string) string {
	if isOllamaComUpstream() {
		return path
	}
	if path == "/v1" || strings.HasPrefix(path, "/v1/") {
		return "/api" + path
	}
	switch path {
	case "/api/embeddings", "/api/embed":
		return "/api/v1/embeddings"
	}
	return path
}

// applyElizaOutboundAuth sets Eliza API key on outbound requests to non-ollama.com upstreams and
// strips conflicting client headers. Any path proxied to Eliza (including /api/v1/... and
// experimental routes under /api/...) uses the same organization key.
func applyElizaOutboundAuth(req *http.Request) {
	if isOllamaComUpstream() {
		return
	}
	key := strings.TrimSpace(envconfig.ElizaCloudAPIKey())
	if key == "" {
		disabled, _ := internalcloud.Status()
		if !disabled {
			elizaAPIKeyMissingWarnOnce.Do(func() {
				slog.Warn("ELIZACLOUD_API_KEY is not set; outbound Eliza requests may fail with 401 until configured")
			})
		}
		return
	}
	for _, h := range []string{
		"Authorization",
		"X-Api-Key", "X-API-Key",
		"X-Wallet-Address", "X-Wallet-Signature", "X-Timestamp",
	} {
		req.Header.Del(h)
	}
	req.Header.Set("X-API-Key", key)
}

// ElizaModelDetailPath returns the Eliza GET path for model metadata (slash-separated ids).
func ElizaModelDetailPath(modelID string) string {
	segs := strings.Split(strings.Trim(modelID, "/"), "/")
	var b strings.Builder
	b.WriteString("/api/v1/models")
	for _, seg := range segs {
		if seg == "" {
			continue
		}
		b.WriteString("/")
		b.WriteString(url.PathEscape(seg))
	}
	return b.String()
}

func writeCloudUnauthorized(c *gin.Context) {
	signinURL, err := cloudProxySigninURL()
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

	c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized", "signin_url": signinURL})
}

func signCloudProxyRequest(ctx context.Context, req *http.Request) error {
	// Only the legacy Ollama cloud host uses Ed25519 request signing. Eliza Cloud
	// and other upstreams use API keys (see applyElizaOutboundAuth).
	if !isOllamaComUpstream() {
		return nil
	}

	ts := strconv.FormatInt(time.Now().Unix(), 10)
	challenge := buildCloudSignatureChallenge(req, ts)
	signature, err := auth.Sign(ctx, []byte(challenge))
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", signature)
	return nil
}

func buildCloudSignatureChallenge(req *http.Request, ts string) string {
	query := req.URL.Query()
	query.Set("ts", ts)
	req.URL.RawQuery = query.Encode()

	return fmt.Sprintf("%s,%s", req.Method, req.URL.RequestURI())
}

func resolveCloudProxyBaseURL(rawOverride string, runMode string) (baseURL string, overridden bool, err error) {
	baseURL = defaultCloudProxyBaseURL

	rawOverride = strings.TrimSpace(rawOverride)
	if rawOverride == "" {
		return baseURL, false, nil
	}

	u, err := url.Parse(rawOverride)
	if err != nil {
		return "", false, fmt.Errorf("invalid URL: %w", err)
	}
	if u.Scheme == "" || u.Host == "" {
		return "", false, fmt.Errorf("invalid URL: scheme and host are required")
	}
	if u.User != nil {
		return "", false, fmt.Errorf("invalid URL: userinfo is not allowed")
	}
	if u.Path != "" && u.Path != "/" {
		return "", false, fmt.Errorf("invalid URL: path is not allowed")
	}
	if u.RawQuery != "" || u.Fragment != "" {
		return "", false, fmt.Errorf("invalid URL: query and fragment are not allowed")
	}

	host := u.Hostname()
	if host == "" {
		return "", false, fmt.Errorf("invalid URL: host is required")
	}

	loopback := isLoopbackHost(host)
	if runMode == gin.ReleaseMode && !loopback {
		return "", false, fmt.Errorf("non-loopback cloud override is not allowed in release mode")
	}
	if !loopback && !strings.EqualFold(u.Scheme, "https") {
		return "", false, fmt.Errorf("non-loopback cloud override must use https")
	}

	u.Path = ""
	u.RawPath = ""
	u.RawQuery = ""
	u.Fragment = ""

	return u.String(), true, nil
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}

	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func copyProxyRequestHeaders(dst, src http.Header) {
	connectionTokens := connectionHeaderTokens(src)
	for key, values := range src {
		if isHopByHopHeader(key) || isConnectionTokenHeader(key, connectionTokens) {
			continue
		}

		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func copyProxyResponseHeaders(dst, src http.Header) {
	connectionTokens := connectionHeaderTokens(src)
	for key, values := range src {
		if isHopByHopHeader(key) || isConnectionTokenHeader(key, connectionTokens) {
			continue
		}

		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func copyProxyResponseBody(dst http.ResponseWriter, src io.Reader) error {
	flusher, canFlush := dst.(http.Flusher)
	buf := make([]byte, 32*1024)

	for {
		n, err := src.Read(buf)
		if n > 0 {
			if _, writeErr := dst.Write(buf[:n]); writeErr != nil {
				return writeErr
			}
			if canFlush {
				// TODO(drifkin): Consider conditional flushing so non-streaming
				// responses don't flush every write and can optimize throughput.
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

type jsonlFramingResponseWriter struct {
	http.ResponseWriter
	pending []byte
}

func (w *jsonlFramingResponseWriter) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (w *jsonlFramingResponseWriter) Write(p []byte) (int, error) {
	w.pending = append(w.pending, p...)
	if err := w.flushCompleteLines(); err != nil {
		return len(p), err
	}
	return len(p), nil
}

func (w *jsonlFramingResponseWriter) FlushPending() error {
	trailing := bytes.TrimSpace(w.pending)
	w.pending = nil
	if len(trailing) == 0 {
		return nil
	}

	_, err := w.ResponseWriter.Write(trailing)
	return err
}

func (w *jsonlFramingResponseWriter) flushCompleteLines() error {
	for {
		newline := bytes.IndexByte(w.pending, '\n')
		if newline < 0 {
			return nil
		}

		line := bytes.TrimSpace(w.pending[:newline])
		w.pending = w.pending[newline+1:]
		if len(line) == 0 {
			continue
		}

		if _, err := w.ResponseWriter.Write(line); err != nil {
			return err
		}
	}
}

func isHopByHopHeader(name string) bool {
	_, ok := hopByHopHeaders[strings.ToLower(name)]
	return ok
}

func connectionHeaderTokens(header http.Header) map[string]struct{} {
	tokens := map[string]struct{}{}
	for _, raw := range header.Values("Connection") {
		for _, token := range strings.Split(raw, ",") {
			token = strings.TrimSpace(strings.ToLower(token))
			if token == "" {
				continue
			}
			tokens[token] = struct{}{}
		}
	}
	return tokens
}

func isConnectionTokenHeader(name string, tokens map[string]struct{}) bool {
	if len(tokens) == 0 {
		return false
	}
	_, ok := tokens[strings.ToLower(name)]
	return ok
}

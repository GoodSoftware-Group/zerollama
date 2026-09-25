package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestIsCLMModelName(t *testing.T) {
	for _, n := range []string{"clm", "clm:latest", "clm-8b", "Contrastive-LM/CLM-v0.1-8B", "contrastive-lm/foo"} {
		require.True(t, isCLMModelName(n), n)
	}
	for _, n := range []string{"laya", "llama3.2", "climax", ""} {
		require.False(t, isCLMModelName(n), n)
	}
}

func TestDecisionsExternalResolve(t *testing.T) {
	t.Setenv("ZEROLLAMA_CLM_URL", "")
	t.Setenv("ZEROLLAMA_LAYA_URL", "")
	t.Setenv("ZEROLLAMA_CLM_HEADS", "")
	t.Setenv("ZEROLLAMA_CLM_EMB_URL", "")

	base, err := decisionsExternalResolve("laya", nil)
	require.NoError(t, err)
	require.Empty(t, base)

	_, err = decisionsExternalResolve("clm", nil)
	require.Error(t, err)

	t.Setenv("ZEROLLAMA_CLM_HEADS", "/tmp/heads.gguf")
	t.Setenv("ZEROLLAMA_CLM_EMB_URL", "http://127.0.0.1:18090")
	base, err = decisionsExternalResolve("clm", nil)
	require.NoError(t, err)
	require.Empty(t, base) // native path — no external URL
	require.True(t, clmWantsNative("clm", nil))

	t.Setenv("ZEROLLAMA_CLM_URL", "http://127.0.0.1:18700/")
	base, err = decisionsExternalResolve("clm:latest", nil)
	require.NoError(t, err)
	require.Equal(t, "http://127.0.0.1:18700", base)
	require.False(t, clmWantsNative("clm", nil)) // URL wins

	t.Setenv("ZEROLLAMA_CLM_URL", "")
	m := &Model{Config: model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityDecisions: model.BackendCLM},
	}}
	base, err = decisionsExternalResolve("my-router", m)
	require.NoError(t, err)
	require.Empty(t, base)
	require.True(t, clmWantsNative("my-router", m))

	t.Setenv("ZEROLLAMA_CLM_HEADS", "")
	t.Setenv("ZEROLLAMA_CLM_EMB_URL", "")
	_, err = decisionsExternalResolve("my-router", m)
	require.Error(t, err)

	t.Setenv("ZEROLLAMA_LAYA_URL", "http://127.0.0.1:18701")
	mLaya := &Model{Config: model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityDecisions: model.BackendLaya},
	}}
	base, err = decisionsExternalResolve("laya-ext", mLaya)
	require.NoError(t, err)
	require.Equal(t, "http://127.0.0.1:18701", base)

	t.Setenv("ZEROLLAMA_LAYA_URL", "")
	base, err = decisionsExternalResolve("laya-ext", mLaya)
	require.NoError(t, err)
	require.Empty(t, base)
}

func TestDecisionsExternalPath(t *testing.T) {
	require.Equal(t, "http://h/v1/systemone", decisionsExternalPath("http://h", "clm", nil))
	require.Equal(t, "http://h/v1/decisions", decisionsExternalPath("http://h", "laya", nil))
	m := &Model{Config: model.ConfigV2{
		ModalityBackends: map[string]string{model.ModalityDecisions: model.BackendCLM},
	}}
	require.Equal(t, "http://h/v1/systemone", decisionsExternalPath("http://h", "router", m))
}

func TestProxyDecisionsExternalPassthrough(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/systemone", r.URL.Path)
		require.Equal(t, http.MethodPost, r.Method)
		var got api.DecisionsRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&got))
		require.Equal(t, "clm", got.Model)
		require.Equal(t, "hello", got.State)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"clm-latest","answers":{"urgent":{"type":"noul","noul":0.41,"confidence":0.2}},"usage":{"input_tokens":12,"output_tokens":0}}`))
	}))
	t.Cleanup(upstream.Close)

	req := api.DecisionsRequest{
		Model: "clm",
		State: "hello",
		Questions: map[string]api.DecisionQuestion{
			"urgent": {Type: "noul", Instructions: "Is this urgent?"},
		},
	}
	out, err := proxyDecisionsExternal(t.Context(), upstream.URL, "clm", nil, req)
	require.NoError(t, err)
	require.Equal(t, "clm-latest", out.Model)
	require.Equal(t, 12, out.Usage.InputTokens)
	require.Contains(t, out.Answers, "urgent")
}

func TestProxyDecisionsExternalUpstreamError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"error":"embedder unreachable"}`))
	}))
	t.Cleanup(upstream.Close)

	_, err := proxyDecisionsExternal(t.Context(), upstream.URL, "clm", nil, api.DecisionsRequest{
		Model: "clm", State: "s",
		Questions: map[string]api.DecisionQuestion{"q": {Type: "noul", Instructions: "?"}},
	})
	require.Error(t, err)
	var se api.StatusError
	require.ErrorAs(t, err, &se)
	require.Equal(t, http.StatusBadGateway, se.StatusCode)
	require.Contains(t, se.ErrorMessage, "embedder unreachable")
}

func TestDecisionsHandlerCLMProxy(t *testing.T) {
	gin.SetMode(gin.TestMode)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/systemone", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"clm","answers":{"q":{"type":"noul","noul":0.5}}}`))
	}))
	t.Cleanup(upstream.Close)
	t.Setenv("ZEROLLAMA_CLM_URL", upstream.URL)

	s := &Server{}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	body := `{"model":"clm","state":"s","questions":{"q":{"type":"noul","instructions":"?"}}}`
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/systemone", bytes.NewBufferString(body))
	c.Request.Header.Set("Content-Type", "application/json")
	s.DecisionsHandler(c)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	var out api.DecisionsResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &out))
	require.Equal(t, "clm", out.Model)
	require.Contains(t, out.Answers, "q")
}

func TestDecisionsHandlerCLMURLMissing(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("ZEROLLAMA_CLM_URL", "")
	t.Setenv("ZEROLLAMA_CLM_HEADS", "")
	t.Setenv("ZEROLLAMA_CLM_EMB_URL", "")
	s := &Server{}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	body := `{"model":"clm","state":"s","questions":{"q":{"type":"noul","instructions":"?"}}}`
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/decisions", bytes.NewBufferString(body))
	c.Request.Header.Set("Content-Type", "application/json")
	s.DecisionsHandler(c)
	require.Equal(t, http.StatusServiceUnavailable, w.Code, w.Body.String())
}

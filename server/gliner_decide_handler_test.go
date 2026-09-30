package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/envconfig"
)

func TestIsGlinerDecideModelName(t *testing.T) {
	require.True(t, isGlinerDecideModelName("gliner-decide"))
	require.True(t, isGlinerDecideModelName("gliner-decide:latest"))
	require.True(t, isGlinerDecideModelName("gliner2.5-decide"))
	require.True(t, isGlinerDecideModelName("gliner2.5-decide-1b"))
	require.False(t, isGlinerDecideModelName("gliner"))
	require.False(t, isGlinerDecideModelName("gliner-small"))
	require.False(t, isGlinerDecideModelName("openjev"))
}

func TestIsGlinerModelNameExcludesDecide(t *testing.T) {
	require.True(t, isGlinerModelName("gliner-small"))
	require.False(t, isGlinerModelName("gliner-decide"))
	require.False(t, isGlinerModelName("gliner2.5-decide"))
}

func TestGlinerDecideHandlerProxiesMechanical(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var gotPath string
	var gotBody map[string]any
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gliner-decide","result":{"intent":"refund_request"},"engine":{"model_id":"x"}}`))
	}))
	defer up.Close()

	t.Setenv("ZEROLLAMA_GLINER_DECIDE_URL", up.URL)
	require.Equal(t, strings.TrimSuffix(up.URL, "/"), envconfig.GlinerDecideURL())

	s := &Server{}
	r := gin.New()
	r.POST("/v1/gliner-decide", s.GlinerDecideHandler)

	body := `{"model":"gliner-decide","text":"refund please","schema":{"intent":["refund_request","other"]},"include_confidence":true}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/gliner-decide", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusOK, w.Code)
	require.Equal(t, "/v1/gliner-decide", gotPath)
	require.Equal(t, "refund please", gotBody["text"])
	require.Contains(t, gotBody, "schema")
}

func TestDecisionsExternalResolveGlinerDecide(t *testing.T) {
	t.Setenv("ZEROLLAMA_GLINER_DECIDE_URL", "http://127.0.0.1:18098")
	base, err := decisionsExternalResolve("gliner-decide", nil)
	require.NoError(t, err)
	require.Equal(t, "http://127.0.0.1:18098", base)
	require.Equal(t, "http://127.0.0.1:18098/v1/systemone", decisionsExternalPath(base, "gliner-decide", nil))
}

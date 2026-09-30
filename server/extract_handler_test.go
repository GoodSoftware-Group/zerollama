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

func TestStripToAbstractDropsEngineKeys(t *testing.T) {
	raw := map[string]any{
		"model": "gliner", "text": "hi", "labels": []any{"person"},
		"threshold": 0.4, "max_width": 12, "flat_ner": true, "device_id": 0,
	}
	out := stripToAbstract(raw)
	require.Contains(t, out, "text")
	require.Contains(t, out, "labels")
	require.Contains(t, out, "threshold")
	require.NotContains(t, out, "max_width")
	require.NotContains(t, out, "flat_ner")
	require.NotContains(t, out, "device_id")
}

func TestIsGlinerModelName(t *testing.T) {
	require.True(t, isGlinerModelName("gliner"))
	require.True(t, isGlinerModelName("gliner:small"))
	require.True(t, isGlinerModelName("gliner-small"))
	require.False(t, isGlinerModelName("openjev"))
	require.False(t, isGlinerModelName("gliner-decide"))
}

func TestExtractHandlerProxiesAbstract(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var gotPath string
	var gotBody map[string]any
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"entities":[{"text":"Kyiv","label":"city","start":0,"end":4,"score":0.9}]}`))
	}))
	defer up.Close()

	t.Setenv("ZEROLLAMA_GLINER_URL", up.URL)
	require.Equal(t, strings.TrimSuffix(up.URL, "/"), envconfig.GlinerURL())

	s := &Server{}
	r := gin.New()
	r.POST("/v1/extract", s.ExtractHandler)
	r.POST("/v1/gliner", s.GlinerHandler)

	body := `{"model":"gliner","text":"Kyiv is capital","labels":["city"],"max_width":99,"flat_ner":false}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/extract", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusOK, w.Code)
	require.Equal(t, "/v1/extract", gotPath)
	require.NotContains(t, gotBody, "max_width")
	require.NotContains(t, gotBody, "flat_ner")

	w2 := httptest.NewRecorder()
	req2 := httptest.NewRequest(http.MethodPost, "/v1/gliner", strings.NewReader(body))
	req2.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w2, req2)
	require.Equal(t, http.StatusOK, w2.Code)
	require.Equal(t, "/v1/gliner", gotPath)
	require.Contains(t, gotBody, "max_width")
}

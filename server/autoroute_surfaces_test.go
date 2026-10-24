package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/server/autoroute"
)

// echoModel is a terminal handler that reports the model the downstream stage received.
func echoModel(c *gin.Context) {
	body, _ := io.ReadAll(c.Request.Body)
	m, _ := extractModelField(body)
	c.JSON(http.StatusOK, gin.H{"model": m})
}

func serveRouteAhead(t *testing.T, s *Server, surface autoroute.Surface, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST(path, s.routeAheadMiddleware(surface), echoModel)
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	r.ServeHTTP(w, req)
	return w
}

func TestRouteAheadMiddlewareRewritesBody(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)

	t.Run("native chat", func(t *testing.T) {
		w := serveRouteAhead(t, s, autoroute.SurfaceChat, "/api/chat",
			`{"model":"auto","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}]}`)
		if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"model":"tooly`) {
			t.Fatalf("status %d body %s", w.Code, w.Body.String())
		}
		if !strings.HasPrefix(w.Header().Get("X-Zerollama-Router-Chosen"), "tooly") {
			t.Fatalf("headers = %v", w.Header())
		}
	})

	t.Run("openai content parts", func(t *testing.T) {
		w := serveRouteAhead(t, s, autoroute.SurfaceChat, "/v1/chat/completions",
			`{"model":"auto","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
		if w.Code != http.StatusOK || w.Header().Get("X-Zerollama-Router") != "auto" {
			t.Fatalf("status %d headers %v body %s", w.Code, w.Header(), w.Body.String())
		}
	})

	t.Run("ordinary model untouched", func(t *testing.T) {
		w := serveRouteAhead(t, s, autoroute.SurfaceChat, "/api/chat", `{"model":"plain","messages":[]}`)
		if !strings.Contains(w.Body.String(), `"model":"plain"`) || w.Header().Get("X-Zerollama-Router") != "" {
			t.Fatalf("body %s headers %v", w.Body.String(), w.Header())
		}
	})

	t.Run("no candidate stops the chain", func(t *testing.T) {
		w := serveRouteAhead(t, s, "", "/api/generate", `{"model":"auto","prompt":"a fox","width":256}`)
		if w.Code != http.StatusServiceUnavailable || !strings.Contains(w.Body.String(), "filtered") {
			t.Fatalf("status %d body %s", w.Code, w.Body.String())
		}
	})
}

func routeEmbed(t *testing.T, s *Server, name string) (string, *routeError) {
	t.Helper()
	c, _ := gin.CreateTestContext(NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/api/embed", nil)
	return s.routeForSurface(c, name, autoroute.Signals{Surface: autoroute.SurfaceEmbed})
}

func resetRouterFile(t *testing.T, yaml string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "router.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_ROUTER_CONFIG", path)
	routerFileMu.Lock()
	routerFileCache = nil
	routerFileMu.Unlock()
	t.Cleanup(func() {
		routerFileMu.Lock()
		routerFileCache = nil
		routerFileMu.Unlock()
	})
}

func TestRouteForSurfaceEmbedIsPinned(t *testing.T) {
	s := newRouteTestServer(t)
	resetRouterFile(t, "routers: {}\n")
	pooling := ggml.KV{"llama.pooling_type": uint32(1)}
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	createRouteTestModelKV(t, s, "embed-a", routePlainTemplate, pooling)

	got, rerr := routeEmbed(t, s, "auto")
	if rerr != nil || !strings.HasPrefix(got, "embed-a") {
		t.Fatalf("single embedder: got %q err %v", got, rerr)
	}

	createRouteTestModelKV(t, s, "embed-b", routePlainTemplate, pooling)
	_, rerr = routeEmbed(t, s, "auto")
	if rerr == nil || !strings.Contains(rerr.Error(), "pinned") {
		t.Fatalf("two embedders: err = %v, want ambiguity refusal", rerr)
	}

	resetRouterFile(t, `
routers:
  rag:
    classifier: unused
    embedder: embed-b
    candidates:
      - model: plain
        labels: [chat]
`)
	got, rerr = routeEmbed(t, s, "rag")
	if rerr != nil || !strings.HasPrefix(got, "embed-b") {
		t.Fatalf("router.yaml pin: got %q err %v", got, rerr)
	}
}

func TestRouteForSurfaceNonTextUsesRouterCandidates(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	createRouteTestModelKV(t, s, "embed-a", routePlainTemplate, ggml.KV{"llama.pooling_type": uint32(1)})
	resetRouterFile(t, `
routers:
  agent:
    classifier: unused
    candidates:
      - model: plain
        labels: [chat]
`)
	// embed-a exists locally but is not one of agent's candidates and there is no embedder pin.
	_, rerr := routeEmbed(t, s, "agent")
	if rerr == nil || !strings.Contains(rerr.Error(), "no local model") {
		t.Fatalf("err = %v", rerr)
	}
	var body map[string]any
	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)
	writeRouterError(c, rerr.dec, rerr.err)
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if w.Code != http.StatusServiceUnavailable || body["filtered"] == nil {
		t.Fatalf("status %d body %v", w.Code, body)
	}
}

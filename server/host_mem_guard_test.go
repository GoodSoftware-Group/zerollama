package server

import (
	"bytes"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestHostMemGuardDisabledPasses(t *testing.T) {
	t.Setenv("ZEROLLAMA_HOST_MEM_GUARD", "0")
	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{}
	r.POST("/api/chat", s.hostMemGuard(), func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/chat", nil)
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestModelAlreadyWarmMatchesPSName(t *testing.T) {
	warm := api.ProcessResponse{Models: []api.ProcessModelResponse{
		{Name: "deepseek-v4-flash:q2_k", Model: "deepseek-v4-flash:q2_k"},
	}}
	if !modelAlreadyWarm(warm, "deepseek-v4-flash:q2_k", nil) {
		t.Fatal("expected warm match")
	}
	if modelAlreadyWarm(warm, "other:latest", nil) {
		t.Fatal("expected miss")
	}
}

func TestPinRequestToLoadedRunnerCapsContext(t *testing.T) {
	opts := api.DefaultOptions()
	opts.NumCtx = 131072
	opts.Temperature = 0.2
	opts.NumBatch = 2048
	snap := loadedReuse{
		found:         true,
		adaptersMatch: true,
		effectiveCtx:  2048,
		runner:        api.Runner{NumCtx: 2048, NumBatch: 512, NumGPU: 99, KvCacheType: "q8_0"},
	}
	got, err := pinRequestToLoadedRunner(opts, snap, false)
	if err != nil {
		t.Fatal(err)
	}
	if got.NumCtx != 2048 {
		t.Fatalf("num_ctx=%d", got.NumCtx)
	}
	if got.NumBatch != 512 || got.NumGPU != 99 || got.KvCacheType != "q8_0" {
		t.Fatalf("runner not pinned: %+v", got.Runner)
	}
	if got.Temperature != 0.2 {
		t.Fatalf("temperature overwritten: %v", got.Temperature)
	}
}

func TestPinRequestToLoadedRunnerRefusesGrowthCases(t *testing.T) {
	opts := api.DefaultOptions()
	if _, err := pinRequestToLoadedRunner(opts, loadedReuse{}, false); !errors.Is(err, ErrHostMemNoReuse) {
		t.Fatalf("missing runner: %v", err)
	}
	if _, err := pinRequestToLoadedRunner(opts, loadedReuse{found: true, dead: true, adaptersMatch: true}, false); !errors.Is(err, ErrHostMemNoReuse) {
		t.Fatalf("dead: %v", err)
	}
	if _, err := pinRequestToLoadedRunner(opts, loadedReuse{found: true, adaptersMatch: false}, false); !errors.Is(err, ErrHostMemNoReuse) {
		t.Fatalf("adapters: %v", err)
	}
	if _, err := pinRequestToLoadedRunner(opts, loadedReuse{found: true, adaptersMatch: true, isImagegen: true}, false); !errors.Is(err, ErrHostMemNoReuse) {
		t.Fatalf("imagegen: %v", err)
	}
	got, err := pinRequestToLoadedRunner(opts, loadedReuse{found: true, loading: true, adaptersMatch: true}, false)
	if err != nil {
		t.Fatal(err)
	}
	if got.NumCtx != opts.NumCtx {
		t.Fatalf("in-flight load should keep request options, ctx=%d", got.NumCtx)
	}
}

func TestHostMemGuardRequestModelFromAccessLog(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/api/chat", bytes.NewReader([]byte(`{"model":"x"}`)))
	c.Set(inferenceAccessLogKey, &inferenceAccessMeta{model: "deepseek-v4-flash:q2_k"})
	if got := hostMemGuardRequestModel(c); got != "deepseek-v4-flash:q2_k" {
		t.Fatalf("got %q", got)
	}
	c2, _ := gin.CreateTestContext(httptest.NewRecorder())
	body := `{"model":"from-body:latest"}`
	c2.Request = httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	if got := hostMemGuardRequestModel(c2); got != "from-body:latest" {
		t.Fatalf("peek got %q", got)
	}
}

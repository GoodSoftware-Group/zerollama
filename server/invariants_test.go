package server

import (
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/version"
)

func TestProxyGGUFPathPrefersTheManifest(t *testing.T) {
	if got := ProxyGGUFPath("/models/blobs/sha256-aa", "/tmp/other.gguf"); got != "/models/blobs/sha256-aa" {
		t.Fatalf("manifest lost: %s", got)
	}
	if got := ProxyGGUFPath("", "  /data/custom.gguf  "); got != "/data/custom.gguf" {
		t.Fatalf("client path: %s", got)
	}
	if got := ProxyGGUFPath("  ", ""); got != "" {
		t.Fatalf("empty: %q", got)
	}
}

func TestInvariantMaxRunnersUnsetIsThreePerGPU(t *testing.T) {
	if got := MaxRunnersForGPUs(0, 2); got != 6 {
		t.Fatalf("two GPUs: got %d", got)
	}
	if got := MaxRunnersForGPUs(0, 0); got != uint(defaultModelsPerGPU) {
		t.Fatalf("no GPU still reserves one GPU worth: got %d", got)
	}
	if got := MaxRunnersForGPUs(1, 8); got != 1 {
		t.Fatalf("explicit cap must win: got %d", got)
	}
}

func TestInvariantIntegratedMemoryCapsDefaultContext(t *testing.T) {
	big := uint64(50) * format.GibiByte
	if got := DefaultNumCtx(big, false); got != 262144 {
		t.Fatalf("discrete 50GiB: got %d", got)
	}
	if got := DefaultNumCtx(big, true); got != 32768 {
		t.Fatalf("unified 50GiB: got %d", got)
	}
	if got := DefaultNumCtx(uint64(30)*format.GibiByte, true); got != 32768 {
		t.Fatalf("mid tier: got %d", got)
	}
	if got := DefaultNumCtx(uint64(8)*format.GibiByte, true); got != 4096 {
		t.Fatalf("small: got %d", got)
	}
}

func TestInvariantOffloadMustCoverEstimate(t *testing.T) {
	if !OffloadFits(0, 0) {
		t.Fatal("missing estimate is not a fit failure")
	}
	if OffloadFits(10, 9) {
		t.Fatal("short free memory must not count as a full fit")
	}
	if !OffloadFits(10, 10) {
		t.Fatal("exact fit is a fit")
	}
}

func TestInvariantRuntimeHealthRequiresOurBody(t *testing.T) {
	ok := []byte(`{"ready":true,"server_revision":"fastapi-body-v4"}`)
	if !IsZerollamaRuntimeHealth(http.StatusOK, ok) {
		t.Fatal("sidecar body")
	}
	if IsZerollamaRuntimeHealth(http.StatusOK, []byte(`{"status":"ok"}`)) {
		t.Fatal("foreign 200")
	}
	if IsZerollamaRuntimeHealth(http.StatusServiceUnavailable, ok) {
		t.Fatal("non-200")
	}
}

func TestInvariantOmittedKeepAliveIsTheDefault(t *testing.T) {
	if got := KeepAliveDuration(nil); got != envconfig.KeepAlive() {
		t.Fatalf("omitted keep_alive %s", got)
	}
	want := 30 * time.Second
	if got := KeepAliveDuration(&api.Duration{Duration: want}); got != want {
		t.Fatalf("explicit keep_alive %s", got)
	}
}

func TestInvariantInFlightContextMustCoverTheRequest(t *testing.T) {
	if !LoadingContextTooSmall(4096, 32768) {
		t.Fatal("smaller in-flight context")
	}
	if LoadingContextTooSmall(32768, 4096) || LoadingContextTooSmall(0, 8192) {
		t.Fatal("larger or unknown in-flight context may be joined")
	}
	if got := LoadingCannotJoin(4096, 8192, false, false, false, false); got != "loading_ctx_short" {
		t.Fatalf("context: %q", got)
	}
	if got := LoadingCannotJoin(8192, 4096, true, false, false, false); got != "loading_context_shift" {
		t.Fatalf("shift: %q", got)
	}
	if got := LoadingCannotJoin(8192, 4096, false, true, false, false); got != "loading_adapters" {
		t.Fatalf("adapters: %q", got)
	}
	if got := LoadingCannotJoin(8192, 4096, false, false, false, true); got != "loading_options" {
		t.Fatalf("options: %q", got)
	}
	if LoadingCannotJoin(8192, 4096, false, false, false, false) != "" {
		t.Fatal("a smaller context with the same options may join")
	}
}

func TestInvariantShortGalleryBodyIsNotTheBlob(t *testing.T) {
	if !InlineBlobComplete(10, 10) || InlineBlobComplete(9, 10) || InlineBlobComplete(0, 0) {
		t.Fatal("inline length")
	}
	if !RangeBodyFitsPart(http.StatusPartialContent, 100, 50, 50, false) {
		t.Fatal("206 is this part")
	}
	if RangeBodyFitsPart(http.StatusOK, 100, 50, 1000, false) {
		t.Fatal("a 200 of the whole object is not a later part")
	}
	if !RangeBodyFitsPart(http.StatusOK, 0, 50, 50, false) {
		t.Fatal("a 200 of exactly the first part is that part")
	}
}

func TestInvariantGGMLLoadReleasesTheRuntime(t *testing.T) {
	if KeepRuntimeResident(true) || KeepRuntimeResident(false) {
		t.Fatal("a weight fit must not keep llama-server resident")
	}
}

func TestInvariantVRAMRecoverySkipsUnifiedMemory(t *testing.T) {
	if VRAMRecoveryNeeded(1, false, "Metal") || VRAMRecoveryNeeded(1, true, "Metal") || VRAMRecoveryNeeded(0, true, "CUDA") {
		t.Fatal("Metal, integrated, and CPU must not poll")
	}
	if !VRAMRecoveryNeeded(1, true, "CUDA") || !VRAMRecoveryNeeded(2, true, "Metal") {
		t.Fatal("discrete non-Metal memory must be polled")
	}
}

func TestInvariantGallery200IsTheBlobURL(t *testing.T) {
	req, err := url.Parse("http://registry.example/v2/library/m/blobs/sha256:ab")
	if err != nil {
		t.Fatal(err)
	}
	got, err := DirectBlobURL(http.StatusOK, nil, req)
	if err != nil || got.String() != req.String() {
		t.Fatalf("inline 200: %v %v", got, err)
	}
	loc, err := url.Parse("https://cdn.example/blob")
	if err != nil {
		t.Fatal(err)
	}
	got, err = DirectBlobURL(http.StatusTemporaryRedirect, loc, req)
	if err != nil || got.String() != loc.String() {
		t.Fatalf("redirect: %v %v", got, err)
	}
	if _, err := DirectBlobURL(http.StatusTemporaryRedirect, nil, req); err == nil {
		t.Fatal("redirect without Location")
	}
}

func TestInvariantShorterLeaseEvictsFirst(t *testing.T) {
	older := time.Now().Add(-time.Hour)
	newer := time.Now()
	short := &runnerRef{sessionDuration: time.Minute, lastUsedAt: newer, modelPath: "b"}
	long := &runnerRef{sessionDuration: time.Hour, lastUsedAt: older, modelPath: "a"}
	if !evictBefore(short, long) || evictBefore(long, short) {
		t.Fatal("shorter keep_alive is the victim")
	}
	cold := &runnerRef{sessionDuration: time.Minute, lastUsedAt: older, modelPath: "z"}
	hot := &runnerRef{sessionDuration: time.Minute, lastUsedAt: newer, modelPath: "a"}
	if !evictBefore(cold, hot) {
		t.Fatal("equal lease evicts the older use")
	}
	forever := &runnerRef{sessionDuration: time.Duration(^uint64(0) >> 1), modelPath: "keep"}
	if evictBefore(forever, short) {
		t.Fatal("forever lease stays resident")
	}
}

func TestInvariantNegativeKeepAliveStaysForever(t *testing.T) {
	forever := time.Duration(^uint64(0) >> 1)
	if got := KeepAliveDuration(&api.Duration{Duration: forever}); got != forever {
		t.Fatalf("negative keep_alive %s", got)
	}
}

func TestInvariantGalleryUserAgentIsOllamaCompat(t *testing.T) {
	h := make(http.Header)
	ApplyGalleryUserAgent(h)
	got := h.Get("User-Agent")
	if got != version.HTTPUserAgent() || got != GalleryUserAgent() {
		t.Fatalf("user-agent %q", got)
	}
	if len(got) < len("ollama/") || got[:len("ollama/")] != "ollama/" {
		t.Fatalf("registry stamp %q", got)
	}
}

func TestInvariantSplitNamesMustBeUnique(t *testing.T) {
	if !ggufSplitBasesCollapse([]string{"library-model.gguf", "library-model.gguf"}) {
		t.Fatal("identical From names collapse")
	}
	if !ggufSplitBasesCollapse([]string{"weights.gguf"}) {
		t.Fatal("a non-split name is not a shard filename")
	}
	ok := []string{
		"model-00001-of-00002.gguf",
		"model-00002-of-00002.gguf",
	}
	if ggufSplitBasesCollapse(ok) {
		t.Fatal("real split names must be kept")
	}
}

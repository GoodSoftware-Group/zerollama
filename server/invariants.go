package server

import (
	"bytes"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/version"
)

// Scheduler and wire invariants. Each function is the rule; call sites must use
// it instead of re-deriving the condition. Storage blob rules are in remotestore.
// Python wire rules are in runtime/runtime/invariants.py.
//
// Accepted on purpose, and not fail-closed here:
//   - A model is a tag path. Names have no @digest pin, so a later pull of the same tag replaces weights.
//   - Lookup folds case. The directory keeps the spelling of the first write.
//   - A chat template may snap to the nearest builtin within edit distance 100.
//   - A name ending in -cloud is the Eliza stub.
//   - Blob PUT HMAC covers method, path, and time. The server hashes the body.
//   - Storage manifest bodies are not signed, and mDNS TXT is not a peer identity.

// MaxRunnersForGPUs is the residency cap. A configured value above zero is that
// cap. Zero means defaultModelsPerGPU for each GPU, and at least one GPU worth
// when discovery returns none. The cap is applied before the eviction branch.
func MaxRunnersForGPUs(configured uint, gpuCount int) uint {
	if configured > 0 {
		return configured
	}
	if gpuCount < 1 {
		gpuCount = 1
	}
	return uint(defaultModelsPerGPU * gpuCount)
}

// DefaultNumCtx is the startup context window for a VRAM total.
// Tiers are 262144 at ≥47 GiB, 32768 at ≥23 GiB, otherwise 4096.
// Integrated-only devices share that memory with the OS, so the 262144 tier
// is capped at 32768. A discrete GPU at the same total keeps 262144.
func DefaultNumCtx(totalVRAM uint64, integratedOnly bool) int {
	var n int
	switch {
	case totalVRAM >= 47*format.GibiByte:
		n = 262144
	case totalVRAM >= 23*format.GibiByte:
		n = 32768
	default:
		n = 4096
	}
	if integratedOnly && n > 32768 {
		return 32768
	}
	return n
}

// OffloadFits is llm.OffloadFits under the scheduler's name.
// requireFull load must fail closed when this is false.
func OffloadFits(need, free uint64) bool {
	return llm.OffloadFits(need, free)
}

// IsZerollamaRuntimeHealth is true only for our sidecar. A foreign listener
// that answers HTTP 200 on the runtime port is not ready.
func IsZerollamaRuntimeHealth(statusCode int, body []byte) bool {
	if statusCode != http.StatusOK {
		return false
	}
	return bytes.Contains(body, []byte(`"server_revision"`))
}

// KeepRuntimeResident is the one-owner rule for a ggml load. A weight-only fit
// is not an exception: llama-server and the ggml runner do not share the GPU.
func KeepRuntimeResident(weightsFit bool) bool {
	return false && weightsFit
}

// VRAMRecoveryNeeded reports whether unload should poll free memory.
// Metal, CPU, and integrated memory do not give that memory back as a rising
// FreeMemory counter, so a poll there only burns the timeout.
func VRAMRecoveryNeeded(gpuCount int, discrete bool, singleLibrary string) bool {
	if gpuCount == 0 || !discrete {
		return false
	}
	if gpuCount == 1 && singleLibrary == "Metal" {
		return false
	}
	return true
}

// DirectBlobURL is where the blob bytes are. A 200 means the requested URL
// already serves them. A redirect means the Location header does.
func DirectBlobURL(status int, location, requested *url.URL) (*url.URL, error) {
	switch status {
	case http.StatusOK, http.StatusPartialContent:
		if requested == nil {
			return nil, fmt.Errorf("blob response %d had no url", status)
		}
		return requested, nil
	case http.StatusMovedPermanently, http.StatusFound, http.StatusSeeOther,
		http.StatusTemporaryRedirect, http.StatusPermanentRedirect:
		if location == nil {
			return nil, http.ErrNoLocation
		}
		return location, nil
	default:
		return nil, fmt.Errorf("unexpected status code %d", status)
	}
}

// KeepAliveDuration is the lease a request applies. Omitted means the server
// default (5 minutes unless OLLAMA_KEEP_ALIVE says otherwise). A negative JSON
// value is already the max duration, which means forever.
func KeepAliveDuration(d *api.Duration) time.Duration {
	if d == nil {
		return envconfig.KeepAlive()
	}
	return d.Duration
}

// ProxyGGUFPath is the GGUF the runtime loads. A resolved manifest blob wins,
// so a client cannot swap weights for a pulled tag. When the tag has no blob,
// the client path is how an unpulled file is named.
func ProxyGGUFPath(manifestPath, clientPath string) string {
	if p := strings.TrimSpace(manifestPath); p != "" {
		return p
	}
	return strings.TrimSpace(clientPath)
}

// LoadingContextTooSmall reports an in-flight load whose context cannot serve
// this request. Joining that load would pin the caller to the smaller KV.
func LoadingContextTooSmall(loaded, requested int) bool {
	return loaded > 0 && requested > loaded
}

// LoadingCannotJoin reports why a request must not attach to an in-flight load.
// An empty string means the load can serve it. Context, shift, adapters,
// projectors, and the other runner options are fixed once the process starts.
func LoadingCannotJoin(loadedCtx, wantCtx int, shiftDiffer, adaptersDiffer, projectorsDiffer, optionsDiffer bool) string {
	if LoadingContextTooSmall(loadedCtx, wantCtx) {
		return "loading_ctx_short"
	}
	if shiftDiffer {
		return "loading_context_shift"
	}
	if adaptersDiffer {
		return "loading_adapters"
	}
	if projectorsDiffer {
		return "loading_projectors"
	}
	if optionsDiffer {
		return "loading_options"
	}
	return ""
}

// InlineBlobComplete reports that a gallery 200 body is the whole object.
// A shorter body is not a blob and must not be committed.
func InlineBlobComplete(n, total int64) bool {
	return total > 0 && n == total
}

// RangeBodyFitsPart reports that a ranged GET returned this part.
// A 200 of the whole object is only that part when the part starts at 0 and
// the body is exactly the part, or the download is a single part.
func RangeBodyFitsPart(status int, start, partSize, contentLength int64, singlePart bool) bool {
	switch status {
	case http.StatusPartialContent:
		return partSize > 0
	case http.StatusOK:
		if start != 0 || partSize <= 0 {
			return false
		}
		if contentLength == partSize {
			return true
		}
		return singlePart && contentLength <= 0
	default:
		return false
	}
}

// GalleryUserAgent is the registry User-Agent stamp (ollama/<OllamaCompat>).
func GalleryUserAgent() string {
	return version.HTTPUserAgent()
}

// ApplyGalleryUserAgent sets that stamp on a gallery request.
// Blob range GETs and the tensor transfer use the same stamp as the manifest client.
func ApplyGalleryUserAgent(h http.Header) {
	if h == nil {
		return
	}
	h.Set("User-Agent", GalleryUserAgent())
}

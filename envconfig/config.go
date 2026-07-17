package envconfig

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Host returns the scheme and host. Host can be configured via the OLLAMA_HOST environment variable.
// Default is scheme "http" and host "0.0.0.0:11434" (all interfaces).
func Host() *url.URL {
	defaultPort := "11434"

	s := strings.TrimSpace(Var("OLLAMA_HOST"))
	scheme, hostport, ok := strings.Cut(s, "://")
	switch {
	case !ok:
		scheme, hostport = "http", s
		if s == "ollama.com" {
			scheme, hostport = "https", "ollama.com:443"
		}
	case scheme == "http":
		defaultPort = "80"
	case scheme == "https":
		defaultPort = "443"
	}

	hostport, path, _ := strings.Cut(hostport, "/")
	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		host, port = "0.0.0.0", defaultPort
		if ip := net.ParseIP(strings.Trim(hostport, "[]")); ip != nil {
			host = ip.String()
		} else if hostport != "" {
			host = hostport
		}
	}

	if n, err := strconv.ParseInt(port, 10, 32); err != nil || n > 65535 || n < 0 {
		slog.Warn("invalid port, using default", "port", port, "default", defaultPort)
		port = defaultPort
	}

	return &url.URL{
		Scheme: scheme,
		Host:   net.JoinHostPort(host, port),
		Path:   path,
	}
}

// ConnectableHost returns Host() with unspecified bind addresses (0.0.0.0, ::)
// replaced by the corresponding loopback address (127.0.0.1, ::1).
// Unspecified addresses are valid for binding a server socket but not for
// connecting as a client, which fails on Windows.
func ConnectableHost() *url.URL {
	u := Host()
	host, port, err := net.SplitHostPort(u.Host)
	if err != nil {
		return u
	}

	if ip := net.ParseIP(host); ip != nil && ip.IsUnspecified() {
		if ip.To4() != nil {
			host = "127.0.0.1"
		} else {
			host = "::1"
		}
		u.Host = net.JoinHostPort(host, port)
	}

	return u
}

// AllowedOrigins returns a list of allowed origins. AllowedOrigins can be configured via the OLLAMA_ORIGINS environment variable.
func AllowedOrigins() (origins []string) {
	if s := Var("OLLAMA_ORIGINS"); s != "" {
		origins = strings.Split(s, ",")
	}

	for _, origin := range []string{"localhost", "127.0.0.1", "0.0.0.0"} {
		origins = append(origins,
			fmt.Sprintf("http://%s", origin),
			fmt.Sprintf("https://%s", origin),
			fmt.Sprintf("http://%s", net.JoinHostPort(origin, "*")),
			fmt.Sprintf("https://%s", net.JoinHostPort(origin, "*")),
		)
	}

	origins = append(origins,
		"app://*",
		"file://*",
		"tauri://*",
		"vscode-webview://*",
		"vscode-file://*",
	)

	return origins
}

// Models returns the path to the models directory. Models directory can be configured via the OLLAMA_MODELS environment variable.
// Default is $HOME/.ollama/models
func Models() string {
	if s := Var("OLLAMA_MODELS"); s != "" {
		return s
	}

	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return filepath.Join(home, ".ollama", "models")
}

// KeepAlive returns the duration that models stay loaded in memory. KeepAlive can be configured via the OLLAMA_KEEP_ALIVE environment variable.
// Negative values are treated as infinite. Zero is treated as no keep alive.
// Default is 5 minutes.
func KeepAlive() (keepAlive time.Duration) {
	keepAlive = 5 * time.Minute
	if s := Var("OLLAMA_KEEP_ALIVE"); s != "" {
		if d, err := time.ParseDuration(s); err == nil {
			keepAlive = d
		} else if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			keepAlive = time.Duration(n) * time.Second
		}
	}

	if keepAlive < 0 {
		return time.Duration(math.MaxInt64)
	}

	return keepAlive
}

// LoadTimeout returns the duration for stall detection during model loads. LoadTimeout can be configured via the OLLAMA_LOAD_TIMEOUT environment variable.
// Zero or Negative values are treated as infinite.
// Default is 5 minutes.
func LoadTimeout() (loadTimeout time.Duration) {
	loadTimeout = 5 * time.Minute
	if s := Var("OLLAMA_LOAD_TIMEOUT"); s != "" {
		if d, err := time.ParseDuration(s); err == nil {
			loadTimeout = d
		} else if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			loadTimeout = time.Duration(n) * time.Second
		}
	}

	if loadTimeout <= 0 {
		return time.Duration(math.MaxInt64)
	}

	return loadTimeout
}

func Remotes() []string {
	var r []string
	raw := strings.TrimSpace(Var("OLLAMA_REMOTES"))
	if raw == "" {
		r = []string{"ollama.com"}
	} else {
		r = strings.Split(raw, ",")
	}
	return r
}

func BoolWithDefault(k string) func(defaultValue bool) bool {
	return func(defaultValue bool) bool {
		if s := Var(k); s != "" {
			b, err := strconv.ParseBool(s)
			if err != nil {
				return true
			}

			return b
		}

		return defaultValue
	}
}

func Bool(k string) func() bool {
	withDefault := BoolWithDefault(k)
	return func() bool {
		return withDefault(false)
	}
}

// LogLevel returns the log level for the application.
// Values are 0 or false INFO (Default), 1 or true DEBUG, 2 TRACE
func LogLevel() slog.Level {
	level := slog.LevelInfo
	if s := Var("OLLAMA_DEBUG"); s != "" {
		if b, _ := strconv.ParseBool(s); b {
			level = slog.LevelDebug
		} else if i, _ := strconv.ParseInt(s, 10, 64); i != 0 {
			level = slog.Level(i * -4)
		}
	}

	return level
}

var (
	// FlashAttention enables the experimental flash attention feature.
	FlashAttention = BoolWithDefault("OLLAMA_FLASH_ATTENTION")
	// DebugLogRequests logs inference requests to disk for replay/debugging.
	DebugLogRequests = Bool("OLLAMA_DEBUG_LOG_REQUESTS")
	// KvCacheType is the quantization type for the K/V cache.
	KvCacheType = String("OLLAMA_KV_CACHE_TYPE")
	// NoHistory disables readline history.
	NoHistory = Bool("OLLAMA_NOHISTORY")
	// NoPrune disables pruning of model blobs on startup.
	NoPrune = Bool("OLLAMA_NOPRUNE")
	// LMStudioImport enables reusing LM Studio caches under ~/.lmstudio/models (and
	// optional OLLAMA_LMSTUDIO_MODELS paths) when pulling a model, and listing them
	// in /api/tags. Supports GGUF and safetensors layouts. Default is on.
	// Why default on: Mac operators commonly have LM Studio installed; opt-out avoids
	// surprise registry downloads when cache already holds the weights.
	LMStudioImport = BoolWithDefault("OLLAMA_LMSTUDIO_IMPORT")
	// LMStudioListAll lists every discoverable LM Studio model even when MLX
	// safetensors import would exceed free disk on OLLAMA_MODELS (pull still checks).
	// Why separate from import: listing is discovery; pull must still fail safely on ENOSPC.
	LMStudioListAll = Bool("OLLAMA_LMSTUDIO_LIST_ALL")
	// SchedSpread allows scheduling models across all GPUs.
	SchedSpread = Bool("OLLAMA_SCHED_SPREAD")
	// MultiUserCache optimizes prompt caching for multi-user scenarios
	MultiUserCache = Bool("OLLAMA_MULTIUSER_CACHE")
	// Enable the new Ollama engine
	NewEngine = Bool("OLLAMA_NEW_ENGINE")
	// ContextLength sets the default context length
	ContextLength = Uint("OLLAMA_CONTEXT_LENGTH", 0)
	// Auth enables authentication between the Ollama client and server
	UseAuth = Bool("OLLAMA_AUTH")
	// EnableVulkan controls experimental Vulkan backend discovery.
	EnableVulkan = BoolWithDefault("OLLAMA_VULKAN")
	// EnableIntegratedGPU controls whether integrated GPUs may be selected.
	// Default false except upstream allowlist (CUDA iGPU, ROCm gfx1151 Strix Halo).
	EnableIntegratedGPU = BoolWithDefault("OLLAMA_IGPU_ENABLE")
	// NoCloudEnv checks the OLLAMA_NO_CLOUD environment variable.
	NoCloudEnv = Bool("OLLAMA_NO_CLOUD")
	// TrainingEnabled starts embedded GPU training (CGO + training.py) and registers training APIs when true.
	// Default true so the capability is discoverable; production without GPU stack sets OLLAMA_TRAINING=false.
	TrainingEnabled = BoolWithDefault("OLLAMA_TRAINING")
)

func String(s string) func() string {
	return func() string {
		return Var(s)
	}
}

var (
	LLMLibrary = String("OLLAMA_LLM_LIBRARY")
	Editor     = String("OLLAMA_EDITOR")

	CudaVisibleDevices    = String("CUDA_VISIBLE_DEVICES")
	HipVisibleDevices     = String("HIP_VISIBLE_DEVICES")
	RocrVisibleDevices    = String("ROCR_VISIBLE_DEVICES")
	VkVisibleDevices      = String("GGML_VK_VISIBLE_DEVICES")
	GpuDeviceOrdinal      = String("GPU_DEVICE_ORDINAL")
	HsaOverrideGfxVersion = String("HSA_OVERRIDE_GFX_VERSION")
)

func Uint(key string, defaultValue uint) func() uint {
	return func() uint {
		if s := Var(key); s != "" {
			if n, err := strconv.ParseUint(s, 10, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return uint(n)
			}
		}

		return defaultValue
	}
}

var (
	// NumParallel sets the number of parallel model requests. NumParallel can be configured via the OLLAMA_NUM_PARALLEL environment variable.
	NumParallel = Uint("OLLAMA_NUM_PARALLEL", 1)
	// MaxRunners sets the maximum number of loaded models. MaxRunners can be configured via the OLLAMA_MAX_LOADED_MODELS environment variable.
	MaxRunners = Uint("OLLAMA_MAX_LOADED_MODELS", 0)
	// MaxQueue sets the maximum number of queued requests. MaxQueue can be configured via the OLLAMA_MAX_QUEUE environment variable.
	MaxQueue = Uint("OLLAMA_MAX_QUEUE", 512)
)

// MemoryReclaimThreshold returns the GPU VRAM usage ratio (0–1) at which the
// scheduler watchdog evicts the least-recently-used idle runner. Zero disables.
func MemoryReclaimThreshold() float64 {
	s := Var("ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD")
	if s == "" {
		return 0
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f <= 0 || f > 1 {
		slog.Warn("invalid ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD, disabled", "value", s)
		return 0
	}
	return f
}

// RunnerBusyTimeout returns how long a runner may stay busy before the watchdog
// forces an unload. Zero disables (default).
func RunnerBusyTimeout() time.Duration {
	s := Var("ZEROLLAMA_RUNNER_BUSY_TIMEOUT")
	if s == "" {
		return 0
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			return time.Duration(n) * time.Second
		}
		slog.Warn("invalid ZEROLLAMA_RUNNER_BUSY_TIMEOUT, disabled", "value", s)
		return 0
	}
	if d <= 0 {
		return 0
	}
	return d
}

// SchedWatchdogInterval is how often the scheduler memory/busy watchdog runs.
func SchedWatchdogInterval() time.Duration {
	s := Var("ZEROLLAMA_SCHED_WATCHDOG_INTERVAL")
	if s == "" {
		return 30 * time.Second
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			return time.Duration(n) * time.Second
		}
		return 30 * time.Second
	}
	if d < 5*time.Second {
		return 5 * time.Second
	}
	return d
}

func Uint64(key string, defaultValue uint64) func() uint64 {
	return func() uint64 {
		if s := Var(key); s != "" {
			if n, err := strconv.ParseUint(s, 10, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return n
			}
		}

		return defaultValue
	}
}

// Set aside VRAM per GPU
var GpuOverhead = Uint64("OLLAMA_GPU_OVERHEAD", 0)

type EnvVar struct {
	Name        string
	Value       any
	Description string
}

func AsMap() map[string]EnvVar {
	ret := map[string]EnvVar{
		"OLLAMA_DEBUG":                             {"OLLAMA_DEBUG", LogLevel(), "Show additional debug information (e.g. OLLAMA_DEBUG=1)"},
		"OLLAMA_DEBUG_LOG_REQUESTS":                {"OLLAMA_DEBUG_LOG_REQUESTS", DebugLogRequests(), "Log inference request bodies and replay curl commands to a temp directory"},
		"OLLAMA_FLASH_ATTENTION":                   {"OLLAMA_FLASH_ATTENTION", FlashAttention(false), "Enabled flash attention"},
		"OLLAMA_KV_CACHE_TYPE":                     {"OLLAMA_KV_CACHE_TYPE", KvCacheType(), "Quantization type for the K/V cache (default: f16)"},
		"OLLAMA_GPU_OVERHEAD":                      {"OLLAMA_GPU_OVERHEAD", GpuOverhead(), "Reserve a portion of VRAM per GPU (bytes)"},
		"OLLAMA_HOST":                              {"OLLAMA_HOST", Host(), "IP Address for the ollama server (default 0.0.0.0:11434)"},
		"OLLAMA_KEEP_ALIVE":                        {"OLLAMA_KEEP_ALIVE", KeepAlive(), "The duration that models stay loaded in memory (default \"5m\")"},
		"OLLAMA_LLM_LIBRARY":                       {"OLLAMA_LLM_LIBRARY", LLMLibrary(), "Set LLM library to bypass autodetection"},
		"OLLAMA_LMSTUDIO_IMPORT":                   {"OLLAMA_LMSTUDIO_IMPORT", LMStudioImport(true), "Reuse LM Studio model caches; auto-sync on list/serve and pull shortcut (default true)"},
		"OLLAMA_LMSTUDIO_LIST_ALL":                 {"OLLAMA_LMSTUDIO_LIST_ALL", LMStudioListAll(), "List all LM Studio models even when MLX import needs more disk than available (pull still enforces space)"},
		"OLLAMA_LMSTUDIO_MODELS":                   {"OLLAMA_LMSTUDIO_MODELS", Var("OLLAMA_LMSTUDIO_MODELS"), "Only scan these LM Studio model directories (comma-separated); unset uses default paths"},
		"OLLAMA_LOAD_TIMEOUT":                      {"OLLAMA_LOAD_TIMEOUT", LoadTimeout(), "How long to allow model loads to stall before giving up (default \"5m\")"},
		"OLLAMA_MAX_LOADED_MODELS":                 {"OLLAMA_MAX_LOADED_MODELS", MaxRunners(), "Maximum number of loaded models per GPU"},
		"OLLAMA_MAX_QUEUE":                         {"OLLAMA_MAX_QUEUE", MaxQueue(), "Maximum number of queued requests"},
		"OLLAMA_MODELS":                            {"OLLAMA_MODELS", Models(), "The path to the models directory"},
		"OLLAMA_NO_CLOUD":                          {"OLLAMA_NO_CLOUD", NoCloud(), "Disable Ollama cloud features (remote inference and web search)"},
		"OLLAMA_NOHISTORY":                         {"OLLAMA_NOHISTORY", NoHistory(), "Do not preserve readline history"},
		"OLLAMA_NOPRUNE":                           {"OLLAMA_NOPRUNE", NoPrune(), "Do not prune model blobs on startup"},
		"OLLAMA_TRAINING":                          {"OLLAMA_TRAINING", TrainingEnabled(true), "Enable GPU training (embedded CPython + training.py; HTTP /api/train and optional TCP) (default true)"},
		"OLLAMA_TRAINING_TCP":                      {"OLLAMA_TRAINING_TCP", Var("OLLAMA_TRAINING_TCP"), "Public training TCP listen address; empty or 1 is :9500; 0 or - disables"},
		"OLLAMA_TRAINING_PYTHONPATH":               {"OLLAMA_TRAINING_PYTHONPATH", Var("OLLAMA_TRAINING_PYTHONPATH"), "Repository root containing training.py; must exist if set (no silent fallback). When unset: walk cwd, then $HOME/zerollama or $HOME/ollama"},
		"ZEROLLAMA_REPO":                           {"ZEROLLAMA_REPO", Var("ZEROLLAMA_REPO"), "Alias for repo root (training.py and runtime/); same rules as OLLAMA_TRAINING_PYTHONPATH"},
		"OLLAMA_NUM_PARALLEL":                      {"OLLAMA_NUM_PARALLEL", NumParallel(), "Maximum number of parallel requests"},
		"OLLAMA_ORIGINS":                           {"OLLAMA_ORIGINS", AllowedOrigins(), "A comma separated list of allowed origins"},
		"OLLAMA_SCHED_SPREAD":                      {"OLLAMA_SCHED_SPREAD", SchedSpread(), "Always schedule model across all GPUs"},
		"OLLAMA_MULTIUSER_CACHE":                   {"OLLAMA_MULTIUSER_CACHE", MultiUserCache(), "Optimize prompt caching for multi-user scenarios"},
		"OLLAMA_CONTEXT_LENGTH":                    {"OLLAMA_CONTEXT_LENGTH", ContextLength(), "Context length to use unless otherwise specified (default: 4k/32k/256k based on VRAM)"},
		"OLLAMA_EDITOR":                            {"OLLAMA_EDITOR", Editor(), "Path to editor for interactive prompt editing (Ctrl+G)"},
		"OLLAMA_NEW_ENGINE":                        {"OLLAMA_NEW_ENGINE", NewEngine(), "Deprecated — ignored for routing; use ZEROLLAMA_LLAMA_SERVER / --edge (Phase 16/17)"},
		"OLLAMA_REMOTES":                           {"OLLAMA_REMOTES", Remotes(), "Allowed hosts for remote models (default \"ollama.com\")"},
		"ELIZACLOUD_API_KEY":                       {"ELIZACLOUD_API_KEY", ElizaCloudAPIKey(), "API key for Eliza Cloud (X-API-Key); required for remote inference when using Eliza"},
		"OLLAMA_SGLANG_URL":                        {"OLLAMA_SGLANG_URL", SGLangURL(), "Base URL for SGLang when modality_backends.video_understanding=sglang"},
		"ZEROLLAMA_RUNTIME_URL":                    {"ZEROLLAMA_RUNTIME_URL", RuntimeURL(), "Base URL for Python GGUF runtime sidecar (PagedAttention)"},
		"ZEROLLAMA_RUNTIME_EMBED":                  {"ZEROLLAMA_RUNTIME_EMBED", RuntimeEmbedDisplay(), "Embed runtime in-process (CGO); default on if URL unset"},
		"ZEROLLAMA_RUNTIME_EMBED_PORT":             {"ZEROLLAMA_RUNTIME_EMBED_PORT", Var("ZEROLLAMA_RUNTIME_EMBED_PORT"), "Loopback port for embedded runtime HTTP (default 8081)"},
		"ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY":   {"ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY", ggmlPauseWhenRuntimeBusyDisplay(), "Pause new ggml loads when Python runtime queue is deep (auto when runtime configured)"},
		"ZEROLLAMA_GGML_PAUSE_RUNTIME_MIN_BACKLOG": {"ZEROLLAMA_GGML_PAUSE_RUNTIME_MIN_BACKLOG", Var("ZEROLLAMA_GGML_PAUSE_RUNTIME_MIN_BACKLOG"), "Runtime waiting+running threshold to pause ggml (default 4)"},
		"ZEROLLAMA_RUNTIME":                        {"ZEROLLAMA_RUNTIME", runtimeEnvDisplay(), "Python runtime proxy for text GGUF: 1/on, 0/off; unset=off on Darwin (ggml default), on on Linux when URL set"},
		"ZEROLLAMA_LLAMA_CPP_BACKEND":              {"ZEROLLAMA_LLAMA_CPP_BACKEND", LlamaCppBackend(), "If 1, route eligible GGUF text inference through llama.cpp (Python runtime) instead of ggml runner"},
		"ZEROLLAMA_LLAMA_SERVER":                   {"ZEROLLAMA_LLAMA_SERVER", LlamaServerBackend(), "Go → llama-server for GGUF (1/on, auto=Linux serve default, 0/off)"},
		"ZEROLLAMA_FLASH_MOE":                      {"ZEROLLAMA_FLASH_MOE", FlashMoEEnabled(), "Flash-MoE slot-bank via anemll-flash-llama.cpp (1/on)"},
		"ZEROLLAMA_FLASH_MOE_SIDECAR":              {"ZEROLLAMA_FLASH_MOE_SIDECAR", FlashMoESidecar(), "Routed-expert sidecar directory for Flash-MoE"},
		"ZEROLLAMA_FLASH_MOE_MODE":                 {"ZEROLLAMA_FLASH_MOE_MODE", FlashMoEMode(), "Flash-MoE runtime mode (default slot-bank)"},
		"ZEROLLAMA_FLASH_MOE_SLOT_BANK":            {"ZEROLLAMA_FLASH_MOE_SLOT_BANK", Var("ZEROLLAMA_FLASH_MOE_SLOT_BANK"), "Resident expert slots per MoE layer"},
		"ZEROLLAMA_FLASH_MOE_TOPK":                 {"ZEROLLAMA_FLASH_MOE_TOPK", Var("ZEROLLAMA_FLASH_MOE_TOPK"), "Routed expert top-k override (0=model default)"},
		"ZEROLLAMA_FLASH_MOE_PREFETCH":             {"ZEROLLAMA_FLASH_MOE_PREFETCH", FlashMoEPrefetchTemporal(), "One-step temporal expert prefetch (1/on)"},
		"ZEROLLAMA_FLASH_MOE_LLAMA_SERVER_BIN":     {"ZEROLLAMA_FLASH_MOE_LLAMA_SERVER_BIN", FlashMoELlamaServerBin(), "Override Flash-MoE llama-server binary path"},
		"FLASH_MOE_REPO":                           {"FLASH_MOE_REPO", FlashMoERepo(), "anemll-flash-llama.cpp checkout for build script"},
		"ANE_REPO":                                 {"ANE_REPO", ANERepo(), "maderix/ane checkout for ANE probe bridge"},
		"ZEROLLAMA_ANE_DRAFT":                      {"ZEROLLAMA_ANE_DRAFT", ANEDraftEnabled(), "Route speculative draft to ANE when wired (default off)"},
		"ZEROLLAMA_EDGE":                           {"ZEROLLAMA_EDGE", EdgeMode(), "Phase 16 upstream-shaped edge: llama-server GGUF, runtime chat off (1/on)"},
		"ZEROLLAMA_RUNTIME_DARWIN_SIDECAR":         {"ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", darwinSidecarEnvDisplay(), "Darwin uv sidecar: unset/on=persist, managed=kill with serve, 0=off"},
		"ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD":       {"ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD", MemoryReclaimThreshold(), "GPU VRAM usage ratio (0–1) to evict idle LRU runner; 0=off"},
		"ZEROLLAMA_RUNNER_BUSY_TIMEOUT":            {"ZEROLLAMA_RUNNER_BUSY_TIMEOUT", RunnerBusyTimeout(), "Force-unload runners busy longer than this; 0=off"},
		"ZEROLLAMA_SCHED_WATCHDOG_INTERVAL":        {"ZEROLLAMA_SCHED_WATCHDOG_INTERVAL", SchedWatchdogInterval(), "Scheduler memory/busy watchdog tick interval (default 30s)"},
		"ZEROLLAMA_ELIZA_NGRAM":                    {"ZEROLLAMA_ELIZA_NGRAM", ElizaNgramDefault(), "Auto ngram-simple for eliza-1-* on llama-server (1/on; default off)"},
		"ZEROLLAMA_FLEET_PEERS":                    {"ZEROLLAMA_FLEET_PEERS", FleetPeers(), "Comma-separated zerollama base URLs for fleet management polling"},
		"ZEROLLAMA_FLEET_LISTEN":                   {"ZEROLLAMA_FLEET_LISTEN", FleetListen(), "Fleet management HTTP listen address (default 0.0.0.0:11450)"},
		"ZEROLLAMA_FLEET_POLL_INTERVAL":            {"ZEROLLAMA_FLEET_POLL_INTERVAL", Var("ZEROLLAMA_FLEET_POLL_INTERVAL"), "Fleet peer poll interval (default 3s)"},
		"ZEROLLAMA_FLEET_ASSIGN_SECRET":            {"ZEROLLAMA_FLEET_ASSIGN_SECRET", "(set/unset)", "F5 HMAC secret for assignment tokens (empty disables)"},
		"ZEROLLAMA_FLEET_ASSIGN_TTL":               {"ZEROLLAMA_FLEET_ASSIGN_TTL", Var("ZEROLLAMA_FLEET_ASSIGN_TTL"), "F5 assign token TTL (default 8s, clamp 2–30s)"},
		"ZEROLLAMA_MDNS":                           {"ZEROLLAMA_MDNS", mdnsEnabledDisplay(), "Advertise this zerollama node via mDNS (_zerollama._tcp) on LAN"},
		"ZEROLLAMA_FLEET_MDNS":                     {"ZEROLLAMA_FLEET_MDNS", fleetMDNSDisplay(), "Browse LAN for zerollama nodes via mDNS (merges with ZEROLLAMA_FLEET_PEERS)"},
		"ZEROLLAMA_FLEET_MDNS_ADVERTISE":           {"ZEROLLAMA_FLEET_MDNS_ADVERTISE", fleetMDNSAdvertiseDisplay(), "Advertise fleet management endpoint via mDNS (_zerollama-fleet._tcp)"},
		"ZEROLLAMA_GGML_CLAMP_NUM_CTX":             {"ZEROLLAMA_GGML_CLAMP_NUM_CTX", ggmlClampNumCtxDisplay(), "Lower ggml merged num_ctx to suggested_max before load (default off; why: silent clamp broke operator trust — parity with Phase 13 runtime clamp)"},
		"ZEROLLAMA_GGML_SUGGEST_CTX_MAX":           {"ZEROLLAMA_GGML_SUGGEST_CTX_MAX", Var("ZEROLLAMA_GGML_SUGGEST_CTX_MAX"), "Upper bound for ggml suggested_max_num_ctx binary search (default 131072)"},
		"ZEROLLAMA_GGML_VRAM_MARGIN":               {"ZEROLLAMA_GGML_VRAM_MARGIN", Var("ZEROLLAMA_GGML_VRAM_MARGIN"), "Multiply ggml load VRAM estimate before compare to free memory (default 1.05; conservative vs GraphSize proxy)"},
		"ZEROLLAMA_LEGACY_RUNNER":                  {"ZEROLLAMA_LEGACY_RUNNER", LegacyRunnerForced(), "If 1, always load ggml runner even for models tagged zerollama-runtime"},
		"OLLAMA_RUNTIME_ALL":                       {"OLLAMA_RUNTIME_ALL", RuntimeProxyAll(), "If 1 and ZEROLLAMA_RUNTIME_URL is set, proxy all local /api/generate to the runtime"},
		"OLLAMA_FFMPEG":                            {"OLLAMA_FFMPEG", FFmpegBin(), "ffmpeg binary for native video frame sampling (default: ffmpeg on PATH)"},
		"OLLAMA_VIDEO_MAX_FRAMES":                  {"OLLAMA_VIDEO_MAX_FRAMES", VideoMaxFrames(), "Max frames sampled per video (default 32)"},
		"OLLAMA_VIDEO_SAMPLE_MODE":                 {"OLLAMA_VIDEO_SAMPLE_MODE", VideoSampleMode(), "Native sampling: fps (time-uniform) or stride (every Nth frame) (default fps)"},
		"OLLAMA_VIDEO_STRIDE":                      {"OLLAMA_VIDEO_STRIDE", VideoStride(), "When OLLAMA_VIDEO_SAMPLE_MODE=stride, emit every Nth frame (default 30)"},
		"OLLAMA_VIDEO_SAMPLE_FPS":                  {"OLLAMA_VIDEO_SAMPLE_FPS", VideoSampleFPS(), "ffmpeg fps filter value for sampling (default 1)"},
		"OLLAMA_VIDEO_MAX_BYTES":                   {"OLLAMA_VIDEO_MAX_BYTES", VideoMaxBytes(), "Max video payload size in bytes (default 256MiB)"},
		"OLLAMA_VIDEO_MAX_PER_MESSAGE":             {"OLLAMA_VIDEO_MAX_PER_MESSAGE", VideoMaxVideosPerMessage(), "Max video_url parts per message (default 1)"},
		"OLLAMA_VIDEO_MAX_IMAGES_PER_MESSAGE":      {"OLLAMA_VIDEO_MAX_IMAGES_PER_MESSAGE", VideoMaxImagesPerMessage(), "Max images after video expansion per message (default 64)"},
		"OLLAMA_VIDEO_FFMPEG_TIMEOUT":              {"OLLAMA_VIDEO_FFMPEG_TIMEOUT", VideoFFmpegTimeout(), "Max duration for ffmpeg sampling (default 5m)"},
		"OLLAMA_VIDEO_ALLOW_INSECURE_HTTP":         {"OLLAMA_VIDEO_ALLOW_INSECURE_HTTP", VideoAllowInsecureHTTP(), "Allow http:// for remote video_url fetches (default: require https)"},
		"OLLAMA_VIDEO_FETCH_TIMEOUT":               {"OLLAMA_VIDEO_FETCH_TIMEOUT", VideoFetchTimeout(), "Max duration for remote video_url HTTP GET (default 10m)"},
		"OLLAMA_LIMIT_MM_DATA_PER_REQUEST":         {"OLLAMA_LIMIT_MM_DATA_PER_REQUEST", "", "JSON caps per latest user turn, e.g. {\"image\":4,\"video\":1,\"audio\":1} (SGLang limit_mm_data_per_request)"},
		"OLLAMA_IMAGE_EMBED_CACHE_SIZE":            {"OLLAMA_IMAGE_EMBED_CACHE_SIZE", ImageEmbedCacheSize(), "Per-runner vision embed (ViT) LRU cache slots (default 4; set 32–64 for video agents)"},
		"OLLAMA_IMAGE_EMBED_CACHE_MAX":             {"OLLAMA_IMAGE_EMBED_CACHE_MAX", ImageEmbedCacheMax(), "Auto-grow ViT embed LRU up to this cap when a turn has more frames (default 64; radix pool may grow further under byte budget)"},
		"OLLAMA_IMAGE_EMBED_CACHE_BYTES":           {"OLLAMA_IMAGE_EMBED_CACHE_BYTES", ImageEmbedCacheBytes(), "ViT embed float byte budget (0=use radix default when OLLAMA_VIT_RADIX on, else slot-only)"},
		"OLLAMA_VIT_RADIX":                         {"OLLAMA_VIT_RADIX", VitRadixEnabled(), "Cross-request ViT embed pool (SGLang MultiModalStaticCache; default on; 100MiB unless BYTES set)"},
		"OLLAMA_KV_MM_PAD_RADIX":                   {"OLLAMA_KV_MM_PAD_RADIX", KVMMPadRadixEnabled(), "Stamp vision pad Token ids with content-hash pad_values (SGLang MM radix; default on)"},

		// Informational
		"HTTP_PROXY":  {"HTTP_PROXY", String("HTTP_PROXY")(), "HTTP proxy"},
		"HTTPS_PROXY": {"HTTPS_PROXY", String("HTTPS_PROXY")(), "HTTPS proxy"},
		"NO_PROXY":    {"NO_PROXY", String("NO_PROXY")(), "No proxy"},
	}

	if runtime.GOOS != "windows" {
		// Windows environment variables are case-insensitive so there's no need to duplicate them
		ret["http_proxy"] = EnvVar{"http_proxy", String("http_proxy")(), "HTTP proxy"}
		ret["https_proxy"] = EnvVar{"https_proxy", String("https_proxy")(), "HTTPS proxy"}
		ret["no_proxy"] = EnvVar{"no_proxy", String("no_proxy")(), "No proxy"}
	}

	if runtime.GOOS != "darwin" {
		ret["CUDA_VISIBLE_DEVICES"] = EnvVar{"CUDA_VISIBLE_DEVICES", CudaVisibleDevices(), "Set which NVIDIA devices are visible"}
		ret["HIP_VISIBLE_DEVICES"] = EnvVar{"HIP_VISIBLE_DEVICES", HipVisibleDevices(), "Set which AMD devices are visible by numeric ID"}
		ret["ROCR_VISIBLE_DEVICES"] = EnvVar{"ROCR_VISIBLE_DEVICES", RocrVisibleDevices(), "Set which AMD devices are visible by UUID or numeric ID"}
		ret["GGML_VK_VISIBLE_DEVICES"] = EnvVar{"GGML_VK_VISIBLE_DEVICES", VkVisibleDevices(), "Set which Vulkan devices are visible by numeric ID"}
		ret["GPU_DEVICE_ORDINAL"] = EnvVar{"GPU_DEVICE_ORDINAL", GpuDeviceOrdinal(), "Set which AMD devices are visible by numeric ID"}
		ret["HSA_OVERRIDE_GFX_VERSION"] = EnvVar{"HSA_OVERRIDE_GFX_VERSION", HsaOverrideGfxVersion(), "Override the gfx used for all detected AMD GPUs"}
		ret["OLLAMA_VULKAN"] = EnvVar{"OLLAMA_VULKAN", strconv.FormatBool(EnableVulkan(true)), "Enable Vulkan support"}
		ret["OLLAMA_IGPU_ENABLE"] = EnvVar{"OLLAMA_IGPU_ENABLE", strconv.FormatBool(EnableIntegratedGPU(false)), "Enable integrated GPU selection"}
	}

	return ret
}

func Values() map[string]string {
	vals := make(map[string]string)
	for k, v := range AsMap() {
		vals[k] = fmt.Sprintf("%v", v.Value)
	}
	return vals
}

// Var returns an environment variable stripped of leading and trailing quotes or spaces
func Var(key string) string {
	return strings.Trim(strings.TrimSpace(os.Getenv(key)), "\"'")
}

// WhisperBin is the path to a whisper.cpp-compatible STT binary. Required when using
// modality_backends.transcribe=whisper unless the binary is discoverable via PATH as "whisper".
func WhisperBin() string {
	return cmp.Or(Var("OLLAMA_WHISPER_BIN"), "whisper")
}

// WhisperModelPath is a default GGML model path when backend_paths.whisper_model is unset.
func WhisperModelPath() string {
	return Var("OLLAMA_WHISPER_MODEL")
}

// WhisperExtraArgs are extra CLI tokens appended to the whisper invocation (split on spaces).
func WhisperExtraArgs() string {
	return Var("OLLAMA_WHISPER_EXTRA_ARGS")
}

// PiperBin is the Piper TTS executable.
func PiperBin() string {
	return cmp.Or(Var("OLLAMA_PIPER_BIN"), "piper")
}

// ExternalImageBin is a user script or binary for modality_backends.image=external-image.
func ExternalImageBin() string {
	return Var("OLLAMA_EXTERNAL_IMAGE_BIN")
}

// ModalityWhisperTimeout bounds Whisper subprocess runtime (default 10m).
func ModalityWhisperTimeout() time.Duration {
	return modalityTimeout("OLLAMA_WHISPER_TIMEOUT", 10*time.Minute)
}

// ModalityPiperTimeout bounds Piper subprocess runtime (default 5m).
func ModalityPiperTimeout() time.Duration {
	return modalityTimeout("OLLAMA_PIPER_TIMEOUT", 5*time.Minute)
}

// TTSURL is a fleet-wide default base or full URL for modality_backends.speech=remote-tts
// when backend_paths.tts_url is unset (e.g. http://cozmic:8090 or …/v1/audio/speech).
func TTSURL() string {
	return strings.TrimSpace(Var("OLLAMA_TTS_URL"))
}

// ModalityRemoteTTSTimeout bounds remote TTS HTTP runtime (default 5m).
func ModalityRemoteTTSTimeout() time.Duration {
	return modalityTimeout("OLLAMA_TTS_TIMEOUT", 5*time.Minute)
}

// ModalityExternalImageTimeout bounds external image hook runtime (default 10m).
func ModalityExternalImageTimeout() time.Duration {
	return modalityTimeout("OLLAMA_EXTERNAL_IMAGE_TIMEOUT", 10*time.Minute)
}

// ComfyUIURL is the base URL for a running ComfyUI server when
// modality_backends.image=comfyui. Defaults to the stock local port.
// WHY a separate process URL (not in-process Diffusers): Comfy owns custom-node
// graphs and HF weight layout; Zerollama only injects agent options and polls PNG output.
func ComfyUIURL() string {
	return strings.TrimSuffix(strings.TrimSpace(cmp.Or(Var("OLLAMA_COMFYUI_URL"), "http://127.0.0.1:8188")), "/")
}

// ModalityComfyUITimeout bounds one Comfy workflow (queue + poll + download).
// WHY default 10m (not the short timeouts used for Whisper/Piper): Qwen-Image GGUF,
// GLM-Image, and FLUX.2-dev on 16GB cards routinely take minutes per image; a 30s
// deadline would abort valid agent jobs.
func ModalityComfyUITimeout() time.Duration {
	return modalityTimeout("OLLAMA_COMFYUI_TIMEOUT", 10*time.Minute)
}

// ComfyUIWorkflowsRoot is an optional base for relative backend_paths.comfy_workflow_dir
// values. WHY: shipped modelfiles use short paths like scripts/comfyui/qwen-image that
// only resolve if the daemon cwd is the repo root (or this root is set). Absolute and
// ~/… paths ignore this variable.
func ComfyUIWorkflowsRoot() string {
	return Var("OLLAMA_COMFYUI_WORKFLOWS_ROOT")
}

func modalityTimeout(envKey string, defaultDur time.Duration) time.Duration {
	if s := Var(envKey); s != "" {
		if d, err := time.ParseDuration(s); err == nil && d > 0 {
			return d
		}
	}
	return defaultDur
}

// SGLangURL is the base URL for optional SGLang HTTP proxy (e.g. http://127.0.0.1:30000).
// Used when modality_backends.video_understanding=sglang.
func SGLangURL() string {
	return strings.TrimSuffix(strings.TrimSpace(Var("OLLAMA_SGLANG_URL")), "/")
}

// RuntimeURL is the base URL for the zerollama Python inference sidecar (e.g. http://127.0.0.1:8081).
// Used when modality_backends.inference=zerollama-runtime or OLLAMA_RUNTIME_ALL=1.
func RuntimeURL() string {
	return strings.TrimSuffix(strings.TrimSpace(Var("ZEROLLAMA_RUNTIME_URL")), "/")
}

// RuntimeProxyAll forwards all local /api/generate requests to RuntimeURL when set.
func RuntimeProxyAll() bool {
	return Var("OLLAMA_RUNTIME_ALL") == "1"
}

// RuntimeDefault is true when ZEROLLAMA_RUNTIME is explicitly enabled (1/true).
func RuntimeDefault() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_RUNTIME"))
	return v == "1" || strings.EqualFold(v, "true")
}

// RuntimeDefaultOn enables the Python runtime for local non-streaming generate/chat
// when RuntimeURL is set. Unset ZEROLLAMA_RUNTIME defaults to on on Linux (CUDA runtime
// path); on Darwin defaults to off so plain GGUF uses ggml Metal (M7 benchmark).
// Set ZEROLLAMA_RUNTIME=1 or --llama-cpp-backend to opt in on Mac.
func RuntimeDefaultOn() bool {
	if RuntimeURL() == "" {
		return false
	}
	v := strings.TrimSpace(Var("ZEROLLAMA_RUNTIME"))
	if v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	if v == "1" || strings.EqualFold(v, "true") {
		return true
	}
	if runtime.GOOS == "darwin" {
		return false
	}
	return true
}

// LegacyRunnerForced keeps the ggml runner for models tagged zerollama-runtime.
func LegacyRunnerForced() bool {
	return Var("ZEROLLAMA_LEGACY_RUNNER") == "1"
}

// LlamaCppBackend routes eligible local GGUF text inference through the Python
// runtime + pinned llama.cpp instead of the in-process ggml runner. Set via
// ZEROLLAMA_LLAMA_CPP_BACKEND=1 or `zerollama serve --llama-cpp-backend`.
func LlamaCppBackend() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_LLAMA_CPP_BACKEND"))
	return v == "1" || strings.EqualFold(v, "true")
}

// AgentCacheRuntimeEnabled routes GGUF /api/chat with prompt_cache_key through the
// Python runtime on Darwin when ZEROLLAMA_RUNTIME is set, even if runtime-default
// is off. Why: L3 slot pinning skips repeat prefill for agent megaprompts; Metal ggml
// default path cannot use L3. Set ZEROLLAMA_AGENT_CACHE_RUNTIME=0 to disable.
func AgentCacheRuntimeEnabled() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_AGENT_CACHE_RUNTIME"))
	if v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	if v == "1" || strings.EqualFold(v, "true") {
		return true
	}
	// auto (default): on when runtime URL is configured and not edge mode.
	return RuntimeURL() != "" && !EdgeMode()
}

// DarwinSidecarKillOnServeExit is true when the child sidecar should exit with zerollama serve.
// Default: persist — sidecar survives serve restarts and is reused on the next bootstrap.
// Set ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=managed to kill the child on serve shutdown.
func DarwinSidecarKillOnServeExit() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR"))
	return strings.EqualFold(v, "managed")
}

func darwinSidecarEnvDisplay() string {
	v := strings.TrimSpace(Var("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR"))
	switch {
	case v == "0" || strings.EqualFold(v, "false"):
		return "off"
	case strings.EqualFold(v, "managed"):
		return "managed"
	default:
		return "persist"
	}
}

// ElizaNgramDefault enables ngram-simple speculative decoding for eliza-1-* tags on the
// llama-server path when no draft model is configured. Set ZEROLLAMA_ELIZA_NGRAM=1.
func ElizaNgramDefault() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_ELIZA_NGRAM"))
	return v == "1" || strings.EqualFold(v, "true")
}

// LlamaServerBackend routes eligible GGUF inference through Go → llama-server
// (upstream shape). Set via ZEROLLAMA_LLAMA_SERVER=1, `auto` (Linux serve default),
// or `zerollama serve --llama-server-backend`.
func LlamaServerBackend() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_LLAMA_SERVER"))
	return v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "auto")
}

// LlamaServerBackendExplicit is true for operator opt-in (all platforms, all GGUF).
func LlamaServerBackendExplicit() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_LLAMA_SERVER"))
	return v == "1" || strings.EqualFold(v, "true")
}

// LlamaServerBackendAuto is true when Linux serve auto-enables upstream routing.
func LlamaServerBackendAuto() bool {
	return strings.EqualFold(strings.TrimSpace(Var("ZEROLLAMA_LLAMA_SERVER")), "auto")
}

// LlamaServerBackendDisabled is true when ZEROLLAMA_LLAMA_SERVER=0 explicitly disables
// Phase 17 auto-default on Linux.
func LlamaServerBackendDisabled() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_LLAMA_SERVER"))
	return v == "0" || strings.EqualFold(v, "false")
}

// EdgeMode enables Phase 16 upstream-shaped deployment: Go → llama-server for GGUF,
// Python runtime off for chat (training embed may still run).
func EdgeMode() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_EDGE"))
	return v == "1" || strings.EqualFold(v, "true")
}

// ApplyEdgeModeDefaults configures upstream-shaped local inference without the Python
// runtime chat middleman. Safe to call when the flag is off (no-op).
func ApplyEdgeModeDefaults() {
	if !EdgeMode() {
		return
	}
	// Always force llama-server when edge mode is active, even if the operator
	// previously set ZEROLLAMA_LLAMA_SERVER=0. Edge with no llama-server would
	// disable the Python runtime but have no inference backend at all.
	_ = os.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	if strings.TrimSpace(Var("ZEROLLAMA_LEGACY_RUNNER")) == "" {
		_ = os.Setenv("ZEROLLAMA_LEGACY_RUNNER", "1")
	}
	// Runtime defaults: do not override an explicit operator value, but default off.
	if strings.TrimSpace(Var("ZEROLLAMA_RUNTIME")) == "" {
		_ = os.Setenv("ZEROLLAMA_RUNTIME", "0")
	}
	if strings.TrimSpace(Var("ZEROLLAMA_RUNTIME_EMBED")) == "" {
		_ = os.Setenv("ZEROLLAMA_RUNTIME_EMBED", "0")
	}
	if strings.TrimSpace(Var("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR")) == "" {
		_ = os.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "0")
	}
}

// ApplyLlamaServerBackendDefaults skips Python runtime routing so sched loads
// models through Go → llama-server. Safe to call when the flag is off (no-op).
func ApplyLlamaServerBackendDefaults() {
	if !LlamaServerBackend() {
		return
	}
	if strings.TrimSpace(Var("ZEROLLAMA_LEGACY_RUNNER")) == "" {
		_ = os.Setenv("ZEROLLAMA_LEGACY_RUNNER", "1")
	}
}

// ApplyLlamaCppBackendDefaults sets runtime env needed for llama.cpp backend testing.
// Safe to call when the flag is off (no-op).
func ApplyLlamaCppBackendDefaults() {
	if !LlamaCppBackend() {
		return
	}
	if strings.TrimSpace(Var("ZEROLLAMA_RUNTIME")) == "" {
		_ = os.Setenv("ZEROLLAMA_RUNTIME", "1")
	}
	if strings.TrimSpace(Var("ZEROLLAMA_AUTO_CONFIG")) == "" &&
		strings.TrimSpace(Var("ZEROLLAMA_RUNTIME_CONFIG")) == "" {
		_ = os.Setenv("ZEROLLAMA_AUTO_CONFIG", "1")
	}
}

func runtimeEnvDisplay() string {
	if RuntimeURL() == "" {
		return Var("ZEROLLAMA_RUNTIME")
	}
	if RuntimeDefaultOn() {
		return "on (default)"
	}
	return "off"
}

// RuntimeEmbedEnabled starts inference runtime inside the zerollama process (CGO).
// Default: on when ZEROLLAMA_RUNTIME_URL is unset; set 0 to use an external sidecar only.
func RuntimeEmbedEnabled() bool {
	// WHY EdgeBuildTag: edge artifacts must not link/spawn runtime embed even when
	// operator sets ZEROLLAMA_EDGE=0 on an edge-marked binary.
	if EdgeBuildTag {
		return false
	}
	if EdgeMode() {
		return false
	}
	v := strings.TrimSpace(Var("ZEROLLAMA_RUNTIME_EMBED"))
	if v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	if v == "1" || strings.EqualFold(v, "true") {
		return true
	}
	if strings.TrimSpace(Var("ZEROLLAMA_RUNTIME_URL")) != "" {
		return false
	}
	// Darwin: system libpython is too old for embed; zerollama serve spawns a uv sidecar.
	if runtime.GOOS == "darwin" {
		return false
	}
	return true
}

func RuntimeEmbedDisplay() string {
	if RuntimeEmbedEnabled() {
		return "on"
	}
	return "off"
}

// Training allowed window: ZEROLLAMA_TRAINING_ALLOWED_WINDOW=22:00-06:00 (optional),
// ZEROLLAMA_TRAINING_WINDOW_TZ (IANA name or "local"). Why: batch fine-tune on a shared
// GPU without a unified FIFO — night-window policy is the first SLO hook.

// Training submit policy (T6): idle-wait, defer queue, and priority interact with
// server/inference_workload.go and server/training_defer_queue.go. See
// docs/scheduling-vram-policy.md for rationale — inference and training are not one
// FIFO because PyTorch epochs are not safely preemptible from Go.

// TrainingWaitInferenceIdle rejects new training job submit while ggml inference is busy.
func TrainingWaitInferenceIdle() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE"))
	return v == "1" || strings.EqualFold(v, "true")
}

// TrainingWaitGgmlLoaded treats resident ggml runners as busy when idle-wait is on.
func TrainingWaitGgmlLoaded() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_WAIT_GGML_LOADED"))
	if v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	if v == "1" || strings.EqualFold(v, "true") {
		return true
	}
	return TrainingWaitInferenceIdle()
}

// TrainingWaitFailClosed rejects training submit when runtime /health cannot be read (idle-wait on).
func TrainingWaitFailClosed() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_WAIT_FAIL_CLOSED"))
	if v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	if v == "1" || strings.EqualFold(v, "true") {
		return true
	}
	return TrainingWaitInferenceIdle()
}

// TrainingQueueOnBusy enqueues training submit instead of HTTP 409 when policy rejects.
// Inference backlog defer requires ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE=1; outside-window
// defer works with ZEROLLAMA_TRAINING_ALLOWED_WINDOW alone.
func TrainingQueueOnBusy() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_QUEUE_ON_BUSY"))
	return v == "1" || strings.EqualFold(v, "true")
}

// TrainingQueuePollInterval returns how often the deferred training queue is polled.
func TrainingQueuePollInterval() time.Duration {
	raw := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_QUEUE_POLL_SECS"))
	if raw == "" {
		return 5 * time.Second
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 {
		return 5 * time.Second
	}
	return time.Duration(n) * time.Second
}

// TrainingQueueMaxDepth caps deferred jobs waiting for inference idle (0 = unlimited).
func TrainingQueueMaxDepth() int {
	raw := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_QUEUE_MAX"))
	if raw == "" {
		return 32
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return 32
	}
	return n
}

// TrainingQueueTombstoneTTL is how long terminal defer-* records stay queryable (0 = keep forever).
func TrainingQueueTombstoneTTL() time.Duration {
	raw := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_QUEUE_TOMBSTONE_SECS"))
	if raw == "" {
		return 24 * time.Hour
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return 24 * time.Hour
	}
	if n == 0 {
		return 0
	}
	return time.Duration(n) * time.Second
}

// TrainingQueueRetryMax is how many times a deferred job may retry promotion after error (0 = no retries).
func TrainingQueueRetryMax() int {
	raw := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_QUEUE_RETRY_MAX"))
	if raw == "" {
		return 3
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return 3
	}
	return n
}

// TrainingQueueRetryInterval is the minimum wait before a failed defer job is promoted again.
func TrainingQueueRetryInterval() time.Duration {
	raw := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_QUEUE_RETRY_SECS"))
	if raw == "" {
		return 30 * time.Second
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 {
		return 30 * time.Second
	}
	return time.Duration(n) * time.Second
}

// TrainingQueueListAll includes terminal defer jobs in GET /api/train/jobs merge (default: waiting only).
func TrainingQueueListAll() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_TRAINING_QUEUE_LIST_ALL"))
	return v == "1" || strings.EqualFold(v, "true")
}

// BlockInferenceDuringTraining rejects runtime proxy requests while training holds the GPU.
func BlockInferenceDuringTraining() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING"))
	if v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	return true
}

// WanVideoTimeoutSec overrides manifest video_generation.timeout_sec when > 0.
func WanVideoTimeoutSec() int {
	raw := strings.TrimSpace(Var("ZEROLLAMA_WAN_VIDEO_TIMEOUT"))
	if raw == "" {
		return 0
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 {
		return 0
	}
	return n
}

// RuntimeDarwinSidecarLikely reports whether macOS serve will start or attach to a uv sidecar.
func RuntimeDarwinSidecarLikely() bool {
	if EdgeBuildTag {
		return false
	}
	if runtime.GOOS != "darwin" {
		return false
	}
	v := strings.TrimSpace(Var("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR"))
	if v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	if RuntimeURL() != "" {
		return false
	}
	v = strings.TrimSpace(Var("ZEROLLAMA_RUNTIME"))
	if v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	return true
}

// RuntimeConfigured reports whether a Python runtime is in use (external URL, embedded, or Darwin sidecar).
func RuntimeConfigured() bool {
	return RuntimeURL() != "" || RuntimeEmbedEnabled() || RuntimeDarwinSidecarLikely()
}

// FleetPeers is a comma-separated list of zerollama base URLs for fleet management polling.
// Optional when ZEROLLAMA_FLEET_MDNS=1; static list still used for K8s and merges with mDNS peers.
func FleetPeers() string {
	return strings.TrimSpace(Var("ZEROLLAMA_FLEET_PEERS"))
}

// FleetListen is the fleet management HTTP listen address.
// Why 0.0.0.0 default: agents on other LAN hosts must reach the manager; use 127.0.0.1 only for all-local dev.
func FleetListen() string {
	if v := strings.TrimSpace(Var("ZEROLLAMA_FLEET_LISTEN")); v != "" {
		return v
	}
	return "0.0.0.0:11450"
}

// FleetPollInterval is how often the fleet manager polls peer /api/status.
func FleetPollInterval() time.Duration {
	v := strings.TrimSpace(Var("ZEROLLAMA_FLEET_POLL_INTERVAL"))
	if v == "" {
		return 3 * time.Second
	}
	d, err := time.ParseDuration(v)
	if err != nil || d <= 0 {
		return 3 * time.Second
	}
	return d
}

// MDNSEnabled controls zerollama serve LAN advertisement (_zerollama._tcp).
// Why default off: explicit opt-in avoids multicast on networks where operators did not expect it.
func MDNSEnabled() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_MDNS"))
	if v == "" {
		return false
	}
	b, _ := strconv.ParseBool(v)
	return b
}

// FleetMDNS enables mDNS browse for inference nodes on the fleet manager.
func FleetMDNS() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_FLEET_MDNS"))
	if v == "" {
		return false
	}
	b, _ := strconv.ParseBool(v)
	return b
}

// FleetMDNSAdvertise registers the fleet management HTTP endpoint on LAN (_zerollama-fleet._tcp).
func FleetMDNSAdvertise() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_FLEET_MDNS_ADVERTISE"))
	if v == "" {
		return false
	}
	b, _ := strconv.ParseBool(v)
	return b
}

func mdnsEnabledDisplay() string {
	if MDNSEnabled() {
		return "1"
	}
	return "0"
}

func fleetMDNSDisplay() string {
	if FleetMDNS() {
		return "1"
	}
	return "0"
}

func fleetMDNSAdvertiseDisplay() string {
	if FleetMDNSAdvertise() {
		return "1"
	}
	return "0"
}

func ggmlPauseWhenRuntimeBusyDisplay() string {
	v := strings.TrimSpace(Var("ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY"))
	if v == "0" || strings.EqualFold(v, "false") {
		return "off"
	}
	if v == "1" || strings.EqualFold(v, "true") {
		return "on"
	}
	if RuntimeConfigured() {
		return "auto (on)"
	}
	return "auto (off)"
}

// GgmlPauseWhenRuntimeBusy pauses new ggml loads while the Python runtime queue is deep.
// Default auto: on when a runtime is configured (URL or embedded sidecar).
func GgmlPauseWhenRuntimeBusy() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY"))
	if v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	if v == "1" || strings.EqualFold(v, "true") {
		return true
	}
	return RuntimeConfigured()
}

// GgmlPauseRuntimeMinBacklog is waiting+running threshold to pause ggml (default 4).
func GgmlPauseRuntimeMinBacklog() int {
	v := strings.TrimSpace(Var("ZEROLLAMA_GGML_PAUSE_RUNTIME_MIN_BACKLOG"))
	if v == "" {
		return 4
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		return 4
	}
	return n
}

// BlockInferenceFailClosed treats training health probe failures as GPU-busy when blocking is enabled.
func BlockInferenceFailClosed() bool {
	v := strings.TrimSpace(Var("ZEROLLAMA_BLOCK_INFERENCE_FAIL_CLOSED"))
	if v == "0" || strings.EqualFold(v, "false") {
		return false
	}
	if v == "1" || strings.EqualFold(v, "true") {
		return true
	}
	return BlockInferenceDuringTraining()
}

// RuntimeEmbedPort is the loopback port for in-process runtime HTTP.
func RuntimeEmbedPort() int {
	p := strings.TrimSpace(Var("ZEROLLAMA_RUNTIME_EMBED_PORT"))
	if p == "" {
		return 8081
	}
	n, err := strconv.Atoi(p)
	if err != nil || n <= 0 {
		return 8081
	}
	return n
}

// FFmpegBin is the ffmpeg executable used for native video frame sampling (default: ffmpeg on PATH).
func FFmpegBin() string {
	return cmp.Or(Var("OLLAMA_FFMPEG"), "ffmpeg")
}

// VideoMaxFrames caps frames sampled per video (default 32).
func VideoMaxFrames() int {
	return int(Uint64("OLLAMA_VIDEO_MAX_FRAMES", 32)())
}

// VideoSampleMode is "fps" or "stride" for native ffmpeg sampling (default fps).
func VideoSampleMode() string {
	s := strings.ToLower(strings.TrimSpace(Var("OLLAMA_VIDEO_SAMPLE_MODE")))
	switch s {
	case "", "fps":
		return "fps"
	case "stride":
		return "stride"
	default:
		return "fps"
	}
}

// VideoStride is N for stride mode: emit every Nth decoded frame (default 30).
func VideoStride() int {
	n := int(Uint64("OLLAMA_VIDEO_STRIDE", 30)())
	if n < 1 {
		return 1
	}
	return n
}

// VideoSampleFPS is the target frame rate for ffmpeg fps filter (default 1.0).
func VideoSampleFPS() float64 {
	if s := Var("OLLAMA_VIDEO_SAMPLE_FPS"); s != "" {
		if f, err := strconv.ParseFloat(s, 64); err == nil && f > 0 {
			return f
		}
	}
	return 1.0
}

// VideoMaxBytes limits downloaded or embedded video payload size (default 256 MiB).
func VideoMaxBytes() int64 {
	return int64(Uint64("OLLAMA_VIDEO_MAX_BYTES", 256<<20)())
}

// VideoMaxVideosPerMessage limits video_url parts per user message (default 1).
func VideoMaxVideosPerMessage() int {
	return int(Uint64("OLLAMA_VIDEO_MAX_PER_MESSAGE", 1)())
}

// VideoMaxImagesPerMessage caps total images after expanding videos (default 64).
func VideoMaxImagesPerMessage() int {
	return int(Uint64("OLLAMA_VIDEO_MAX_IMAGES_PER_MESSAGE", 64)())
}

// VideoFFmpegTimeout bounds a single ffmpeg invocation (default 5m).
func VideoFFmpegTimeout() time.Duration {
	return modalityTimeout("OLLAMA_VIDEO_FFMPEG_TIMEOUT", 5*time.Minute)
}

// VideoAllowInsecureHTTP allows video_url to use http:// for remote fetches (default false; prefer https).
func VideoAllowInsecureHTTP() bool {
	s := strings.ToLower(strings.TrimSpace(Var("OLLAMA_VIDEO_ALLOW_INSECURE_HTTP")))
	return s == "1" || s == "true" || s == "yes"
}

// VideoFetchTimeout bounds the entire remote GET for video_url (connect + response headers + body read, default 10m).
func VideoFetchTimeout() time.Duration {
	return modalityTimeout("OLLAMA_VIDEO_FETCH_TIMEOUT", 10*time.Minute)
}

// LimitMMDataPerRequest parses OLLAMA_LIMIT_MM_DATA_PER_REQUEST JSON, e.g.
// {"image":4,"video":1,"audio":1}. Matches SGLang limit_mm_data_per_request; zero
// or missing keys mean no cap for that modality. Why env JSON: operators tune per
// deployment without recompiling; same shape as SGLang server args.
func LimitMMDataPerRequest() (image, video, audio int) {
	raw := strings.TrimSpace(Var("OLLAMA_LIMIT_MM_DATA_PER_REQUEST"))
	if raw == "" {
		return 0, 0, 0
	}
	var m map[string]int
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		slog.Warn("OLLAMA_LIMIT_MM_DATA_PER_REQUEST invalid JSON; ignoring", "error", err)
		return 0, 0, 0
	}
	return m["image"], m["video"], m["audio"]
}

// ImageEmbedCacheSize is the number of LRU slots in the per-runner vision embed cache.
// Default 4 matches the upstream llamarunner constant; raise to 32–64 for video agents
// that resend the same clip frames across turns.
func ImageEmbedCacheSize() int {
	n := int(Uint64("OLLAMA_IMAGE_EMBED_CACHE_SIZE", 4)())
	if n < 1 {
		return 1
	}
	return n
}

// ImageEmbedCacheMax is the upper bound for auto-grown ViT embed LRU slots per turn.
// WHY: video agents may send 32 frames while OLLAMA_IMAGE_EMBED_CACHE_SIZE stays at 4;
// llamarunner grows the cache up to this cap for the loaded runner session.
// Also caps per-session ViT overlay hashes (SGLang-style capacity under agent churn).
func ImageEmbedCacheMax() int {
	n := int(Uint64("OLLAMA_IMAGE_EMBED_CACHE_MAX", 64)())
	if n < 1 {
		return 1
	}
	return n
}

// ImageEmbedCacheBytes is an explicit float32-byte budget for the ViT embed pool
// on ollama-engine and ggml llamarunner (PNG + precomputed share one total on llamarunner).
// WHY: SGLang #28441 moved MM caches to capacity-aware pools — slot count alone treats a
// tiny audio embed the same as a 32-frame video. 0 means "unset"; see EffectiveImageEmbedCacheBytes.
func ImageEmbedCacheBytes() int64 {
	return int64(Uint64("OLLAMA_IMAGE_EMBED_CACHE_BYTES", 0)())
}

// DefaultVitRadixBytes matches SGLang's default SGLANG_VLM_CACHE_SIZE_MB (100 MiB).
const DefaultVitRadixBytes = int64(100 << 20)

// VitRadixEnabled turns on the cross-request content-addressed ViT embed pool
// (SGLang MultiModalStaticCache). Default on; set OLLAMA_VIT_RADIX=0 for slot-only LRU.
func VitRadixEnabled() bool {
	return BoolWithDefault("OLLAMA_VIT_RADIX")(true)
}

// EffectiveImageEmbedCacheBytes is the runtime byte budget for the ViT pool.
// Explicit OLLAMA_IMAGE_EMBED_CACHE_BYTES wins; else radix default 100MiB when enabled; else 0 (slot-only).
func EffectiveImageEmbedCacheBytes() int64 {
	if b := ImageEmbedCacheBytes(); b > 0 {
		return b
	}
	if VitRadixEnabled() {
		return DefaultVitRadixBytes
	}
	return 0
}

// KVMMPadRadixEnabled stamps vision soft/pad Token ids with SGLang-style pad_values
// (1_000_000 + hash%2^30) so input-cache / token-id radix keys distinguish images.
// Default on; set OLLAMA_KV_MM_PAD_RADIX=0 to keep vocabulary image_pad ids only.
func KVMMPadRadixEnabled() bool {
	return BoolWithDefault("OLLAMA_KV_MM_PAD_RADIX")(true)
}

// serverConfigData holds the parsed fields from ~/.ollama/server.json.
type serverConfigData struct {
	DisableOllamaCloud bool `json:"disable_ollama_cloud,omitempty"`
}

var (
	serverCfgMu     sync.RWMutex
	serverCfgLoaded bool
	serverCfg       serverConfigData
)

func loadServerConfig() {
	serverCfgMu.RLock()
	if serverCfgLoaded {
		serverCfgMu.RUnlock()
		return
	}
	serverCfgMu.RUnlock()

	cfg := serverConfigData{}
	home, err := os.UserHomeDir()
	if err == nil {
		path := filepath.Join(home, ".ollama", "server.json")
		data, err := os.ReadFile(path)
		if err != nil {
			if !errors.Is(err, os.ErrNotExist) {
				slog.Debug("envconfig: could not read server config", "error", err)
			}
		} else if err := json.Unmarshal(data, &cfg); err != nil {
			slog.Debug("envconfig: could not parse server config", "error", err)
		}
	}

	serverCfgMu.Lock()
	defer serverCfgMu.Unlock()
	if serverCfgLoaded {
		return
	}
	serverCfg = cfg
	serverCfgLoaded = true
}

func cachedServerConfig() serverConfigData {
	serverCfgMu.RLock()
	defer serverCfgMu.RUnlock()
	return serverCfg
}

// ElizaCloudAPIKey returns the API key for Eliza Cloud. The server sends it as X-API-Key on
// outbound proxied requests to non-ollama.com hosts because Eliza’s contract is API-key based
// (unlike legacy ollama.com Ed25519 signing). Empty means no key is sent; the server may log once
// when cloud features are enabled and the first such request is made.
func ElizaCloudAPIKey() string {
	return Var("ELIZACLOUD_API_KEY")
}

// ReloadServerConfig refreshes the cached ~/.ollama/server.json settings.
func ReloadServerConfig() {
	serverCfgMu.Lock()
	serverCfgLoaded = false
	serverCfg = serverConfigData{}
	serverCfgMu.Unlock()

	loadServerConfig()
}

// NoCloud returns true if Ollama cloud features are disabled,
// checking both the OLLAMA_NO_CLOUD environment variable and
// the disable_ollama_cloud field in ~/.ollama/server.json.
func NoCloud() bool {
	if NoCloudEnv() {
		return true
	}
	loadServerConfig()
	return cachedServerConfig().DisableOllamaCloud
}

// NoCloudSource returns the source of the cloud-disabled decision.
// Returns "none", "env", "config", or "both".
func NoCloudSource() string {
	envDisabled := NoCloudEnv()
	loadServerConfig()
	configDisabled := cachedServerConfig().DisableOllamaCloud

	switch {
	case envDisabled && configDisabled:
		return "both"
	case envDisabled:
		return "env"
	case configDisabled:
		return "config"
	default:
		return "none"
	}
}

// GemmaAgentLogPath is the JSONL path for MLX Gemma agent diagnostics (Hermes loops).
// Set ZEROLLAMA_GEMMA_AGENT_LOG=0 to disable. Default: ~/.ollama/logs/gemma-agent.jsonl
// The file is truncated on each serve start.
func GemmaAgentLogPath() string {
	if raw := strings.TrimSpace(Var("ZEROLLAMA_GEMMA_AGENT_LOG")); raw != "" {
		if raw == "0" || strings.EqualFold(raw, "off") || strings.EqualFold(raw, "false") {
			return ""
		}
		return raw
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(Models(), "..", "logs", "gemma-agent.jsonl")
	}
	return filepath.Join(home, ".ollama", "logs", "gemma-agent.jsonl")
}

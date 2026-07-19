package server

import (
	"regexp"
	"strings"
)

// mlxSessionClass is the scheduling/KV intent tier for an MLX request.
type mlxSessionClass int

const (
	mlxClassUnknown mlxSessionClass = iota
	mlxClassInteractive // live agent thread — holds runner, fast_path
	mlxClassAuxiliary   // subagents, compression, keyed side work
	mlxClassBackground  // batch generate, best-effort
)

func (c mlxSessionClass) String() string {
	switch c {
	case mlxClassInteractive:
		return "interactive"
	case mlxClassAuxiliary:
		return "auxiliary"
	case mlxClassBackground:
		return "background"
	default:
		return "unknown"
	}
}

// fulfillmentMode is a request-scoped guarantee above normal qos_class.
// WHY separate from qos_class: interactive already means "agent thread priority";
// fulfillment adds no-degradation / exclusive-GPU contracts for benches and critical work.
type fulfillmentMode int

const (
	fulfillmentNone fulfillmentMode = iota
	fulfillmentComplete  // finish without preemption / eviction; allow other idle models
	fulfillmentBenchmark // exclusive GPU speed path; unload peers; block all other traffic
)

func (m fulfillmentMode) String() string {
	switch m {
	case fulfillmentComplete:
		return "complete"
	case fulfillmentBenchmark:
		return "benchmark"
	default:
		return ""
	}
}

func (m fulfillmentMode) Exclusive() bool {
	return m == fulfillmentBenchmark
}

func (m fulfillmentMode) Active() bool {
	return m == fulfillmentComplete || m == fulfillmentBenchmark
}

// mlxQoS is the harness-facing scheduling contract (options.zerollama).
type mlxQoS struct {
	Class        mlxSessionClass
	Priority     int    // 0–100; used when Class unknown
	SessionGroup string // shared trie branch namespace (harness id)
	ParentKey    string // parent thread key (observability / future parent-aware defer)
	ProjectID    string // client harness id (zerollama ps / fleet)
	ProjectName  string // human label (repo, Discord bot, etc.)
	CacheScope   string // auto | thread | shared
	Fulfillment  fulfillmentMode
	Explicit     bool // client set qos_class or legacy mlx_session_class
}

const (
	qosCacheScopeAuto   = "auto"
	qosCacheScopeThread = "thread"
	qosCacheScopeShared = "shared"
)

// qosClassAliases map harness-friendly names to canonical classes.
var qosClassAliases = map[string]mlxSessionClass{
	"interactive": mlxClassInteractive,
	"primary":     mlxClassInteractive,
	"foreground":  mlxClassInteractive,
	"auxiliary":   mlxClassAuxiliary,
	"aux":         mlxClassAuxiliary,
	"background":  mlxClassBackground,
	"bg":          mlxClassBackground,
}

// fulfillmentAliases map client-facing names to canonical modes.
var fulfillmentAliases = map[string]fulfillmentMode{
	"complete":  fulfillmentComplete,
	"guarantee": fulfillmentComplete,
	"reliable":  fulfillmentComplete,
	"benchmark": fulfillmentBenchmark,
	"bench":     fulfillmentBenchmark,
	"speed":     fulfillmentBenchmark,
	"exclusive": fulfillmentBenchmark,
}

func parseFulfillmentName(name string) (fulfillmentMode, bool) {
	key := strings.ToLower(strings.TrimSpace(name))
	if key == "" || key == "none" || key == "off" {
		return fulfillmentNone, key != ""
	}
	if mode, ok := fulfillmentAliases[key]; ok {
		return mode, true
	}
	return fulfillmentNone, false
}

// ephemeralSessionKey matches timestamped one-off session ids (Hermes spawns, etc.).
var ephemeralSessionKey = regexp.MustCompile(`^[a-z][a-z0-9_-]*:\d{8}_`)

func parseQoSClassName(name string) (mlxSessionClass, bool) {
	key := strings.ToLower(strings.TrimSpace(name))
	if key == "" {
		return mlxClassUnknown, false
	}
	if cls, ok := qosClassAliases[key]; ok {
		return cls, true
	}
	return mlxClassUnknown, false
}

func qosClassFromPriority(priority int) mlxSessionClass {
	switch {
	case priority >= 70:
		return mlxClassInteractive
	case priority >= 30:
		return mlxClassAuxiliary
	default:
		return mlxClassBackground
	}
}

func stringFromMap(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	v, ok := m[key].(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(v)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func intFromMap(m map[string]any, key string) (int, bool) {
	if m == nil {
		return 0, false
	}
	v, ok := m[key]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	default:
		return 0, false
	}
}

// zerollamaBlockFromOptions returns the nested options.zerollama map.
func zerollamaBlockFromOptions(opts map[string]any) map[string]any {
	if opts == nil {
		return nil
	}
	if z, ok := opts["zerollama"].(map[string]any); ok && len(z) > 0 {
		return z
	}
	return nil
}

// mlxQoSFromOptions parses harness QoS from options.zerollama and legacy flat fields.
func mlxQoSFromOptions(opts map[string]any) mlxQoS {
	q := mlxQoS{CacheScope: qosCacheScopeAuto}
	if opts == nil {
		return q
	}
	z := zerollamaBlockFromOptions(opts)

	className := stringFromMap(z, "qos_class")
	if className == "" {
		className = stringFromMap(opts, "mlx_session_class")
	}
	if cls, ok := parseQoSClassName(className); ok {
		q.Class = cls
		q.Explicit = true
	}

	if p, ok := intFromMap(z, "qos_priority"); ok {
		q.Priority = clampQoSPriority(p)
		if !q.Explicit {
			q.Class = qosClassFromPriority(q.Priority)
		}
	}

	q.SessionGroup = stringFromMap(z, "session_group")
	if q.SessionGroup == "" {
		q.SessionGroup = stringFromMap(z, "harness")
	}

	q.ParentKey = stringFromMap(z, "session_parent")
	if q.ParentKey == "" {
		q.ParentKey = stringFromMap(opts, "mlx_session_parent")
	}

	q.ProjectID = firstNonEmpty(
		stringFromMap(z, "project_id"),
		stringFromMap(z, "client_id"),
		stringFromMap(z, "project"),
	)
	q.ProjectName = firstNonEmpty(
		stringFromMap(z, "project_name"),
		stringFromMap(z, "client_name"),
	)

	scope := strings.ToLower(stringFromMap(z, "cache_scope"))
	switch scope {
	case qosCacheScopeThread, qosCacheScopeShared, qosCacheScopeAuto:
		q.CacheScope = scope
	}

	fulfillName := firstNonEmpty(
		stringFromMap(z, "fulfillment"),
		stringFromMap(z, "fulfill_mode"),
		stringFromMap(z, "priority_mode"),
	)
	if mode, ok := parseFulfillmentName(fulfillName); ok && mode.Active() {
		q.Fulfillment = mode
		// Fulfillment always elevates to interactive so defer policy treats it as a holder.
		q.Class = mlxClassInteractive
		q.Explicit = true
	}
	return q
}

func clampQoSPriority(p int) int {
	if p < 0 {
		return 0
	}
	if p > 100 {
		return 100
	}
	return p
}

// mergeZerollamaIntoOptions lifts top-level/extra_body zerollama blocks into options.
func mergeZerollamaIntoOptions(opts map[string]any, sources ...map[string]any) map[string]any {
	if opts == nil {
		opts = map[string]any{}
	}
	merged := make(map[string]any)
	for _, src := range sources {
		if src == nil {
			continue
		}
		if z, ok := src["zerollama"].(map[string]any); ok && len(z) > 0 {
			for k, v := range z {
				merged[k] = v
			}
		}
	}
	if len(merged) == 0 {
		return opts
	}
	existing, _ := opts["zerollama"].(map[string]any)
	opts["zerollama"] = mergeOptionsMaps(existing, merged)
	return opts
}

func zerollamaVersionCapabilities() map[string]any {
	// Advertised on GET /api/version so clients probe once and send Tier 2 hints only
	// on zerollama nodes. runner_paths prevents sending MLX-specific options to nodes
	// that only run gguf_ggml or vanilla-compatible paths.
	//
	// Phase A/B capacity flags: progressive probe without SSH.
	// WHY stable_multi_model_swap=false: B0/B1 + soft-pin dampen thrash; they do not make
	// Python multi-resident. Hire maps that key on this flag must stay conservative.
	// WHY pin_reserve/propose_sidecar=true: advertise shipped surfaces; clients still honor serialize_required.
	return map[string]any{
		"mlx_qos":                true,
		"prompt_cache_key":       true,
		"mlx_live_kv":            true,
		"session_qos_gate":       true,
		"fulfillment":            true,
		"runner_paths":           zerollamaVersionRunnerPaths(),
		"runtime_config":         true,
		"can_load":               true,
		"metrics":                true,
		"admission_retry_after":  true,
		"error_timings":          true,
		"empty_gen_classify":       true,
		"priority_queues": map[string]any{
			"session_qos_class": []string{"interactive", "auxiliary", "background"},
			"runtime_priority":  []string{"high", "normal", "low"},
			"note":              "existing; Phase A does not add new queue code",
		},
		"residency_policy": map[string]any{
			"same_model_multi_copy": false,
			"max_loaded_models_env": "OLLAMA_MAX_LOADED_MODELS",
			"num_parallel_means":    "slots_inside_one_runner",
		},
		"stable_multi_model_swap": false, // B0/B1 dampen thrash only; Python still single-GGUF
		"pin_reserve":             true,
		"propose_sidecar":         true,
	}
}

func zerollamaVersionQoS() map[string]any {
	return map[string]any{
		"classes":      []string{"interactive", "auxiliary", "background"},
		"fulfillment":  []string{"complete", "benchmark"},
		"modalities":   []string{mlxModalityText, mlxModalityVision, mlxModalityVideoUnderstanding, mlxModalityImageGeneration, mlxModalityVideoGeneration},
		"options": map[string]any{
			"path": "options.zerollama",
			"fields": map[string]string{
				"qos_class":      "interactive | auxiliary | background (aliases: primary, aux, bg)",
				"qos_priority":   "0-100; class inferred when qos_class omitted (>=70 interactive)",
				"fulfillment":    "complete | benchmark — no-degradation / exclusive speed (aliases: guarantee, reliable / bench, speed, exclusive)",
				"session_group":  "harness id for shared cache branch (alias: harness)",
				"session_parent": "parent thread prompt_cache_key",
				"project_id":     "client harness id (alias: client_id, project)",
				"project_name":   "human label for zerollama ps (alias: client_name)",
				"cache_scope":    "auto | thread | shared",
			},
			"legacy": []string{"mlx_session_class", "mlx_session_parent", "fulfill_mode", "priority_mode"},
		},
		"openai": map[string]any{
			"extra_body": []string{"zerollama", "options", "prompt_cache_key"},
		},
		"routes": map[string]string{
			"text":                "/api/generate, /api/chat, /v1/chat/completions",
			"vision":              "/api/chat, /v1/chat/completions (images in messages)",
			"video_understanding": "/api/chat, /v1/chat/completions (videos in messages)",
			"image_generation":    "/api/generate (image models), /v1/images/generations, /v1/images/edits",
			"video_generation":    "POST /v1/videos",
		},
		"defaults": map[string]string{
			"image_generation":    "background",
			"video_generation":    "background",
			"vision":              "auxiliary (stream) / background (non-stream)",
			"video_understanding": "auxiliary (stream) / background (non-stream)",
		},
	}
}

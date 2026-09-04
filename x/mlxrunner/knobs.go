package mlxrunner

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// Knob is one MLX operator setting: current effective value, default, and
// when to touch it. Doctor prints these; decode logs a one-line hint when a
// gate fires.
type Knob struct {
	Env        string `json:"env"`
	Value      string `json:"value"`
	Default    string `json:"default"`
	Overridden bool   `json:"overridden"`
	Why        string `json:"why"`
	Tune       string `json:"tune"`
}

func envSet(key string) bool {
	return strings.TrimSpace(os.Getenv(key)) != ""
}

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

// KnobSnapshot lists MLX decode/prefill knobs and how to tune them.
func KnobSnapshot() []Knob {
	pldRaw := strings.TrimSpace(os.Getenv("ZEROLLAMA_MLX_PLD"))
	pldVal := "on"
	if !pldEnabled() {
		pldVal = "off"
	}
	mtpVal := envOr("ZEROLLAMA_MLX_MTP", "auto")
	histVal := envOr("ZEROLLAMA_MLX_MTP_HISTORY", "committed")

	return []Knob{
		{
			Env:        "ZEROLLAMA_MLX_PLD",
			Value:      pldVal,
			Default:    "on",
			Overridden: pldRaw != "",
			Why:        "n-gram speculative decode without a draft head",
			Tune:       "off only for AR benches vs mlx-serve; leave on for agents/code",
		},
		{
			Env:        "ZEROLLAMA_MLX_MTP",
			Value:      mtpVal,
			Default:    "auto",
			Overridden: envSet("ZEROLLAMA_MLX_MTP"),
			Why:        "draft-head load policy (not PLD)",
			Tune:       "require = fail load if no MTP/assistant; auto = warn and keep PLD/AR. MoE parks MTP per request unless enable_mtp=true",
		},
		{
			Env:        "ZEROLLAMA_MLX_MTP_HISTORY",
			Value:      histVal,
			Default:    "committed",
			Overridden: envSet("ZEROLLAMA_MLX_MTP_HISTORY"),
			Why:        "how much prompt the draft KV sees",
			Tune:       "auto or last_window on 32k+ prompts if draft RAM spikes; committed for max accept",
		},
		{
			Env:        "ZEROLLAMA_MLX_DRAFT_QUANT",
			Value:      map[bool]string{true: "on", false: "off"}[draftQuantEnabled()],
			Default:    "on",
			Overridden: envSet("ZEROLLAMA_MLX_DRAFT_QUANT"),
			Why:        "4-bit dense MTP/assistant weights at load (target still verifies)",
			Tune:       "off only if a companion's accept rate collapses after load",
		},
		{
			Env:        "ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW",
			Value:      fmt.Sprintf("%d", mtpHistoryLastWindowTokens()),
			Default:    fmt.Sprintf("%d", defaultMTPHistoryLastWindow),
			Overridden: envSet("ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW"),
			Why:        "tokens kept when history=last_window",
			Tune:       "raise if MTP accept falls on long chats; lower if UMA is tight",
		},
		{
			Env:        "OLLAMA_MLX_PREFILL_CHUNK",
			Value:      envOr("OLLAMA_MLX_PREFILL_CHUNK", fmt.Sprintf("%d (auto)", defaultPrefillChunk)),
			Default:    fmt.Sprintf("%d, or 2×sliding_window, or 1024 if RAM≥75%% working set", defaultPrefillChunk),
			Overridden: envSet("OLLAMA_MLX_PREFILL_CHUNK"),
			Why:        "prompt tokens per forward",
			Tune:       "set only to pin a bench; unset lets rotating-KV and working-set caps pick",
		},
		{
			Env:        "OLLAMA_MLX_PREFILL_SNAPSHOT_INTERVAL",
			Value:      envOr("OLLAMA_MLX_PREFILL_SNAPSHOT_INTERVAL", "auto (window or 32k)"),
			Default:    "auto",
			Overridden: envSet("OLLAMA_MLX_PREFILL_SNAPSHOT_INTERVAL"),
			Why:        "trie restore points during prefill",
			Tune:       "0 disables MTP interior snaps; smaller = better agent rewind, more RAM",
		},
		{
			Env:        "OLLAMA_MLX_PREFILL_CLEAR_CACHE_EVERY",
			Value:      envOr("OLLAMA_MLX_PREFILL_CLEAR_CACHE_EVERY", "4 (1 when RAM tight)"),
			Default:    "4",
			Overridden: envSet("OLLAMA_MLX_PREFILL_CLEAR_CACHE_EVERY"),
			Why:        "MLX allocator sweep cadence in prefill",
			Tune:       "1 if peak memory climbs; 0 to skip (fast but leaky)",
		},
		{
			Env:        "OLLAMA_MLX_PREFILL_MATERIALIZE_EVERY",
			Value:      envOr("OLLAMA_MLX_PREFILL_MATERIALIZE_EVERY", "4 (1 when RAM tight)"),
			Default:    "4",
			Overridden: envSet("OLLAMA_MLX_PREFILL_MATERIALIZE_EVERY"),
			Why:        "eval KV every N prefill chunks",
			Tune:       "1 on OOM during long prefill; higher on hot agent tails",
		},
		{
			Env:        "ZEROLLAMA_MLX_GREEDY_DRAFT_CHAIN",
			Value:      map[bool]string{true: "on", false: "off"}[envEnabledDefaultOn("ZEROLLAMA_MLX_GREEDY_DRAFT_CHAIN")],
			Default:    "on",
			Overridden: envSet("ZEROLLAMA_MLX_GREEDY_DRAFT_CHAIN"),
			Why:        "T=0: keep draft argmax on-device (no Categorical on a one-hot)",
			Tune:       "off to A/B vs sampled draft; fence at ZEROLLAMA_MLX_GREEDY_TRIO_MAX_CONTEXT",
		},
		{
			Env:        "ZEROLLAMA_MLX_BATCHED_GREEDY_ACCEPT",
			Value:      map[bool]string{true: "on", false: "off"}[envEnabledDefaultOn("ZEROLLAMA_MLX_BATCHED_GREEDY_ACCEPT")],
			Default:    "on",
			Overridden: envSet("ZEROLLAMA_MLX_BATCHED_GREEDY_ACCEPT"),
			Why:        "T=0: accept by token equality instead of Bernoulli p/q",
			Tune:       "off to A/B vs Leviathan path; same fence as draft chain",
		},
		{
			Env:        "ZEROLLAMA_MLX_GREEDY_TRIO_MAX_CONTEXT",
			Value:      fmt.Sprintf("%d", greedyTrioMaxContext()),
			Default:    fmt.Sprintf("%d", defaultGreedyTrioMaxContext),
			Overridden: envSet("ZEROLLAMA_MLX_GREEDY_TRIO_MAX_CONTEXT"),
			Why:        "prompt-token fence for the T=0 greedy stack (0 = no fence)",
			Tune:       "mtplx measured a small loss at 16k/32k; raise only after a local A/B",
		},
		{
			Env:        "ZEROLLAMA_MLX_SUPPRESS_RESERVED",
			Value:      map[bool]string{true: "on", false: "off"}[suppressReservedEnabled()],
			Default:    "on",
			Overridden: envSet("ZEROLLAMA_MLX_SUPPRESS_RESERVED"),
			Why:        "never sample FIM/reserved tokenizer ids (think/tool tags stay legal)",
			Tune:       "off only to match a bench that must emit FIM holes",
		},
	}
}

func FormatKnobs(knobs []Knob) string {
	parts := make([]string, 0, len(knobs))
	for _, k := range knobs {
		mark := ""
		if k.Overridden {
			mark = " [set]"
		}
		parts = append(parts, fmt.Sprintf("%s=%s%s — %s. Tune: %s", k.Env, k.Value, mark, k.Why, k.Tune))
	}
	return strings.Join(parts, " | ")
}

// RoundCostTuneReport reads persisted spec-width tables and says whether
// they look healthy.
func RoundCostTuneReport() (detail string, warn bool) {
	dir := roundCostDir()
	ents, err := os.ReadDir(dir)
	if err != nil {
		return "no mlx-round-cost tables yet (first MLX Load writes them next to OLLAMA_MODELS)", false
	}
	var notes []string
	for _, e := range ents {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") || strings.HasSuffix(e.Name(), ".last.json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		var f roundCostFile
		if json.Unmarshal(data, &f) != nil {
			continue
		}
		name := strings.TrimSuffix(e.Name(), ".json")
		note := fmt.Sprintf("%s scheduled=%d probe=%d cost_depths=%d", name, f.Scheduled, f.ProbeInterval, len(f.Cost))
		if len(f.ByCtx) > 0 {
			buckets := make([]string, 0, len(f.ByCtx))
			for k := range f.ByCtx {
				buckets = append(buckets, k)
			}
			slices.Sort(buckets)
			note += " ctx=[" + strings.Join(buckets, ",") + "]"
		} else if f.CtxBucket > 0 {
			note += fmt.Sprintf(" ctx=%d", f.CtxBucket)
		}
		if f.Scheduled == 0 && len(f.Cost) >= 2 {
			note += " (stuck at AR depth 0 — try an echo/code prompt, or unset ZEROLLAMA_MLX_PLD=off)"
			warn = true
		} else if f.Scheduled > 0 {
			note += " (next Load starts at this draft width)"
		}
		notes = append(notes, note)
	}
	if len(notes) == 0 {
		return "mlx-round-cost dir empty", false
	}
	return strings.Join(notes, " | "), warn
}

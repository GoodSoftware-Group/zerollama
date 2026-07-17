package envconfig

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// detectVisibleGPUCount counts NVIDIA GPUs via `nvidia-smi -L`.
// Overridable in tests. ok=false when the probe is unavailable.
var detectVisibleGPUCount = detectVisibleGPUCountNvidiaSMI

// ApplyHardwareLaneDefaults reads ``serve:`` and ``vram:`` from the runtime topology
// YAML and sets Go process env only when the operator has not already set the
// corresponding variable.
//
// Config selection mirrors runtime/autoconfig.py: explicit ZEROLLAMA_RUNTIME_CONFIG,
// else GPU-count pick (1 → single_gpu.yaml, ≥2 → dual_4090.yaml). Do not hardcode
// dual_4090 — that mis-lanes single-card hosts (e.g. RTX 5080).
func ApplyHardwareLaneDefaults() {
	path := resolvedRuntimeConfigPath()
	if path == "" {
		return
	}
	if serve := readYAMLBlock(path, "serve"); len(serve) > 0 {
		applyServeEnv(serve, filepath.Base(path))
	}
	if vram := readYAMLBlock(path, "vram"); len(vram) > 0 {
		applyVRAMEnv(vram, filepath.Base(path))
	}
}

func resolvedRuntimeConfigPath() string {
	if p := strings.TrimSpace(Var("ZEROLLAMA_RUNTIME_CONFIG")); p != "" {
		return p
	}
	if v := strings.TrimSpace(Var("ZEROLLAMA_AUTO_CONFIG")); v == "0" || strings.EqualFold(v, "false") {
		return ""
	}
	return resolveAutoRuntimeConfigPath()
}

func resolveAutoRuntimeConfigPath() string {
	configs := runtimeConfigsDir()
	if configs == "" {
		return ""
	}
	apple := filepath.Join(configs, "apple_silicon.yaml")
	single := filepath.Join(configs, "single_gpu.yaml")
	dual := filepath.Join(configs, "dual_4090.yaml")

	if runtime.GOOS == "darwin" && fileExists(apple) {
		slog.Info("hardware lane autoconfig pick",
			"pick", "apple_silicon",
			"config", filepath.Base(apple),
			"reason", "darwin",
		)
		return apple
	}

	n, ok := detectVisibleGPUCount()
	switch {
	case ok && n <= 1 && fileExists(single):
		slog.Info("hardware lane autoconfig pick",
			"pick", "single_gpu",
			"config", filepath.Base(single),
			"visible_gpu_count", n,
		)
		return single
	case ok && n >= 2 && fileExists(dual):
		slog.Info("hardware lane autoconfig pick",
			"pick", "dual_4090",
			"config", filepath.Base(dual),
			"visible_gpu_count", n,
		)
		return dual
	}

	// Probe failed: prefer single_gpu (safe on one card) over dual tensor-split topology.
	if fileExists(single) {
		slog.Warn("hardware lane autoconfig pick",
			"pick", "single_gpu",
			"config", filepath.Base(single),
			"reason", "nvidia-smi unavailable; defaulting to single_gpu (set ZEROLLAMA_RUNTIME_CONFIG to override)",
		)
		return single
	}
	if fileExists(dual) {
		slog.Warn("hardware lane autoconfig pick",
			"pick", "dual_4090",
			"config", filepath.Base(dual),
			"reason", "single_gpu.yaml missing; falling back to dual_4090",
		)
		return dual
	}
	if fileExists(apple) {
		return apple
	}
	return ""
}

func runtimeConfigsDir() string {
	repo := strings.TrimSpace(Var("ZEROLLAMA_REPO"))
	candidates := []string{}
	if repo != "" {
		candidates = append(candidates, filepath.Join(repo, "runtime/configs"))
	}
	candidates = append(candidates, "/opt/zerollama/runtime/configs")
	for _, d := range candidates {
		if st, err := os.Stat(d); err == nil && st.IsDir() {
			return d
		}
	}
	return ""
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func detectVisibleGPUCountNvidiaSMI() (n int, ok bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-smi", "-L")
	out, err := cmd.Output()
	if err != nil {
		return 0, false
	}
	count := 0
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "GPU ") {
			count++
		}
	}
	return count, true
}

func readYAMLBlock(path string, blockName string) map[string]any {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var root map[string]any
	if err := yaml.Unmarshal(data, &root); err != nil {
		slog.Debug("hardware lane: skip yaml block", "config", path, "block", blockName, "err", err)
		return nil
	}
	block, ok := root[blockName].(map[string]any)
	if !ok || len(block) == 0 {
		return nil
	}
	return block
}

func applyServeEnv(serve map[string]any, configName string) {
	type mapping struct {
		yamlKey string
		envKey  string
	}
	for _, m := range []mapping{
		{"max_loaded_models", "OLLAMA_MAX_LOADED_MODELS"},
		{"keep_alive", "OLLAMA_KEEP_ALIVE"},
		{"num_parallel", "OLLAMA_NUM_PARALLEL"},
		{"max_queue", "OLLAMA_MAX_QUEUE"},
		{"llama_fork", "ZEROLLAMA_LLAMA_FORK"},
		{"llama_fork_profile", "ZEROLLAMA_LLAMA_FORK_PROFILE"},
	} {
		if strings.TrimSpace(os.Getenv(m.envKey)) != "" {
			continue
		}
		raw, ok := serve[m.yamlKey]
		if !ok || raw == nil {
			continue
		}
		val := strings.TrimSpace(formatServeValue(raw))
		if val == "" {
			continue
		}
		_ = os.Setenv(m.envKey, val)
		slog.Info("hardware lane autoconfig",
			"config", configName,
			"env", m.envKey,
			"value", val,
		)
	}
}

func applyVRAMEnv(vram map[string]any, configName string) {
	type mapping struct {
		yamlKey string
		envKey  string
	}
	for _, m := range []mapping{
		{"clamp_num_ctx", "ZEROLLAMA_GGML_CLAMP_NUM_CTX"},
	} {
		if strings.TrimSpace(os.Getenv(m.envKey)) != "" {
			continue
		}
		raw, ok := vram[m.yamlKey]
		if !ok || raw == nil {
			continue
		}
		val := strings.TrimSpace(formatServeValue(raw))
		if val == "" {
			continue
		}
		_ = os.Setenv(m.envKey, val)
		slog.Info("hardware lane autoconfig",
			"config", configName,
			"env", m.envKey,
			"value", val,
		)
	}
}

func formatServeValue(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case int:
		return fmt.Sprintf("%d", t)
	case int64:
		return fmt.Sprintf("%d", t)
	case float64:
		return fmt.Sprintf("%g", t)
	default:
		return fmt.Sprint(v)
	}
}

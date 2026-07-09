package envconfig

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// ApplyHardwareLaneDefaults reads ``serve:`` and ``vram:`` from the runtime topology
// YAML (e.g. dual_4090.yaml) and sets Go process env only when the operator has not
// already set the corresponding variable.
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
	repo := strings.TrimSpace(Var("ZEROLLAMA_REPO"))
	candidates := []string{"/opt/zerollama/runtime/configs/dual_4090.yaml"}
	if repo != "" {
		candidates = append(candidates, filepath.Join(repo, "runtime/configs/dual_4090.yaml"))
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
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

package discover

import (
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// filterANEDraftEnv drops env entries whose key satisfies dropKey (e.g. stale WEIGHT_FILE2 from shell).
func filterANEDraftEnv(env []string, dropKey func(string) bool) []string {
	if dropKey == nil {
		return env
	}
	out := env[:0]
	for _, e := range env {
		k, _, _ := strings.Cut(e, "=")
		if dropKey(k) {
			continue
		}
		out = append(out, e)
	}
	return out
}

func applyConvManifestEnv(env *[]string, m ANEDraftWeightManifest, convDepth int) {
	paths := []func() string{
		m.ConvWeightPath, m.Conv2WeightPath, m.Conv3WeightPath, m.Conv4WeightPath,
		m.Conv5WeightPath, m.Conv6WeightPath, m.Conv7WeightPath, m.Conv8WeightPath,
	}
	limit := len(paths)
	if convDepth > 0 && convDepth < limit {
		limit = convDepth
	}
	for i := 0; i < limit; i++ {
		p := paths[i]()
		if p == "" {
			continue
		}
		key := "ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE"
		if i > 0 {
			key += strconv.Itoa(i + 1)
		}
		*env = append(*env, key+"="+p)
	}
}

func aneDraftEnvHasKey(env []string, key string) bool {
	prefix := key + "="
	for _, e := range env {
		if strings.HasPrefix(e, prefix) {
			return true
		}
	}
	return false
}

func aneDraftEnvValue(env []string, key string) (string, bool) {
	prefix := key + "="
	for i := len(env) - 1; i >= 0; i-- {
		if strings.HasPrefix(env[i], prefix) {
			return strings.TrimPrefix(env[i], prefix), true
		}
	}
	return "", false
}

// stripANEDraftEnv removes ZEROLLAMA_ANE_DRAFT_* entries so lab wiring from Go wins
// over stale shell exports (duplicate keys make getenv pick the wrong value).
// Keeps diagnostic toggles so agent/lab debug flags survive.
func stripANEDraftEnv(env []string) []string {
	return filterANEDraftEnv(env, func(k string) bool {
		switch k {
		case "ZEROLLAMA_ANE_DRAFT_BISECT_DEBUG",
			"ZEROLLAMA_ANE_DRAFT_METAL_ATTN_OUT",
			"ZEROLLAMA_ANE_DRAFT_METAL_CTX_KV",
			"ZEROLLAMA_ANE_DRAFT_METAL_NOISE_KV",
			"ZEROLLAMA_ANE_DRAFT_METAL_CAT_NOISE",
			"ZEROLLAMA_ANE_DRAFT_NOISE_OFF",
			"ZEROLLAMA_ANE_DRAFT_NOISE_V_ZERO",
			"ZEROLLAMA_ANE_DRAFT_WO_HOST_FP32",
			"ZEROLLAMA_ANE_HANDOFF_SKIP",
			"ZEROLLAMA_ANE_HANDOFF_STAGE",
			"ZEROLLAMA_ANE_B7_SKIP",
			"ZEROLLAMA_ANE_IOSURFACE_LIFECYCLE_LOG",
			"ZEROLLAMA_ANE_KEEP_AB_LOG",
			"ZEROLLAMA_ANE_LAB_PORT":
			return false
		}
		return strings.HasPrefix(k, "ZEROLLAMA_ANE_DRAFT")
	})
}

// dedupeEnvLastWins removes duplicate keys, keeping the last value (matches shell export order).
func dedupeEnvLastWins(env []string) []string {
	lastIdx := map[string]int{}
	for i, e := range env {
		k, _, ok := strings.Cut(e, "=")
		if ok {
			lastIdx[k] = i
		}
	}
	out := make([]string, 0, len(lastIdx))
	for i, e := range env {
		k, _, ok := strings.Cut(e, "=")
		if !ok {
			out = append(out, e)
			continue
		}
		if lastIdx[k] != i {
			continue
		}
		out = append(out, e)
	}
	return out
}

// upsertEnv replaces key=value or appends when missing (avoids duplicate getenv winners).
func upsertEnv(env []string, key, val string) []string {
	for i, e := range env {
		k, _, ok := strings.Cut(e, "=")
		if ok && k == key {
			env[i] = key + "=" + val
			return env
		}
	}
	return append(env, key+"="+val)
}

// prependLlamaServerLibPath puts llama-server's bin dir first on the platform library path.
// Why: libllama-common may reference libane_bridge.dylib without @loader_path; dyld needs bin/.
func prependLlamaServerLibPath(env []string, serverBin string) []string {
	binDir := filepath.Dir(serverBin)
	if abs, err := filepath.Abs(binDir); err == nil && abs != "" {
		binDir = abs
	}
	if binDir == "" || binDir == "." {
		return env
	}
	pathKey := "LD_LIBRARY_PATH"
	switch runtime.GOOS {
	case "darwin":
		pathKey = "DYLD_LIBRARY_PATH"
	case "windows":
		pathKey = "PATH"
	}
	sep := string(filepath.ListSeparator)
	for i, e := range env {
		k, v, ok := strings.Cut(e, "=")
		if ok && k == pathKey {
			if v == "" {
				env[i] = pathKey + "=" + binDir
			} else {
				env[i] = pathKey + "=" + binDir + sep + v
			}
			return env
		}
	}
	return append(env, pathKey+"="+binDir)
}

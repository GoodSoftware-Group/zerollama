package discover

import "strconv"
import "strings"

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
	for _, e := range env {
		if strings.HasPrefix(e, prefix) {
			return strings.TrimPrefix(e, prefix), true
		}
	}
	return "", false
}

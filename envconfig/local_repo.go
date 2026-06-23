package envconfig

import (
	"os"
	"path/filepath"
	"strings"
)

// defaultInferenceCheckout returns ~/Sites/inference/<parts...>.
// Why this default: Mac operator layout used by anemll/maderix scout repos and build scripts.
func defaultInferenceCheckout(parts ...string) string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	segs := append([]string{home, "Sites", "inference"}, parts...)
	return filepath.Join(segs...)
}

// localRepoPath resolves an env override or the default inference checkout path.
func localRepoPath(envKey string, defaultParts ...string) string {
	if p := strings.TrimSpace(Var(envKey)); p != "" {
		return filepath.Clean(p)
	}
	return defaultInferenceCheckout(defaultParts...)
}

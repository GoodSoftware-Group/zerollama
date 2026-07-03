package envconfig

import (
	"os"
	"path/filepath"
)

const systemModelsDir = "/usr/share/ollama/.ollama/models"

// ModelsSearchDirs returns candidate OLLAMA_MODELS roots for local manifest inspection.
//
// WHY multi-root: production installs often use /usr/share/ollama/.ollama/models (systemd
// service) while dev shells use ~/.ollama/models. Bench health and doctor must find manifests
// in both without requiring OLLAMA_MODELS in every invocation — especially when run as root
// for service parity checks.
//
// When OLLAMA_MODELS is set, only that directory is used.
func ModelsSearchDirs() []string {
	if s := Var("OLLAMA_MODELS"); s != "" {
		abs, err := filepath.Abs(s)
		if err != nil {
			abs = s
		}
		return []string{abs}
	}

	var dirs []string
	seen := map[string]struct{}{}
	add := func(p string) {
		if p == "" {
			return
		}
		abs, err := filepath.Abs(p)
		if err != nil {
			abs = p
		}
		if _, ok := seen[abs]; ok {
			return
		}
		seen[abs] = struct{}{}
		dirs = append(dirs, abs)
	}

	add(systemModelsDir)
	if home, err := os.UserHomeDir(); err == nil {
		add(filepath.Join(home, ".ollama", "models"))
	}
	return dirs
}

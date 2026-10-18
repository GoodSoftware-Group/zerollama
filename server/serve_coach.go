package server

import (
	"log/slog"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
)

// logServeCoach prints a short tier-1 DX hint after the HTTP listener is up.
// Never starts serve, never binds extra ports — slog only.
func logServeCoach() {
	n := localManifestCount()
	switch {
	case n == 0:
		slog.Info("tier1 coach: no local models yet",
			"hint_pull", "zerollama pull llama3.2:3b",
			"hint_remote", "ZEROLLAMA_MDNS=1 on storage host; same ZEROLLAMA_STORAGE_SECRET here",
			"hint_doctor", "zerollama doctor",
		)
	case n > 0:
		slog.Info("tier1 coach: models ready",
			"local_manifests", n,
			"hint_run", "zerollama run <name>",
			"hint_share", "ZEROLLAMA_MDNS=1 zerollama storage serve  # other machines browse with ZEROLLAMA_MDNS=1",
		)
	}

	if !libllamaLikelyPresent() {
		slog.Info("tier1 coach: runtime sidecar may be limited",
			"detail", "Metal libllama not found — ggml chat on :11434 still works",
			"hint", "MAC_SETUP_LLAMA_OPTIONAL=0 ./scripts/runtime/dev_bootstrap.sh",
		)
	}

	if envconfig.StorageServers() == "" && envconfig.StorageSecret() == "" && n > 0 {
		// Large local tree? Nudge sharing without requiring size scan every boot.
		if modelsDirBytesRough() >= 5<<30 {
			slog.Info("tier1 coach: large local model tree",
				"hint", "share on LAN: ZEROLLAMA_MDNS=1 ZEROLLAMA_STORAGE_SECRET=… zerollama storage serve",
			)
		}
	}
}

func localManifestCount() int {
	ms, err := manifest.Manifests(true)
	if err != nil || ms == nil {
		return 0
	}
	return len(ms)
}

func libllamaLikelyPresent() bool {
	if p := os.Getenv("LLAMA_CPP_LIB"); p != "" {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return true
		}
	}
	candidates := []string{
		filepath.Join("..", "llama.cpp", "build", "bin", "libllama.dylib"),
		filepath.Join("..", "llama.cpp", "build", "bin", "libllama.so"),
	}
	if root := os.Getenv("LLAMA_CPP_ROOT"); root != "" {
		candidates = append([]string{
			filepath.Join(root, "build", "bin", "libllama.dylib"),
			filepath.Join(root, "build", "bin", "libllama.so"),
		}, candidates...)
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return true
		}
	}
	return false
}

func modelsDirBytesRough() int64 {
	root := envconfig.Models()
	blobs := filepath.Join(root, "blobs")
	var total int64
	_ = filepath.Walk(blobs, func(_ string, info os.FileInfo, err error) error {
		if err != nil || info == nil || info.IsDir() {
			return nil
		}
		total += info.Size()
		if total >= 5<<30 {
			return filepath.SkipAll
		}
		return nil
	})
	return total
}

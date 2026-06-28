package discover

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

// GGMLIOSurfaceHookStatus reports in-tree ggml IOSurface map API availability.
type GGMLIOSurfaceHookStatus struct {
	Platform       string `json:"platform"`
	ANEDraftEnv    bool   `json:"ane_draft_env"`
	APIAvailable   bool   `json:"api_available"`
	CFunction      string `json:"c_function"`
	BackendFunction string `json:"backend_function"`
	Note           string `json:"note"`
}

// ProbeGGMLIOSurfaceHookStatus returns ggml hook readiness (darwin + env + lab binaries).
func ProbeGGMLIOSurfaceHookStatus() GGMLIOSurfaceHookStatus {
	st := GGMLIOSurfaceHookStatus{
		Platform:        runtime.GOOS,
		ANEDraftEnv:     envconfig.ANEDraftEnabled(),
		CFunction:       "ggml_metal_buffer_map_iosurface",
		BackendFunction: "ggml_backend_dev_buffer_from_iosurface",
		Note:            "same-process IOSurface only; pair with discover.ANEDraftRouter surface_id",
	}
	st.APIAvailable = runtime.GOOS == "darwin" &&
		ggmlIOSurfaceHookInTree() &&
		FindANEDraftDaemonBin() != "" &&
		FindANEGGMLMapSmokeBin() != ""
	return st
}

func ggmlIOSurfaceHookInTree() bool {
	if runtime.GOOS != "darwin" {
		return false
	}
	headerPaths := []string{
		"ml/backend/ggml/ggml/include/ggml-metal.h",
		filepath.Join("ml", "backend", "ggml", "ggml", "src", "ggml-metal", "ggml-metal-device.h"),
		filepath.Join("..", "ml", "backend", "ggml", "ggml", "include", "ggml-metal.h"),
		filepath.Join("..", "ml", "backend", "ggml", "ggml", "src", "ggml-metal", "ggml-metal-device.h"),
	}
	var hasBackend, hasMetalMap bool
	for _, rel := range headerPaths {
		b, err := os.ReadFile(rel)
		if err != nil {
			continue
		}
		s := string(b)
		if strings.Contains(s, "ggml_backend_dev_buffer_from_iosurface") {
			hasBackend = true
		}
		if strings.Contains(s, "ggml_metal_buffer_map_iosurface") {
			hasMetalMap = true
		}
	}
	return hasBackend && hasMetalMap
}

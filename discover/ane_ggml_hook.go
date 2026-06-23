package discover

import (
	"runtime"

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
	st.APIAvailable = runtime.GOOS == "darwin" && FindANEDraftDaemonBin() != "" && FindANEGGMLMapSmokeBin() != ""
	return st
}

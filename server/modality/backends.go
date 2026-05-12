// Package modality routes multimodal requests to built-in runners or optional subprocess backends.
//
// Video: raw containers are expanded to raster frames here (not in the OpenAI layer) so ffmpeg,
// caps, and policy stay in one place, and this package can depend on [types/model.ConfigV2] without
// importing server.Model (avoids cycles and keeps tests small). See docs/video-understanding.md
// for why merges, sampling modes, and preflight work the way they do.
package modality

import (
	"github.com/ollama/ollama/types/model"
)

// BackendFor returns the configured backend name for a modality key ([model.ModalityImage], etc.).
// An empty string means use the default built-in implementation.
func BackendFor(cfg model.ConfigV2, key string) string {
	if cfg.ModalityBackends == nil {
		return ""
	}
	return cfg.ModalityBackends[key]
}

// PathFor returns a filesystem path from [model.ConfigV2.BackendPaths].
func PathFor(cfg model.ConfigV2, key string) string {
	if cfg.BackendPaths == nil {
		return ""
	}
	return cfg.BackendPaths[key]
}

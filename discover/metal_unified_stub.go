//go:build !darwin || !arm64

package discover

import "github.com/ollama/ollama/ml"

func applyMetalUnifiedFreeMemory(devices []ml.DeviceInfo, updated []bool) {}

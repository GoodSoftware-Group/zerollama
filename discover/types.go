package discover

import (
	"log/slog"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/version"
)

type memInfo struct {
	TotalMemory uint64 `json:"total_memory,omitempty"`
	FreeMemory  uint64 `json:"free_memory,omitempty"`
	FreeSwap    uint64 `json:"free_swap,omitempty"` // TODO split this out for system only
}

// CPU type represents a CPU Package occupying a socket
type CPU struct {
	ID                  string `cpuinfo:"processor"`
	VendorID            string `cpuinfo:"vendor_id"`
	ModelName           string `cpuinfo:"model name"`
	CoreCount           int
	EfficiencyCoreCount int // Performance = CoreCount - Efficiency
	ThreadCount         int
}

// LogStartupBanner prints version + wall-clock start time before any other serve noise.
// WHY early: operators need to know which binary/time they launched when grepping long DEBUG logs.
func LogStartupBanner() {
	slog.Info("zerollama starting",
		"version", version.Version,
		"started_at", time.Now().Format(time.RFC3339),
		"goos", runtime.GOOS,
		"goarch", runtime.GOARCH,
	)
}

// LogStartupHardware prints a single clear GPU/CUDA (or CPU-only) summary after discovery.
// Complements LogDetails (per-device rows) so serve boot answers: GPU? which CUDA? which llm library?
func LogStartupHardware(devices []ml.DeviceInfo) {
	requested := strings.TrimSpace(envconfig.LLMLibrary())
	var gpus []ml.DeviceInfo
	for _, d := range devices {
		lib := strings.ToLower(d.Library)
		if lib == "" || lib == "cpu" {
			continue
		}
		gpus = append(gpus, d)
	}

	if len(gpus) == 0 {
		slog.Warn("startup hardware: no GPU found — inference will use CPU",
			"gpu_found", false,
			"ollama_llm_library", requested,
			"hint", "check nvidia-smi, /root/nvidia-host in LD_LIBRARY_PATH, and OLLAMA_LLM_LIBRARY (cuda_v12/cuda_v13)",
		)
		return
	}

	// Primary device = highest free memory (same preference as LogDetails).
	sort.Sort(sort.Reverse(ml.ByFreeMemory(gpus)))
	dev := gpus[0]
	var libs []string
	for _, dir := range dev.LibraryPath {
		if strings.Contains(dir, filepath.Join("lib", "ollama")) {
			libs = append(libs, filepath.Base(dir))
		}
	}
	libdirs := strings.Join(libs, ",")
	if libdirs == "" && len(dev.LibraryPath) > 0 {
		libdirs = strings.Join(dev.LibraryPath, ",")
	}

	slog.Info("startup hardware: GPU ready",
		"gpu_found", true,
		"gpu_count", len(gpus),
		"device", firstNonEmpty(dev.Description, dev.Name),
		"library", dev.Library,
		"compute", dev.Compute(),
		"driver", dev.Driver(),
		"vram_total", format.HumanBytes2(dev.TotalMemory),
		"vram_available", format.HumanBytes2(dev.FreeMemory),
		"ollama_llm_library", requested,
		"libdirs", libdirs,
		"pci_id", dev.PCIID,
	)
	if requested != "" && libdirs != "" && !strings.Contains(libdirs, requested) {
		slog.Warn("startup hardware: OLLAMA_LLM_LIBRARY does not match discovered libdirs",
			"ollama_llm_library", requested,
			"libdirs", libdirs,
		)
	}
}

func LogDetails(devices []ml.DeviceInfo) {
	sort.Sort(sort.Reverse(ml.ByFreeMemory(devices))) // Report devices in order of scheduling preference
	for _, dev := range devices {
		var libs []string
		for _, dir := range dev.LibraryPath {
			if strings.Contains(dir, filepath.Join("lib", "ollama")) {
				libs = append(libs, filepath.Base(dir))
			}
		}
		typeStr := "discrete"
		if dev.Integrated {
			typeStr = "iGPU"
		}
		slog.Info("inference compute",
			"id", dev.ID,
			"filter_id", dev.FilterID,
			"library", dev.Library,
			"compute", dev.Compute(),
			"name", dev.Name,
			"description", dev.Description,
			"libdirs", strings.Join(libs, ","),
			"driver", dev.Driver(),
			"pci_id", dev.PCIID,
			"type", typeStr,
			"total", format.HumanBytes2(dev.TotalMemory),
			"available", format.HumanBytes2(dev.FreeMemory),
		)
	}
	// CPU inference fallback when bootstrap discovery found no GPUs.
	// Why log CPU here: discover uses GPU list for scheduling; empty list means CPU-only
	// layout. On Mac, if you expected Metal, rebuild — Jun 2026 fixed bootstrap /info
	// disabling Metal via zero-layer dummy load (see docs/apple-silicon-metal.md).
	if len(devices) == 0 {
		dev, _ := GetCPUMem()
		slog.Info("inference compute",
			"id", "cpu",
			"library", "cpu",
			"compute", "",
			"name", "cpu",
			"description", "cpu",
			"libdirs", "ollama",
			"driver", "",
			"pci_id", "",
			"type", "",
			"total", format.HumanBytes2(dev.TotalMemory),
			"available", format.HumanBytes2(dev.FreeMemory),
		)
	}
}

package discover

import (
	"context"
	"log/slog"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// defaultIntegratedROCmGFXTargets lists integrated AMD GPUs upstream allowlists
// by default (without OLLAMA_IGPU_ENABLE=1). Strix Halo 8060S needs gfx1151 or
// scheduling drops the only usable iGPU on Ryzen AI Max+ 395 boxes.
var defaultIntegratedROCmGFXTargets = map[string]struct{}{
	"gfx1151": {},
}

func filterIntegratedGPUs(devices []ml.DeviceInfo) []ml.DeviceInfo {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		return devices
	}

	allow, explicit := integratedGPUAdmission()
	filtered := devices[:0]
	for _, device := range devices {
		if !device.Integrated {
			filtered = append(filtered, device)
			continue
		}

		if explicit {
			if allow {
				filtered = append(filtered, device)
			}
			continue
		}
		if integratedGPUAllowedByDefault(device) {
			filtered = append(filtered, device)
			continue
		}

		slog.Info("dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1",
			"id", device.ID,
			"library", device.Library,
			"compute", device.Compute(),
			"name", device.Name,
			"description", device.Description,
			"pci_id", device.PCIID)
	}

	return filtered
}

func integratedGPUAdmission() (allow, explicit bool) {
	enabledWithTrueDefault := envconfig.EnableIntegratedGPU(true)
	enabledWithFalseDefault := envconfig.EnableIntegratedGPU(false)
	if enabledWithTrueDefault == enabledWithFalseDefault {
		return enabledWithTrueDefault, true
	}
	return false, false
}

func integratedGPUAllowedByDefault(device ml.DeviceInfo) bool {
	switch device.Library {
	case "CUDA":
		return true
	case "ROCm":
		_, ok := defaultIntegratedROCmGFXTargets[device.GFXTarget]
		return ok
	default:
		return false
	}
}

func bootstrapDevicesWithMetalRetry(firstAttemptCtx, retryParentCtx context.Context, timeout time.Duration, ollamaLibDirs []string, extraEnvs map[string]string) []ml.DeviceInfo {
	extraEnvs = normalizeDiscoveryEnv(ollamaLibDirs, extraEnvs)

	if llm.LlamaServerDiscoverable() && (envconfig.LlamaServerBackend() || runtime.GOOS == "linux") {
		devices, status, err := runBootstrapDevicesWithStatusWatchdog(firstAttemptCtx, ollamaLibDirs, extraEnvs, llamaServerBootstrapDevicesWithStatus)
		if err == nil && len(devices) > 0 {
			recordPersistentRunnerEnv(devices, extraEnvs)
			return devices
		}

		if llm.ShouldRetryWithMetalTensorDisabled(err, status) && (extraEnvs == nil || extraEnvs["GGML_METAL_TENSOR_DISABLE"] != "1") {
			retryEnvs := map[string]string{}
			for k, v := range extraEnvs {
				retryEnvs[k] = v
			}
			retryEnvs["GGML_METAL_TENSOR_DISABLE"] = "1"
			slog.Warn("retrying llama-server GPU discovery with Metal tensor API disabled", "error", err, "detail", lastDiscoveryStatusError(status))

			retryCtx, cancel := context.WithTimeout(retryParentCtx, timeout)
			devices, status, err = runBootstrapDevicesWithStatusWatchdog(retryCtx, ollamaLibDirs, retryEnvs, llamaServerBootstrapDevicesWithStatus)
			cancel()
			if err == nil && len(devices) > 0 {
				recordPersistentRunnerEnv(devices, retryEnvs)
				return devices
			}
		}

	if err != nil {
		slog.Debug("llama-server discovery unavailable, falling back to ggml runner", "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "error", err, "detail", lastDiscoveryStatusError(status))
		}
	}

	if !envconfig.GgmlRunnerLinked() {
		// WHY: edge builds stub StartRunner; falling back would spawn a subprocess that always fails
		// and hides the real fix (build/install llama-server, set ZEROLLAMA_LLAMA_SERVER).
		slog.Debug("ggml runner unlinked; skipping ggml discovery bootstrap")
		return nil
	}

	return bootstrapGgmlDevices(firstAttemptCtx, ollamaLibDirs, extraEnvs)
}

type bootstrapDevicesResult struct {
	devices []ml.DeviceInfo
	status  *llm.StatusWriter
	err     error
}

func runBootstrapDevicesWithStatusWatchdog(
	ctx context.Context,
	ollamaLibDirs []string,
	extraEnvs map[string]string,
	discover func(context.Context, []string, map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error),
) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
	resultCh := make(chan bootstrapDevicesResult, 1)
	go func() {
		devices, status, err := discover(ctx, ollamaLibDirs, extraEnvs)
		resultCh <- bootstrapDevicesResult{devices: devices, status: status, err: err}
	}()

	select {
	case result := <-resultCh:
		return result.devices, result.status, result.err
	case <-ctx.Done():
		slog.Warn("llama-server GPU discovery watchdog timed out", "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "extra_envs", extraEnvs, "error", ctx.Err())
		return nil, nil, ctx.Err()
	}
}

func normalizeDiscoveryEnv(ollamaLibDirs []string, extraEnvs map[string]string) map[string]string {
	return normalizeDiscoveryEnvForGOOS(runtime.GOOS, ollamaLibDirs, extraEnvs)
}

func normalizeDiscoveryEnvForGOOS(goos string, ollamaLibDirs []string, extraEnvs map[string]string) map[string]string {
	if goos != "linux" || len(ollamaLibDirs) == 0 || !isROCmLibraryDir(filepath.Base(ollamaLibDirs[len(ollamaLibDirs)-1])) {
		return extraEnvs
	}

	if extraEnvs["ROCR_VISIBLE_DEVICES"] != "" || envconfig.RocrVisibleDevices() != "" {
		return extraEnvs
	}

	source, tokens := rocmNumericVisibleDeviceSource(extraEnvs)
	if len(tokens) == 0 {
		return extraEnvs
	}

	env := make(map[string]string, len(extraEnvs)+1)
	for k, v := range extraEnvs {
		env[k] = v
	}
	env["ROCR_VISIBLE_DEVICES"] = strings.Join(tokens, ",")
	env[source] = visibleDeviceOrdinals(len(tokens))
	slog.Debug("normalizing AMD visible devices for ROCm discovery", "from_env", source, "ROCR_VISIBLE_DEVICES", env["ROCR_VISIBLE_DEVICES"], "visible_ordinals", env[source])
	return env
}

func isROCmLibraryDir(name string) bool {
	return strings.HasPrefix(name, "rocm")
}

func remapFilterIDForUserVisibleDevices(device *ml.DeviceInfo) {
	tokens := visibleDeviceFilterTokens(runtime.GOOS, device.Library)
	if len(tokens) == 0 {
		return
	}

	id := device.FilterID
	if id == "" {
		id = device.ID
	}
	index, err := strconv.Atoi(id)
	if err != nil || index < 0 || index >= len(tokens) {
		return
	}

	device.FilterID = tokens[index]
}

func visibleDeviceFilterTokens(goos, library string) []string {
	switch library {
	case "CUDA":
		return splitVisibleDeviceList(envconfig.CudaVisibleDevices())
	case "ROCm":
		if goos == "linux" {
			if tokens := splitVisibleDeviceList(envconfig.RocrVisibleDevices()); len(tokens) > 0 {
				return tokens
			}
			if _, tokens := rocmNumericVisibleDeviceSource(nil); len(tokens) > 0 {
				return tokens
			}
			return nil
		}
		for _, value := range []string{envconfig.HipVisibleDevices(), envconfig.GpuDeviceOrdinal(), envconfig.CudaVisibleDevices()} {
			if tokens := splitNumericVisibleDeviceList(value); len(tokens) > 0 {
				return tokens
			}
		}
	case "Vulkan":
		return splitVisibleDeviceList(envconfig.VkVisibleDevices())
	}

	return nil
}

func rocmNumericVisibleDeviceSource(extraEnvs map[string]string) (string, []string) {
	for _, name := range []string{"HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "CUDA_VISIBLE_DEVICES"} {
		value := extraEnvs[name]
		if value == "" {
			switch name {
			case "HIP_VISIBLE_DEVICES":
				value = envconfig.HipVisibleDevices()
			case "GPU_DEVICE_ORDINAL":
				value = envconfig.GpuDeviceOrdinal()
			case "CUDA_VISIBLE_DEVICES":
				value = envconfig.CudaVisibleDevices()
			}
		}
		if tokens := splitNumericVisibleDeviceList(value); len(tokens) > 0 {
			return name, tokens
		}
	}
	return "", nil
}

func splitVisibleDeviceList(value string) []string {
	fields := strings.Split(value, ",")
	tokens := make([]string, 0, len(fields))
	for _, field := range fields {
		field = strings.TrimSpace(field)
		if field != "" {
			tokens = append(tokens, field)
		}
	}
	return tokens
}

func splitNumericVisibleDeviceList(value string) []string {
	tokens := splitVisibleDeviceList(value)
	if len(tokens) == 0 {
		return nil
	}
	for _, token := range tokens {
		index, err := strconv.Atoi(token)
		if err != nil || index < 0 {
			return nil
		}
	}
	return tokens
}

func visibleDeviceOrdinals(count int) string {
	ordinals := make([]string, count)
	for i := range ordinals {
		ordinals[i] = strconv.Itoa(i)
	}
	return strings.Join(ordinals, ",")
}

func lastDiscoveryStatusError(status *llm.StatusWriter) string {
	if status == nil {
		return ""
	}
	return status.LastError()
}

func recordPersistentRunnerEnv(devices []ml.DeviceInfo, extraEnvs map[string]string) {
	if extraEnvs["GGML_METAL_TENSOR_DISABLE"] != "1" {
		return
	}
	for i := range devices {
		if devices[i].Library != "Metal" {
			continue
		}
		if devices[i].RunnerEnvOverrides == nil {
			devices[i].RunnerEnvOverrides = map[string]string{}
		}
		devices[i].RunnerEnvOverrides["GGML_METAL_TENSOR_DISABLE"] = "1"
	}
}

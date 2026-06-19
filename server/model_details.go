package server

import (
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
	xserver "github.com/ollama/ollama/x/server"
)

func enrichModelDetailsFromPath(details *api.ModelDetails, modelPath string) {
	if modelPath == "" {
		return
	}
	defer func() {
		if r := recover(); r != nil {
			slog.Warn("model details enrichment skipped after panic", "path", modelPath, "panic", r)
		}
	}()

	data, err := llm.LoadModel(modelPath, 0)
	if err != nil {
		return
	}

	enrichModelDetailsFromGGML(details, data.KV(), data.Tensors())
}

func enrichModelDetailsFromGGML(details *api.ModelDetails, kv ggml.KV, tensors ggml.Tensors) {
	if details == nil {
		return
	}

	if architecture := kv.Architecture(); architecture != "" && architecture != "unknown" && details.Family == "" {
		details.Family = architecture
	}

	if total := kv.ParameterCount(); total > 0 {
		details.ParameterCount = total
		if details.ParameterSize == "" {
			details.ParameterSize = format.HumanNumber(total)
		}
	}

	expertCount := kv.Uint("expert_count")
	expertUsedCount := kv.Uint("expert_used_count")
	if expertCount == 0 {
		if details.ParameterCount > 0 {
			details.ArchitectureType = "dense"
		}
		return
	}

	details.ArchitectureType = "moe"
	details.ExpertCount = expertCount
	details.ExpertUsedCount = expertUsedCount

	if details.ParameterCount == 0 || expertUsedCount == 0 || expertUsedCount > expertCount {
		return
	}

	expertParams := routedExpertParameters(tensors)
	activeParams := activeParameterCount(details.ParameterCount, expertParams, expertCount, expertUsedCount)
	if activeParams == 0 {
		return
	}

	details.ActiveParameterCount = activeParams
}

func enrichModelDetailsFromSafetensors(details *api.ModelDetails, name model.Name) {
	if details == nil {
		return
	}

	info, err := xserver.GetSafetensorsLLMInfo(name)
	if err != nil {
		return
	}

	architecture, _ := info["general.architecture"].(string)
	if architecture != "" && architecture != "unknown" && details.Family == "" {
		details.Family = architecture
	}

	if total := modelInfoUint64(info["general.parameter_count"]); total > 0 {
		details.ParameterCount = total
		if details.ParameterSize == "" {
			details.ParameterSize = format.HumanNumber(total)
		}
	}

	expertCount := modelInfoUint32(info[architecture+".expert_count"])
	expertUsedCount := modelInfoUint32(info[architecture+".expert_used_count"])
	if expertCount == 0 {
		if details.ParameterCount > 0 {
			details.ArchitectureType = "dense"
		}
		return
	}

	details.ArchitectureType = "moe"
	details.ExpertCount = expertCount
	details.ExpertUsedCount = expertUsedCount

	if details.ParameterCount == 0 || expertUsedCount == 0 || expertUsedCount > expertCount {
		return
	}

	tensors, err := xserver.GetSafetensorsTensorInfo(name)
	if err != nil {
		return
	}

	expertParams := routedExpertAPITensors(tensors)
	activeParams := activeParameterCount(details.ParameterCount, expertParams, expertCount, expertUsedCount)
	if activeParams == 0 {
		return
	}

	details.ActiveParameterCount = activeParams
}

func routedExpertParameters(tensors ggml.Tensors) uint64 {
	return routedExpertParameterItems(tensors.Items())
}

func routedExpertParameterItems(tensors []*ggml.Tensor) uint64 {
	var n uint64
	for _, t := range tensors {
		name := t.Name
		if strings.Contains(name, "_exps.") || strings.Contains(name, "_exps_") || strings.Contains(name, ".experts.") {
			n += t.Elements()
		}
	}
	return n
}

func routedExpertAPITensors(tensors []api.Tensor) uint64 {
	var n uint64
	for _, t := range tensors {
		name := t.Name
		if !strings.Contains(name, "_exps.") && !strings.Contains(name, "_exps_") && !strings.Contains(name, ".experts.") {
			continue
		}

		elements := uint64(1)
		for _, dim := range t.Shape {
			if dim == 0 {
				elements = 0
				break
			}
			elements *= dim
		}
		n += elements
	}
	return n
}

func modelInfoUint64(value any) uint64 {
	switch v := value.(type) {
	case uint64:
		return v
	case uint32:
		return uint64(v)
	case uint:
		return uint64(v)
	case int64:
		if v > 0 {
			return uint64(v)
		}
	case int:
		if v > 0 {
			return uint64(v)
		}
	case float64:
		if v > 0 {
			return uint64(v)
		}
	}
	return 0
}

func modelInfoUint32(value any) uint32 {
	v := modelInfoUint64(value)
	if v > 0 && v <= uint64(^uint32(0)) {
		return uint32(v)
	}
	return 0
}

func activeParameterCount(total, expertParams uint64, expertCount, expertUsedCount uint32) uint64 {
	if total == 0 || expertParams == 0 || expertParams > total || expertCount == 0 || expertUsedCount == 0 || expertUsedCount > expertCount {
		return 0
	}

	activeExpertParams := expertParams * uint64(expertUsedCount) / uint64(expertCount)
	return total - expertParams + activeExpertParams
}

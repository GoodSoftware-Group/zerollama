package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// ANEDraftMILExtractResult reports GGUF → MIL BLOBFILE weight extract for the lab conv proxy.
type ANEDraftMILExtractResult struct {
	OK                  bool     `json:"ok"`
	Mode                string   `json:"mode"`
	Tag                 string   `json:"tag,omitempty"`
	DraftGGUF           string   `json:"draft_gguf,omitempty"`
	DraftSidecarPresent bool     `json:"draft_sidecar_present"`
	SourceTensor        string   `json:"source_tensor"`
	SourceShape         []uint64 `json:"source_shape,omitempty"`
	SourceKind          string   `json:"source_kind,omitempty"`
	ProxyChannels       int      `json:"proxy_channels"`
	ProxySpatial        int      `json:"proxy_spatial"`
	FP16SquareBytes     int      `json:"fp16_square_bytes"`
	WeightBlobBytes     int      `json:"weight_blob_bytes"`
	OutputPath          string   `json:"output_path,omitempty"`
	WeightSource        string   `json:"weight_source"`
	Blockers            []string `json:"blockers,omitempty"`
	NextStep            string   `json:"next_step,omitempty"`
	Note                string   `json:"note,omitempty"`
}

// ExtractProxyConvWeightBlob reads a sidecar GGUF tensor and packs the ANE MIL weight blob.
func ExtractProxyConvWeightBlob(ggufPath, tensorName string, channels int) ([]byte, *ggml.Tensor, error) {
	if channels <= 0 {
		return nil, nil, fmt.Errorf("channels must be positive")
	}
	if tensorName == "" {
		tensorName = DefaultProxyConvTensor()
	}

	raw, tensor, err := ggml.ReadTensorBytes(ggufPath, tensorName)
	if err != nil {
		return nil, nil, err
	}

	fp16, err := ExtractTopLeftSquareFP16(raw, tensor, channels)
	if err != nil {
		return nil, tensor, err
	}

	blob, err := PackANEMILWeightBlob(fp16)
	if err != nil {
		return nil, tensor, err
	}
	return blob, tensor, nil
}

// ProbeANEDraftMILExtract builds the proxy conv weight blob from an Eagle3 sidecar GGUF.
func ProbeANEDraftMILExtract(_ context.Context, preferred, tensorName, outputPath string) (ANEDraftMILExtractResult, error) {
	out := ANEDraftMILExtractResult{
		Mode:         "draft_mil_extract",
		SourceTensor: strings.TrimSpace(tensorName),
		WeightSource: "sidecar",
		Note:         "top-left channels×channels slice → maderix BLOBFILE layout for ane-draft-daemon --weight-file",
	}
	if out.SourceTensor == "" {
		out.SourceTensor = DefaultProxyConvTensor()
	}

	if runtime.GOOS != "darwin" {
		out.Blockers = append(out.Blockers, "darwin only")
		return out, nil
	}

	entries, err := ListANEDraftInventory()
	if err != nil {
		return out, err
	}
	entry, ok := SelectANEDraftModel(entries, preferred)
	if !ok {
		out.Blockers = append(out.Blockers, "no ANE draft target in local inventory")
		return out, fmt.Errorf("no ANE draft target in local inventory")
	}

	out.Tag = entry.Tag
	out.ProxyChannels = entry.ProxyChannels
	out.ProxySpatial = entry.ProxySpatial
	out.WeightBlobBytes = draftMILWeightBlobBytes(entry.ProxyChannels)
	out.FP16SquareBytes = entry.ProxyChannels * entry.ProxyChannels * 2

	draftPath, present := resolveDraftGGUFPath(entry)
	out.DraftGGUF = draftPath
	out.DraftSidecarPresent = present
	if !present || draftPath == "" {
		out.Blockers = append(out.Blockers, "eagle3 drafter GGUF missing")
		out.NextStep = "download drafter (scripts/setup_mtp_models.sh) then re-run extract"
		return out, fmt.Errorf("eagle3 drafter GGUF missing")
	}

	blob, tensor, err := ExtractProxyConvWeightBlob(draftPath, out.SourceTensor, entry.ProxyChannels)
	if err != nil {
		out.Blockers = append(out.Blockers, err.Error())
		out.NextStep = "zerollama ane-draft-mil-map --model " + strings.SplitN(entry.Tag, ":", 2)[0]
		return out, err
	}
	if tensor != nil {
		out.SourceShape = append([]uint64(nil), tensor.Shape...)
		out.SourceKind = ggml.TensorType(tensor.Kind).String()
	}
	if len(blob) != out.WeightBlobBytes {
		out.Blockers = append(out.Blockers, fmt.Sprintf("blob size %d != expected %d", len(blob), out.WeightBlobBytes))
		return out, fmt.Errorf("unexpected blob size")
	}

	if outputPath = strings.TrimSpace(outputPath); outputPath != "" {
		if err := os.WriteFile(outputPath, blob, 0o644); err != nil {
			out.Blockers = append(out.Blockers, "write output: "+err.Error())
			return out, err
		}
		out.OutputPath = outputPath
	}

	if FindANEDraftDaemonBin() == "" {
		out.Blockers = append(out.Blockers, "ane-draft-daemon not built")
	} else if out.OutputPath != "" {
		out.NextStep = fmt.Sprintf("ane-draft-daemon --channels %d --spatial %d --weight-file %s", entry.ProxyChannels, entry.ProxySpatial, out.OutputPath)
	} else {
		out.NextStep = "re-run with --out to write blob, then pass --weight-file to ane-draft-daemon"
	}

	if len(out.Blockers) == 0 {
		out.OK = true
	}
	return out, nil
}

// RunANEDraftMILExtractJSON writes draft MIL extract JSON to w.
func RunANEDraftMILExtractJSON(ctx context.Context, w io.Writer, preferred, tensorName, outputPath string) error {
	res, err := ProbeANEDraftMILExtract(ctx, preferred, tensorName, outputPath)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err != nil {
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}

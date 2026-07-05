package discover

// B7: tied-embedding argmax head for ANE-driven draft tokens (lab).
// Why token_embd not output.weight: dflash sidecars use tied embeddings; logits = h @ embed.
import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	mlggml "github.com/ollama/ollama/ml/backend/ggml"
)

const driveEmbdMagic = "ZANE1"

// ANEDraftDriveHeadManifest describes mmap-friendly tied-embed + output_norm caches for B7 drive.
type ANEDraftDriveHeadManifest struct {
	TokenEmbdPath string `json:"token_embd_path"`
	OutputNormPath string `json:"output_norm_path,omitempty"`
	NEmbd         int    `json:"n_embd"`
	NVocab        int    `json:"n_vocab"`
	EmbdBytes     int    `json:"embd_bytes"`
	NormBytes     int    `json:"norm_bytes,omitempty"`
	TensorEmbd    string `json:"tensor_embd"`
	TensorNorm    string `json:"tensor_norm,omitempty"`
}

func defaultDriveEmbdTensor() string  { return "token_embd.weight" }
func defaultDriveNormTensor() string { return "output_norm.weight" }

func driveEmbdTensorCandidates() []string {
	return []string{"token_embd.weight", "output.weight"}
}

func resolveDriveEmbdGGUF(entry ANEDraftEntry, tensorName string) (path string, ok bool) {
	if draftPath, present := resolveDraftGGUFPath(entry); present && draftPath != "" {
		if _, _, err := ggml.ReadTensorBytes(draftPath, tensorName); err == nil {
			return draftPath, true
		}
	}
	if base := strings.TrimSpace(entry.BaseGGUF); base != "" {
		if _, _, err := ggml.ReadTensorBytes(base, tensorName); err == nil {
			return base, true
		}
	}
	return "", false
}

func resolveDriveEmbdSource(entry ANEDraftEntry) (ggufPath, tensorName string, err error) {
	for _, t := range driveEmbdTensorCandidates() {
		if p, ok := resolveDriveEmbdGGUF(entry, t); ok {
			return p, t, nil
		}
	}
	return "", "", fmt.Errorf("no token_embd/output.weight in sidecar or base for %s", entry.Tag)
}

func resolveDriveNormSource(entry ANEDraftEntry, nEmbd int) (ggufPath, tensorName string) {
	for _, t := range []string{defaultDriveNormTensor(), "blk.0.attn_norm.weight"} {
		if draftPath, present := resolveDraftGGUFPath(entry); present && draftPath != "" {
			if raw, tensor, err := ggml.ReadTensorBytes(draftPath, t); err == nil {
				if _, err := extractVectorF32(raw, tensor, nEmbd); err == nil {
					return draftPath, t
				}
			}
		}
		if base := strings.TrimSpace(entry.BaseGGUF); base != "" {
			if raw, tensor, err := ggml.ReadTensorBytes(base, t); err == nil {
				if _, err := extractVectorF32(raw, tensor, nEmbd); err == nil {
					return base, t
				}
			}
		}
	}
	return "", ""
}

func driveEmbdCachePath(sidecarPath string) string {
	base := strings.TrimSuffix(filepath.Base(sidecarPath), filepath.Ext(sidecarPath))
	return filepath.Join(aneDraftWeightCacheDir(), base+"-drive-embd.fp16")
}

func driveEmbdMetaPath(sidecarPath string) string {
	return driveEmbdCachePath(sidecarPath) + ".json"
}

func driveNormCachePath(sidecarPath string) string {
	base := strings.TrimSuffix(filepath.Base(sidecarPath), filepath.Ext(sidecarPath))
	return filepath.Join(aneDraftWeightCacheDir(), base+"-drive-output_norm.f32")
}

// MaterializeANEDraftDriveHead extracts tied token_embd + output_norm for host-side B7 argmax.
func MaterializeANEDraftDriveHead(entry ANEDraftEntry) (ANEDraftDriveHeadManifest, bool, error) {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return ANEDraftDriveHeadManifest{}, false, fmt.Errorf("draft sidecar GGUF missing for %s", entry.Tag)
	}

	metaPath := driveEmbdMetaPath(draftPath)
	embdPath := driveEmbdCachePath(draftPath)
	normPath := driveNormCachePath(draftPath)

	sidecarStat, err := os.Stat(draftPath)
	if err != nil {
		return ANEDraftDriveHeadManifest{}, false, err
	}
	if metaStat, err := os.Stat(metaPath); err == nil {
		if !metaStat.ModTime().Before(sidecarStat.ModTime()) {
			var cached ANEDraftDriveHeadManifest
			data, err := os.ReadFile(metaPath)
			if err == nil && json.Unmarshal(data, &cached) == nil && cached.TokenEmbdPath != "" {
				if st, err := os.Stat(cached.TokenEmbdPath); err == nil && st.Size() > 0 {
					return cached, true, nil
				}
			}
		}
	}

	embdGGUF, embdTensor, err := resolveDriveEmbdSource(entry)
	if err != nil {
		return ANEDraftDriveHeadManifest{}, false, err
	}
	rawEmbd, tensorEmbd, err := ggml.ReadTensorBytes(embdGGUF, embdTensor)
	if err != nil {
		return ANEDraftDriveHeadManifest{}, false, err
	}
	if len(tensorEmbd.Shape) < 2 {
		return ANEDraftDriveHeadManifest{}, false, fmt.Errorf("token_embd rank %v", tensorEmbd.Shape)
	}
	nEmbd := int(tensorEmbd.Shape[0])
	nVocab := int(tensorEmbd.Shape[1])
	if nEmbd <= 0 || nVocab <= 0 {
		return ANEDraftDriveHeadManifest{}, false, fmt.Errorf("invalid token_embd shape %v", tensorEmbd.Shape)
	}

	fp16Embd, err := extractTokenEmbdFP16(rawEmbd, tensorEmbd)
	if err != nil {
		return ANEDraftDriveHeadManifest{}, false, err
	}
	if err := writeDriveEmbdFile(embdPath, fp16Embd); err != nil {
		return ANEDraftDriveHeadManifest{}, false, err
	}

	out := ANEDraftDriveHeadManifest{
		TokenEmbdPath: embdPath,
		NEmbd:         nEmbd,
		NVocab:        nVocab,
		EmbdBytes:     len(fp16Embd),
		TensorEmbd:    embdTensor,
	}

	if normGGUF, normTensor := resolveDriveNormSource(entry, nEmbd); normGGUF != "" {
		if normRaw, tensorNorm, err := ggml.ReadTensorBytes(normGGUF, normTensor); err == nil {
			normF32, err := extractVectorF32(normRaw, tensorNorm, nEmbd)
			if err == nil {
				if err := os.WriteFile(normPath, normF32, 0o644); err == nil {
					out.OutputNormPath = normPath
					out.NormBytes = len(normF32)
					out.TensorNorm = tensorNorm.Name
				}
			}
		}
	}

	metaBytes, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return ANEDraftDriveHeadManifest{}, false, err
	}
	if err := os.WriteFile(metaPath, metaBytes, 0o644); err != nil {
		return ANEDraftDriveHeadManifest{}, false, err
	}
	return out, false, nil
}

func writeDriveEmbdFile(path string, fp16 []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	hdr := make([]byte, 16)
	copy(hdr, driveEmbdMagic)
	binary.LittleEndian.PutUint32(hdr[8:], uint32(len(fp16)))
	if _, err := f.Write(hdr); err != nil {
		f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if _, err := f.Write(fp16); err != nil {
		f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

func extractTokenEmbdFP16(raw []byte, tensor *ggml.Tensor) ([]byte, error) {
	if len(tensor.Shape) < 2 {
		return nil, fmt.Errorf("token_embd needs rank 2")
	}
	nEmbd := int(tensor.Shape[0])
	nVocab := int(tensor.Shape[1])
	need := nEmbd * nVocab * 2
	kind := ggml.TensorType(tensor.Kind)
	switch kind {
	case ggml.TensorTypeF16:
		if len(raw) < need {
			return nil, fmt.Errorf("token_embd truncated")
		}
		return append([]byte(nil), raw[:need]...), nil
	case ggml.TensorTypeBF16:
		out := make([]byte, need)
		for e := 0; e < nEmbd; e++ {
			for v := 0; v < nVocab; v++ {
				off := (e*nVocab + v) * 2
				h := binary.LittleEndian.Uint16(raw[off : off+2])
				binary.LittleEndian.PutUint16(out[off:off+2], float32ToFloat16Bits(bfloat16BitsToFloat32(h)))
			}
		}
		return out, nil
	case ggml.TensorTypeF32:
		out := make([]byte, need)
		for e := 0; e < nEmbd; e++ {
			for v := 0; v < nVocab; v++ {
				off := (e*nVocab + v) * 4
				f := math.Float32frombits(binary.LittleEndian.Uint32(raw[off : off+4]))
				binary.LittleEndian.PutUint16(out[(e*nVocab+v)*2:], float32ToFloat16Bits(f))
			}
		}
		return out, nil
	default:
		if !ggml.TensorType(kind).IsQuantized() {
			return nil, fmt.Errorf("token_embd kind %v unsupported for B7 drive", kind)
		}
		f32 := mlggml.ConvertToF32(raw, tensor.Kind, uint64(nEmbd*nVocab))
		if len(f32) < nEmbd*nVocab {
			return nil, fmt.Errorf("token_embd dequant truncated")
		}
		out := make([]byte, need)
		for i := 0; i < nEmbd*nVocab; i++ {
			binary.LittleEndian.PutUint16(out[i*2:], float32ToFloat16Bits(f32[i]))
		}
		return out, nil
	}
}

func extractVectorF32(raw []byte, tensor *ggml.Tensor, n int) ([]byte, error) {
	if len(tensor.Shape) != 1 || int(tensor.Shape[0]) < n {
		return nil, fmt.Errorf("norm vector shape %v", tensor.Shape)
	}
	out := make([]byte, n*4)
	kind := ggml.TensorType(tensor.Kind)
	switch kind {
	case ggml.TensorTypeF32:
		copy(out, raw[:n*4])
	case ggml.TensorTypeF16:
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint16(raw[i*2:])
			f := fp16BitsToFloat32(bits)
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(f))
		}
	case ggml.TensorTypeBF16:
		for i := 0; i < n; i++ {
			h := binary.LittleEndian.Uint16(raw[i*2:])
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(bfloat16BitsToFloat32(h)))
		}
	default:
		return nil, fmt.Errorf("norm kind %v unsupported", kind)
	}
	return out, nil
}

// ExportDriveEnvForHead returns env vars for B7 host argmax head.
func ExportDriveEnvForHead(h ANEDraftDriveHeadManifest) map[string]string {
	out := map[string]string{}
	if h.TokenEmbdPath != "" {
		out["ZEROLLAMA_ANE_DRAFT_TOKEN_EMBD_FILE"] = h.TokenEmbdPath
	}
	if h.OutputNormPath != "" {
		out["ZEROLLAMA_ANE_DRAFT_OUTPUT_NORM_FILE"] = h.OutputNormPath
	}
	if h.NEmbd > 0 {
		out["ZEROLLAMA_ANE_DRAFT_N_EMBD"] = fmt.Sprintf("%d", h.NEmbd)
	}
	if h.NVocab > 0 {
		out["ZEROLLAMA_ANE_DRAFT_N_VOCAB"] = fmt.Sprintf("%d", h.NVocab)
	}
	out["ZEROLLAMA_ANE_DRAFT_DRIVE_VOCAB_CAP"] = "8192"
	return out
}

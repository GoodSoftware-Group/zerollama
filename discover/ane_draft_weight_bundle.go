package discover

// ANE dflash weight bundle — extracts sidecar GGUF tensors into maderix BLOBFILE caches.
// Why host-side gamma: ANE MIL broadcast mul (conv × norm) failed compile; scaling at pack
// time preserves B3 intent without blocking IOSurface handoff. Why manifest v2: optional
// second conv weight (ffn_up) for B6 subgraph expansion toward full dflash block0.
import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ANEDraftWeightEntry is one cached MIL BLOBFILE on disk.
type ANEDraftWeightEntry struct {
	Slot   string `json:"slot"`
	Role   string `json:"role"`
	Tensor string `json:"tensor,omitempty"`
	Path   string `json:"path"`
	Bytes  int    `json:"bytes,omitempty"`
}

// ANEDraftWeightManifest lists conv proxy + optional norm gamma for in-process ANE session init.
type ANEDraftWeightManifest struct {
	Version  int                   `json:"version"`
	Tag      string                `json:"tag,omitempty"`
	Channels int                   `json:"channels"`
	Spatial  int                   `json:"spatial"`
	Weights  []ANEDraftWeightEntry `json:"weights"`
	Note     string                `json:"note,omitempty"`
}

// ConvWeightPath returns the lab conv proxy blob path when present.
func (m ANEDraftWeightManifest) ConvWeightPath() string {
	for _, w := range m.Weights {
		if w.Slot == "proxy_conv_w0" || w.Role == "conv_w0" {
			return w.Path
		}
	}
	if len(m.Weights) > 0 {
		return m.Weights[0].Path
	}
	return ""
}

// Conv2WeightPath returns the optional second conv proxy (B6 subgraph) when present.
func (m ANEDraftWeightManifest) Conv2WeightPath() string {
	for _, w := range m.Weights {
		if w.Slot == "proxy_conv_w1" || w.Role == "conv_w1" {
			return w.Path
		}
	}
	return ""
}

// GammaWeightPath returns the optional RMS-norm gamma blob for conv output scaling.
func (m ANEDraftWeightManifest) GammaWeightPath() string {
	for _, w := range m.Weights {
		if w.Slot == "decoder_ffn_norm" || w.Role == "ffn_norm_gamma" {
			return w.Path
		}
	}
	return ""
}

func aneDraftWeightManifestPath(sidecarPath string, channels int) string {
	base := strings.TrimSuffix(filepath.Base(sidecarPath), filepath.Ext(sidecarPath))
	return filepath.Join(aneDraftWeightCacheDir(), fmt.Sprintf("%s-%d-manifest.json", base, channels))
}

// MaterializeANEDraftWeightBundle extracts proxy conv + optional ffn norm gamma into cache.
func MaterializeANEDraftWeightBundle(entry ANEDraftEntry) (ANEDraftWeightManifest, bool, error) {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return ANEDraftWeightManifest{}, false, fmt.Errorf("draft sidecar GGUF missing for %s", entry.Tag)
	}

	ch := entry.ProxyChannels
	if ch <= 0 {
		ch, _ = DraftANEProxyDims(entry.EmbeddingLength)
	}
	sp := entry.ProxySpatial
	if sp <= 0 {
		sp = 16
	}

	manifestPath := aneDraftWeightManifestPath(draftPath, ch)
	sidecarStat, err := os.Stat(draftPath)
	if err != nil {
		return ANEDraftWeightManifest{}, false, err
	}
	if manifestStat, err := os.Stat(manifestPath); err == nil {
		if !manifestStat.ModTime().Before(sidecarStat.ModTime()) {
			var cached ANEDraftWeightManifest
			data, err := os.ReadFile(manifestPath)
			if err == nil && json.Unmarshal(data, &cached) == nil && cached.ConvWeightPath() != "" {
				if cached.Conv2WeightPath() != "" && cached.Version >= 3 {
					return cached, true, nil
				}
			}
		}
	}

	proxyTensor, _, err := DefaultProxyConvTensorForSidecar(draftPath)
	if err != nil {
		proxyTensor = DefaultProxyConvTensor()
	}

	convPath, _, err := MaterializeANEDraftWeightFile(entry, proxyTensor)
	if err != nil {
		return ANEDraftWeightManifest{}, false, err
	}

	manifest := ANEDraftWeightManifest{
		Version:  1,
		Tag:      entry.Tag,
		Channels: ch,
		Spatial:  sp,
		Note:     "B3/B6: conv proxy + optional ffn_up conv2 + ffn norm gamma",
		Weights: []ANEDraftWeightEntry{
			{
				Slot:   "proxy_conv_w0",
				Role:   "conv_w0",
				Tensor: proxyTensor,
				Path:   convPath,
				Bytes:  draftMILWeightBlobBytes(ch),
			},
		},
	}

	if conv2Path, _, err := MaterializeANEDraftWeightFile(entry, DefaultProxyConv2Tensor()); err == nil && conv2Path != "" {
		manifest.Weights = append(manifest.Weights, ANEDraftWeightEntry{
			Slot:   "proxy_conv_w1",
			Role:   "conv_w1",
			Tensor: DefaultProxyConv2Tensor(),
			Path:   conv2Path,
			Bytes:  draftMILWeightBlobBytes(ch),
		})
		manifest.Version = 3
		manifest.Note = "B6 v3: convT [out,in] from GGUF FFN + maderix blob header (wsize@72, payload@128)"
	}

	for _, normTensor := range []string{"blk.0.ffn_norm.weight", "blk.0.attn_norm.weight"} {
		gammaPath := aneDraftWeightCachePath(draftPath, ch, normTensor)
		if gammaStat, err := os.Stat(gammaPath); err == nil {
			if !gammaStat.ModTime().Before(sidecarStat.ModTime()) && gammaStat.Size() > 0 {
				manifest.Weights = append(manifest.Weights, ANEDraftWeightEntry{
					Slot:   "decoder_ffn_norm",
					Role:   "ffn_norm_gamma",
					Tensor: normTensor,
					Path:   gammaPath,
					Bytes:  int(gammaStat.Size()),
				})
				break
			}
		}

		blob, tensor, err := ExtractNormVectorWeightBlob(draftPath, normTensor, ch)
		if err != nil {
			continue
		}
		if err := os.MkdirAll(filepath.Dir(gammaPath), 0o755); err != nil {
			return ANEDraftWeightManifest{}, false, err
		}
		tmp := gammaPath + ".tmp"
		if err := os.WriteFile(tmp, blob, 0o644); err != nil {
			return ANEDraftWeightManifest{}, false, err
		}
		if err := os.Rename(tmp, gammaPath); err != nil {
			_ = os.Remove(tmp)
			return ANEDraftWeightManifest{}, false, err
		}
		manifest.Weights = append(manifest.Weights, ANEDraftWeightEntry{
			Slot:   "decoder_ffn_norm",
			Role:   "ffn_norm_gamma",
			Tensor: tensor.Name,
			Path:   gammaPath,
			Bytes:  len(blob),
		})
		break
	}

	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return ANEDraftWeightManifest{}, false, err
	}
	tmpManifest := manifestPath + ".tmp"
	if err := os.WriteFile(tmpManifest, data, 0o644); err != nil {
		return ANEDraftWeightManifest{}, false, err
	}
	if err := os.Rename(tmpManifest, manifestPath); err != nil {
		_ = os.Remove(tmpManifest)
		return ANEDraftWeightManifest{}, false, err
	}

	return manifest, false, nil
}

// MaterializeANEDraftWeightBundleWithDrive is like MaterializeANEDraftWeightBundle but also
// extracts B7 token_embd when ZEROLLAMA_ANE_DRAFT_DRIVE is set or forceDrive is true.
func MaterializeANEDraftWeightBundleWithDrive(entry ANEDraftEntry, forceDrive bool) (ANEDraftWeightManifest, bool, error) {
	manifest, cached, err := MaterializeANEDraftWeightBundle(entry)
	if err != nil {
		return manifest, cached, err
	}
	wantDrive := forceDrive || envTruthy(os.Getenv("ZEROLLAMA_ANE_DRAFT_DRIVE"))
	if !wantDrive {
		return manifest, cached, nil
	}
	if manifest.TokenEmbdPath() != "" && manifest.Version >= 4 {
		return manifest, cached, nil
	}
	head, _, err := MaterializeANEDraftDriveHead(entry)
	if err != nil {
		return manifest, cached, err
	}
	manifest.Version = 4
	manifest.Note = "B7: conv proxy + optional drive token_embd argmax head"
	manifest.Weights = append(manifest.Weights, ANEDraftWeightEntry{
		Slot:   "drive_token_embd",
		Role:   "token_embd",
		Tensor: head.TensorEmbd,
		Path:   head.TokenEmbdPath,
		Bytes:  head.EmbdBytes,
	})
	if head.OutputNormPath != "" {
		manifest.Weights = append(manifest.Weights, ANEDraftWeightEntry{
			Slot:   "drive_output_norm",
			Role:   "output_norm",
			Tensor: head.TensorNorm,
			Path:   head.OutputNormPath,
			Bytes:  head.NormBytes,
		})
	}
	manifestPath := aneDraftWeightManifestPath(draftPathFromEntry(entry), manifest.Channels)
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return manifest, false, err
	}
	_ = os.WriteFile(manifestPath, data, 0o644)
	return manifest, false, nil
}

func draftPathFromEntry(entry ANEDraftEntry) string {
	p, _ := resolveDraftGGUFPath(entry)
	return p
}

func envTruthy(v string) bool {
	v = strings.TrimSpace(strings.ToLower(v))
	return v == "1" || v == "true" || v == "yes" || v == "on" || v == "force" || v == "shadow"
}

// ExportEnvForManifest returns env vars for llama-server in-process ANE session init.
func ExportEnvForManifest(m ANEDraftWeightManifest, manifestPath string) map[string]string {
	out := map[string]string{
		"ZEROLLAMA_ANE_DRAFT":         "1",
		"ZEROLLAMA_ANE_DRAFT_CHANNELS": fmt.Sprintf("%d", m.Channels),
		"ZEROLLAMA_ANE_DRAFT_SPATIAL":  fmt.Sprintf("%d", m.Spatial),
	}
	if manifestPath != "" {
		out["ZEROLLAMA_ANE_DRAFT_WEIGHT_MANIFEST"] = manifestPath
	}
	if conv := m.ConvWeightPath(); conv != "" {
		out["ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE"] = conv
	}
	if conv2 := m.Conv2WeightPath(); conv2 != "" {
		out["ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2"] = conv2
	}
	if gamma := m.GammaWeightPath(); gamma != "" {
		out["ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"] = gamma
	}
	applyDriveHeadEnv(out, m)
	return out
}

func applyDriveHeadEnv(out map[string]string, m ANEDraftWeightManifest) {
	embd := m.TokenEmbdPath()
	if embd == "" {
		return
	}
	out["ZEROLLAMA_ANE_DRAFT_TOKEN_EMBD_FILE"] = embd
	for _, w := range m.Weights {
		if w.Slot == "drive_output_norm" && w.Path != "" {
			out["ZEROLLAMA_ANE_DRAFT_OUTPUT_NORM_FILE"] = w.Path
		}
	}
	metaPath := embd + ".json"
	data, err := os.ReadFile(metaPath)
	if err != nil {
		return
	}
	var head ANEDraftDriveHeadManifest
	if json.Unmarshal(data, &head) != nil {
		return
	}
	out["ZEROLLAMA_ANE_DRAFT_DRIVE_VOCAB_CAP"] = "8192"
	for k, v := range ExportDriveEnvForHead(head) {
		out[k] = v
	}
}

// TokenEmbdPath returns optional B7 tied-embed cache path when manifest includes drive head.
func (m ANEDraftWeightManifest) TokenEmbdPath() string {
	for _, w := range m.Weights {
		if w.Slot == "drive_token_embd" || w.Role == "token_embd" {
			return w.Path
		}
	}
	return ""
}

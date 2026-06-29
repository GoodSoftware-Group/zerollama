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
	"strconv"
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

// Conv3WeightPath returns the optional third conv proxy (B8 attn_gate on dflash-draft) when present.
func (m ANEDraftWeightManifest) Conv3WeightPath() string {
	for _, w := range m.Weights {
		if w.Slot == "proxy_conv_w2" || w.Role == "conv_w2" {
			return w.Path
		}
	}
	return ""
}

// Conv4WeightPath returns the optional fourth conv proxy (B9 ffn_down) when present.
func (m ANEDraftWeightManifest) Conv4WeightPath() string {
	for _, w := range m.Weights {
		if w.Slot == "proxy_conv_w3" || w.Role == "conv_w3" {
			return w.Path
		}
	}
	return ""
}

// Conv5WeightPath returns the optional fifth conv proxy (B10 blk.1 ffn_gate) when present.
func (m ANEDraftWeightManifest) Conv5WeightPath() string {
	for _, w := range m.Weights {
		if w.Slot == "proxy_conv_w4" || w.Role == "conv_w4" {
			return w.Path
		}
	}
	return ""
}

// Conv6WeightPath returns the optional sixth conv proxy (B11 blk.1 ffn_up) when present.
func (m ANEDraftWeightManifest) Conv6WeightPath() string {
	for _, w := range m.Weights {
		if w.Slot == "proxy_conv_w5" || w.Role == "conv_w5" {
			return w.Path
		}
	}
	return ""
}

// Conv7WeightPath returns the optional seventh conv proxy (B12 blk.1 attn_gate) when present.
func (m ANEDraftWeightManifest) Conv7WeightPath() string {
	for _, w := range m.Weights {
		if w.Slot == "proxy_conv_w6" || w.Role == "conv_w6" {
			return w.Path
		}
	}
	return ""
}

// Conv8WeightPath returns the optional eighth conv proxy (B13 blk.1 ffn_down) when present.
func (m ANEDraftWeightManifest) Conv8WeightPath() string {
	for _, w := range m.Weights {
		if w.Slot == "proxy_conv_w7" || w.Role == "conv_w7" {
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
			if err == nil && json.Unmarshal(data, &cached) == nil && cached.ConvWeightPath() != "" && cached.Version >= 10 {
				return cached, true, nil
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

	if conv3Path, _, err := MaterializeANEDraftWeightFile(entry, conv3TensorForEntry(entry)); err == nil && conv3Path != "" {
		manifest.Weights = append(manifest.Weights, ANEDraftWeightEntry{
			Slot:   "proxy_conv_w2",
			Role:   "conv_w2",
			Tensor: conv3TensorForEntry(entry),
			Path:   conv3Path,
			Bytes:  draftMILWeightBlobBytes(ch),
		})
		manifest.Version = 5
		manifest.Note = "B8 v5: gate+up+attn_gate triple conv1 chain (lab proxy toward block0)"
	}

	if conv4Path, _, err := MaterializeANEDraftWeightFile(entry, conv4TensorForEntry(entry)); err == nil && conv4Path != "" {
		manifest.Weights = append(manifest.Weights, ANEDraftWeightEntry{
			Slot:   "proxy_conv_w3",
			Role:   "conv_w3",
			Tensor: conv4TensorForEntry(entry),
			Path:   conv4Path,
			Bytes:  draftMILWeightBlobBytes(ch),
		})
		manifest.Version = 6
		manifest.Note = "B9 v6: gate+up+attn_gate+ffn_down quad conv1 chain (block0 lab proxy)"
	}

	if conv5Path, _, err := MaterializeANEDraftWeightFile(entry, conv5TensorForEntry(entry)); err == nil && conv5Path != "" {
		manifest.Weights = append(manifest.Weights, ANEDraftWeightEntry{
			Slot:   "proxy_conv_w4",
			Role:   "conv_w4",
			Tensor: conv5TensorForEntry(entry),
			Path:   conv5Path,
			Bytes:  draftMILWeightBlobBytes(ch),
		})
		manifest.Version = 7
		manifest.Note = "B10 v7: block0 quad + blk.1 ffn_gate pent conv1 chain"
	}

	if conv6Path, _, err := MaterializeANEDraftWeightFile(entry, conv6TensorForEntry(entry)); err == nil && conv6Path != "" {
		manifest.Weights = append(manifest.Weights, ANEDraftWeightEntry{
			Slot:   "proxy_conv_w5",
			Role:   "conv_w5",
			Tensor: conv6TensorForEntry(entry),
			Path:   conv6Path,
			Bytes:  draftMILWeightBlobBytes(ch),
		})
		manifest.Version = 8
		manifest.Note = "B11 v8: block0 quad + blk.1 gate/up hex conv1 chain"
	}

	if conv7Path, _, err := MaterializeANEDraftWeightFile(entry, conv7TensorForEntry(entry)); err == nil && conv7Path != "" {
		manifest.Weights = append(manifest.Weights, ANEDraftWeightEntry{
			Slot:   "proxy_conv_w6",
			Role:   "conv_w6",
			Tensor: conv7TensorForEntry(entry),
			Path:   conv7Path,
			Bytes:  draftMILWeightBlobBytes(ch),
		})
		manifest.Version = 9
		manifest.Note = "B12 v9: block0 quad + blk.1 gate/up/attn_gate hept conv1 chain"
	}

	if conv8Path, _, err := MaterializeANEDraftWeightFile(entry, conv8TensorForEntry(entry)); err == nil && conv8Path != "" {
		manifest.Weights = append(manifest.Weights, ANEDraftWeightEntry{
			Slot:   "proxy_conv_w7",
			Role:   "conv_w7",
			Tensor: conv8TensorForEntry(entry),
			Path:   conv8Path,
			Bytes:  draftMILWeightBlobBytes(ch),
		})
		manifest.Version = 10
		manifest.Note = "B13 v10: block0 quad + blk.1 full quad oct conv1 chain"
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
	if manifest.Conv8WeightPath() != "" {
		manifest.Version = 10
		manifest.Note = "B13 v10 + B7 drive token_embd argmax head"
	} else if manifest.Conv7WeightPath() != "" {
		manifest.Version = 9
		manifest.Note = "B12 v9 + B7 drive token_embd argmax head"
	} else if manifest.Conv6WeightPath() != "" {
		manifest.Version = 8
		manifest.Note = "B11 v8 + B7 drive token_embd argmax head"
	} else if manifest.Conv5WeightPath() != "" {
		manifest.Version = 7
		manifest.Note = "B10 v7 + B7 drive token_embd argmax head"
	} else if manifest.Conv4WeightPath() != "" {
		manifest.Version = 6
		manifest.Note = "B9 v6 + B7 drive token_embd argmax head"
	} else if manifest.Conv3WeightPath() != "" {
		manifest.Version = 5
		manifest.Note = "B8 v5 + B7 drive token_embd argmax head"
	} else {
		manifest.Version = 4
		manifest.Note = "B7: conv proxy + optional drive token_embd argmax head"
	}
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

func conv3TensorForEntry(entry ANEDraftEntry) string {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return DefaultProxyConv3Tensor()
	}
	if t, _, err := ResolveProxyConv3TensorForSidecar(draftPath); err == nil {
		return t
	}
	return DefaultProxyConv3Tensor()
}

func conv4TensorForEntry(entry ANEDraftEntry) string {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return DefaultProxyConv4Tensor()
	}
	if t, _, err := ResolveProxyConv4TensorForSidecar(draftPath); err == nil {
		return t
	}
	return DefaultProxyConv4Tensor()
}

func conv5TensorForEntry(entry ANEDraftEntry) string {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return DefaultProxyConv5Tensor()
	}
	if t, _, err := ResolveProxyConv5TensorForSidecar(draftPath); err == nil {
		return t
	}
	return DefaultProxyConv5Tensor()
}

func conv6TensorForEntry(entry ANEDraftEntry) string {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return DefaultProxyConv6Tensor()
	}
	if t, _, err := ResolveProxyConv6TensorForSidecar(draftPath); err == nil {
		return t
	}
	return DefaultProxyConv6Tensor()
}

func conv7TensorForEntry(entry ANEDraftEntry) string {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return DefaultProxyConv7Tensor()
	}
	if t, _, err := ResolveProxyConv7TensorForSidecar(draftPath); err == nil {
		return t
	}
	return DefaultProxyConv7Tensor()
}

func conv8TensorForEntry(entry ANEDraftEntry) string {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return DefaultProxyConv8Tensor()
	}
	if t, _, err := ResolveProxyConv8TensorForSidecar(draftPath); err == nil {
		return t
	}
	return DefaultProxyConv8Tensor()
}

func envTruthy(v string) bool {
	v = strings.TrimSpace(strings.ToLower(v))
	return v == "1" || v == "true" || v == "yes" || v == "on" || v == "force" || v == "shadow"
}

// ManifestConvCount returns how many conv proxy weights are in the manifest (1..7).
func ManifestConvCount(m ANEDraftWeightManifest) int {
	n := 0
	for _, fn := range []func() string{
		m.ConvWeightPath, m.Conv2WeightPath, m.Conv3WeightPath, m.Conv4WeightPath,
		m.Conv5WeightPath, m.Conv6WeightPath, m.Conv7WeightPath, m.Conv8WeightPath,
	} {
		if fn() != "" {
			n++
		}
	}
	return n
}

// ApplyConvDepthEnv sets ZEROLLAMA_ANE_DRAFT_CONV_DEPTH when depth > 0 (cap active ANE kernels).
func ApplyConvDepthEnv(out map[string]string, depth int) {
	if depth > 0 {
		out["ZEROLLAMA_ANE_DRAFT_CONV_DEPTH"] = strconv.Itoa(depth)
	}
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
	if conv3 := m.Conv3WeightPath(); conv3 != "" {
		out["ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3"] = conv3
	}
	if conv4 := m.Conv4WeightPath(); conv4 != "" {
		out["ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4"] = conv4
	}
	if conv5 := m.Conv5WeightPath(); conv5 != "" {
		out["ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5"] = conv5
	}
	if conv6 := m.Conv6WeightPath(); conv6 != "" {
		out["ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6"] = conv6
	}
	if conv7 := m.Conv7WeightPath(); conv7 != "" {
		out["ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7"] = conv7
	}
	if conv8 := m.Conv8WeightPath(); conv8 != "" {
		out["ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8"] = conv8
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

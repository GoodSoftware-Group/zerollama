package discover

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// FlashMoEInventoryEntry is a local MoE model zerollama can run with Flash-MoE.
type FlashMoEInventoryEntry struct {
	Name         string `json:"name"`
	Tag          string `json:"tag"`
	GGUFPath     string `json:"gguf_path"`
	Sidecar      string `json:"sidecar,omitempty"`
	SidecarReady bool   `json:"sidecar_ready"`
	ExpertCount  uint32 `json:"expert_count"`
	Family       string `json:"family"`
	SizeBytes    int64  `json:"size_bytes"`
}

// ListFlashMoEInventory scans ~/.ollama/models for pulled MoE GGUF tags.
// Why local inventory: operators should not hand-copy blob paths — smoke and doctor
// resolve the same way `zerollama show` does.
func ListFlashMoEInventory() ([]FlashMoEInventoryEntry, error) {
	ms, err := manifest.Manifests(true)
	if err != nil {
		return nil, err
	}

	var out []FlashMoEInventoryEntry
	for name, mf := range ms {
		entry, ok, err := flashMoEEntryFromManifest(name, mf)
		if err != nil || !ok {
			continue
		}
		out = append(out, entry)
	}

	sortFlashMoEInventory(out)
	return out, nil
}

func sortFlashMoEInventory(out []FlashMoEInventoryEntry) {
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.SidecarReady != b.SidecarReady {
			return a.SidecarReady
		}
		if a.SizeBytes != b.SizeBytes {
			if a.SizeBytes == 0 {
				return false
			}
			if b.SizeBytes == 0 {
				return true
			}
			return a.SizeBytes < b.SizeBytes
		}
		return a.Tag < b.Tag
	})
}

// SelectFlashMoEModel picks an inventory entry. When preferred is non-empty, match
// tag or full name (case-insensitive). Otherwise prefer sidecar-ready, then smallest.
func SelectFlashMoEModel(entries []FlashMoEInventoryEntry, preferred string) (FlashMoEInventoryEntry, bool) {
	if len(entries) == 0 {
		return FlashMoEInventoryEntry{}, false
	}
	entries = append([]FlashMoEInventoryEntry(nil), entries...)
	sortFlashMoEInventory(entries)
	preferred = strings.TrimSpace(preferred)
	if preferred != "" {
		want := strings.ToLower(preferred)
		for _, e := range entries {
			if strings.EqualFold(e.Name, preferred) ||
				strings.EqualFold(e.Tag, preferred) ||
				strings.EqualFold(strings.TrimSuffix(e.Tag, ":latest"), want) {
				return e, true
			}
		}
	}
	return entries[0], true
}

// ResolveFlashMoEModel returns the best local MoE model for Flash-MoE smoke/serve.
func ResolveFlashMoEModel(preferred string) (FlashMoEInventoryEntry, error) {
	entries, err := ListFlashMoEInventory()
	if err != nil {
		return FlashMoEInventoryEntry{}, err
	}
	if e, ok := SelectFlashMoEModel(entries, preferred); ok {
		return e, nil
	}
	return FlashMoEInventoryEntry{}, os.ErrNotExist
}

// FlashMoEEntryForName inspects one freshly pulled/created manifest and reports
// whether it is a Flash-MoE candidate. Why a single-name path instead of
// reusing ListFlashMoEInventory: pull-time auto-extract must not walk every
// local manifest on each pull — only the tag that just landed.
func FlashMoEEntryForName(name model.Name) (FlashMoEInventoryEntry, bool, error) {
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return FlashMoEInventoryEntry{}, false, err
	}
	return flashMoEEntryFromManifest(name, mf)
}

func flashMoEEntryFromManifest(name model.Name, mf *manifest.Manifest) (FlashMoEInventoryEntry, bool, error) {
	if mf == nil {
		return FlashMoEInventoryEntry{}, false, nil
	}

	var (
		modelPath string
		modelSize int64
		params    map[string]any
		family    string
	)

	if mf.Config.Digest != "" {
		cfgPath, err := manifest.BlobsPath(mf.Config.Digest)
		if err == nil {
			if f, err := os.Open(cfgPath); err == nil {
				var cfg model.ConfigV2
				if json.NewDecoder(f).Decode(&cfg) == nil {
					family = strings.TrimSpace(cfg.ModelFamily)
					if family == "" && len(cfg.ModelFamilies) > 0 {
						family = cfg.ModelFamilies[0]
					}
					if cfg.ModelFormat == "safetensors" {
						f.Close()
						return FlashMoEInventoryEntry{}, false, nil
					}
				}
				f.Close()
			}
		}
	}

	for _, layer := range mf.Layers {
		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			path, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return FlashMoEInventoryEntry{}, false, err
			}
			modelPath = path
			modelSize = layer.Size
		case "application/vnd.ollama.image.projector":
			// Vision bundles are not Flash-MoE smoke targets today.
			return FlashMoEInventoryEntry{}, false, nil
		case "application/vnd.ollama.image.params":
			path, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return FlashMoEInventoryEntry{}, false, err
			}
			f, err := os.Open(path)
			if err != nil {
				return FlashMoEInventoryEntry{}, false, err
			}
			_ = json.NewDecoder(f).Decode(&params)
			f.Close()
		}
	}

	if modelPath == "" {
		return FlashMoEInventoryEntry{}, false, nil
	}

	meta, err := llm.LoadModelMetadata(modelPath)
	if err != nil {
		return FlashMoEInventoryEntry{}, false, nil
	}
	kv := meta.KV()
	if !isMoEGGUF(kv, family) {
		return FlashMoEInventoryEntry{}, false, nil
	}

	tag := name.DisplayShortest()
	if tag == "" {
		tag = name.String()
	}
	short := strings.SplitN(tag, ":", 2)[0]

	sidecar := flashMoESidecarFromParams(params)
	if sidecar == "" {
		sidecar = flashMoEDefaultSidecarPath(short, modelPath)
	}
	ready := flashMoESidecarReady(sidecar)

	return FlashMoEInventoryEntry{
		Name:         name.String(),
		Tag:          tag,
		GGUFPath:     modelPath,
		Sidecar:      sidecar,
		SidecarReady: ready,
		ExpertCount:  kv.Uint("expert_count"),
		Family:       firstNonEmpty(family, kv.Architecture()),
		SizeBytes:    modelSize,
	}, true, nil
}

func isMoEGGUF(kv ggml.KV, family string) bool {
	if kv.Uint("expert_count") > 0 {
		return true
	}
	arch := strings.ToLower(kv.Architecture())
	if strings.Contains(arch, "moe") {
		return true
	}
	family = strings.ToLower(family)
	return strings.Contains(family, "moe")
}

func flashMoESidecarFromParams(params map[string]any) string {
	if params == nil {
		return ""
	}
	if v, ok := params["moe_sidecar"].(string); ok {
		return strings.TrimSpace(v)
	}
	return ""
}

func flashMoEDefaultSidecarPath(shortName, ggufPath string) string {
	shortName = strings.TrimSpace(shortName)
	candidates := []string{
		filepath.Join(envconfig.Models(), "flash", shortName),
		filepath.Join(os.Getenv("HOME"), "Models", "flash", shortName),
	}
	if ggufPath != "" {
		candidates = append(candidates,
			filepath.Join(filepath.Dir(ggufPath), "flash", shortName),
			filepath.Join(filepath.Dir(ggufPath), "..", "flash", shortName),
		)
	}
	for _, c := range candidates {
		if flashMoESidecarReady(c) {
			return c
		}
	}
	if shortName != "" {
		return filepath.Join(os.Getenv("HOME"), "Models", "flash", shortName)
	}
	return ""
}

func flashMoESidecarReady(path string) bool {
	path = strings.TrimSpace(path)
	if path == "" {
		return false
	}
	st, err := os.Stat(path)
	if err != nil || !st.IsDir() {
		if st, err := os.Stat(filepath.Join(path, "manifest.json")); err == nil && !st.IsDir() {
			return true
		}
		return false
	}
	st, err = os.Stat(filepath.Join(path, "manifest.json"))
	return err == nil && !st.IsDir()
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if s := strings.TrimSpace(v); s != "" {
			return s
		}
	}
	return ""
}

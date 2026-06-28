package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"runtime"
)

// ProbeANEDraftMILBundle materializes conv + norm gamma weights and writes manifest JSON.
func ProbeANEDraftMILBundle(_ context.Context, preferred string) (ANEDraftWeightManifest, string, bool, error) {
	if runtime.GOOS != "darwin" {
		return ANEDraftWeightManifest{}, "", false, fmt.Errorf("ane draft mil bundle: darwin only")
	}
	entries, err := ListANEDraftInventory()
	if err != nil {
		return ANEDraftWeightManifest{}, "", false, err
	}
	entry, ok := SelectANEDraftModel(entries, preferred)
	if !ok {
		return ANEDraftWeightManifest{}, "", false, fmt.Errorf("no ANE draft target in local inventory")
	}
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present {
		return ANEDraftWeightManifest{}, "", false, fmt.Errorf("draft sidecar GGUF missing")
	}
	ch := entry.ProxyChannels
	if ch <= 0 {
		ch, _ = DraftANEProxyDims(entry.EmbeddingLength)
	}
	manifest, cached, err := MaterializeANEDraftWeightBundle(entry)
	if err != nil {
		return ANEDraftWeightManifest{}, "", false, err
	}
	return manifest, aneDraftWeightManifestPath(draftPath, ch), cached, nil
}

// RunANEDraftMILBundleJSON writes bundle manifest JSON to w.
func RunANEDraftMILBundleJSON(ctx context.Context, w io.Writer, preferred string) error {
	manifest, manifestPath, cached, err := ProbeANEDraftMILBundle(ctx, preferred)
	type out struct {
		ANEDraftWeightManifest
		ManifestPath string            `json:"manifest_path"`
		Cached       bool              `json:"cached"`
		ExportEnv    map[string]string `json:"export_env,omitempty"`
	}
	res := out{
		ANEDraftWeightManifest: manifest,
		ManifestPath:           manifestPath,
		Cached:                 cached,
		ExportEnv:              ExportEnvForManifest(manifest, manifestPath),
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err != nil {
		res.Note = err.Error()
		_ = enc.Encode(res)
		return err
	}
	return enc.Encode(res)
}

// Package blobaudit summarizes OLLAMA_MODELS blob storage: referenced vs orphan
// bytes, per-tag rollups, and content-addressed dedupe across tags.
package blobaudit

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// Report is the full blob storage audit for OLLAMA_MODELS.
type Report struct {
	ModelsRoot      string       `json:"models_root"`
	BlobDir         string       `json:"blob_dir"`
	TagCount        int          `json:"tag_count"`
	BlobFileCount   int          `json:"blob_file_count"`
	TotalBytes      int64        `json:"total_bytes"`
	ReferencedBytes int64        `json:"referenced_bytes"`
	OrphanBytes     int64        `json:"orphan_bytes"`
	OrphanFileCount int          `json:"orphan_file_count"`
	SharedDigests   int          `json:"shared_digests"`
	DedupeSaved     int64        `json:"dedupe_saved_bytes"`
	Tags            []TagRollup  `json:"tags"`
	TopOrphans      []BlobEntry  `json:"top_orphans,omitempty"`
}

// TagRollup is per-manifest storage attribution.
type TagRollup struct {
	Name         string `json:"name"`
	Kind         string `json:"kind"` // mlx, gguf, mixed
	LayerCount   int    `json:"layer_count"`
	TensorLayers int    `json:"tensor_layers"`
	ManifestSize int64  `json:"manifest_size"`
	UniqueBytes  int64  `json:"unique_bytes"`
	SharedBytes  int64  `json:"shared_bytes"`
}

// BlobEntry describes one on-disk blob file.
type BlobEntry struct {
	Digest string `json:"digest"`
	Path   string `json:"path,omitempty"`
	Bytes  int64  `json:"bytes"`
}

// Audit walks manifests and the blobs directory under OLLAMA_MODELS.
func Audit() (*Report, error) {
	root := envconfig.Models()
	blobDir, err := manifest.BlobsPath("")
	if err != nil {
		return nil, err
	}

	mfs, err := manifest.Manifests(true)
	if err != nil {
		return nil, err
	}

	type digestInfo struct {
		size     int64
		tags     map[string]struct{}
		manifest int64 // manifest-declared size (first seen)
	}

	onDisk := map[string]int64{} // file key sha256-hex -> bytes
	entries, err := os.ReadDir(blobDir)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		onDisk[e.Name()] = info.Size()
	}

	digests := map[string]*digestInfo{}
	tagLayers := map[string][]manifest.Layer{}
	tagNames := make([]model.Name, 0, len(mfs))

	for n, mf := range mfs {
		tagNames = append(tagNames, n)
		display := n.DisplayShortest()
		var layers []manifest.Layer
		layers = append(layers, mf.Config)
		layers = append(layers, mf.Layers...)
		tagLayers[display] = layers

		addDigest := func(d string, size int64) {
			if d == "" {
				return
			}
			canon := canonicalDigest(d)
			di, ok := digests[canon]
			if !ok {
				di = &digestInfo{tags: map[string]struct{}{}}
				digests[canon] = di
			}
			if size > 0 && di.manifest == 0 {
				di.manifest = size
			}
			di.tags[display] = struct{}{}
		}
		addDigest(mf.Config.Digest, mf.Config.Size)
		for _, layer := range mf.Layers {
			addDigest(layer.Digest, layer.Size)
		}
	}

	sort.Slice(tagNames, func(i, j int) bool {
		return strings.Compare(tagNames[i].DisplayShortest(), tagNames[j].DisplayShortest()) < 0
	})

	// Attach on-disk sizes to canonical digests.
	referencedKeys := map[string]struct{}{}
	for canon, di := range digests {
		key := digestFileKey(canon)
		if sz, ok := onDisk[key]; ok {
			di.size = sz
			referencedKeys[key] = struct{}{}
		} else if di.manifest > 0 {
			di.size = di.manifest
		}
		_ = canon
	}

	report := &Report{
		ModelsRoot:    root,
		BlobDir:       blobDir,
		TagCount:      len(mfs),
		BlobFileCount: len(onDisk),
	}

	var sharedDigests int
	var dedupeSaved int64
	for _, di := range digests {
		n := len(di.tags)
		if n > 1 && di.size > 0 {
			sharedDigests++
			dedupeSaved += di.size * int64(n-1)
		}
	}
	report.SharedDigests = sharedDigests
	report.DedupeSaved = dedupeSaved

	var orphans []BlobEntry
	report.ReferencedBytes = 0
	for key, sz := range onDisk {
		report.TotalBytes += sz
		if _, ok := referencedKeys[key]; ok {
			report.ReferencedBytes += sz
			continue
		}
		report.OrphanBytes += sz
		report.OrphanFileCount++
		d := "sha256:" + strings.TrimPrefix(key, "sha256-")
		orphans = append(orphans, BlobEntry{Digest: d, Bytes: sz})
	}

	sort.Slice(orphans, func(i, j int) bool {
		if orphans[i].Bytes == orphans[j].Bytes {
			return orphans[i].Digest < orphans[j].Digest
		}
		return orphans[i].Bytes > orphans[j].Bytes
	})
	if len(orphans) > 10 {
		orphans = orphans[:10]
	}
	report.TopOrphans = orphans

	// Per-tag unique vs shared attribution (count full blob size once per tag).
	tagUnique := map[string]int64{}
	tagShared := map[string]int64{}
	for _, di := range digests {
		if di.size <= 0 {
			continue
		}
		n := len(di.tags)
		for tag := range di.tags {
			if n <= 1 {
				tagUnique[tag] += di.size
			} else {
				tagShared[tag] += di.size
			}
		}
	}

	rollups := make([]TagRollup, 0, len(tagNames))
	for _, n := range tagNames {
		display := n.DisplayShortest()
		layers := tagLayers[display]
		var manifestSize int64
		tensors := 0
		hasModel := false
		for _, layer := range layers {
			manifestSize += layer.Size
			switch layer.MediaType {
			case manifest.MediaTypeImageTensor, manifest.MediaTypeImageDraft:
				tensors++
			case "application/vnd.ollama.image.model":
				hasModel = true
			}
		}
		kind := "mixed"
		switch {
		case tensors > 0 && !hasModel:
			kind = "mlx"
		case hasModel && tensors == 0:
			kind = "gguf"
		case tensors > 0 && hasModel:
			kind = "mixed"
		}
		rollups = append(rollups, TagRollup{
			Name:         display,
			Kind:         kind,
			LayerCount:   len(layers),
			TensorLayers: tensors,
			ManifestSize: manifestSize,
			UniqueBytes:  tagUnique[display],
			SharedBytes:  tagShared[display],
		})
	}
	sort.Slice(rollups, func(i, j int) bool {
		ai := rollups[i].UniqueBytes + rollups[i].SharedBytes
		aj := rollups[j].UniqueBytes + rollups[j].SharedBytes
		if ai == aj {
			return rollups[i].Name < rollups[j].Name
		}
		return ai > aj
	})
	report.Tags = rollups
	return report, nil
}

func canonicalDigest(d string) string {
	d = strings.TrimSpace(d)
	if strings.HasPrefix(d, "sha256-") {
		return "sha256:" + strings.TrimPrefix(d, "sha256-")
	}
	return d
}

func digestFileKey(digest string) string {
	return strings.ReplaceAll(canonicalDigest(digest), ":", "-")
}

// FormatHuman renders a concise terminal report.
func FormatHuman(r *Report) string {
	if r == nil {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Blob audit: %s\n", r.BlobDir)
	fmt.Fprintf(&b, "  tags:           %d\n", r.TagCount)
	fmt.Fprintf(&b, "  blob files:     %d\n", r.BlobFileCount)
	fmt.Fprintf(&b, "  on-disk total:  %s\n", humanGiB(r.TotalBytes))
	fmt.Fprintf(&b, "  referenced:     %s (manifest digests with files)\n", humanGiB(r.ReferencedBytes))
	fmt.Fprintf(&b, "  orphan:         %s (%d files)\n", humanGiB(r.OrphanBytes), r.OrphanFileCount)
	if r.SharedDigests > 0 {
		fmt.Fprintf(&b, "  shared digests: %d (dedupe saves ~%s vs naive duplicate)\n", r.SharedDigests, humanGiB(r.DedupeSaved))
	}
	if len(r.Tags) > 0 {
		b.WriteString("\nPer tag (unique + shared blob bytes):\n")
		for _, t := range r.Tags {
			total := t.UniqueBytes + t.SharedBytes
			fmt.Fprintf(&b, "  %-36s %8s  %5d layers  %4d tensors  [%s]\n",
				t.Name, humanGiB(total), t.LayerCount, t.TensorLayers, t.Kind)
		}
	}
	if len(r.TopOrphans) > 0 {
		b.WriteString("\nLargest orphan blobs (safe to prune after grace period):\n")
		for _, o := range r.TopOrphans {
			fmt.Fprintf(&b, "  %s  %s\n", humanGiB(o.Bytes), o.Digest)
		}
	}
	return strings.TrimRight(b.String(), "\n") + "\n"
}

func humanGiB(n int64) string {
	if n <= 0 {
		return "0 B"
	}
	const gib = 1024 * 1024 * 1024
	if n >= gib {
		return fmt.Sprintf("%.1f GiB", float64(n)/float64(gib))
	}
	const mib = 1024 * 1024
	if n >= mib {
		return fmt.Sprintf("%.1f MiB", float64(n)/float64(mib))
	}
	return fmt.Sprintf("%d B", n)
}

// PruneCandidateCount returns orphan files eligible for PruneLayers-style cleanup.
func PruneCandidateCount(r *Report) int {
	if r == nil {
		return 0
	}
	return r.OrphanFileCount
}

// BlobDirFromRoot returns the blobs path for tests.
func BlobDirFromRoot(root string) string {
	return filepath.Join(root, "blobs")
}

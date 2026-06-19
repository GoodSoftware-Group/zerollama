package manifest

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// ManifestWeights provides fast weight loading from tensor blobs.
// Uses native mmap loading with synthetic safetensors headers for zero-copy.
type ManifestWeights struct {
	manifest    *ModelManifest
	component   string
	tensors     map[string]ManifestLayer // name -> layer
	cache       map[string]*mlx.Array    // name -> loaded array
	nativeCache []*mlx.SafetensorsFile   // keep native handles alive
	quantType   string                   // quantization type from blob metadata (e.g., "int4", "int8")
	groupSize   int                      // quantization group size from blob metadata
}

// LoadWeightsFromManifest creates a weight loader from manifest storage.
// If component is empty, loads all tensors (for LLM models).
// If component is specified, loads only tensors for that component and strips the prefix.
func LoadWeightsFromManifest(manifest *ModelManifest, component string) (*ManifestWeights, error) {
	layers := manifest.GetTensorLayers(component)
	if len(layers) == 0 {
		if component == "" {
			return nil, fmt.Errorf("no tensor layers found in manifest")
		}
		return nil, fmt.Errorf("no tensor layers found for component %q", component)
	}

	// Strip component prefix from tensor names for model loading
	// e.g., "text_encoder/model.embed_tokens.weight" -> "model.embed_tokens.weight"
	tensors := make(map[string]ManifestLayer, len(layers))
	for _, layer := range layers {
		if component == "" {
			tensors[layer.Name] = layer
		} else {
			tensorName := strings.TrimPrefix(layer.Name, component+"/")
			tensors[tensorName] = layer
		}
	}

	return &ManifestWeights{
		manifest:  manifest,
		component: component,
		tensors:   tensors,
		cache:     make(map[string]*mlx.Array),
	}, nil
}

// Load loads all tensor blobs using native mmap (zero-copy) on the GPU stream.
// Callers must keep nativeCache alive until ReleaseAll — see load() below.
func (mw *ManifestWeights) Load(dtype mlx.Dtype) error {
	return mw.load(dtype, mlx.LoadSafetensorsNative)
}

// LoadOnCPU loads all tensor blobs onto the CPU stream (for post-denoise VAE decode on CUDA).
func (mw *ManifestWeights) LoadOnCPU(dtype mlx.Dtype) error {
	return mw.load(dtype, mlx.LoadSafetensorsCPU)
}

func (mw *ManifestWeights) load(dtype mlx.Dtype, open func(string) (*mlx.SafetensorsFile, error)) error {
	// Track native handles to free after batch eval
	nativeHandles := make([]*mlx.SafetensorsFile, 0, len(mw.tensors))
	arrays := make([]*mlx.Array, 0, len(mw.tensors))

	// Group tensors by digest to avoid loading the same blob multiple times
	type blobEntry struct {
		name  string
		layer ManifestLayer
	}
	blobGroups := make(map[string][]blobEntry)
	for name, layer := range mw.tensors {
		blobGroups[layer.Digest] = append(blobGroups[layer.Digest], blobEntry{name, layer})
	}

	for digest, entries := range blobGroups {
		path := mw.manifest.BlobPath(digest)

		// Load blob as safetensors (native mmap, zero-copy)
		sf, err := open(path)
		if err != nil {
			for _, h := range nativeHandles {
				h.Free()
			}
			return fmt.Errorf("load %s: %w", entries[0].name, err)
		}
		nativeHandles = append(nativeHandles, sf)

		// Read quantization metadata from blob
		if qt := sf.GetMetadata("quant_type"); qt != "" && mw.quantType == "" {
			mw.quantType = qt
			if gs := sf.GetMetadata("group_size"); gs != "" {
				mw.groupSize, _ = strconv.Atoi(gs)
			}
		}

		for _, entry := range entries {
			name := entry.name

			// Try to get tensor by stripped name first, then with component prefix,
			// then fall back to "data" for legacy blobs created by older versions
			// that stored all tensors with the generic key "data".
			lookupName := name
			arr := sf.Get(lookupName)
			if arr == nil && mw.component != "" {
				lookupName = mw.component + "/" + name
				arr = sf.Get(lookupName)
			}
			if arr == nil {
				// Legacy blob format: tensor stored as "data"
				lookupName = "data"
				arr = sf.Get(lookupName)
			}
			if arr != nil {
				// Single-tensor blob or tensor found by name
				if dtype != 0 && arr.Dtype() != dtype {
					arr = mlx.AsType(arr, dtype)
				}
				arr = mlx.Contiguous(arr)
				mw.cache[name] = arr
				arrays = append(arrays, arr)

				// Check for scale tensor
				if scale := sf.Get(lookupName + ".scale"); scale != nil {
					scale = mlx.Contiguous(scale)
					mw.cache[name+"_scale"] = scale
					arrays = append(arrays, scale)
				}

				// Check for bias tensor
				if bias := sf.Get(lookupName + ".bias"); bias != nil {
					bias = mlx.Contiguous(bias)
					mw.cache[name+"_qbias"] = bias
					arrays = append(arrays, bias)
				}
			} else {
				// Packed blob: manifest name is a group prefix, not a tensor name.
				// Load all individual tensors from the blob.
				tensorNames, err := ParseBlobTensorNames(path)
				if err != nil {
					for _, h := range nativeHandles {
						h.Free()
					}
					return fmt.Errorf("parse packed blob for %s: %w", name, err)
				}

				for _, tensorName := range tensorNames {
					tArr := sf.Get(tensorName)
					if tArr == nil {
						continue
					}

					if dtype != 0 && tArr.Dtype() != dtype {
						tArr = mlx.AsType(tArr, dtype)
					}
					tArr = mlx.Contiguous(tArr)

					// Strip component prefix from blob-internal names so cache keys
					// match the stripped names used by LoadModule.
					cacheName := tensorName
					if mw.component != "" {
						cacheName = strings.TrimPrefix(tensorName, mw.component+"/")
					}
					mw.cache[cacheName] = tArr
					arrays = append(arrays, tArr)

					// Check for scale tensor
					if scale := sf.Get(tensorName + ".scale"); scale != nil {
						scale = mlx.Contiguous(scale)
						mw.cache[cacheName+"_scale"] = scale
						arrays = append(arrays, scale)
					}

					// Check for bias tensor
					if bias := sf.Get(tensorName + ".bias"); bias != nil {
						bias = mlx.Contiguous(bias)
						mw.cache[cacheName+"_qbias"] = bias
						arrays = append(arrays, bias)
					}
				}
			}
		}
	}

	// Materialize weights in batches — a single mlx_eval over the full transformer
	// spikes VRAM on 16GB GPUs when text-encoder weights were just released.
	if err := mlx.EvalErrBatched(16, arrays); err != nil {
		for _, h := range nativeHandles {
			h.Free()
		}
		return fmt.Errorf("materialize weights: %w", err)
	}
	for _, arr := range arrays {
		mlx.Untrack(arr)
	}
	// Keep mmap handles alive until ReleaseAll(); freeing immediately after Eval can
	// invalidate weight buffers still referenced by model structs on CUDA.
	mw.nativeCache = append(mw.nativeCache, nativeHandles...)

	return nil
}

// GetTensor returns a tensor from cache. Call Load() first.
func (mw *ManifestWeights) GetTensor(name string) (*mlx.Array, error) {
	if mw.cache == nil {
		return nil, fmt.Errorf("cache not initialized: call Load() first")
	}
	arr, ok := mw.cache[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found", name)
	}
	return arr, nil
}

// ListTensors returns all tensor names in sorted order.
// Includes both manifest tensor names and scale/bias entries from combined blobs.
func (mw *ManifestWeights) ListTensors() []string {
	seen := make(map[string]bool, len(mw.tensors)+len(mw.cache))
	for name := range mw.tensors {
		seen[name] = true
	}
	// Also include cache entries (scale/bias from combined blobs)
	for name := range mw.cache {
		seen[name] = true
	}
	names := make([]string, 0, len(seen))
	for name := range seen {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// HasTensor checks if a tensor exists in the manifest or cache.
func (mw *ManifestWeights) HasTensor(name string) bool {
	if _, ok := mw.tensors[name]; ok {
		return true
	}
	// Also check cache for scale/bias entries from combined blobs
	if _, ok := mw.cache[name]; ok {
		return true
	}
	return false
}

// Quantization returns the model's quantization type.
// Returns the quant_type from blob metadata (e.g., "int4", "int8", "nvfp4", "mxfp8").
// Returns empty string if not quantized.
// Falls back to model_index.json for image gen models.
func (mw *ManifestWeights) Quantization() string {
	if mw.quantType != "" {
		return strings.ToUpper(mw.quantType)
	}

	if mw.manifest == nil {
		return ""
	}

	// Fallback: read from model_index.json (for image gen models)
	var index struct {
		Quantization string `json:"quantization"`
	}
	if err := mw.manifest.ReadConfigJSON("model_index.json", &index); err == nil && index.Quantization != "" {
		return index.Quantization
	}

	return ""
}

// GroupSize returns the quantization group size.
// Returns the group_size from blob metadata.
// Returns 0 if not specified (caller should use default based on quantization type).
func (mw *ManifestWeights) GroupSize() int {
	if mw.groupSize > 0 {
		return mw.groupSize
	}

	if mw.manifest == nil {
		return 0
	}

	// Fallback: read from model_index.json (for image gen models)
	var index struct {
		GroupSize int `json:"group_size"`
	}
	if err := mw.manifest.ReadConfigJSON("model_index.json", &index); err == nil && index.GroupSize > 0 {
		return index.GroupSize
	}

	return 0
}

// ReleaseAll frees all native handles and clears the tensor cache.
func (mw *ManifestWeights) ReleaseAll() {
	for _, sf := range mw.nativeCache {
		sf.Free()
	}
	mw.nativeCache = nil
	mw.cache = nil
}

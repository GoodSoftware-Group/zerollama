package server

import (
	"maps"
	"slices"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
	"golang.org/x/sync/singleflight"
)

// inferenceModelCache stores fully resolved model metadata and capabilities.
type inferenceModelCache struct {
	mu      sync.RWMutex
	entries map[inferenceModelCacheKey]inferenceModelCacheEntry
	loads   singleflight.Group

	loadModel func(string) (*Model, error)
}

type inferenceModelCacheKey struct {
	name       string
	modelsRoot string
}

type inferenceModelCacheEntry struct {
	digest string
	model  *Model
}

func newInferenceModelCache() *inferenceModelCache {
	return &inferenceModelCache{
		entries: make(map[inferenceModelCacheKey]inferenceModelCacheEntry),
	}
}

var inferenceModelCacheDefault = newInferenceModelCache()

func init() {
	inferenceModelCacheDefault.loadModel = loadModelUncached
}

func (c *inferenceModelCache) Get(name string) (*Model, error) {
	n := model.ParseName(name)
	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		return nil, err
	}

	key := inferenceModelCacheKey{
		name:       n.String(),
		modelsRoot: envconfig.Models(),
	}
	digest := mf.Digest()

	c.mu.RLock()
	entry, ok := c.entries[key]
	c.mu.RUnlock()
	if ok && entry.digest == digest {
		return cloneInferenceModel(entry.model), nil
	}

	loadKey := key.name + "\x00" + key.modelsRoot + "\x00" + digest
	v, err, _ := c.loads.Do(loadKey, func() (any, error) {
		c.mu.RLock()
		entry, ok := c.entries[key]
		c.mu.RUnlock()
		if ok && entry.digest == digest {
			return entry.model, nil
		}

		m, err := c.loadModel(name)
		if err != nil {
			return nil, err
		}

		c.mu.Lock()
		c.entries[key] = inferenceModelCacheEntry{digest: m.Digest, model: m}
		c.mu.Unlock()
		return m, nil
	})
	if err != nil {
		return nil, err
	}

	return cloneInferenceModel(v.(*Model)), nil
}

func cloneInferenceModel(src *Model) *Model {
	if src == nil {
		return nil
	}

	dst := *src
	dst.Config.ModelFamilies = slices.Clone(src.Config.ModelFamilies)
	dst.Config.Capabilities = slices.Clone(src.Config.Capabilities)
	if src.Config.Draft != nil {
		draft := *src.Config.Draft
		dst.Config.Draft = &draft
	}
	dst.AdapterPaths = slices.Clone(src.AdapterPaths)
	dst.ProjectorPaths = slices.Clone(src.ProjectorPaths)
	dst.License = slices.Clone(src.License)
	dst.Options = maps.Clone(src.Options)
	dst.Messages = slices.Clone(src.Messages)

	return &dst
}

func (s *Server) getModel(name string) (*Model, error) {
	return inferenceModelCacheDefault.Get(name)
}

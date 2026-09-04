package server

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

func TestPruneLayersSkipsRecentOrphans(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	recentDigest := "sha256:0000000000000000000000000000000000000000000000000000000000000001"
	oldDigest := "sha256:0000000000000000000000000000000000000000000000000000000000000002"

	for _, digest := range []string{recentDigest, oldDigest} {
		p, err := manifest.BlobsPath(digest)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	oldPath, err := manifest.BlobsPath(oldDigest)
	if err != nil {
		t.Fatal(err)
	}
	oldTime := time.Now().Add(-layerPruneGracePeriod - time.Hour)
	if err := os.Chtimes(oldPath, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}

	if err := PruneLayers(); err != nil {
		t.Fatal(err)
	}

	recentPath, err := manifest.BlobsPath(recentDigest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(recentPath); err != nil {
		t.Fatalf("recent orphan was pruned: %v", err)
	}
	if _, err := os.Stat(oldPath); !os.IsNotExist(err) {
		t.Fatalf("old orphan still exists: %v", err)
	}
}

func TestModelCapabilities(t *testing.T) {
	// Create completion model (llama architecture without vision)
	completionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
	}, []*ggml.Tensor{})

	// Create vision model (llama architecture with vision block count)
	visionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	// Create embedding model (bert architecture with pooling type)
	embeddingModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, []*ggml.Tensor{})

	toolsInsertTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	chatTemplate, err := template.Parse("{{ .prompt }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	toolsTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	testModels := []struct {
		name         string
		model        Model
		expectedCaps []model.Capability
	}{
		{
			name: "model with image generation capability via config",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image"},
				},
			},
			expectedCaps: []model.Capability{model.CapabilityImage},
		},
		{
			name: "model with image and vision capability (image editing)",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image", "vision"},
				},
			},
			expectedCaps: []model.Capability{model.CapabilityImage, model.CapabilityVision, model.CapabilityVideo},
		},
		{
			name: "model with completion capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion},
		},

		{
			name: "model with completion, tools, and insert capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsInsertTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model with tools capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools},
		},
		{
			name: "missing gguf blob does not panic",
			model: Model{
				Name:      "toy:latest",
				ModelPath: filepath.Join(t.TempDir(), "missing.gguf"),
			},
			expectedCaps: nil,
		},
		{
			name: "model with vision capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityVideo},
		},
		{
			name: "model with vision, tools, and insert capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  toolsInsertTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityVideo, model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model with embedding capability",
			model: Model{
				ModelPath: embeddingModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityEmbedding},
		},
		{
			name: "gemma4 small safetensors suppresses vision and audio",
			model: Model{
				Config: model.ConfigV2{
					ModelFormat:  "safetensors",
					Renderer:     gemma4RendererSmall,
					Capabilities: []string{"vision", "audio"},
				},
				Template: chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityInsert},
		},
		{
			name: "gemma4 large safetensors suppresses vision and audio",
			model: Model{
				Config: model.ConfigV2{
					ModelFormat:  "safetensors",
					Renderer:     gemma4RendererLarge,
					Capabilities: []string{"vision", "audio"},
				},
				Template: chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityInsert},
		},
		{
			name: "legacy gemma4 safetensors suppresses vision and audio",
			model: Model{
				Config: model.ConfigV2{
					ModelFormat:  "safetensors",
					Renderer:     gemma4RendererLegacy,
					Capabilities: []string{"vision", "audio"},
				},
				Template: chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityInsert},
		},
	}

	// compare two slices of model.Capability regardless of order
	compareCapabilities := func(a, b []model.Capability) bool {
		if len(a) != len(b) {
			return false
		}

		aCount := make(map[model.Capability]int)
		for _, cap := range a {
			aCount[cap]++
		}

		bCount := make(map[model.Capability]int)
		for _, cap := range b {
			bCount[cap]++
		}

		for cap, count := range aCount {
			if bCount[cap] != count {
				return false
			}
		}

		return true
	}

	for _, tt := range testModels {
		t.Run(tt.name, func(t *testing.T) {
			// Test Capabilities method
			caps := tt.model.Capabilities()
			if !compareCapabilities(caps, tt.expectedCaps) {
				t.Errorf("Expected capabilities %v, got %v", tt.expectedCaps, caps)
			}
		})
	}
}

func TestModelCheckCapabilities(t *testing.T) {
	// Create simple model file for tests that don't depend on GGUF content
	completionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
	}, []*ggml.Tensor{})

	// Create vision model (llama architecture with vision block count)
	visionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	// Create embedding model (bert architecture with pooling type)
	embeddingModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, []*ggml.Tensor{})

	toolsInsertTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	chatTemplate, err := template.Parse("{{ .prompt }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	toolsTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	tests := []struct {
		name           string
		model          Model
		checkCaps      []model.Capability
		expectedErrMsg string
	}{
		{
			name: "completion model without tools capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityTools},
			expectedErrMsg: "does not support tools",
		},
		{
			name: "model with all needed capabilities",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsInsertTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model missing insert capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityInsert},
			expectedErrMsg: "does not support insert",
		},
		{
			name: "model missing vision capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityVision},
			expectedErrMsg: "does not support vision",
		},
		{
			name: "model with vision capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  chatTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityVision},
		},
		{
			name: "model with embedding capability",
			model: Model{
				ModelPath: embeddingModelPath,
				Template:  chatTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityEmbedding},
		},
		{
			name: "model missing speech capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilitySpeech},
			expectedErrMsg: "does not support speech",
		},
		{
			name: "model with speech capability in config",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"speech"},
				},
			},
			checkCaps: []model.Capability{model.CapabilitySpeech},
		},
		{
			name: "unknown capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{"unknown"},
			expectedErrMsg: "unknown capability",
		},
		{
			name: "model missing image generation capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityImage},
			expectedErrMsg: "does not support image generation",
		},
		{
			name: "model with image generation capability",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image"},
				},
			},
			checkCaps: []model.Capability{model.CapabilityImage},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test CheckCapabilities method
			err := tt.model.CheckCapabilities(tt.checkCaps...)
			if tt.expectedErrMsg == "" {
				if err != nil {
					t.Errorf("Expected no error, got: %v", err)
				}
			} else {
				if err == nil {
					t.Errorf("Expected error containing %q, got nil", tt.expectedErrMsg)
				} else if !strings.Contains(err.Error(), tt.expectedErrMsg) {
					t.Errorf("Expected error containing %q, got: %v", tt.expectedErrMsg, err)
				}
			}
		})
	}
}

func TestPullModelManifest(t *testing.T) {
	cases := []struct {
		name     string
		manifest string
	}{
		{
			name: "pretty printed",
			manifest: `{  "schemaVersion": 2,  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": { "digest": "sha256:abc", "mediaType": "application/vnd.docker.container.image.v1+json", "size": 50 },
  "layers": [{ "digest": "sha256:t1", "mediaType": "application/vnd.ollama.image.tensor", "size": 1024, "name": "model.weight" }]
}`,
		},
		{
			name:     "non-standard field order",
			manifest: `{"layers":[{"size":999,"digest":"sha256:def","mediaType":"application/vnd.ollama.image.model"}],"schemaVersion":2,"config":{"size":50,"digest":"sha256:abc","mediaType":"application/vnd.docker.container.image.v1+json"},"mediaType":"application/vnd.docker.distribution.manifest.v2+json"}`,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(tt.manifest))
			}))
			defer ts.Close()

			n := model.ParseName("test/model:latest")
			n.ProtocolScheme = "http"
			n.Host = strings.TrimPrefix(ts.URL, "http://")

			mf, data, err := pullModelManifest(t.Context(), n, &registryOptions{})
			if err != nil {
				t.Fatal(err)
			}

			// Raw bytes must be byte-for-byte identical to what the server sent
			if string(data) != tt.manifest {
				t.Fatalf("raw bytes differ from server response")
			}

			// SHA256 of returned data must match the expected registry digest
			expectedDigest := fmt.Sprintf("%x", sha256.Sum256([]byte(tt.manifest)))
			gotDigest := fmt.Sprintf("%x", sha256.Sum256(data))
			if gotDigest != expectedDigest {
				t.Fatalf("digest mismatch\ngot:  %s\nwant: %s", gotDigest, expectedDigest)
			}

			// Parsed manifest must still be usable
			if mf.SchemaVersion != 2 {
				t.Fatalf("schemaVersion = %d, want 2", mf.SchemaVersion)
			}
			if mf.Config.Digest == "" {
				t.Fatal("config digest is empty")
			}
			if len(mf.Layers) == 0 {
				t.Fatal("expected at least one layer")
			}
		})
	}
}

// TestPullModelDuplicateDigestVerifiesBlob pulls a manifest whose config and
// layer share a digest. The registry redirects blob downloads via Location to
// an "internal" path serving bytes that don't match the digest, so PullModel
// must reject the pull with errDigestMismatch.
func TestPullModelDuplicateDigestVerifiesBlob(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	const bogusDigest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

	backendURL := ""
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/manifests/"):
			w.Header().Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
			fmt.Fprintf(w, `{
				"schemaVersion": 2,
				"mediaType": "application/vnd.docker.distribution.manifest.v2+json",
				"config": {
					"mediaType": "application/vnd.ollama.image.config",
					"digest": %q,
					"size": 5
				},
				"layers": [{
					"mediaType": "application/vnd.ollama.image.model",
					"digest": %q,
					"size": 5
				}]
			}`, bogusDigest, bogusDigest)
		case strings.Contains(r.URL.Path, "/internal/blobs/"):
			w.Write([]byte("attacker-controlled-bytes"))
		case strings.Contains(r.URL.Path, "/blobs/"):
			w.Header().Set("Location", backendURL+"/internal"+r.URL.Path)
			w.WriteHeader(http.StatusOK)
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()
	backendURL = ts.URL

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	n := model.ParseName(u.Host + "/test/attack")
	n.ProtocolScheme = "http"

	err = PullModel(t.Context(), n.String(), &registryOptions{Insecure: true}, func(api.ProgressResponse) {})
	if !errors.Is(err, errDigestMismatch) {
		t.Fatalf("PullModel = %v, want errDigestMismatch (unverified blob would persist)", err)
	}
}

func TestTextSurfaceWrongModalityMessage(t *testing.T) {
	got := textSurfaceWrongModalityMessage(&Model{
		Config: model.ConfigV2{Capabilities: []string{string(model.CapabilityEmbedding)}},
	}, "bert", "chat")
	if !strings.Contains(got, "embedding model") || !strings.Contains(got, "/v1/embeddings") {
		t.Fatalf("got %q", got)
	}
	got = textSurfaceWrongModalityMessage(&Model{
		Config: model.ConfigV2{Capabilities: []string{string(model.CapabilityCompletion)}},
	}, "llama", "chat")
	if got != `"llama" does not support chat` {
		t.Fatalf("got %q", got)
	}
}

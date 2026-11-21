package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/weightedit"
	"github.com/ollama/ollama/x/weightedit/gpatch"
)

func TestWeightEditApplyLinked(t *testing.T) {
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)
	gin.SetMode(gin.TestMode)

	ckpt := filepath.Join(t.TempDir(), "ckpt")
	if err := os.MkdirAll(ckpt, 0o755); err != nil {
		t.Fatal(err)
	}
	writeTinyLinkedCheckpoint(t, ckpt)
	registerLinkedTag(t, "tiny-linked:latest", ckpt)

	store, err := weightedit.OpenDirectory(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	edit := buildTestEdit(t, store)
	var patch bytes.Buffer
	if err := gpatch.Write(&patch, edit); err != nil {
		t.Fatal(err)
	}
	// Refresh ID from write.
	decoded, _, err := weightedit.Decode(bytes.NewReader(patch.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	s := &Server{}
	r := gin.New()
	s.registerWeightEditRoutes(r)

	body, _ := json.Marshal(map[string]any{
		"model":       "tiny-linked:latest",
		"destination": "tiny-linked:patched",
		"patch":       patch.Bytes(),
	})
	req := httptest.NewRequest(http.MethodPost, "/api/weights/edits", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	var resp weightEditApplyResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.ChangedWeights != decoded.ChangedWeights() {
		t.Fatalf("changed=%d", resp.ChangedWeights)
	}

	dst, err := GetModel("tiny-linked:patched")
	if err != nil {
		t.Fatal(err)
	}
	if dst.Config.SourceDir == "" || dst.Config.SourceDir == ckpt {
		t.Fatalf("destination source_dir=%q", dst.Config.SourceDir)
	}
	if _, err := os.Stat(filepath.Join(dst.Config.SourceDir, "geodesic_applied.json")); err != nil {
		t.Fatal(err)
	}
}

func TestWeightEditRejectsMissingSourceDir(t *testing.T) {
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)
	gin.SetMode(gin.TestMode)

	cfgData, err := json.Marshal(model.ConfigV2{
		ModelFormat:  "safetensors",
		Capabilities: []string{"completion"},
		FileType:     "F16",
	})
	if err != nil {
		t.Fatal(err)
	}
	layer, err := manifest.NewLayer(bytes.NewReader(cfgData), "application/vnd.docker.container.image.v1+json")
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifest(model.ParseName("blob-only:latest"), layer, nil); err != nil {
		t.Fatal(err)
	}

	s := &Server{}
	r := gin.New()
	s.registerWeightEditRoutes(r)
	body, _ := json.Marshal(map[string]any{
		"model":       "blob-only:latest",
		"destination": "blob-only:patched",
		"patch":       []byte("not-a-real-patch"),
	})
	req := httptest.NewRequest(http.MethodPost, "/api/weights/edits", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestWeightEditInspect(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ckpt := filepath.Join(t.TempDir(), "ckpt")
	_ = os.MkdirAll(ckpt, 0o755)
	writeTinyLinkedCheckpoint(t, ckpt)
	store, err := weightedit.OpenDirectory(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	edit := buildTestEdit(t, store)
	var patch bytes.Buffer
	if err := gpatch.Write(&patch, edit); err != nil {
		t.Fatal(err)
	}

	s := &Server{}
	r := gin.New()
	s.registerWeightEditRoutes(r)
	req := httptest.NewRequest(http.MethodPost, "/api/weights/edits/inspect", bytes.NewReader(patch.Bytes()))
	req.Header.Set("Content-Type", "application/octet-stream")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}

func registerLinkedTag(t *testing.T, name, sourceDir string) {
	t.Helper()
	cfgData, err := json.Marshal(model.ConfigV2{
		ModelFormat:  "safetensors",
		Capabilities: []string{"completion"},
		FileType:     "linked",
		SourceDir:    sourceDir,
	})
	if err != nil {
		t.Fatal(err)
	}
	layer, err := manifest.NewLayer(bytes.NewReader(cfgData), "application/vnd.docker.container.image.v1+json")
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifest(model.ParseName(name), layer, nil); err != nil {
		t.Fatal(err)
	}
}

func writeTinyLinkedCheckpoint(t *testing.T, dir string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["TinyForCausalLM"],"model_type":"tiny"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"model_max_length":128}`), 0o644); err != nil {
		t.Fatal(err)
	}
	embed := make([]byte, 4*2*4)
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(embed[i*4:], math.Float32bits(float32(i)+0.5))
	}
	other := make([]byte, 2*2*4)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint32(other[i*4:], math.Float32bits(float32(10+i)))
	}
	header := map[string]any{
		"model.embed_tokens.weight": map[string]any{
			"dtype": "F32", "shape": []int{4, 2}, "data_offsets": []int{0, len(embed)},
		},
		"other.weight": map[string]any{
			"dtype": "F32", "shape": []int{2, 2}, "data_offsets": []int{len(embed), len(embed) + len(other)},
		},
	}
	headerJSON, _ := json.Marshal(header)
	pad := (8 - len(headerJSON)%8) % 8
	headerJSON = append(headerJSON, bytes.Repeat([]byte{' '}, pad)...)
	var sizeBuf [8]byte
	binary.LittleEndian.PutUint64(sizeBuf[:], uint64(len(headerJSON)))
	var buf bytes.Buffer
	buf.Write(sizeBuf[:])
	buf.Write(headerJSON)
	buf.Write(embed)
	buf.Write(other)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func buildTestEdit(t *testing.T, store *weightedit.DirectoryStore) *weightedit.Edit {
	t.Helper()
	name := "model.embed_tokens.weight"
	info, err := store.Info(name)
	if err != nil {
		t.Fatal(err)
	}
	srcHash, err := store.TensorHash(name)
	if err != nil {
		t.Fatal(err)
	}
	rc, err := store.OpenTensor(name)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		t.Fatal(err)
	}
	const size = 4
	indices := []uint64{2, 3}
	old := make([]byte, len(indices)*size)
	neu := make([]byte, len(indices)*size)
	for i, idx := range indices {
		copy(old[i*size:(i+1)*size], raw[int(idx)*size:])
		binary.LittleEndian.PutUint32(neu[i*size:], math.Float32bits(float32(100+i)))
	}
	resultRaw := append([]byte(nil), raw...)
	for i, idx := range indices {
		copy(resultRaw[int(idx)*size:], neu[i*size:(i+1)*size])
	}
	sum := sha256Sum(resultRaw)

	layout, _ := store.LayoutDigest()
	fp, hashes, _ := store.Fingerprint()
	cfg, tok, _ := store.SideFiles()
	resultHashes := map[string]string{}
	for k, v := range hashes {
		resultHashes[k] = v
	}
	resultHashes[name] = sum
	names, _ := store.TensorNames()
	infos := make([]weightedit.TensorInfo, 0, len(names))
	for _, n := range names {
		info, _ := store.Info(n)
		infos = append(infos, info)
	}
	resultFP := fingerprintLinesLocal(infos, resultHashes)

	return &weightedit.Edit{
		Name: "http-test",
		Source: weightedit.Identity{
			LayoutSHA256: layout,
			Fingerprint:  fp,
			NTensors:     len(names),
			Config:       cfg,
			Tokenizer:    tok,
		},
		Result: weightedit.Identity{Fingerprint: resultFP},
		Changes: []weightedit.Change{{
			Name: name, Dtype: info.Dtype, Shape: info.Shape,
			Indices: indices, Old: old, New: neu,
			SourceSHA256: srcHash, ResultSHA256: sum,
		}},
	}
}

func sha256Sum(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func fingerprintLinesLocal(infos []weightedit.TensorInfo, hashes map[string]string) string {
	var lines string
	for i, info := range infos {
		shape := ""
		for j, d := range info.Shape {
			if j > 0 {
				shape += ","
			}
			shape += fmt.Sprintf("%d", d)
		}
		if i > 0 {
			lines += "\n"
		}
		lines += info.Name + "\t" + info.Dtype + "\t" + shape + "\t" + hashes[info.Name]
	}
	return sha256Sum([]byte(lines))
}

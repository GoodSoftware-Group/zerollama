package weightedit_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/weightedit"
	"github.com/ollama/ollama/x/weightedit/gpatch"
)

func TestDirectoryApplyRestoreRoundTrip(t *testing.T) {
	src := t.TempDir()
	writeTinyCheckpoint(t, src)

	store, err := weightedit.OpenDirectory(src)
	if err != nil {
		t.Fatal(err)
	}
	edit := buildEmbeddingEdit(t, store)

	var patch bytes.Buffer
	if err := gpatch.Write(&patch, edit); err != nil {
		t.Fatal(err)
	}

	decoded, codec, err := weightedit.Decode(bytes.NewReader(patch.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if codec.Format() != "geodesic-patch" {
		t.Fatalf("format=%s", codec.Format())
	}
	if decoded.ID != edit.ID {
		t.Fatalf("id=%s want %s", decoded.ID, edit.ID)
	}

	out := filepath.Join(t.TempDir(), "patched")
	bound, err := weightedit.Bind(store, decoded, "source")
	if err != nil {
		t.Fatal(err)
	}
	res, err := weightedit.Apply(bound, out)
	if err != nil {
		t.Fatal(err)
	}
	if res.ChangedWeights != 2 {
		t.Fatalf("changed=%d", res.ChangedWeights)
	}

	srcBefore := fileSHA(t, filepath.Join(src, "model.safetensors"))
	if fileSHA(t, filepath.Join(src, "model.safetensors")) != srcBefore {
		t.Fatal("source shard mutated")
	}

	srcInfo, err := os.Stat(filepath.Join(src, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	dstInfo, err := os.Stat(filepath.Join(out, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if os.SameFile(srcInfo, dstInfo) {
		t.Fatal("touched shard shares inode with source")
	}

	if _, err := os.Stat(filepath.Join(out, "geodesic_applied.json")); err != nil {
		t.Fatal("missing provenance")
	}

	patched, err := weightedit.OpenDirectory(out)
	if err != nil {
		t.Fatal(err)
	}
	restored := filepath.Join(t.TempDir(), "restored")
	if _, err := weightedit.Restore(patched, decoded, restored); err != nil {
		t.Fatal(err)
	}
	if fileSHA(t, filepath.Join(restored, "model.safetensors")) != srcBefore {
		t.Fatal("restore did not return source bytes")
	}
}

func TestRestoreRejectsMutatedTokenizer(t *testing.T) {
	src := t.TempDir()
	writeTinyCheckpoint(t, src)
	store, err := weightedit.OpenDirectory(src)
	if err != nil {
		t.Fatal(err)
	}
	edit := buildEmbeddingEdit(t, store)

	out := filepath.Join(t.TempDir(), "patched")
	bound, err := weightedit.Bind(store, edit, "source")
	if err != nil {
		t.Fatal(err)
	}
	res, err := weightedit.Apply(bound, out)
	if err != nil {
		t.Fatal(err)
	}
	if !res.FingerprintOK || res.OutputFingerprint == "" {
		t.Fatalf("expected measured output fingerprint, got ok=%v fp=%q", res.FingerprintOK, res.OutputFingerprint)
	}
	outStore, err := weightedit.OpenDirectory(out)
	if err != nil {
		t.Fatal(err)
	}
	gotFP, _, err := outStore.Fingerprint()
	if err != nil {
		t.Fatal(err)
	}
	if gotFP != res.OutputFingerprint {
		t.Fatal("ApplyResult fingerprint does not match rehash of output directory")
	}

	// Break the hardlink and rewrite the tokenizer so weights still match result
	// but source sidecar claims no longer hold.
	tok := filepath.Join(out, "tokenizer_config.json")
	if err := os.Remove(tok); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(tok, []byte(`{"model_max_length":999}`), 0o644); err != nil {
		t.Fatal(err)
	}
	patched, err := weightedit.OpenDirectory(out)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := weightedit.Bind(patched, edit, "result"); err == nil {
		t.Fatal("expected restore Bind to refuse mutated tokenizer")
	}
}

func TestInvalidPatchWrapped(t *testing.T) {
	src := t.TempDir()
	writeTinyCheckpoint(t, src)
	store, err := weightedit.OpenDirectory(src)
	if err != nil {
		t.Fatal(err)
	}
	edit := buildEmbeddingEdit(t, store)
	var patch bytes.Buffer
	if err := gpatch.Write(&patch, edit); err != nil {
		t.Fatal(err)
	}
	data := append([]byte(nil), patch.Bytes()...)
	// Corrupt a byte inside the JSON header so Detect still sees geodesic-patch
	// metadata but Decode fails the patch_id check.
	headerSize := binary.LittleEndian.Uint64(data[:8])
	if headerSize < 32 || int(headerSize)+8 > len(data) {
		t.Fatalf("unexpected header size %d", headerSize)
	}
	data[8+int(headerSize)/2] ^= 0x01
	_, _, err = weightedit.Decode(bytes.NewReader(data))
	if err == nil {
		t.Fatal("expected decode failure")
	}
	if !errors.Is(err, weightedit.ErrInvalidPatch) && !errors.Is(err, weightedit.ErrUnknownFormat) {
		t.Fatalf("want ErrInvalidPatch or ErrUnknownFormat, got %v", err)
	}
}

func TestBindRejectsWrongCheckpoint(t *testing.T) {
	src := t.TempDir()
	writeTinyCheckpoint(t, src)
	store, err := weightedit.OpenDirectory(src)
	if err != nil {
		t.Fatal(err)
	}
	edit := buildEmbeddingEdit(t, store)

	other := t.TempDir()
	writeTinyCheckpoint(t, other)
	st := filepath.Join(other, "model.safetensors")
	raw, err := os.ReadFile(st)
	if err != nil {
		t.Fatal(err)
	}
	raw[len(raw)-1] ^= 0xff
	if err := os.WriteFile(st, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	otherStore, err := weightedit.OpenDirectory(other)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := weightedit.Bind(otherStore, edit, "source"); err == nil {
		t.Fatal("expected incompatible checkpoint")
	}
}

func TestRejectBadPatchID(t *testing.T) {
	src := t.TempDir()
	writeTinyCheckpoint(t, src)
	store, err := weightedit.OpenDirectory(src)
	if err != nil {
		t.Fatal(err)
	}
	edit := buildEmbeddingEdit(t, store)
	var patch bytes.Buffer
	if err := gpatch.Write(&patch, edit); err != nil {
		t.Fatal(err)
	}
	data := append([]byte(nil), patch.Bytes()...)
	if len(data) < 64 {
		t.Fatal("patch too small")
	}
	data[20] ^= 0x01
	if _, _, err := weightedit.Decode(bytes.NewReader(data)); err == nil {
		t.Fatal("expected decode failure on corrupted patch")
	}
}

func TestRejectOutInsideSource(t *testing.T) {
	src := t.TempDir()
	writeTinyCheckpoint(t, src)
	store, err := weightedit.OpenDirectory(src)
	if err != nil {
		t.Fatal(err)
	}
	edit := buildEmbeddingEdit(t, store)
	bound, err := weightedit.Bind(store, edit, "source")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := weightedit.Apply(bound, filepath.Join(src, "nested")); err == nil {
		t.Fatal("expected out-inside-source rejection")
	}
}

func TestRejectIndexPastNumel(t *testing.T) {
	c := weightedit.Change{
		Name:    "t",
		Dtype:   "F32",
		Shape:   []int64{2, 2},
		Indices: []uint64{0, 4},
		Old:     make([]byte, 8),
		New:     make([]byte, 8),
	}
	if err := weightedit.ValidateChange(c); err == nil {
		t.Fatal("expected index past numel")
	}
}

func TestUnknownFormat(t *testing.T) {
	if _, _, err := weightedit.Decode(bytes.NewReader([]byte("not-a-patch"))); err == nil {
		t.Fatal("expected unknown format")
	}
}

func TestTestCodecDoesNotChangeApply(t *testing.T) {
	src := t.TempDir()
	writeTinyCheckpoint(t, src)
	store, err := weightedit.OpenDirectory(src)
	if err != nil {
		t.Fatal(err)
	}
	edit := buildEmbeddingEdit(t, store)

	weightedit.RegisterCodec(&inlineCodec{edit: edit})
	raw := []byte("INLINE_TEST_CODEC_v1")
	decoded, codec, err := weightedit.Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if codec.Format() != "inline-test" {
		t.Fatalf("codec=%s", codec.Format())
	}
	out := filepath.Join(t.TempDir(), "from-inline")
	bound, err := weightedit.Bind(store, decoded, "source")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := weightedit.Apply(bound, out); err != nil {
		t.Fatal(err)
	}
}

type inlineCodec struct {
	edit *weightedit.Edit
}

func (c *inlineCodec) Format() string { return "inline-test" }
func (c *inlineCodec) Detect(prefix []byte) bool {
	return bytes.HasPrefix(prefix, []byte("INLINE_TEST_CODEC_v1"))
}
func (c *inlineCodec) Decode(r io.Reader) (*weightedit.Edit, error) {
	_, _ = io.ReadAll(r)
	cp := *c.edit
	cp.Changes = append([]weightedit.Change(nil), c.edit.Changes...)
	return &cp, nil
}

func writeTinyCheckpoint(t *testing.T, dir string) {
	t.Helper()
	cfg := []byte(`{"architectures":["TinyForCausalLM"],"model_type":"tiny"}`)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfg, 0o644); err != nil {
		t.Fatal(err)
	}
	tok := []byte(`{"model_max_length":128}`)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), tok, 0o644); err != nil {
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
			"dtype":        "F32",
			"shape":        []int{4, 2},
			"data_offsets": []int{0, len(embed)},
		},
		"other.weight": map[string]any{
			"dtype":        "F32",
			"shape":        []int{2, 2},
			"data_offsets": []int{len(embed), len(embed) + len(other)},
		},
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
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

func buildEmbeddingEdit(t *testing.T, store *weightedit.DirectoryStore) *weightedit.Edit {
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
	sum := sha256.Sum256(resultRaw)
	resultHash := hex.EncodeToString(sum[:])

	layout, err := store.LayoutDigest()
	if err != nil {
		t.Fatal(err)
	}
	fp, hashes, err := store.Fingerprint()
	if err != nil {
		t.Fatal(err)
	}
	cfg, tok, err := store.SideFiles()
	if err != nil {
		t.Fatal(err)
	}
	resultHashes := map[string]string{}
	for k, v := range hashes {
		resultHashes[k] = v
	}
	resultHashes[name] = resultHash
	names, _ := store.TensorNames()
	infos := make([]weightedit.TensorInfo, 0, len(names))
	for _, n := range names {
		info, _ := store.Info(n)
		infos = append(infos, info)
	}
	resultFP := fingerprintOf(infos, resultHashes)

	edit := &weightedit.Edit{
		Name: "test-embed-row",
		Source: weightedit.Identity{
			LayoutSHA256: layout,
			Fingerprint:  fp,
			NTensors:     len(names),
			Config:       cfg,
			Tokenizer:    tok,
		},
		Result: weightedit.Identity{
			Fingerprint: resultFP,
		},
		Changes: []weightedit.Change{{
			Name:         name,
			Dtype:        info.Dtype,
			Shape:        info.Shape,
			Indices:      indices,
			Old:          old,
			New:          neu,
			SourceSHA256: srcHash,
			ResultSHA256: resultHash,
		}},
	}
	var buf bytes.Buffer
	if err := gpatch.Write(&buf, edit); err != nil {
		t.Fatal(err)
	}
	decoded, _, err := weightedit.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	return decoded
}

func fingerprintOf(infos []weightedit.TensorInfo, hashes map[string]string) string {
	lines := make([]string, 0, len(infos))
	for _, info := range infos {
		shape := ""
		for i, d := range info.Shape {
			if i > 0 {
				shape += ","
			}
			shape += fmt.Sprintf("%d", d)
		}
		lines = append(lines, info.Name+"\t"+info.Dtype+"\t"+shape+"\t"+hashes[info.Name])
	}
	joined := ""
	for i, l := range lines {
		if i > 0 {
			joined += "\n"
		}
		joined += l
	}
	sum := sha256.Sum256([]byte(joined))
	return hex.EncodeToString(sum[:])
}

func fileSHA(t *testing.T, path string) [32]byte {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return sha256.Sum256(raw)
}

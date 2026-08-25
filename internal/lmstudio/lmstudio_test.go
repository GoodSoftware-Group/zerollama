package lmstudio

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestMatchDir_gemma(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "gemma-4-31B-it-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "gemma-4-31B-it-Q8_0.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	n := model.ParseName("gemma4:31b")
	got, ok := MatchDir(n)
	if !ok {
		t.Fatal("expected match")
	}
	if got != modelDir {
		t.Fatalf("got %q want %q", got, modelDir)
	}
}

func TestMatchDir_shardedGGUF(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "Qwen", "Qwen2.5-Coder-32B-Instruct-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"qwen2.5-coder-32b-instruct-fp16-00001-of-00009.gguf",
		"qwen2.5-coder-32b-instruct-fp16-00002-of-00009.gguf",
	} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	n := model.ParseName("qwen2.5-coder:32b")
	got, ok := MatchDir(n)
	if !ok {
		t.Fatal("expected sharded GGUF layout to match")
	}
	if got != modelDir {
		t.Fatalf("got %q want %q", got, modelDir)
	}
}

func TestMatchDir_safetensors(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "Hermes-4-70B-MLX-8bit")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"model-00001-of-00002.safetensors",
		"model-00002-of-00002.safetensors",
		"config.json",
	} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	n := model.ParseName("lmstudio-community/hermes-4-70b:8bit")
	got, ok := MatchDir(n)
	if !ok {
		t.Fatal("expected safetensors layout to match suggested name")
	}
	if got != modelDir {
		t.Fatalf("got %q want %q", got, modelDir)
	}
}

func TestMatchDir_ambiguous(t *testing.T) {
	root := t.TempDir()
	for _, sub := range []string{"a/gemma-4-31B-it-GGUF", "b/gemma-4-31B-it-GGUF"} {
		modelDir := filepath.Join(root, sub)
		if err := os.MkdirAll(modelDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(modelDir, "x.gguf"), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	n := model.ParseName("gemma4:31b")
	_, ok := MatchDir(n)
	if ok {
		t.Fatal("expected ambiguous match to be rejected")
	}
}

func TestDirLooksLikeLMStudioModel_mmproj(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "mmproj-model.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !dirLooksLikeLMStudioModel(dir) {
		t.Fatal("expected model + mmproj to be accepted")
	}
}

func TestSuggestedName(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "gemma-4-31B-it-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "gemma-4-31B-it-Q8_0.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	got := SuggestedName(root, modelDir)
	want := "lmstudio-community/gemma-4-31b-it:q8_0"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestList(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "Qwen", "Qwen3.6-27B-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "Qwen3.6-27B-Q8_0.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	entries := List()
	if len(entries) != 1 {
		t.Fatalf("len=%d want 1", len(entries))
	}
	if entries[0].Format != "gguf" {
		t.Fatalf("format=%q want gguf", entries[0].Format)
	}
	if entries[0].Name != "qwen/qwen3.6-27b:q8_0" {
		t.Fatalf("name=%q", entries[0].Name)
	}
}

func TestMatchDir_multiQuantGGUF(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "driaforall", "Tiny-Agent-a-0.5B")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"dria-agent-a-0.5b.Q4_K_M.gguf",
		"dria-agent-a-0.5b.Q8_0.gguf",
	} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	entries := List()
	if len(entries) != 2 {
		t.Fatalf("len=%d want 2 quant variants", len(entries))
	}

	n := model.ParseName("driaforall/tiny-agent-a-0.5b:q8_0")
	dir, weight, ok := MatchSelection(n)
	if !ok || dir != modelDir || weight != "dria-agent-a-0.5b.Q8_0.gguf" {
		t.Fatalf("match=%v dir=%q weight=%q", ok, dir, weight)
	}
}

func TestListRealLMStudioCache(t *testing.T) {
	cache := "/Users/user1/.lmstudio/models"
	if st, err := os.Stat(cache); err != nil || !st.IsDir() {
		t.Skip("real LM Studio cache not present")
	}
	t.Setenv("OLLAMA_LMSTUDIO_MODELS", cache)

	entries := List()
	if len(entries) == 0 {
		t.Fatal("expected at least one discoverable LM Studio model")
	}

	// Every publisher/model dir or symlink-to-dir with weights must appear.
	wantDirs := map[string]bool{}
	publishers, err := os.ReadDir(cache)
	if err != nil {
		t.Fatal(err)
	}
	for _, pub := range publishers {
		if !pub.IsDir() {
			continue
		}
		pubPath := filepath.Join(cache, pub.Name())
		models, err := os.ReadDir(pubPath)
		if err != nil {
			continue
		}
		for _, m := range models {
			modelPath := filepath.Join(pubPath, m.Name())
			info, err := m.Info()
			if err != nil {
				continue
			}
			isDir := m.IsDir()
			if !isDir && info.Mode()&os.ModeSymlink != 0 {
				if st, err := os.Stat(modelPath); err == nil && st.IsDir() {
					isDir = true
				}
			}
			if !isDir {
				continue
			}
			scan := modelPath
			if resolved, err := filepath.EvalSymlinks(modelPath); err == nil && resolved != "" {
				scan = resolved
			}
			if _, ok := dirModelFormat(scan); !ok {
				continue
			}
			wantDirs[modelPath] = true
		}
	}
	if len(wantDirs) == 0 {
		t.Skip("no weighted model dirs in real LM Studio cache")
	}

	gotDirs := map[string]bool{}
	formats := map[string]int{}
	for _, e := range entries {
		formats[e.Format]++
		if e.Name == "" || e.Dir == "" {
			t.Fatalf("incomplete entry: %+v", e)
		}
		gotDirs[e.Dir] = true
	}
	for dir := range wantDirs {
		if !gotDirs[dir] {
			t.Fatalf("missing discovery for weighted model dir %q (got %d entries)", dir, len(entries))
		}
	}
	if formats["gguf"] == 0 && formats["safetensors"] == 0 {
		t.Fatal("expected gguf or safetensors entries")
	}
}

func TestList_symlinkSafetensors(t *testing.T) {
	root := t.TempDir()
	realDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(realDir, "config.json"), []byte(`{"model_type":"lfm2"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	// Non-model* name exercises the broad *.safetensors glob.
	if err := os.WriteFile(filepath.Join(realDir, "weights.safetensors"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	linkDir := filepath.Join(root, "mlx-community")
	if err := os.MkdirAll(linkDir, 0o755); err != nil {
		t.Fatal(err)
	}
	linkPath := filepath.Join(linkDir, "LFM2-350M-4bit")
	if err := os.Symlink(realDir, linkPath); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	entries := List()
	if len(entries) != 1 {
		t.Fatalf("len=%d want 1 (symlink MLX tree)", len(entries))
	}
	if entries[0].Format != "safetensors" {
		t.Fatalf("format=%q want safetensors", entries[0].Format)
	}
	if entries[0].Dir != linkPath {
		t.Fatalf("Dir=%q want LM Studio symlink path %q", entries[0].Dir, linkPath)
	}
	wantName := "mlx-community/lfm2-350m:4bit"
	if entries[0].Name != wantName {
		t.Fatalf("Name=%q want %q", entries[0].Name, wantName)
	}

	n := model.ParseName(wantName)
	dir, _, ok := MatchSelection(n)
	if !ok || dir != linkPath {
		t.Fatalf("MatchSelection ok=%v dir=%q want %q", ok, dir, linkPath)
	}
}

func TestSafetensorsWeightFiles_genericName(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "weights.safetensors"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	weights, ok := safetensorsWeightFiles(dir)
	if !ok || len(weights) != 1 {
		t.Fatalf("ok=%v len=%d want 1", ok, len(weights))
	}
	if filepath.Base(weights[0]) != "weights.safetensors" {
		t.Fatalf("got %q", weights[0])
	}
}

// makeMLXDir creates a temporary directory that looks like an MLX safetensors
// model (config.json + one .safetensors weight file).
func makeMLXDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte(""), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestImportCopyBytes(t *testing.T) {
	// GGUF: always zero.
	gguf := Entry{Format: "gguf", Size: 30 << 30}
	if got := ImportCopyBytes(gguf); got != 0 {
		t.Fatalf("gguf copy bytes = %d, want 0", got)
	}

	// Legacy safetensors without config.json: symlinked, so zero.
	legacyDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(legacyDir, "model.safetensors"), []byte(""), 0o644); err != nil {
		t.Fatal(err)
	}
	legacy := Entry{Format: "safetensors", Size: 30 << 30, Dir: legacyDir}
	if got := ImportCopyBytes(legacy); got != 0 {
		t.Fatalf("legacy safetensors (no config.json) copy bytes = %d, want 0", got)
	}

	// MLX safetensors with config.json: full copy.
	mlxDir := makeMLXDir(t)
	st := Entry{Format: "safetensors", Size: 30 << 30, Dir: mlxDir}
	want := int64(30<<30) + ImportHeadroomBytes
	if got := ImportCopyBytes(st); got != want {
		t.Fatalf("mlx safetensors copy bytes = %d, want %d", got, want)
	}
}

func TestDirImportCopyBytes(t *testing.T) {
	legacyDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(legacyDir, "model.safetensors"), []byte(""), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := DirImportCopyBytes(legacyDir); got != 0 {
		t.Fatalf("legacy safetensors copy bytes = %d, want 0", got)
	}

	mlxDir := makeMLXDir(t)
	if got := DirImportCopyBytes(mlxDir); got <= ImportHeadroomBytes {
		t.Fatalf("mlx safetensors copy bytes = %d, want > headroom", got)
	}
}

func TestHasDiskForImport_Safetensors(t *testing.T) {
	mlxDir := makeMLXDir(t)

	orig := modelsFreeBytes
	t.Cleanup(func() { modelsFreeBytes = orig })
	modelsFreeBytes = func(string) (int64, error) { return 1 << 30, nil }

	e := Entry{Format: "safetensors", Size: 80 << 30, Dir: mlxDir}
	ok, free, need, err := HasDiskForImport(e)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatalf("expected insufficient disk: free=%d need=%d", free, need)
	}

	modelsFreeBytes = func(string) (int64, error) { return 100 << 30, nil }
	ok, _, need, err = HasDiskForImport(e)
	if err != nil || !ok {
		t.Fatalf("expected sufficient disk: ok=%v err=%v need=%d", ok, err, need)
	}
}

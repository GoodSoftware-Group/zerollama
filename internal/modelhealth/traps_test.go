package modelhealth

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestTrap21NoGenerationConfig(t *testing.T) {
	r := trap21NoGenerationConfig("m:latest", "", nil)
	if r.Status != StatusRepairable {
		t.Fatalf("empty params: status=%s want repairable", r.Status)
	}

	r = trap21NoGenerationConfig("m:latest", "/tmp/params", map[string]any{"num_ctx": 4096, "stop": []string{"</s>"}})
	if r.Status != StatusRepairable {
		t.Fatalf("no sampling keys: status=%s want repairable", r.Status)
	}

	r = trap21NoGenerationConfig("m:latest", "/tmp/params", map[string]any{"temperature": 0.7})
	if r.Status != StatusOK {
		t.Fatalf("with temperature: status=%s want ok", r.Status)
	}
}

func TestTrap10QuantLabel(t *testing.T) {
	r := trap10QuantLabel("llama3:q4_k_m", "Q4_K_M", "Q4_K_M", "Q4_K_M")
	if r.Status != StatusOK {
		t.Fatalf("match: %s %s", r.Status, r.Detail)
	}

	r = trap10QuantLabel("llama3:q8_0", "Q8_0", "Q4_K_M", "Q4_K_M")
	if r.Status != StatusRepairable {
		t.Fatalf("mismatch: status=%s want repairable (%s)", r.Status, r.Detail)
	}

	r = trap10QuantLabel("llama3:latest", "", "Q4_K_M", "Q8_0")
	if r.Status != StatusRepairable {
		t.Fatalf("config vs gguf: status=%s want repairable", r.Status)
	}
}

func TestTrap56NoChatTemplate(t *testing.T) {
	r := trap56NoChatTemplate("m", true, "", "")
	if r.Status != StatusOK {
		t.Fatal("go template should be ok")
	}

	r = trap56NoChatTemplate("m", false, "{{ messages }}", "")
	if r.Status != StatusOK {
		t.Fatal("jinja should be ok")
	}

	r = trap56NoChatTemplate("m", false, "def apply_chat_template(messages):\n  raise Exception('no')\n", "")
	if r.Status != StatusRepairable {
		t.Fatalf("python-only: status=%s", r.Status)
	}

	r = trap56NoChatTemplate("m", false, "", "")
	if r.Status != StatusRepairable {
		t.Fatalf("missing: status=%s", r.Status)
	}

	r = trap56NoChatTemplate("m", false, "", "qwen35")
	if r.Status != StatusOK {
		t.Fatal("renderer should be ok")
	}
}

func TestTrap55ContextMismatch(t *testing.T) {
	r := trap55ContextMismatch("m", 131072, 4096, 131072)
	if r.Status != StatusRepairable {
		t.Fatalf("advertised vs served: status=%s detail=%s", r.Status, r.Detail)
	}

	r = trap55ContextMismatch("m", 8192, 8192, 8192)
	if r.Status != StatusOK {
		t.Fatalf("aligned: status=%s", r.Status)
	}

	r = trap55ContextMismatch("m", 0, 32768, 8192)
	if r.Status != StatusRepairable {
		t.Fatalf("served > trained: status=%s", r.Status)
	}
}

func TestNTagQuant(t *testing.T) {
	cases := map[string]string{
		"llama3.2:q4_k_m":   "Q4_K_M",
		"foo:Q8_0":          "Q8_0",
		"bar:latest":        "",
		"model-iq4_xs-chat": "IQ4_XS",
		"lib/gemma:f16":     "F16",
	}
	for in, want := range cases {
		if got := nTagQuant(in); got != want {
			t.Errorf("nTagQuant(%q)=%q want %q", in, got, want)
		}
	}
}

func TestQuantEqual(t *testing.T) {
	if !quantEqual("Q4_K_M", "q4_k_m") {
		t.Fatal("case fold")
	}
	if !quantEqual("Q4_K", "Q4_K_M") {
		t.Fatal("Q4_K alias")
	}
	if quantEqual("Q8_0", "Q4_K_M") {
		t.Fatal("should differ")
	}
}

func TestDivergeContext(t *testing.T) {
	if !divergeContext(4096, 131072) {
		t.Fatal("expected diverge")
	}
	if divergeContext(4096, 8192) {
		t.Fatal("exact 2x should not diverge (requires hi > lo*2)")
	}
	if !divergeContext(4096, 8193) {
		t.Fatal("8193 should diverge from 4096")
	}
}

func TestCheckConfigTrapsInWithFixture(t *testing.T) {
	modelsDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", modelsDir)

	n := model.ParseName("registry.ollama.ai/library/traptest:q8_0")
	cfgDigest := "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	modelDigest := "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	paramsDigest := "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	tmplDigest := "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"

	cfg := typesModelConfig(map[string]any{
		"model_format":   "gguf",
		"model_family":   "llama",
		"file_type":      "Q4_K_M", // deliberate mismatch with tag q8_0
		"context_length": 131072,
		"architecture":   "amd64",
		"os":             "linux",
		"rootfs":         map[string]any{"type": "layers", "diff_ids": []string{}},
	})
	writeBlob(t, modelsDir, cfgDigest, cfg)
	// Not a real GGUF — looksLikeGGUF will fail; config-only path still runs.
	writeBlob(t, modelsDir, modelDigest, []byte("not-gguf"))
	writeBlob(t, modelsDir, paramsDigest, []byte(`{"num_ctx":4096}`))
	writeBlob(t, modelsDir, tmplDigest, []byte(`{{ .Prompt }}`))

	if err := manifest.WriteManifest(n, manifest.Layer{
		MediaType: "application/vnd.ollama.image.model",
		Digest:    cfgDigest,
		Size:      int64(len(cfg)),
	}, []manifest.Layer{
		{MediaType: "application/vnd.ollama.image.model", Digest: modelDigest, Size: 8},
		{MediaType: "application/vnd.ollama.image.params", Digest: paramsDigest, Size: 16},
		{MediaType: "application/vnd.ollama.image.template", Digest: tmplDigest, Size: 14},
	}); err != nil {
		t.Fatal(err)
	}

	mf, err := manifest.ParseNamedManifestIn(modelsDir, n)
	if err != nil {
		t.Fatal(err)
	}
	reports := checkConfigTrapsIn(modelsDir, "traptest:q8_0", mf)
	if len(reports) != 4 {
		t.Fatalf("got %d reports, want 4", len(reports))
	}

	byTrap := map[string]Report{}
	for _, r := range reports {
		byTrap[r.Name] = r
	}

	q10 := byTrap["model traptest:q8_0 trap-10 (quant label)"]
	if q10.Status != StatusRepairable {
		t.Fatalf("trap-10: want repairable for tag/config mismatch, got %s (%s)", q10.Status, q10.Detail)
	}

	q21 := byTrap["model traptest:q8_0 trap-21 (generation defaults)"]
	if q21.Status != StatusRepairable {
		t.Fatalf("trap-21: want repairable (no sampling keys), got %s", q21.Status)
	}

	q56 := byTrap["model traptest:q8_0 trap-56 (chat template)"]
	if q56.Status != StatusOK {
		t.Fatalf("trap-56: template present, got %s", q56.Status)
	}

	q55 := byTrap["model traptest:q8_0 trap-55/61 (context)"]
	if q55.Status != StatusRepairable {
		t.Fatalf("trap-55: advertised 131072 vs served 4096, got %s (%s)", q55.Status, q55.Detail)
	}
}

func typesModelConfig(v map[string]any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

func writeBlob(t *testing.T, modelsDir, digest string, data []byte) {
	t.Helper()
	path, err := manifest.BlobsPathIn(modelsDir, digest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

//go:build darwin && uma

package uma_test

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/uma"
)

// TestOptiqTokenTailPrefillRematch — post-norm prefill_hidden → GEMV→ARGMAX
// rematches got_gen[0]=12675 (lab dump is post-norm; recipe=gemv_argmax).
// Live serve path uses SkipFinalNorm + full NORM→GEMV→ARGMAX (F0700 / m32).
func TestOptiqTokenTailPrefillRematch(t *testing.T) {
	if os.Getenv("ZEROLLAMA_UMA_OPTIQ_TOKEN_SMOKE") == "" {
		t.Skip("set ZEROLLAMA_UMA_OPTIQ_TOKEN_SMOKE=1")
	}
	dump := os.Getenv("UMA_OPTIQ_TOKEN_TAIL_DIR")
	if dump == "" {
		dump = "/tmp/uma_optiq_token_tail_dump"
	}
	if _, err := os.Stat(filepath.Join(dump, "lm_q8.bin")); err != nil {
		t.Skip("no token-tail dump (make optiq-token-tail-dump)")
	}
	genDir := os.Getenv("ORNITH_OPTIQ_GENERATE_DIR")
	if genDir == "" {
		genDir = "/tmp/uma_optiq_generate_dump"
	}
	hb, err := os.ReadFile(filepath.Join(genDir, "prefill_hidden.bin"))
	if err != nil {
		t.Skipf("no prefill_hidden at %s (run ornith_generate_parity): %v", genDir, err)
	}
	wantTok := int32(12675)
	if mb, err := os.ReadFile(filepath.Join(genDir, "meta.json")); err == nil {
		var meta struct {
			GotGen []int32 `json:"got_gen"`
		}
		if json.Unmarshal(mb, &meta) == nil && len(meta.GotGen) > 0 {
			wantTok = meta.GotGen[0]
		}
	}

	_ = os.Setenv("ZEROLLAMA_UMA_SCHED", "require")
	_ = os.Setenv("ZEROLLAMA_UMA_OPTIQ_TOKEN_TAIL", "require")
	_ = os.Setenv("ZEROLLAMA_UMA_OPTIQ_TOKEN_RECIPE", "gemv_argmax")
	_ = os.Setenv("UMA_OPTIQ_TOKEN_TAIL_DIR", dump)
	_ = os.Setenv("UMA_JOB_NAME", "xuma-optiq-token")

	uma.ResetOptiqTokenTailForTest()
	if err := uma.Acquire(); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer uma.Release()

	n := len(hb) / 4
	x := make([]float32, n)
	for i := 0; i < n; i++ {
		x[i] = math.Float32frombits(binary.LittleEndian.Uint32(hb[i*4:]))
	}
	tok, err := uma.OptiqTokenTailArgmax(x)
	if err != nil {
		t.Fatalf("OptiqTokenTailArgmax: %v", err)
	}
	if tok != wantTok {
		t.Fatalf("token-tail tok=%d want=%d", tok, wantTok)
	}
	t.Logf("PASS: GRAPH token-tail rematch tok=%d (post-norm dump / gemv_argmax)", tok)
}

package discover

import (
	"encoding/json"
	"testing"
)

func TestParseDflashStatistics(t *testing.T) {
	log := `statistics dflash: #calls(b,g,a) = 1 3 3, #gen drafts = 12, #acc drafts = 4, #gen tokens = 48, #acc tokens = 12, dur(b,g,a) = 1.0, 2.0, 3.0 ms`
	g, a, gt, at, ok := ParseDflashStatisticsFromLog(log)
	if !ok {
		t.Fatal("expected match")
	}
	if g != 12 || a != 4 || gt != 48 || at != 12 {
		t.Fatalf("got drafts=%d/%d tokens=%d/%d", g, a, gt, at)
	}
	if draftAcceptance(gt, at) != 0.25 {
		t.Fatalf("acceptance = %v", draftAcceptance(gt, at))
	}

	multi := log + "\nstatistics dflash: #calls(b,g,a) = 2 6 5, #gen drafts = 20, #acc drafts = 8, #gen tokens = 26, #acc tokens = 7, dur(b,g,a) = 2.0, 4.0, 5.0 ms"
	_, _, gt2, at2, ok := ParseDflashStatisticsFromLog(multi)
	if !ok || gt2 != 26 || at2 != 7 {
		t.Fatalf("last stats gt/at=%d/%d ok=%v", gt2, at2, ok)
	}
}

func TestFillServerRunFromCompletion(t *testing.T) {
	var cc chatCompletionResp
	if err := json.Unmarshal([]byte(`{
		"usage":{"completion_tokens":16,"prompt_tokens":17},
		"timings":{"predicted_n":16,"predicted_ms":162.0,"predicted_per_second":98.7,"draft_n":3,"draft_n_accepted":3}
	}`), &cc); err != nil {
		t.Fatal(err)
	}
	var run ANEDraftServerRun
	fillServerRunFromCompletion(&run, cc, 0)
	if run.EvalCount != 16 || run.TokensPerSec != 98.7 {
		t.Fatalf("eval=%d tps=%v", run.EvalCount, run.TokensPerSec)
	}
	if run.DraftAcceptance != 1.0 {
		t.Fatalf("acceptance=%v", run.DraftAcceptance)
	}
}

func TestParseMatmulChainFromLog(t *testing.T) {
	if got := parseMatmulChainFromLog(`P1 matmul kernel active (blk.0 ffn_gate h@W, seq=16 oc=256 dynamic_mil)`); got != 1 {
		t.Fatalf("P1 chain = %d", got)
	}
	if got := parseMatmulChainFromLog(`P1 matmul kernel active chain2=gate+silu+up`); got != 2 {
		t.Fatalf("P2 chain = %d", got)
	}
	if got := parseMatmulChainFromLog(`P1 matmul kernel active chain5=swiglu+down+attn_gate+ssm_out`); got != 5 {
		t.Fatalf("P5 chain = %d", got)
	}
	if got := parseMatmulChainFromLog(`P1 matmul kernel active chain4=swiglu+down+attn_gate`); got != 4 {
		t.Fatalf("P4 chain = %d", got)
	}
	if got := parseMatmulChainFromLog(`B6 golden step=1 mode=matmul_chain4 mse_ref_vs_ane=0 cosine=1.0`); got != 4 {
		t.Fatalf("golden chain4 = %d", got)
	}
	if got := parseMatmulChainFromLog(`P1 matmul kernel active chain3=swiglu+down`); got != 3 {
		t.Fatalf("P3 chain = %d", got)
	}
	if got := parseMatmulChainFromLog(`B6 golden step=1 mode=matmul_chain3 mse_ref_vs_ane=0 cosine=1.0`); got != 3 {
		t.Fatalf("golden chain3 = %d", got)
	}
}

func TestCountANEHandoffsFromLog(t *testing.T) {
	log := `common_ane_draft_handoff_after_decode: step=1 ggml iosurface handoff ok
common_ane_draft_handoff_after_decode: step=2 ggml iosurface handoff ok
other line`
	if n := CountANEHandoffsFromLog(log); n != 2 {
		t.Fatalf("handoffs=%d want 2", n)
	}
}

func TestParseGoldenCosineFromLog(t *testing.T) {
	log := `log_ane_golden_telemetry: B6 golden step=1 mode=conv2 mse_ref_vs_ane=0.001 cosine=0.9987 ane_steps=1
log_ane_golden_telemetry: B6 golden step=2 mode=conv2 mse_ref_vs_ane=0.002 cosine=0.9912 ane_steps=2`
	c, n := parseGoldenCosineFromLog(log)
	if n != 2 || c < 0.99 {
		t.Fatalf("cosine=%v count=%d", c, n)
	}
}

func TestParseB7ShadowFromLog(t *testing.T) {
	log := `B7 shadow step=1 seq=0 ane_tok=42 metal_tok=42 match=1 hidden_cos=0.8123
B7 shadow step=2 seq=0 ane_tok=1 metal_tok=2 match=0 hidden_cos=0.4500`
	steps, matches, cosSum, cosN := parseB7ShadowFromLog(log)
	if steps != 2 || matches != 1 {
		t.Fatalf("steps=%d matches=%d", steps, matches)
	}
	if cosN != 2 || cosSum < 1.2 {
		t.Fatalf("hidden cos sum=%v n=%d", cosSum, cosN)
	}
}

func TestDraftANEMatmulDims(t *testing.T) {
	ic, oc, seq := DraftANEMatmulDims(ANEDraftEntry{EmbeddingLength: 768, ProxyChannels: 256, ProxySpatial: 16})
	if ic != 768 || oc != 256 || seq != 16 {
		t.Fatalf("got ic=%d oc=%d seq=%d", ic, oc, seq)
	}
}

func TestDraftANEHandoffStride(t *testing.T) {
	if got := DraftANEHandoffStride("matmul", 1); got != 4 {
		t.Fatalf("matmul P1 stride = %d, want 4", got)
	}
	if got := DraftANEHandoffStride("matmul", 2); got != 4 {
		t.Fatalf("matmul P2 stride = %d, want 4", got)
	}
	if got := DraftANEHandoffStride("matmul", 3); got != 8 {
		t.Fatalf("matmul P3 stride = %d, want 8", got)
	}
	if got := DraftANEHandoffStride("matmul", 5); got != 12 {
		t.Fatalf("matmul P5 stride = %d, want 12", got)
	}
	if got := DraftANEHandoffStride("conv", 0); got != 2 {
		t.Fatalf("conv stride = %d, want 2", got)
	}
}

func TestDraftANEMatmulChain2Dims(t *testing.T) {
	ic2, oc2 := DraftANEMatmulChain2Dims(768, 256)
	if ic2 != 256 || oc2 != 768 {
		t.Fatalf("got ic2=%d oc2=%d", ic2, oc2)
	}
}

func TestDraftANEMatmulChain3Dims(t *testing.T) {
	icUp, ocUp := DraftANEMatmulChain3UpDims(768, 256)
	if icUp != 768 || ocUp != 256 {
		t.Fatalf("up dims ic=%d oc=%d", icUp, ocUp)
	}
	icDown, ocDown := DraftANEMatmulChain3DownDims(256, 768)
	if icDown != 256 || ocDown != 768 {
		t.Fatalf("down dims ic=%d oc=%d", icDown, ocDown)
	}
}

func TestANEDraftNeedsDriveHead(t *testing.T) {
	if ANEDraftNeedsDriveHead("matmul", "shadow") {
		t.Fatal("matmul shadow hidden-only should skip drive head")
	}
	if !ANEDraftNeedsDriveHeadWithMetrics("matmul", "shadow", "both") {
		t.Fatal("matmul shadow both needs drive head")
	}
	if !ANEDraftNeedsDriveHead("matmul", "force") {
		t.Fatal("matmul force needs drive head")
	}
	if !ANEDraftNeedsDriveHead("conv", "shadow") {
		t.Fatal("conv shadow needs drive head")
	}
}

func TestCalcHookOverheadPct(t *testing.T) {
	if got := calcHookOverheadPct(100, 80); got < 19.9 || got > 20.1 {
		t.Fatalf("overhead = %v, want ~20", got)
	}
}

func TestAcceptanceClose(t *testing.T) {
	if !acceptanceClose(0.25, 0.24) {
		t.Fatal("expected close")
	}
	if acceptanceClose(0.25, 0.10) {
		t.Fatal("expected not close")
	}
}

func TestAcceptanceParity(t *testing.T) {
	if !acceptanceParity(ANEDraftServerRun{}, ANEDraftServerRun{GenTokens: 10}) {
		t.Fatal("incomparable legs should not fail parity")
	}
	if !acceptanceParity(
		ANEDraftServerRun{GenTokens: 10, AccTokens: 3, DraftAcceptance: 0.3},
		ANEDraftServerRun{GenTokens: 10, AccTokens: 3, DraftAcceptance: 0.29},
	) {
		t.Fatal("comparable close legs should pass")
	}
}

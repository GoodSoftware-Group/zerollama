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
	log := `B7 shadow step=1 seq=0 ane_tok=42 metal_tok=42 match=1
B7 shadow step=2 seq=0 ane_tok=1 metal_tok=2 match=0`
	steps, matches := parseB7ShadowFromLog(log)
	if steps != 2 || matches != 1 {
		t.Fatalf("steps=%d matches=%d", steps, matches)
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

package discover

import "testing"

func TestManifestConvCount(t *testing.T) {
	m := ANEDraftWeightManifest{
		Weights: []ANEDraftWeightEntry{
			{Slot: "proxy_conv_w0", Path: "/a"},
			{Slot: "proxy_conv_w1", Path: "/b"},
			{Slot: "proxy_conv_w2", Path: "/c"},
		},
	}
	if got := ManifestConvCount(m); got != 3 {
		t.Fatalf("ManifestConvCount() = %d, want 3", got)
	}

	out := map[string]string{}
	ApplyConvDepthEnv(out, 0)
	if _, ok := out["ZEROLLAMA_ANE_DRAFT_CONV_DEPTH"]; ok {
		t.Fatal("ApplyConvDepthEnv(0) should not set env")
	}
	ApplyConvDepthEnv(out, 4)
	if out["ZEROLLAMA_ANE_DRAFT_CONV_DEPTH"] != "4" {
		t.Fatalf("ApplyConvDepthEnv(4) = %q, want 4", out["ZEROLLAMA_ANE_DRAFT_CONV_DEPTH"])
	}
}

func TestInferActiveConvDepthFromLog(t *testing.T) {
	if got := inferActiveConvDepthFromLog("B8 triple conv1 chain active"); got != 3 {
		t.Fatalf("inferActiveConvDepthFromLog() = %d, want 3", got)
	}
	if got := inferActiveConvDepthFromLog("no chain"); got != 0 {
		t.Fatalf("inferActiveConvDepthFromLog() = %d, want 0", got)
	}
}

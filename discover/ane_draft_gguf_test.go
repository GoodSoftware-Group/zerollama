package discover

import "testing"

func TestDraftSidecarHasTensor(t *testing.T) {
	entries, err := ListANEDraftInventory()
	if err != nil {
		t.Skip(err)
	}
	entry, ok := SelectANEDraftModel(entries, "eliza-1-2b-dflash")
	if !ok {
		t.Skip("eliza dflash not in inventory")
	}
	if !DraftSidecarHasTensor(entry, "blk.0.ffn_gate.weight") {
		t.Fatal("expected blk.0.ffn_gate.weight in eliza dflash sidecar")
	}
	if DraftSidecarHasTensor(entry, "dflash_fc.weight") {
		t.Fatal("eliza qwen35 drafter sidecar should not have dflash_fc.weight")
	}
}

func TestDraftANEMatmulChain10Blk1DownMaterialize(t *testing.T) {
	entries, err := ListANEDraftInventory()
	if err != nil {
		t.Skip(err)
	}
	entry, ok := SelectANEDraftModel(entries, "eliza-1-2b-dflash")
	if !ok {
		t.Skip("eliza dflash not in inventory")
	}
	ic10, oc10 := DraftANEMatmulChain10Blk1DownDims(entry.ProxyChannels, entry.EmbeddingLength)
	if entry.ProxyChannels <= 0 {
		ic10, oc10 = DraftANEMatmulChain10Blk1DownDims(256, 768)
	}
	if _, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.1.ffn_down.weight", ic10, oc10); err != nil {
		t.Fatalf("blk.1.ffn_down materialize: %v", err)
	}
}

func TestDraftANEMatmulChain10Blk1DownDims(t *testing.T) {
	ic, oc := DraftANEMatmulChain10Blk1DownDims(256, 768)
	if ic != 256 || oc != 768 {
		t.Fatalf("got ic=%d oc=%d", ic, oc)
	}
}

func TestDraftANEMatmulChain9Blk1UpMaterialize(t *testing.T) {
	entries, err := ListANEDraftInventory()
	if err != nil {
		t.Skip(err)
	}
	entry, ok := SelectANEDraftModel(entries, "eliza-1-2b-dflash")
	if !ok {
		t.Skip("eliza dflash not in inventory")
	}
	ic9, oc9 := DraftANEMatmulChain9Blk1UpDims(entry.EmbeddingLength, entry.ProxyChannels)
	if entry.ProxyChannels <= 0 {
		ic9, oc9 = DraftANEMatmulChain9Blk1UpDims(768, 256)
	}
	if _, _, err := MaterializeANEDraftMatmulWeightFile(entry, "blk.1.ffn_up.weight", ic9, oc9); err != nil {
		t.Fatalf("blk.1.ffn_up materialize: %v", err)
	}
}

func TestDraftANEMatmulChain9Blk1UpDims(t *testing.T) {
	ic, oc := DraftANEMatmulChain9Blk1UpDims(768, 256)
	if ic != 768 || oc != 256 {
		t.Fatalf("got ic=%d oc=%d", ic, oc)
	}
}

func TestDraftANEMatmulChain7Blk1GateDims(t *testing.T) {
	ic, oc := DraftANEMatmulChain7Blk1GateDims(768, 256)
	if ic != 768 || oc != 256 {
		t.Fatalf("got ic=%d oc=%d", ic, oc)
	}
}

func TestDraftANEMatmulChain11AttnQMaterialize(t *testing.T) {
	entries, err := ListANEDraftInventory()
	if err != nil {
		t.Skip(err)
	}
	entry, ok := SelectANEDraftModel(entries, "eliza-1-2b-dflash")
	if !ok {
		t.Skip("eliza dflash not in inventory")
	}
	tensor := ResolveChain11AttnQTensor(entry)
	if tensor != "blk.0.attn_qkv.weight" {
		t.Fatalf("expected attn_qkv fallback for qwen35 sidecar, got %q", tensor)
	}
	icQ, ocQ := DraftANEMatmulChain11AttnQDims(entry, entry.ProxyChannels)
	if entry.ProxyChannels <= 0 {
		icQ, ocQ = DraftANEMatmulChain11AttnQDims(entry, 256)
	}
	if _, _, err := MaterializeANEDraftMatmulWeightFile(entry, tensor, icQ, ocQ); err != nil {
		t.Fatalf("chain11 attn_q materialize: %v", err)
	}
}

func TestDraftANEMatmulChain7DflashFcDimsMissing(t *testing.T) {
	entries, err := ListANEDraftInventory()
	if err != nil {
		t.Skip(err)
	}
	entry, ok := SelectANEDraftModel(entries, "eliza-1-2b-dflash")
	if !ok {
		t.Skip("eliza dflash not in inventory")
	}
	if _, _, ok := DraftANEMatmulChain7DflashFcDims(entry); ok {
		t.Fatal("expected no dflash_fc dims for eliza qwen35 sidecar")
	}
	if env := ExportDflashTargetMetaEnv(entry); len(env) != 0 {
		t.Fatalf("expected no dflash meta env for eliza sidecar, got %v", env)
	}
	if IsNativeDflashDraftSidecar(entry) {
		t.Fatal("eliza 2b sidecar should not be native dflash-draft")
	}
}

func TestNativeDflash27bSidecarInventory(t *testing.T) {
	entries, err := ListANEDraftInventory()
	if err != nil {
		t.Skip(err)
	}
	entry, ok := SelectANEDraftModel(entries, "eliza-1-27b-256k-dflash")
	if !ok {
		t.Skip("eliza 27b dflash not in inventory")
	}
	if !IsNativeDflashDraftSidecar(entry) {
		t.Fatalf("expected native dflash-draft sidecar, got %q (base %q)", DraftSidecarArchitecture(entry), entry.Architecture)
	}
	if !DraftSidecarHasTensor(entry, "dflash_fc.weight") {
		t.Fatal("expected dflash_fc.weight in 27b native sidecar")
	}
	ic, oc, ok := DraftANEMatmulChain7DflashFcDims(entry)
	if !ok || ic <= 0 || oc <= 0 {
		t.Fatalf("dflash_fc dims missing: ic=%d oc=%d ok=%v", ic, oc, ok)
	}
	env := ExportDflashTargetMetaEnv(entry)
	if env["ZEROLLAMA_ANE_DRAFT_DFLASH_N_TARGET_FEATURES"] == "" {
		t.Fatalf("expected export env for native sidecar, got %v", env)
	}
	if env["ZEROLLAMA_ANE_DRAFT_DFLASH_N_TARGET_LAYERS"] != "5" {
		t.Fatalf("expected 5 target layers in export env, got %v", env)
	}
	nFeat, layers, ok := DraftDflashTargetMeta(entry)
	if !ok || nFeat != 25600 || len(layers) != 5 {
		t.Fatalf("DraftDflashTargetMeta = nFeat=%d layers=%v ok=%v", nFeat, layers, ok)
	}
	if layers[0] != 1 || layers[4] != 61 {
		t.Fatalf("unexpected target_layer_ids: %v", layers)
	}
	if ResolveChain11HiddenNormTensor(entry) != "dflash_hidden_norm.weight" {
		t.Fatalf("expected dflash_hidden_norm gamma, got %q", ResolveChain11HiddenNormTensor(entry))
	}
	if dim := DraftANEDflashHiddenNormDim(entry); dim != oc {
		t.Fatalf("DraftANEDflashHiddenNormDim = %d want fcOut %d", dim, oc)
	}
	if w, _, err := MaterializeANEDraftMatmulWeightFile(entry, "dflash_fc.weight", ic, oc); err != nil || w == "" {
		t.Fatalf("dflash_fc materialize ic=%d oc=%d: %v path=%q", ic, oc, err, w)
	}
	head, _, err := MaterializeANEDraftDriveHead(entry)
	if err != nil || head.TokenEmbdPath == "" || head.NEmbd <= 0 {
		t.Fatalf("drive head materialize for native sidecar: head=%+v err=%v", head, err)
	}
}

func TestDraftANEDraftAttnFullDimsChain13(t *testing.T) {
	entries, err := ListANEDraftInventory()
	if err != nil {
		t.Skip(err)
	}
	entry, ok := SelectANEDraftModel(entries, "eliza-1-27b-256k-dflash")
	if !ok {
		t.Skip("eliza-27b dflash not in inventory")
	}
	if !IsNativeDflashDraftSidecar(entry) {
		t.Skip("not a native dflash sidecar")
	}
	fcOut := entry.EmbeddingLength
	if _, ocFc, ok := DraftANEMatmulChain7DflashFcDims(entry); ok && ocFc > 0 {
		fcOut = ocFc
	}
	_, ocQ := DraftANEMatmulChain11AttnQDimsForChain(entry, fcOut, 17)
	_, ocKV := DraftANEMatmulChain12AttnKVDimsForChain(entry, fcOut, 17)
	icWo, ocWo := DraftANEMatmulChain14AttnWoDimsForChain(entry, fcOut, 17)
	if ocQ <= 512 {
		t.Fatalf("chain17 Q oc=%d expected full head width > 512", ocQ)
	}
	if ocKV <= 512 {
		t.Fatalf("chain17 KV oc=%d expected full KV width > 512", ocKV)
	}
	if icWo != ocQ || ocWo != fcOut {
		t.Fatalf("attn_wo dims ic=%d oc=%d want ic=%d oc=%d", icWo, ocWo, ocQ, fcOut)
	}
	meta, err := DraftANEDraftAttnHeadMeta(entry)
	if err != nil {
		t.Fatalf("head meta: %v", err)
	}
	if nh := DraftANEDraftAttnQueryHeadCount(entry); nh <= 0 || nh*meta.HeadDim != ocQ {
		t.Fatalf("Q heads mismatch nh=%d head_dim=%d ocQ=%d", nh, meta.HeadDim, ocQ)
	}
	if wq, _, err := MaterializeANEDraftMatmulWeightFile(entry, ResolveChain11AttnQTensor(entry), fcOut, ocQ); err != nil || wq == "" {
		t.Fatalf("attn_q materialize ic=%d oc=%d: %v path=%q", fcOut, ocQ, err, wq)
	}
	if wwo, _, err := MaterializeANEDraftMatmulWeightFile(entry, ResolveChain14AttnWoTensor(entry), icWo, ocWo); err != nil || wwo == "" {
		t.Fatalf("attn_wo materialize ic=%d oc=%d: %v path=%q", icWo, ocWo, err, wwo)
	}
}

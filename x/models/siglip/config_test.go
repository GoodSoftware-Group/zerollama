package siglip

import "testing"

func TestParseSparseSiglipConfig(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"model_type": "siglip",
		"text_config": {"model_type": "siglip_text_model", "vocab_size": 32000},
		"vision_config": {"model_type": "siglip_vision_model", "patch_size": 16}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Family != "siglip" || cfg.Vision.NaFlex || cfg.Vision.ImageSize != 224 || cfg.Vision.NumPatches != 196 {
		t.Fatalf("vision = %+v", cfg.Vision)
	}
	if cfg.Text.MaxPositionEmbeddings != 64 || cfg.Text.PadTokenID != 1 || cfg.Text.ProjectionSize != 768 {
		t.Fatalf("text = %+v", cfg.Text)
	}
}

func TestParseSiglip2NaFlex(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"architectures": ["Siglip2Model"],
		"model_type": "siglip2",
		"text_config": {"model_type": "siglip2_text_model", "vocab_size": 256000},
		"vision_config": {"model_type": "siglip2_vision_model"}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Family != "siglip2" || !cfg.Vision.NaFlex || cfg.Vision.NumPatches != 256 || cfg.Vision.ImageSize != 0 {
		t.Fatalf("vision = %+v", cfg.Vision)
	}
	if cfg.Text.VocabSize != 256000 {
		t.Fatalf("vocab = %d", cfg.Text.VocabSize)
	}
}

func TestParseTextOnlySkipsVision(t *testing.T) {
	cfg, err := parseConfig([]byte(`{"architectures":["SiglipTextModel"],"model_type":"siglip_text_model","vocab_size":32000}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Vision != nil || cfg.Text == nil {
		t.Fatalf("text-only parse kept vision=%v text=%v", cfg.Vision != nil, cfg.Text != nil)
	}
}

func TestImageSizeForMaxPatches(t *testing.T) {
	h, w := imageSizeForMaxPatches(224, 224, 16, 256)
	if h != 256 || w != 256 {
		t.Fatalf("224 square -> %dx%d, want 256x256", h, w)
	}
	h, w = imageSizeForMaxPatches(1000, 500, 16, 256)
	if h != 352 || w != 176 {
		t.Fatalf("1000x500 -> %dx%d, want 352x176", h, w)
	}
	if (h/16)*(w/16) > 256 {
		t.Fatalf("patch count %d exceeds 256", (h/16)*(w/16))
	}
}

func TestBilinearPosWeightsIdentity(t *testing.T) {
	idx, wt := bilinearPosWeights(4, 4, 4)
	for i := 0; i < 16; i++ {
		if idx[i*4] != int32(i) || wt[i*4] != 1 {
			t.Fatalf("cell %d idx=%v wt=%v, want identity on the first corner", i, idx[i*4:i*4+4], wt[i*4:i*4+4])
		}
	}
}

func TestPatchifyLayout(t *testing.T) {
	// 4x4 image, patch 2, one channel. Values are the pixel index.
	pixels := make([]float32, 16)
	for i := range pixels {
		pixels[i] = float32(i)
	}
	got := patchify(pixels, 4, 4, 2, 1)
	// Patch (0,0) is pixels 0,1 / 4,5. Patch (0,1) is 2,3 / 6,7.
	want := []float32{0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}
	if len(got) != len(want) {
		t.Fatalf("len %d", len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("patchify[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

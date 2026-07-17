package deepseekocr

import "testing"

func TestDeepseekOCRValidateProcessorPixels_ok(t *testing.T) {
	rows, cols := 2, 2
	local := rows * cols * deepseekOCRChannels * deepseekOCRTileSize * deepseekOCRTileSize
	global := deepseekOCRChannels * deepseekOCRBaseSize * deepseekOCRBaseSize
	pixels := make([]float32, local+global)

	gotRows, gotCols, localElems, globalElems, crop, err := deepseekOCRValidateProcessorPixels(pixels, []int{1, rows, cols})
	if err != nil {
		t.Fatalf("validate: %v", err)
	}
	if gotRows != rows || gotCols != cols || localElems != local || globalElems != global {
		t.Fatalf("rows=%d cols=%d local=%d global=%d", gotRows, gotCols, localElems, globalElems)
	}
	if crop[0] != cols || crop[1] != rows {
		t.Fatalf("crop=%v want [%d %d]", crop, cols, rows)
	}
}

func TestDeepseekOCRValidateProcessorPixels_errors(t *testing.T) {
	cases := []struct {
		name   string
		grid   []int
		pixels int
	}{
		{"bad_t", []int{2, 2, 2}, 100},
		{"too_few_tiles", []int{1, 1, 1}, 100},
		{"too_many_tiles", []int{1, 4, 3}, 100},
		{"len_mismatch", []int{1, 2, 2}, 10},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, _, _, _, err := deepseekOCRValidateProcessorPixels(make([]float32, tc.pixels), tc.grid)
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

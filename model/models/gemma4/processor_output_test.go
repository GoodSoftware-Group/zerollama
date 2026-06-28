package gemma4

import "testing"

func TestGemma4PixelSizeFromGrid(t *testing.T) {
	w, h, err := gemma4PixelSizeFromGrid([]int{1, 224, 224}, 3, 3*224*224)
	if err != nil || w != 224 || h != 224 {
		t.Fatalf("got w=%d h=%d err=%v", w, h, err)
	}
	_, _, err = gemma4PixelSizeFromGrid([]int{2, 224, 224}, 3, 3*224*224)
	if err == nil {
		t.Fatal("expected error for T>1")
	}
	_, _, err = gemma4PixelSizeFromGrid([]int{1, 224, 224}, 3, 100)
	if err == nil {
		t.Fatal("expected length mismatch error")
	}
}

package llamarunner

import "testing"

func TestVisionTokensFromGridTHW_qwenStyle(t *testing.T) {
	got := visionTokensFromGridTHW([]int{1, 24, 32}, 2)
	if got != 192 {
		t.Fatalf("got %d want 192", got)
	}
}

func TestMultimodalTokenize_nilImageContext(t *testing.T) {
	var c *ImageContext
	chunks, err := c.MultimodalTokenize(nil, []byte{1}, "sess", []int{1, 8, 8}, false)
	if err != nil || chunks != nil {
		t.Fatalf("nil context: chunks=%v err=%v", chunks, err)
	}
}

func TestLogVisionGridHint_noPanic(t *testing.T) {
	logVisionGridHint(0, []int{1, 24, 32}, []visionChunk{{embed: make([]float32, 4)}})
	logVisionGridHint(1, nil, nil)
}

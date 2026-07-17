package ollamarunner

import "testing"

func TestHashImage_includesGrid(t *testing.T) {
	c := &VisionEmbedCache{}
	data := []byte("png-bytes")
	a := c.hashImage(data, nil)
	b := c.hashImage(data, []int{1, 24, 32})
	d := c.hashImage(data, []int{1, 24, 32})
	if a == b {
		t.Fatal("grid hint must change vision embed cache key")
	}
	if b != d {
		t.Fatal("same grid must hash identically")
	}
}

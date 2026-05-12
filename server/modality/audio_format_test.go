package modality

import (
	"testing"
)

func TestAudioInputExt(t *testing.T) {
	t.Parallel()
	if g, w := AudioInputExt("clip.webm", nil), "webm"; g != w {
		t.Fatalf("filename: got %q want %q", g, w)
	}
	wavHdr := []byte("RIFFxxxxWAVE")
	if g, w := AudioInputExt("", wavHdr), "wav"; g != w {
		t.Fatalf("magic wav: got %q want %q", g, w)
	}
	id3 := append([]byte("ID3"), make([]byte, 20)...)
	if g, w := AudioInputExt("", id3), "mp3"; g != w {
		t.Fatalf("magic mp3: got %q want %q", g, w)
	}
}

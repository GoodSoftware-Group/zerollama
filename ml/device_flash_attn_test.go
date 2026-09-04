package ml

import "testing"

func TestMetalLikeLibrary(t *testing.T) {
	t.Parallel()
	cases := map[string]bool{
		"Metal": true,
		"MTL":   true,
		"mtl":   true,
		"metal": true,
		"CUDA":  false,
		"cpu":   false,
		"":      false,
	}
	for in, want := range cases {
		if got := MetalLikeLibrary(in); got != want {
			t.Fatalf("MetalLikeLibrary(%q)=%v want %v", in, got, want)
		}
	}
}

func TestFlashAttentionSupportedMTL(t *testing.T) {
	t.Parallel()
	if !FlashAttentionSupported([]DeviceInfo{{DeviceID: DeviceID{Library: "MTL"}, Name: "Apple M4 Max"}}) {
		t.Fatal("MTL library must support flash attention (ggml reports MTL, not Metal)")
	}
	if !FlashAttentionSupported([]DeviceInfo{{DeviceID: DeviceID{Library: "Metal"}, Name: "Metal"}}) {
		t.Fatal("Metal library must support flash attention")
	}
	if FlashAttentionSupported([]DeviceInfo{
		{DeviceID: DeviceID{Library: "MTL"}},
		{DeviceID: DeviceID{Library: "UnknownAccel"}},
	}) {
		t.Fatal("mixed unsupported device must fail")
	}
}

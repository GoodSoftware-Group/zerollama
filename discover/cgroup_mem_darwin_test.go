package discover

import "testing"

func TestParseDarwinSwapUsage(t *testing.T) {
	total, used := parseDarwinSwapUsage("vm.swapusage: total = 2048.00M  used = 512.00M  free = 1536.00M")
	if total != 2048*1024*1024 {
		t.Fatalf("total=%d", total)
	}
	if used != 512*1024*1024 {
		t.Fatalf("used=%d", used)
	}
	gTotal, gUsed := parseDarwinSwapUsage("vm.swapusage: total = 8.00G  used = 1.50G  free = 6.50G")
	if gTotal != 8*1024*1024*1024 {
		t.Fatalf("G total=%d", gTotal)
	}
	wantUsed := uint64(3) * 1024 * 1024 * 1024 / 2
	if gUsed != wantUsed {
		t.Fatalf("G used=%d want %d", gUsed, wantUsed)
	}
}

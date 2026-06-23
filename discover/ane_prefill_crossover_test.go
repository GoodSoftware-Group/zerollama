package discover

import "testing"

func TestSummarizeWidthCrossover(t *testing.T) {
	widths := []int{512, 768, 1024}
	points := []ANEPrefillCompareResult{
		{OK: true, IC: 512, Faster: "ane"},
		{OK: true, IC: 768, Faster: "metal_mps"},
		{OK: true, IC: 1024, Faster: "metal_mps"},
	}
	out := summarizeWidthCrossover(512, widths, points)
	if out.ANEWins != 1 || out.MetalWins != 2 || out.WidthCrossover != 768 {
		t.Fatalf("summarize = %+v", out)
	}
}

func TestParseCrossoverWidthsRange(t *testing.T) {
	got, err := ParseCrossoverWidths("512:768:128")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] != 512 || got[2] != 768 {
		t.Fatalf("ParseCrossoverWidths = %v", got)
	}
}

func TestCrossoverWidthsAround(t *testing.T) {
	got := crossoverWidthsAround(896, true)
	if len(got) == 0 || got[len(got)-1] != 896 {
		t.Fatalf("crossoverWidthsAround = %v", got)
	}
}

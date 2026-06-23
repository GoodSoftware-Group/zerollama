package discover

import "testing"

func TestParsePrefillSweepSeqs(t *testing.T) {
	got, err := ParsePrefillSweepSeqs("128, 512,2048")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] != 128 || got[2] != 2048 {
		t.Fatalf("ParsePrefillSweepSeqs = %v", got)
	}
}

func TestDefaultPrefillSweepSeqs(t *testing.T) {
	q := DefaultPrefillSweepSeqs(true)
	if len(q) != 3 {
		t.Fatalf("quick sweep len = %d", len(q))
	}
	f := DefaultPrefillSweepSeqs(false)
	if len(f) != 6 {
		t.Fatalf("full sweep len = %d", len(f))
	}
}

func TestSummarizePrefillSweepCrossover(t *testing.T) {
	points := []ANEPrefillCompareResult{
		{OK: true, Seq: 128, Faster: "ane"},
		{OK: true, Seq: 512, Faster: "ane"},
		{OK: true, Seq: 2048, Faster: "metal"},
	}
	out := summarizePrefillSweep(256, 256, points)
	if out.ANEWins != 2 || out.MetalWins != 1 || out.CrossoverSeq != 2048 {
		t.Fatalf("summarize = %+v", out)
	}
}

func TestSummarizePrefillSweepMPSCountsAsMetal(t *testing.T) {
	points := []ANEPrefillCompareResult{
		{OK: true, Seq: 128, Faster: "metal_mps"},
	}
	out := summarizePrefillSweep(256, 256, points)
	if out.MetalWins != 1 || out.ANEWins != 0 {
		t.Fatalf("summarize = %+v", out)
	}
}

package whispercli

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

func TestTextArgsSkipsTimestamps(t *testing.T) {
	got := TextArgs("m.bin", "-", "out", "en")
	want := []string{"-m", "m.bin", "-f", "-", "-otxt", "-of", "out", "-l", "en", "--beam-size", "1", "--best-of", "1", "--no-timestamps"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("%q", got)
	}
}

func TestArgs(t *testing.T) {
	got := Args("m.bin", "in.wav", "out", "en", " names and places ", "0.2", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", true, false, false, false, false, false, false, false, false, false, false, false)
	want := []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "-l", "en", "--prompt", "names and places", "--temperature", "0.2", "-oj"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("args %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "  ", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("defaults %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", true, true, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--split-on-word", "-ojf"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("words %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", true, true, false, false, false, false, false, false, false, false, false, true)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--split-on-word", "-ojf"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("words-and-split %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "es", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, true, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "-l", "es", "--translate"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("translate %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, true, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--diarize"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("diarize %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, true, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--suppress-nst"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("suppress %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, true, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--no-fallback"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("no-fallback %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "0.2", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--temperature-inc", "0.2"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("temperature-inc %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "2.4", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--entropy-thold", "2.4"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("entropy-thold %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "-1", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--logprob-thold", "-1"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("logprob-thold %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "0.6", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--no-speech-thold", "0.6"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("no-speech-thold %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "5", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--beam-size", "5"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("beam-size %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "5", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--best-of", "5"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("best-of %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, true, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--carry-initial-prompt"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("carry-initial-prompt %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "80", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--max-len", "80"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("max-len %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "80", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, true)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--max-len", "80", "--split-on-word"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("split-on-word %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "0.2", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--word-thold", "0.2"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("word-thold %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "0", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--max-context", "0"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("max-context %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "64", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--audio-ctx", "64"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("audio-ctx %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "1500", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--offset-t", "1500"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("offset-t %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "2500", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--duration", "2500"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("duration %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", `\[MUSIC\]`, "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--suppress-regex", `\[MUSIC\]`}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("suppress-regex %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, true, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--no-timestamps"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("no-timestamps %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", `root ::= "yes"`, "root", "0", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--grammar", `root ::= "yes"`, "--grammar-rule", "root", "--grammar-penalty", "0"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("grammar %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, true, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--print-special"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("print-special %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", true, false, false, false, false, false, false, false, false, true, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--detect-language", "-oj"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("detect-language %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "large.v3", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--dtw", "large.v3", "--no-flash-attn"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("dtw %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "/models/silero.bin", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--vad", "--vad-model", "/models/silero.bin"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("vad %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "/models/silero.bin", "0.5", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--vad", "--vad-model", "/models/silero.bin", "--vad-threshold", "0.5"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("vad-threshold %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "/models/silero.bin", "0.5", "250", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--vad", "--vad-model", "/models/silero.bin", "--vad-threshold", "0.5", "--vad-min-speech-duration-ms", "250"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("vad-min-speech %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "/models/silero.bin", "0.5", "250", "100", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--vad", "--vad-model", "/models/silero.bin", "--vad-threshold", "0.5", "--vad-min-speech-duration-ms", "250", "--vad-min-silence-duration-ms", "100"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("vad-min-silence %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "/models/silero.bin", "0.5", "250", "100", "30", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--vad", "--vad-model", "/models/silero.bin", "--vad-threshold", "0.5", "--vad-min-speech-duration-ms", "250", "--vad-min-silence-duration-ms", "100", "--vad-max-speech-duration-s", "30"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("vad-max-speech %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "/models/silero.bin", "0.5", "250", "100", "30", "0", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--vad", "--vad-model", "/models/silero.bin", "--vad-threshold", "0.5", "--vad-min-speech-duration-ms", "250", "--vad-min-silence-duration-ms", "100", "--vad-max-speech-duration-s", "30", "--vad-speech-pad-ms", "0"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("vad-speech-pad %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "/models/silero.bin", "0.5", "250", "100", "30", "0", "0", false, false, false, false, false, false, false, false, false, false, false, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--vad", "--vad-model", "/models/silero.bin", "--vad-threshold", "0.5", "--vad-min-speech-duration-ms", "250", "--vad-min-silence-duration-ms", "100", "--vad-max-speech-duration-s", "30", "--vad-speech-pad-ms", "0", "--vad-samples-overlap", "0"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("vad-samples-overlap %q", got)
	}
	got = Args("m.bin", "in.wav", "out", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, true, false)
	want = []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "--tinydiarize"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("tinydiarize %q", got)
	}
	g, rule, err := Grammar("", "")
	if err != nil || g != "" || rule != "" {
		t.Fatal(err, g, rule)
	}
	if _, _, err = Grammar(`root ::= "yes"`, ""); err == nil {
		t.Fatal("expected reject")
	}
	if g, rule, err = Grammar(`root ::= "yes"`, "root"); err != nil || g != `root ::= "yes"` || rule != "root" {
		t.Fatal(err, g, rule)
	}
	if _, _, err = Grammar(`root ::= "yes"`, "other"); err == nil {
		t.Fatal("expected reject")
	}
	if _, _, err = Grammar("nope", "root"); err == nil {
		t.Fatal("expected reject")
	}
	if pen, perr := GrammarPenalty(""); perr != nil || pen != "" {
		t.Fatal(perr, pen)
	}
	if pen, perr := GrammarPenalty("0"); perr != nil || pen != "0" {
		t.Fatal(perr, pen)
	}
	if pen, perr := GrammarPenalty("0100"); perr != nil || pen != "100" {
		t.Fatal(perr, pen)
	}
	if _, perr := GrammarPenalty("-1"); perr == nil {
		t.Fatal("expected reject")
	}
	if _, perr := GrammarPenalty("1001"); perr == nil {
		t.Fatal("expected reject")
	}
	ps, pserr := PrintSpecialFlag("true")
	if pserr != nil || !ps {
		t.Fatal(pserr, ps)
	}
	if ps, pserr = PrintSpecialFlag(""); pserr != nil || ps {
		t.Fatal(pserr, ps)
	}
	if _, pserr = PrintSpecialFlag("maybe"); pserr == nil {
		t.Fatal("expected reject")
	}
	dl, dlerr := DetectLanguageFlag("true")
	if dlerr != nil || !dl {
		t.Fatal(dlerr, dl)
	}
	if dl, dlerr = DetectLanguageFlag(""); dlerr != nil || dl {
		t.Fatal(dlerr, dl)
	}
	if _, dlerr = DetectLanguageFlag("maybe"); dlerr == nil {
		t.Fatal("expected reject")
	}
	nt, err := NoTimestampsFlag("true")
	if err != nil || !nt {
		t.Fatal(err, nt)
	}
	if nt, err = NoTimestampsFlag(""); err != nil || nt {
		t.Fatal(err, nt)
	}
	if _, err = NoTimestampsFlag("maybe"); err == nil {
		t.Fatal("expected reject")
	}
	rx, err := SuppressRegex("   ")
	if err != nil || rx != "" {
		t.Fatal(err, rx)
	}
	if rx, err = SuppressRegex(`\[MUSIC\]`); err != nil || rx != `\[MUSIC\]` {
		t.Fatal(err, rx)
	}
	if _, err = SuppressRegex(strings.Repeat("a", 257)); err == nil {
		t.Fatal("expected reject")
	}
	dm, err := DurationMS(" 02500 ")
	if err != nil || dm != "2500" {
		t.Fatal(err, dm)
	}
	if dm, err = DurationMS("0"); err != nil || dm != "" {
		t.Fatal(err, dm)
	}
	if _, err = DurationMS("86400001"); err == nil {
		t.Fatal("expected reject")
	}
	off, err := OffsetT(" 01500 ")
	if err != nil || off != "1500" {
		t.Fatal(err, off)
	}
	if off, err = OffsetT("0"); err != nil || off != "" {
		t.Fatal(err, off)
	}
	if _, err = OffsetT("86400001"); err == nil {
		t.Fatal("expected reject")
	}
	ac, err := AudioCtx(" 064 ")
	if err != nil || ac != "64" {
		t.Fatal(err, ac)
	}
	if ac, err = AudioCtx("0"); err != nil || ac != "" {
		t.Fatal(err, ac)
	}
	if _, err = AudioCtx("1501"); err == nil {
		t.Fatal("expected reject")
	}
	mc, err := MaxContext("0")
	if err != nil || mc != "0" {
		t.Fatal(err, mc)
	}
	if mc, err = MaxContext("-1"); err != nil || mc != "" {
		t.Fatal(err, mc)
	}
	if _, err = MaxContext("16385"); err == nil {
		t.Fatal("expected reject")
	}
	wt, err := WordThold(" 0.20 ")
	if err != nil || wt != "0.2" {
		t.Fatal(err, wt)
	}
	if wt, err = WordThold(""); err != nil || wt != "" {
		t.Fatal(err, wt)
	}
	if _, err = WordThold("2"); err == nil {
		t.Fatal("expected reject")
	}
	ml, err := MaxLen(" 080 ")
	if err != nil || ml != "80" {
		t.Fatal(err, ml)
	}
	if ml, err = MaxLen("0"); err != nil || ml != "" {
		t.Fatal(err, ml)
	}
	if _, err = MaxLen("1001"); err == nil {
		t.Fatal("expected reject")
	}
	carry, err := CarryInitialFlag("true")
	if err != nil || !carry {
		t.Fatal(err, carry)
	}
	if carry, err = CarryInitialFlag(""); err != nil || carry {
		t.Fatal(err, carry)
	}
	if _, err = CarryInitialFlag("maybe"); err == nil {
		t.Fatal("expected reject")
	}
	best, err := BestOf(" 03 ")
	if err != nil || best != "3" {
		t.Fatal(err, best)
	}
	if best, err = BestOf(""); err != nil || best != "" {
		t.Fatal(err, best)
	}
	if _, err = BestOf("17"); err == nil {
		t.Fatal("expected reject")
	}
	beam, err := BeamSize(" 05 ")
	if err != nil || beam != "5" {
		t.Fatal(err, beam)
	}
	if beam, err = BeamSize(""); err != nil || beam != "" {
		t.Fatal(err, beam)
	}
	if _, err = BeamSize("0"); err == nil {
		t.Fatal("expected reject")
	}
	ns, err := NoSpeechThold(" 0.60 ")
	if err != nil || ns != "0.6" {
		t.Fatal(err, ns)
	}
	if ns, err = NoSpeechThold(""); err != nil || ns != "" {
		t.Fatal(err, ns)
	}
	if _, err = NoSpeechThold("2"); err == nil {
		t.Fatal("expected reject")
	}
	lp, err := LogprobThold(" -1.00 ")
	if err != nil || lp != "-1" {
		t.Fatal(err, lp)
	}
	if lp, err = LogprobThold(""); err != nil || lp != "" {
		t.Fatal(err, lp)
	}
	if _, err = LogprobThold("1"); err == nil {
		t.Fatal("expected reject")
	}
	thold, err := EntropyThold(" 2.40 ")
	if err != nil || thold != "2.4" {
		t.Fatal(err, thold)
	}
	if thold, err = EntropyThold(""); err != nil || thold != "" {
		t.Fatal(err, thold)
	}
	if _, err = EntropyThold("5"); err == nil {
		t.Fatal("expected reject")
	}
	inc, err := TemperatureInc(" 0.20 ")
	if err != nil || inc != "0.2" {
		t.Fatal(err, inc)
	}
	if inc, err = TemperatureInc(""); err != nil || inc != "" {
		t.Fatal(err, inc)
	}
	if _, err = TemperatureInc("2"); err == nil {
		t.Fatal("expected reject")
	}
	if SpeakerLabel(" (speaker 1) hello") != "1" || SpeakerLabel("(speaker ?)") != "?" || SpeakerLabel("hello") != "" {
		t.Fatal("speaker label")
	}
	if StripSpeakerPrefix(" (speaker 1) hello") != "hello" || StripSpeakerPrefix("(speaker ?) there") != "there" || StripSpeakerPrefix("hello") != "hello" {
		t.Fatal("strip speaker")
	}
	on, err := DiarizeFlag("true")
	if err != nil || !on {
		t.Fatal(err, on)
	}
	on, err = DiarizeFlag("")
	if err != nil || on {
		t.Fatal(err, on)
	}
	if _, err := DiarizeFlag("maybe"); err == nil {
		t.Fatal("expected reject")
	}
	on, err = SuppressNSTFlag("1")
	if err != nil || !on {
		t.Fatal(err, on)
	}
	if _, err := SuppressNSTFlag("maybe"); err == nil {
		t.Fatal("expected reject")
	}
	on, err = NoFallbackFlag("true")
	if err != nil || !on {
		t.Fatal(err, on)
	}
	if _, err := NoFallbackFlag("maybe"); err == nil {
		t.Fatal("expected reject")
	}
}

func TestWordTimestamps(t *testing.T) {
	words, err := WordTimestamps([]string{"segment", "word"})
	if err != nil || !words {
		t.Fatal(err, words)
	}
	words, err = WordTimestamps(nil)
	if err != nil || words {
		t.Fatal(err, words)
	}
	if _, err := WordTimestamps([]string{"phone"}); err == nil {
		t.Fatal("expected reject")
	}
}

func TestDetectedLanguage(t *testing.T) {
	code, p := DetectedLanguage("")
	if code != "" || p != nil {
		t.Fatal(code, p)
	}
	code, p = DetectedLanguage("whisper_full_with_state: auto-detected language: es (p = 0.000000)\n")
	if code != "es" || p == nil || *p != 0 {
		t.Fatal(code, p)
	}
	code, p = DetectedLanguage("older: auto-detected language: en (p = 0.250000)\nwhisper_full_with_state: auto-detected language: fr (p = 0.500000)\n")
	if code != "fr" || p == nil || *p != 0.5 {
		t.Fatal(code, p)
	}
	if code, p = DetectedLanguage("auto-detected language: en (p = 1.500000)\n"); code != "en" || p != nil {
		t.Fatal(code, p)
	}
}

func TestLanguageCode(t *testing.T) {
	if c, err := LanguageCode(""); err != nil || c != "" {
		t.Fatal(err, c)
	}
	if c, err := LanguageCode(" auto "); err != nil || c != "auto" {
		t.Fatal(err, c)
	}
	if c, err := LanguageCode("EN"); err != nil || c != "en" {
		t.Fatal(err, c)
	}
	if c, err := LanguageCode("English"); err != nil || c != "en" {
		t.Fatal(err, c)
	}
	if c, err := LanguageCode("haitian creole"); err != nil || c != "ht" {
		t.Fatal(err, c)
	}
	if c, err := LanguageCode("yue"); err != nil || c != "yue" {
		t.Fatal(err, c)
	}
	if _, err := LanguageCode("en-US"); err == nil {
		t.Fatal("expected reject")
	}
	if _, err := LanguageCode("zz"); err == nil {
		t.Fatal("expected reject")
	}
	seen := map[string]struct{}{}
	for _, code := range whisperLanguages {
		seen[code] = struct{}{}
	}
	if len(seen) != 100 {
		t.Fatalf("languages %d", len(seen))
	}
	got := Args("m.bin", "in.wav", "out", "auto", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, false, false, false, false, false)
	want := []string{"-m", "m.bin", "-f", "in.wav", "-otxt", "-of", "out", "-l", "auto"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("auto %q", got)
	}
}

func TestParseJSON(t *testing.T) {
	raw := []byte(`{
		"result": {"language": "en"},
		"transcription": [
			{"offsets": {"from": 0, "to": 1500}, "text": " (speaker 0) hello"},
			{"offsets": {"from": 1500, "to": 3200}, "text": " there"}
		]
	}`)
	got, err := ParseJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got.Text != "(speaker 0) hello there" || got.Language != "en" || len(got.Segments) != 2 {
		t.Fatalf("%+v", got)
	}
	empty, err := ParseJSON([]byte(`{"result":{"language":"es"},"transcription":[]}`))
	if err != nil || empty.Language != "es" || empty.Text != "" || len(empty.Segments) != 0 {
		t.Fatal(err, empty)
	}
	if got.Segments[0].Speaker != "0" || got.Segments[1].Speaker != "" {
		t.Fatalf("speakers %+v", got.Segments)
	}
	if got.Segments[0].Start != 0 || got.Segments[0].End != 1.5 || got.Segments[1].Start != 1.5 || got.Segments[1].End != 3.2 {
		t.Fatalf("%+v", got.Segments)
	}
	if got.Segments[0].Tokens != nil {
		t.Fatal("missing token ids")
	}
	srt := WriteSRT(got.Segments)
	wantSRT := "1\n00:00:00,000 --> 00:00:01,500\n(speaker 0) hello\n\n2\n00:00:01,500 --> 00:00:03,200\nthere\n\n"
	if srt != wantSRT {
		t.Fatalf("srt %q", srt)
	}
	shifted := WriteSRTOffset(got.Segments, 5)
	wantShift := "6\n00:00:00,000 --> 00:00:01,500\n(speaker 0) hello\n\n7\n00:00:01,500 --> 00:00:03,200\nthere\n\n"
	if shifted != wantShift {
		t.Fatalf("offset srt %q", shifted)
	}
	if n, err := OffsetN(" 05 "); err != nil || n != 5 {
		t.Fatal(err, n)
	}
	if n, err := OffsetN("0"); err != nil || n != 0 {
		t.Fatal(err, n)
	}
	if _, err := OffsetN("1000001"); err == nil {
		t.Fatal("expected reject")
	}
	vtt := WriteVTT(got.Segments)
	wantVTT := "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n(speaker 0) hello\n\n00:00:01.500 --> 00:00:03.200\nthere\n\n"
	if vtt != wantVTT {
		t.Fatalf("vtt %q", vtt)
	}
	if WriteSRT(nil) != "" || WriteVTT(nil) != "WEBVTT\n\n" {
		t.Fatal("empty")
	}
	raw = []byte(`{
		"result": {"language": "en"},
		"transcription": [{
			"offsets": {"from": 0, "to": 900},
			"text": " Hello",
			"tokens": [
				{"offsets": {"from": 0, "to": 200}, "text": "[_BEG_]", "id": 50364},
				{"offsets": {"from": 0, "to": 400}, "text": " Hel", "id": 2425},
				{"offsets": {"from": 400, "to": 500}, "text": "lo"},
				{"offsets": {"from": 500, "to": 900}, "text": " world", "id": 0}
			]
		}]
	}`)
	got, err = ParseJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Words) != 2 || got.Words[0].Word != "Hello" || got.Words[0].End != 0.5 || got.Words[1].Word != "world" || got.Words[1].Start != 0.5 {
		t.Fatalf("%+v", got.Words)
	}
	if got.Segments[0].AvgLogprob != nil || got.Segments[0].NoSpeechProb != nil {
		t.Fatal("missing probabilities")
	}
	ids := got.Segments[0].Tokens
	if len(ids) != 3 || ids[0] != 50364 || ids[1] != 2425 || ids[2] != 0 {
		t.Fatalf("token ids %v", ids)
	}
	raw = []byte(`{
		"result": {"language": "en"},
		"transcription": [{
			"offsets": {"from": 0, "to": 900},
			"text": " Hello",
			"tokens": [
				{"offsets": {"from": 0, "to": 100}, "text": " Hel", "t_dtw": 50},
				{"offsets": {"from": 100, "to": 200}, "text": "lo", "t_dtw": 80},
				{"offsets": {"from": 200, "to": 400}, "text": " world", "t_dtw": 150}
			]
		}]
	}`)
	got, err = ParseJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Words) != 2 || got.Words[0].Word != "Hello" || got.Words[0].Start != 0.5 || got.Words[0].End != 1.5 || got.Words[1].Word != "world" || got.Words[1].Start != 1.5 {
		t.Fatalf("dtw %+v", got.Words)
	}
	if preset, err := DTWPreset(" Large.v3 "); err != nil || preset != "large.v3" {
		t.Fatal(err, preset)
	}
	if _, err := DTWPreset("large"); err == nil {
		t.Fatal("expected reject")
	}
	if on, err := VADFlag("true"); err != nil || !on {
		t.Fatal(err, on)
	}
	if on, err := VADFlag(""); err != nil || on {
		t.Fatal(err, on)
	}
	if _, err := VADFlag("maybe"); err == nil {
		t.Fatal("expected reject")
	}
	if th, err := VADThreshold("0.50"); err != nil || th != "0.5" {
		t.Fatal(err, th)
	}
	if th, err := VADThreshold("0"); err != nil || th != "0" {
		t.Fatal(err, th)
	}
	if th, err := VADThreshold(""); err != nil || th != "" {
		t.Fatal(err, th)
	}
	if _, err := VADThreshold("1.1"); err == nil {
		t.Fatal("expected reject")
	}
	if ms, err := VADMinSpeechMS("0250"); err != nil || ms != "250" {
		t.Fatal(err, ms)
	}
	if ms, err := VADMinSpeechMS("0"); err != nil || ms != "0" {
		t.Fatal(err, ms)
	}
	if ms, err := VADMinSpeechMS(""); err != nil || ms != "" {
		t.Fatal(err, ms)
	}
	if _, err := VADMinSpeechMS("-1"); err == nil {
		t.Fatal("expected reject")
	}
	if ms, err := VADMinSilenceMS("0100"); err != nil || ms != "100" {
		t.Fatal(err, ms)
	}
	if ms, err := VADMinSilenceMS("0"); err != nil || ms != "0" {
		t.Fatal(err, ms)
	}
	if ms, err := VADMinSilenceMS(""); err != nil || ms != "" {
		t.Fatal(err, ms)
	}
	if _, err := VADMinSilenceMS("-1"); err == nil {
		t.Fatal("expected reject")
	}
	if sec, err := VADMaxSpeechS("030"); err != nil || sec != "30" {
		t.Fatal(err, sec)
	}
	if sec, err := VADMaxSpeechS(""); err != nil || sec != "" {
		t.Fatal(err, sec)
	}
	if _, err := VADMaxSpeechS("0.5"); err == nil {
		t.Fatal("expected reject")
	}
	if ms, err := VADSpeechPadMS("00"); err != nil || ms != "0" {
		t.Fatal(err, ms)
	}
	if ms, err := VADSpeechPadMS(""); err != nil || ms != "" {
		t.Fatal(err, ms)
	}
	if _, err := VADSpeechPadMS("-1"); err == nil {
		t.Fatal("expected reject")
	}
	if sec, err := VADSamplesOverlap("0.20"); err != nil || sec != "0.2" {
		t.Fatal(err, sec)
	}
	if sec, err := VADSamplesOverlap("0"); err != nil || sec != "0" {
		t.Fatal(err, sec)
	}
	if sec, err := VADSamplesOverlap(""); err != nil || sec != "" {
		t.Fatal(err, sec)
	}
	if _, err := VADSamplesOverlap("-0.1"); err == nil {
		t.Fatal("expected reject")
	}
	if on, err := SplitOnWordFlag("true"); err != nil || !on {
		t.Fatal(err, on)
	}
	if on, err := SplitOnWordFlag(""); err != nil || on {
		t.Fatal(err, on)
	}
	if _, err := SplitOnWordFlag("yes"); err == nil {
		t.Fatal("expected reject")
	}
	if on, err := TinyDiarizeFlag("true"); err != nil || !on {
		t.Fatal(err, on)
	}
	if on, err := TinyDiarizeFlag(""); err != nil || on {
		t.Fatal(err, on)
	}
	raw = []byte(`{
		"result": {"language": "en"},
		"transcription": [{
			"offsets": {"from": 0, "to": 400},
			"text": " Hello",
			"speaker_turn_next": true
		}]
	}`)
	got, err = ParseJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Segments) != 1 || !got.Segments[0].SpeakerTurnNext || got.Segments[0].Text != " Hello [SPEAKER_TURN]" || got.Text != "Hello [SPEAKER_TURN]" {
		t.Fatalf("turn %+v %q", got.Segments, got.Text)
	}
	raw = []byte(`{
		"result": {"language": "en"},
		"transcription": [{
			"offsets": {"from": 0, "to": 400},
			"text": " Hi",
			"tokens": [
				{"text": "[_BEG_]", "p": 0.99},
				{"text": "<|nospeech|>", "p": 0.05},
				{"text": " Hi", "p": 0.5},
				{"text": "!", "p": 0.5},
				{"text": " x", "p": 0}
			]
		}]
	}`)
	got, err = ParseJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	want := math.Log(0.5)
	if got.Segments[0].AvgLogprob == nil || math.Abs(*got.Segments[0].AvgLogprob-want) > 1e-9 {
		t.Fatalf("avg %+v", got.Segments[0].AvgLogprob)
	}
	if got.Segments[0].NoSpeechProb == nil || *got.Segments[0].NoSpeechProb != 0.05 {
		t.Fatalf("nospeech %+v", got.Segments[0].NoSpeechProb)
	}
	if len(got.Logprobs) != 2 || got.Logprobs[0].Token != " Hi" || math.Abs(got.Logprobs[0].Logprob-want) > 1e-9 || got.Logprobs[1].Token != "!" {
		t.Fatalf("%+v", got.Logprobs)
	}
	on, err := IncludeLogprobs([]string{"logprobs"})
	if err != nil || !on {
		t.Fatal(err, on)
	}
	on, err = IncludeLogprobs(nil)
	if err != nil || on {
		t.Fatal(err, on)
	}
	if _, err := IncludeLogprobs([]string{"words"}); err == nil {
		t.Fatal("expected reject")
	}
	if _, ok := WavSeconds([]byte("not-a-wav")); ok {
		t.Fatal("expected non-wav")
	}
	sec, ok := WavSeconds(testWav(16000, 1, 16, 32000))
	if !ok || math.Abs(sec-1) > 1e-9 {
		t.Fatalf("seconds %v %v", sec, ok)
	}
	sec, ok = AudioSeconds("audio/pcm;rate=24000;encoding=float;bits=32", []byte{0, 0, 0, 0, 0, 0, 0, 0}, 0)
	if !ok || math.Abs(sec-2.0/24000) > 1e-12 {
		t.Fatalf("pcm seconds %v %v", sec, ok)
	}
	sec, ok = AudioSeconds("audio/pcm", []byte{0, 0, 0, 0}, 16000)
	if !ok || math.Abs(sec-1.0/16000) > 1e-12 {
		t.Fatalf("fallback rate %v %v", sec, ok)
	}
	if _, ok := AudioSeconds("audio/mpeg", []byte{1, 2, 3}, 24000); ok {
		t.Fatal("mpeg has no duration")
	}
	cue := CueSpan("hello", 1.5)
	if cue.Start != 0 || cue.End != 1.5 {
		t.Fatal(cue)
	}
	span := WriteSRT([]Segment{cue})
	if !strings.Contains(span, "00:00:00,000 --> 00:00:01,500") || !strings.Contains(span, "hello") {
		t.Fatal(span)
	}
	if CueSpan("hello", 0).End != 0 {
		t.Fatal("unknown length")
	}
	if KnownDuration(5, 3.2) != 5 || KnownDuration(0, 3.2) != 3.2 || KnownDuration(0, 0) != 0 {
		t.Fatal("duration")
	}
	if sec, ok := HeaderAudioSeconds("1.5"); !ok || sec != 1.5 {
		t.Fatal(sec, ok)
	}
	if _, ok := HeaderAudioSeconds("0"); ok {
		t.Fatal("zero duration")
	}
	if _, ok := HeaderAudioSeconds("nan"); ok {
		t.Fatal("nan")
	}
	meter := NewStreamMeter("audio/pcm;rate=24000;encoding=float;bits=32", 0)
	meter.Add([]byte{0, 0, 0, 0})
	meter.Add([]byte{0, 0, 0, 0})
	sec, ok = meter.Seconds()
	if !ok || math.Abs(sec-2.0/24000) > 1e-12 {
		t.Fatalf("stream pcm %v %v", sec, ok)
	}
	head := testWav(16000, 1, 16, 0)
	binary.LittleEndian.PutUint32(head[4:8], ^uint32(0))
	binary.LittleEndian.PutUint32(head[40:44], ^uint32(0))
	wavMeter := NewStreamMeter("audio/wav", 0)
	wavMeter.Add(head)
	wavMeter.Add(make([]byte, 32000))
	sec, ok = wavMeter.Seconds()
	if !ok || math.Abs(sec-1) > 1e-9 {
		t.Fatalf("stream wav %v %v", sec, ok)
	}
}

func testWav(rate int, channels, bits int, dataLen int) []byte {
	// 12 byte RIFF header, 8+16 fmt, 8+data.
	fmtSize := 16
	total := 4 + (8 + fmtSize) + (8 + dataLen)
	b := make([]byte, 12+8+fmtSize+8+dataLen)
	copy(b[0:4], "RIFF")
	binary.LittleEndian.PutUint32(b[4:8], uint32(total))
	copy(b[8:12], "WAVE")
	copy(b[12:16], "fmt ")
	binary.LittleEndian.PutUint32(b[16:20], uint32(fmtSize))
	binary.LittleEndian.PutUint16(b[20:22], 1)
	binary.LittleEndian.PutUint16(b[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(b[24:28], uint32(rate))
	byteRate := rate * channels * bits / 8
	binary.LittleEndian.PutUint32(b[28:32], uint32(byteRate))
	binary.LittleEndian.PutUint16(b[32:34], uint16(channels*bits/8))
	binary.LittleEndian.PutUint16(b[34:36], uint16(bits))
	copy(b[36:40], "data")
	binary.LittleEndian.PutUint32(b[40:44], uint32(dataLen))
	return b
}

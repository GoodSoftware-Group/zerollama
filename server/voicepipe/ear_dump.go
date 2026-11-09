package voicepipe

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ZEROLLAMA_EAR_DUMP is a directory. When it is set, each duplex clip is
// written there as a WAV plus a sidecar that records the transcript.
// guess-*.wav is the audio Qwen (or Whisper) actually heard. commit-*.wav
// is the turn as it closed. transcribed=false on a commit means the guess
// text was used and this longer file was not sent to the transcriber.
const earDumpEnv = "ZEROLLAMA_EAR_DUMP"

const earDumpKeep = 80

var (
	earDumpMu  sync.Mutex
	earDumpSeq atomic.Uint64
)

// DumpEarWAV writes wav into ZEROLLAMA_EAR_DUMP. An empty directory, a
// missing env, or a short buffer writes nothing and returns "".
func DumpEarWAV(kind string, wav []byte) string {
	dir := strings.TrimSpace(os.Getenv(earDumpEnv))
	if dir == "" || len(wav) < 44 {
		return ""
	}
	kind = safeDumpKind(kind)
	name := fmt.Sprintf("%s-%s-%03d-%dms.wav",
		time.Now().Format("20060102-150405.000"),
		kind,
		earDumpSeq.Add(1)%1000,
		wavPCMMS(wav),
	)
	path := filepath.Join(dir, name)
	earDumpMu.Lock()
	defer earDumpMu.Unlock()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		slog.Warn("duplex ear dump", "err", err)
		return ""
	}
	if err := os.WriteFile(path, wav, 0o644); err != nil {
		slog.Warn("duplex ear dump", "path", path, "err", err)
		return ""
	}
	pruneEarDump(dir)
	return path
}

// NoteEar records the transcript next to a dump. path comes from DumpEarWAV.
func NoteEar(path, transcript, backend string, transcribed bool, transcribeErr error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return
	}
	errText := ""
	if transcribeErr != nil {
		errText = transcribeErr.Error()
	}
	body := fmt.Sprintf("transcribed: %t\nbackend: %s\ntranscript: %s\nerror: %s\n",
		transcribed, strings.TrimSpace(backend), strings.TrimSpace(transcript), errText)
	note := strings.TrimSuffix(path, filepath.Ext(path)) + ".txt"
	if err := os.WriteFile(note, []byte(body), 0o644); err != nil {
		slog.Warn("duplex ear dump", "path", note, "err", err)
		return
	}
	slog.Info("duplex ear clip",
		"path", path,
		"ms", wavFileMS(path),
		"transcribed", transcribed,
		"backend", backend,
		"transcript", strings.TrimSpace(transcript),
	)
}

func safeDumpKind(kind string) string {
	kind = strings.ToLower(strings.TrimSpace(kind))
	var b strings.Builder
	for _, r := range kind {
		if r >= 'a' && r <= 'z' {
			b.WriteRune(r)
		}
	}
	if b.Len() == 0 {
		return "clip"
	}
	return b.String()
}

func wavPCMMS(wav []byte) int {
	rate, pcm, err := ParseWAV(wav)
	if err != nil || rate <= 0 {
		return 0
	}
	return len(pcm) / 2 * 1000 / rate
}

func wavFileMS(path string) int {
	in, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	return wavPCMMS(in)
}

func pruneEarDump(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	var wavs []string
	for _, ent := range entries {
		if ent.IsDir() {
			continue
		}
		if strings.HasSuffix(ent.Name(), ".wav") {
			wavs = append(wavs, ent.Name())
		}
	}
	if len(wavs) <= earDumpKeep {
		return
	}
	sort.Strings(wavs)
	for _, name := range wavs[:len(wavs)-earDumpKeep] {
		_ = os.Remove(filepath.Join(dir, name))
		_ = os.Remove(filepath.Join(dir, strings.TrimSuffix(name, ".wav")+".txt"))
	}
}

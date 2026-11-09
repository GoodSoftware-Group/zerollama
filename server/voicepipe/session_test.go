package voicepipe

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
)

func TestReplyPredictCapsALongReply(t *testing.T) {
	if ReplyPredict(-1) != ReplyTokens || ReplyPredict(0) != ReplyTokens || ReplyPredict(200) != ReplyTokens {
		t.Fatalf("cap %d %d %d", ReplyPredict(-1), ReplyPredict(0), ReplyPredict(200))
	}
	if ReplyPredict(12) != 12 || ReplyPredict(ReplyTokens) != ReplyTokens {
		t.Fatalf("keep %d %d", ReplyPredict(12), ReplyPredict(ReplyTokens))
	}
}

func TestReplyNumCtxCapsGalleryDefaults(t *testing.T) {
	if ReplyNumCtx(0) != MaxNumCtx || ReplyNumCtx(-1) != MaxNumCtx || ReplyNumCtx(32768) != MaxNumCtx {
		t.Fatalf("cap %d %d %d", ReplyNumCtx(0), ReplyNumCtx(-1), ReplyNumCtx(32768))
	}
	if ReplyNumCtx(2048) != 2048 || ReplyNumCtx(MaxNumCtx) != MaxNumCtx {
		t.Fatalf("keep %d %d", ReplyNumCtx(2048), ReplyNumCtx(MaxNumCtx))
	}
}

func TestWarmReplyPrefillsTheFinishedTurn(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	var mu sync.Mutex
	var priors [][]Exchange
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "hello", nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Sure.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}}, nil
		},
		WarmReply: func(ctx context.Context, instructions string, prior []Exchange) error {
			mu.Lock()
			priors = append(priors, append([]Exchange(nil), prior...))
			mu.Unlock()
			return nil
		},
	})
	br := bufio.NewReader(client)
	if ev := readJSON(t, br); ev["type"] != "session.created" {
		t.Fatal(ev)
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": "AQA="})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.commit"})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "response.done" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	deadline := time.Now().Add(time.Second)
	for {
		mu.Lock()
		n := countFilled(priors)
		mu.Unlock()
		if n >= 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("prefix was not prefilled")
		}
		time.Sleep(5 * time.Millisecond)
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": "AQA="})
	time.Sleep(40 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	var filled []Exchange
	for _, prior := range priors {
		if len(prior) > 0 {
			filled = prior
		}
	}
	if countFilled(priors) != 1 || len(filled) != 1 || filled[0].User != "hello" || filled[0].Assistant != "Sure." {
		t.Fatalf("priors %#v", priors)
	}
}

func countFilled(priors [][]Exchange) int {
	n := 0
	for _, prior := range priors {
		if len(prior) > 0 {
			n++
		}
	}
	return n
}

func TestSpeechTurn(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	var spoken []string
	go Serve(context.Background(), server, nil, Engine{
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			spoken = append(spoken, text)
			pcm := []byte{0, 0, 1, 0}
			return Clip{MIME: "audio/pcm", Rate: 22050, PCM: pcm}, nil
		},
	})
	br := bufio.NewReader(client)
	ev := readJSON(t, br)
	if ev["type"] != "session.created" {
		t.Fatal(ev)
	}
	sendJSON(t, client, map[string]any{"type": "conversation.item.create", "item": map[string]any{"text": "Hello there."}})
	sendJSON(t, client, map[string]any{"type": "response.create"})
	var deltas int
	for {
		ev = readJSON(t, br)
		if ev["type"] == "response.output_audio.delta" {
			deltas++
			if ev["mime"] != "audio/pcm" || ev["rate"] != float64(22050) {
				t.Fatal(ev)
			}
		}
		if ev["type"] == "response.done" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	if deltas != 1 || len(spoken) != 1 || spoken[0] != "Hello there." {
		t.Fatalf("deltas %d spoken %q", deltas, spoken)
	}
}

func TestSpeechItemEmotionReachesTheSpeaker(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	var got string
	go Serve(context.Background(), server, nil, Engine{
		Emotion: "calm",
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			got = emotion
			return Clip{MIME: "audio/pcm", Rate: 24000, PCM: []byte{0, 0}}, nil
		},
	})
	br := bufio.NewReader(client)
	if ev := readJSON(t, br); ev["type"] != "session.created" {
		t.Fatal(ev)
	}
	sendJSON(t, client, map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{"text": "Hello.", "emotion": " Excited "},
	})
	sendJSON(t, client, map[string]any{"type": "response.create"})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "response.done" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	if got != "Excited" {
		t.Fatalf("emotion %q", got)
	}
}

func TestSpeakMoreSendsTheFirstChunk(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	release := make(chan struct{})
	go Serve(context.Background(), server, nil, Engine{
		SpeakMore: func(ctx context.Context, text, voice, language, emotion string, emit func(Clip) error) error {
			if text != "Hello there." {
				t.Errorf("text %q", text)
			}
			if err := emit(Clip{MIME: "audio/pcm", Rate: 16000, PCM: []byte{1, 0}}); err != nil {
				return err
			}
			select {
			case <-release:
			case <-ctx.Done():
				return ctx.Err()
			}
			return emit(Clip{MIME: "audio/pcm", Rate: 16000, PCM: []byte{2, 0}})
		},
	})
	br := bufio.NewReader(client)
	if ev := readJSON(t, br); ev["type"] != "session.created" {
		t.Fatal(ev)
	}
	sendJSON(t, client, map[string]any{"type": "conversation.item.create", "item": map[string]any{"text": "Hello there."}})
	sendJSON(t, client, map[string]any{"type": "response.create"})
	var deltas int
	for {
		ev := readJSON(t, br)
		if ev["type"] == "response.output_audio.delta" {
			deltas++
			if deltas == 1 {
				close(release)
			}
		}
		if ev["type"] == "response.done" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	if deltas != 2 {
		t.Fatalf("deltas %d", deltas)
	}
}

func TestDuplexASRBackendOnWire(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	go Serve(context.Background(), server, nil, Engine{
		Duplex:     true,
		InputRate:  16000,
		ASR:        "qwen3-asr",
		ASRBackend: "qwen3-asr",
		HearEx: func(ctx context.Context, wav []byte, language string) (string, string, error) {
			return "hey", "whisper_fallback", nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Hi.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{Rate: 16000, PCM: []byte{0, 0}}, nil
		},
	})
	br := bufio.NewReader(client)
	created := readJSON(t, br)
	if created["type"] != "session.created" {
		t.Fatal(created)
	}
	sess, _ := created["session"].(map[string]any)
	if sess["asr"] != "qwen3-asr" || sess["asr_backend"] != "qwen3-asr" {
		t.Fatalf("session.created %#v", sess)
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": "AQA="})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.commit"})
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "conversation.item.input_audio_transcription.completed":
			if ev["transcript"] != "hey" || ev["asr"] != "qwen3-asr" || ev["asr_backend"] != "whisper_fallback" {
				t.Fatalf("completed %#v", ev)
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func TestDuplexCommit(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	wav, err := WrapPCM16(16000, []byte{1, 0, 2, 0})
	if err != nil {
		t.Fatal(err)
	}
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, got []byte, language string) (string, error) {
			if language != "en" || string(got) != string(wav) {
				t.Fatalf("lang %q wav %d", language, len(got))
			}
			return "what I heard", nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			if instructions != "be brief" || user != "what I heard" {
				t.Fatalf("inst %q user %q", instructions, user)
			}
			return "Sure. Next.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{Rate: 16000, PCM: []byte{0, 0}}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "session.update", "session": map[string]any{
		"instructions": "be brief",
		"audio":        map[string]any{"input": map[string]any{"transcription": map[string]any{"language": "en"}}},
	}})
	upd := readJSON(t, br)
	if upd["type"] != "session.updated" {
		t.Fatal(upd)
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": "AQA="}) // 1, 0
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": "AgA="}) // 2, 0
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.commit"})
	var transcript string
	var spokenText string
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "conversation.item.input_audio_transcription.completed":
			transcript, _ = ev["transcript"].(string)
		case "response.output_text.delta":
			delta, _ := ev["delta"].(string)
			spokenText += delta
		case "response.done":
			if transcript != "what I heard" || spokenText != "Sure. Next." {
				t.Fatalf("transcript %q text %q", transcript, spokenText)
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func TestBargeKeepsSentencesAlreadySent(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "", nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			if len(prior) == 0 {
				return "One. Two.", nil
			}
			if len(prior) != 1 || prior[0].User != "go" || prior[0].Assistant != "One." {
				return "", fmt.Errorf("prior %+v", prior)
			}
			return "Next.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			if text == "One." {
				<-ctx.Done()
				return Clip{}, ctx.Err()
			}
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "conversation.item.create", "item": map[string]any{"text": "go"}})
	sendJSON(t, client, map[string]any{"type": "response.create"})
	var spoken string
	cancelled := false
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "response.output_text.delta":
			delta, _ := ev["delta"].(string)
			spoken += delta
			if spoken == "One." && !cancelled {
				sendJSON(t, client, map[string]any{"type": "response.cancel"})
				cancelled = true
			}
		case "response.done":
			resp, _ := ev["response"].(map[string]any)
			status, _ := resp["status"].(string)
			if !cancelled {
				t.Fatalf("done before cancel: %q", spoken)
			}
			if status != "cancelled" || spoken != "One." {
				t.Fatalf("status %q text %q", status, spoken)
			}
			sendJSON(t, client, map[string]any{"type": "response.create"})
			spoken = ""
			for {
				ev = readJSON(t, br)
				switch ev["type"] {
				case "response.output_text.delta":
					delta, _ := ev["delta"].(string)
					spoken += delta
				case "response.done":
					if spoken != "Next." {
						t.Fatalf("text %q", spoken)
					}
					return
				case "error":
					t.Fatal(ev)
				}
			}
		case "error":
			t.Fatal(ev)
		}
	}
}

func TestBargeDropsLaterSentences(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	started := make(chan struct{})
	var n int
	go Serve(context.Background(), server, nil, Engine{
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "One. Two.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			n++
			if n == 1 {
				close(started)
				<-ctx.Done()
				return Clip{}, ctx.Err()
			}
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "go", nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	go func() {
		for {
			_, _, err := readFrame(br)
			if err != nil {
				return
			}
		}
	}()
	sendJSON(t, client, map[string]any{"type": "conversation.item.create", "item": map[string]any{"text": "go"}})
	sendJSON(t, client, map[string]any{"type": "response.create"})
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("speak did not start")
	}
	sendJSON(t, client, map[string]any{"type": "response.cancel"})
	time.Sleep(50 * time.Millisecond)
	if n != 1 {
		t.Fatalf("spoke %d", n)
	}
}

func TestTrailingSilenceInASpeechPacketCounts(t *testing.T) {
	g := Gate{Rate: 16000, Turn: Turn{Threshold: 0.02, SilenceMS: 20, PrefixMS: 0}}
	mixed := make([]byte, (1+320)*2)
	binary.LittleEndian.PutUint16(mixed[:2], 4000)
	step := g.Push(mixed, false)
	if step.Commit == nil || step.Draft != nil || !step.Started {
		t.Fatalf("commit %#v", step)
	}

	g.Reset()
	short := make([]byte, (1+10)*2)
	binary.LittleEndian.PutUint16(short[:2], 4000)
	step = g.Push(short, false)
	if step.Commit != nil || step.Draft == nil || !step.Quiet || g.quiet != 10 {
		t.Fatalf("tail %#v quiet %d", step, g.quiet)
	}
	step = g.Push(make([]byte, 310*2), false)
	if step.Commit == nil {
		t.Fatal("remainder did not commit")
	}

	g.Reset()
	lead := make([]byte, 100*2)
	binary.LittleEndian.PutUint16(lead[len(lead)-2:], 4000)
	step = g.Push(lead, false)
	if step.Commit != nil || step.Draft != nil || g.quiet != 0 || !step.Started {
		t.Fatalf("lead %#v quiet %d", step, g.quiet)
	}
}

func TestTrimQuietTailKeepsAShortTail(t *testing.T) {
	long := make([]byte, (1+8000)*2)
	binary.LittleEndian.PutUint16(long[:2], 4000)
	if got := TrimQuietTail(long, 16000, 0.02, hearTailMS); len(got) != (1+3200)*2 {
		t.Fatalf("trimmed %d", len(got))
	}
	short := make([]byte, (1+100)*2)
	binary.LittleEndian.PutUint16(short[:2], 4000)
	if len(TrimQuietTail(short, 16000, 0.02, hearTailMS)) != len(short) {
		t.Fatal("short tail was cut")
	}
	quiet := make([]byte, 8000)
	if len(TrimQuietTail(quiet, 16000, 0.02, hearTailMS)) != len(quiet) {
		t.Fatal("quiet clip was cut")
	}
	lead := make([]byte, (50+1+8000)*2)
	binary.LittleEndian.PutUint16(lead[50*2:], 4000)
	if got := TrimQuietTail(lead, 16000, 0.02, hearTailMS); len(got) != (50+1+3200)*2 {
		t.Fatalf("lead %d", len(got))
	}
}

func TestTranscriberDropsTheLongQuietTail(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	heard := make(chan struct{})
	var got int
	turn := Turn{Threshold: 0.02, SilenceMS: 500, PrefixMS: 0}
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Turn:      &turn,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			_, pcm, err := ParseWAV(wav)
			if err == nil {
				got = len(pcm)
			}
			close(heard)
			return "heard", nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{Rate: 16000, PCM: []byte{0, 0}}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	go func() {
		for {
			if _, _, err := readFrame(br); err != nil {
				return
			}
		}
	}()
	pcm := make([]byte, (1+8000)*2)
	binary.LittleEndian.PutUint16(pcm[:2], 4000)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": base64.StdEncoding.EncodeToString(pcm)})
	select {
	case <-heard:
	case <-time.After(2 * time.Second):
		t.Fatal("silence did not commit")
	}
	if got != (1+3200)*2 {
		t.Fatalf("pcm %d", got)
	}
}

func TestServerVADCommitsOnSilence(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	heard := make(chan struct{})
	var got int
	turn := Turn{Threshold: 0.02, SilenceMS: 500, PrefixMS: 1}
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Turn:      &turn,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			_, pcm, err := ParseWAV(wav)
			if err == nil {
				got = len(pcm)
			}
			close(heard)
			return "heard", nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{Rate: 16000, PCM: []byte{0, 0}}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	go func() {
		for {
			if _, _, err := readFrame(br); err != nil {
				return
			}
		}
	}()
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 20)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 8000)})
	select {
	case <-heard:
	case <-time.After(2 * time.Second):
		t.Fatal("silence did not commit")
	}
	if got != 32+2+6400 {
		t.Fatalf("pcm %d", got)
	}
}

func TestLoudPacketCommitsWhenTheSenderStops(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(2 * time.Second))
	heard := make(chan struct{})
	vad := Turn{Threshold: 0.02, SilenceMS: 80, PrefixMS: 0}
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Turn:      &vad,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			close(heard)
			return "heard", nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	go func() {
		for {
			if _, _, err := readFrame(br); err != nil {
				return
			}
		}
	}()
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	select {
	case <-heard:
	case <-time.After(time.Second):
		t.Fatal("loud packet did not commit")
	}
}

func TestLoudPauseStartsTranscription(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(2 * time.Second))
	started := make(chan struct{})
	release := make(chan struct{})
	defer close(release)
	turn := Turn{Threshold: 0.02, SilenceMS: 400, PrefixMS: 0}
	go Serve(context.Background(), server, nil, Engine{
		Duplex: true, InputRate: 16000, Turn: &turn,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			return func(ctx context.Context, wav []byte) (string, error) {
				close(started)
				select {
				case <-release:
					return "early", nil
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}, nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "input_audio_buffer.speech_started" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	select {
	case <-started:
	case <-time.After(250 * time.Millisecond):
		t.Fatal("transcription waited for the silence to finish")
	}
}

func TestLoadingTranscriberIsKept(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	entered := make(chan struct{})
	var once sync.Once
	release := make(chan struct{})
	var mu sync.Mutex
	var warmN, hearN int
	used := make(chan string, 1)
	turn := Turn{Threshold: 0.02, SilenceMS: 80, PrefixMS: 0}
	go Serve(context.Background(), server, nil, Engine{
		Duplex: true, InputRate: 16000, Turn: &turn,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			mu.Lock()
			hearN++
			mu.Unlock()
			used <- "cold"
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			mu.Lock()
			warmN++
			mu.Unlock()
			once.Do(func() { close(entered) })
			select {
			case <-release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			return func(ctx context.Context, wav []byte) (string, error) {
				used <- "warm"
				return "warm", nil
			}, nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("transcriber did not start loading")
	}
	go func() {
		for {
			if _, _, err := readFrame(br); err != nil {
				return
			}
		}
	}()
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	select {
	case got := <-used:
		t.Fatalf("transcribed with %s before the load finished", got)
	case <-time.After(200 * time.Millisecond):
	}
	close(release)
	select {
	case got := <-used:
		if got != "warm" {
			t.Fatalf("used %s", got)
		}
	case <-time.After(time.Second):
		t.Fatal("loaded transcriber was not used")
	}
	mu.Lock()
	defer mu.Unlock()
	// Socket-open warm plus the next-ear warm after the reply both call WarmHear.
	if warmN < 1 || hearN != 0 {
		t.Fatalf("warm %d hear %d", warmN, hearN)
	}
}

func TestRepeatedLoudPacketsWaitToTranscribe(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(2 * time.Second))
	release := make(chan struct{})
	defer close(release)
	var mu sync.Mutex
	var n int
	turn := Turn{Threshold: 0.02, SilenceMS: 500, PrefixMS: 0}
	go Serve(context.Background(), server, nil, Engine{
		Duplex: true, InputRate: 16000, Turn: &turn,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			return func(ctx context.Context, wav []byte) (string, error) {
				mu.Lock()
				n++
				mu.Unlock()
				select {
				case <-release:
					return "early", nil
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}, nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "input_audio_buffer.speech_started" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	time.Sleep(250 * time.Millisecond)
	mu.Lock()
	got := n
	mu.Unlock()
	if got != 1 {
		t.Fatalf("transcriptions %d", got)
	}
}

func TestIdleSilenceCommitsWithoutMoreAudio(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(2 * time.Second))
	heard := make(chan struct{})
	vad := Turn{Threshold: 0.02, SilenceMS: 80, PrefixMS: 0}
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Turn:      &vad,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			close(heard)
			return "heard", nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	go func() {
		for {
			if _, _, err := readFrame(br); err != nil {
				return
			}
		}
	}()
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 160)})
	select {
	case <-heard:
	case <-time.After(time.Second):
		t.Fatal("idle audio did not commit")
	}
}

func TestQuietDoesNotCommit(t *testing.T) {
	if Loud([]byte{0, 0}, 0) {
		t.Fatal("zero")
	}
	if !Loud([]byte{1, 0}, 0) {
		t.Fatal("threshold 0")
	}
	on, turn, err := ParseTurn(map[string]any{"type": "server_vad", "threshold": 0.0, "silence_duration_ms": 0.0})
	if err != nil || !on || turn.Threshold != 0 || turn.SilenceMS != 0 || turn.PrefixMS != 300 {
		t.Fatal(err, on, turn)
	}
	client, server := net.Pipe()
	defer client.Close()
	heard := make(chan struct{}, 1)
	vad := Turn{Threshold: 0.02, SilenceMS: 20, PrefixMS: 0}
	go Serve(context.Background(), server, nil, Engine{
		Duplex: true, InputRate: 16000, Turn: &vad,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			heard <- struct{}{}
			return "x", nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "x", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	go func() {
		for {
			if _, _, err := readFrame(br); err != nil {
				return
			}
		}
	}()
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 320)})
	select {
	case <-heard:
		t.Fatal("quiet audio committed")
	case <-time.After(150 * time.Millisecond):
	}
}

func b64pcm(sample int16, extraZeros ...int) string {
	n := 1
	if len(extraZeros) == 1 && sample == 0 {
		n = extraZeros[0]
	}
	buf := make([]byte, n*2)
	if sample != 0 {
		buf[0] = byte(sample)
		buf[1] = byte(uint16(sample) >> 8)
	}
	return base64.StdEncoding.EncodeToString(buf)
}

func TestSpeechRejectsTurnDetection(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	go Serve(context.Background(), server, nil, Engine{
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "session.update", "session": map[string]any{
		"audio": map[string]any{"input": map[string]any{"turn_detection": map[string]any{"type": "server_vad"}}},
	}})
	ev := readJSON(t, br)
	msg, _ := ev["error"].(map[string]any)["message"].(string)
	if ev["type"] != "error" || !strings.Contains(msg, "duplex") {
		t.Fatal(ev)
	}
}

func TestLoudBargeQuietDoesNot(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	started := make(chan struct{})
	released := make(chan struct{})
	var n int
	vad := Turn{Threshold: 0.02, SilenceMS: 500, PrefixMS: 0}
	go Serve(context.Background(), server, nil, Engine{
		Duplex: true, InputRate: 16000, Turn: &vad,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "go", nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "One. Two.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			n++
			if n == 1 {
				close(started)
				<-ctx.Done()
				close(released)
				return Clip{}, ctx.Err()
			}
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	go func() {
		for {
			if _, _, err := readFrame(br); err != nil {
				return
			}
		}
	}()
	sendJSON(t, client, map[string]any{"type": "conversation.item.create", "item": map[string]any{"text": "go"}})
	sendJSON(t, client, map[string]any{"type": "response.create"})
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("speak did not start")
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 160)})
	select {
	case <-released:
		t.Fatal("quiet audio cancelled the reply")
	case <-time.After(80 * time.Millisecond):
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	select {
	case <-released:
	case <-time.After(2 * time.Second):
		t.Fatal("loud audio did not cancel the reply")
	}
	if n != 1 {
		t.Fatalf("spoke %d", n)
	}
}

func TestHistoryCap(t *testing.T) {
	s := &session{}
	for i := 0; i < maxHistory+1; i++ {
		s.remember(string(rune('a'+i)), "z")
	}
	if len(s.history) != maxHistory || s.history[0].User != "b" || s.history[maxHistory-1].User != string(rune('a'+maxHistory)) {
		t.Fatalf("%+v", s.history)
	}
}

func TestDuplexRemembersTurn(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	var calls int
	var second []Exchange
	var secondUser string
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			calls++
			if calls == 1 {
				return "one", nil
			}
			return "two", nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			if calls == 2 {
				second = append([]Exchange(nil), prior...)
				secondUser = user
			}
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	commit := func() {
		sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(1)})
		sendJSON(t, client, map[string]any{"type": "input_audio_buffer.commit"})
	}
	waitDone := func() {
		for {
			ev := readJSON(t, br)
			if ev["type"] == "response.done" {
				return
			}
			if ev["type"] == "error" {
				t.Fatal(ev)
			}
		}
	}
	commit()
	waitDone()
	commit()
	waitDone()
	if secondUser != "two" || len(second) != 1 || second[0].User != "one" || second[0].Assistant != "Yes." {
		t.Fatalf("user %q prior %+v", secondUser, second)
	}
}

func TestWAVRoundTrip(t *testing.T) {
	pcm := []byte{0, 0, 0xff, 0x7f}
	wav, err := WrapPCM16(16000, pcm)
	if err != nil {
		t.Fatal(err)
	}
	rate, got, err := ParseWAV(wav)
	if err != nil || rate != 16000 || string(got) != string(pcm) {
		t.Fatalf("rate %d pcm %v err %v", rate, got, err)
	}
	parts := Sentences("Hello. Next?\nTail")
	if strings.Join(parts, "|") != "Hello.|Next?|Tail" {
		t.Fatal(parts)
	}
	var stream SentenceStream
	var sentences []string
	emit := func(s string) error {
		sentences = append(sentences, s)
		return nil
	}
	if err := stream.Push("Su", emit); err != nil {
		t.Fatal(err)
	}
	if err := stream.Push("re. Ne", emit); err != nil {
		t.Fatal(err)
	}
	if err := stream.Push("xt.\nTail", emit); err != nil {
		t.Fatal(err)
	}
	if err := stream.Flush(emit); err != nil {
		t.Fatal(err)
	}
	if strings.Join(sentences, "|") != "Sure.|Next.|Tail" {
		t.Fatal(sentences)
	}
}

func TestSentenceStreamSpeaksTheFirstWords(t *testing.T) {
	var stream SentenceStream
	var got []string
	emit := func(s string) error {
		got = append(got, s)
		return nil
	}
	text := "I can help you now"
	if err := stream.Push(text, emit); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || utf8.RuneCountInString(got[0]) >= clauseRunes {
		t.Fatalf("first %q", got)
	}
	if err := stream.Flush(emit); err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, " ") != text {
		t.Fatalf("%q", got)
	}
}

func TestSentenceStreamKeepsALeadingFiller(t *testing.T) {
	var stream SentenceStream
	var got []string
	emit := func(s string) error {
		got = append(got, s)
		return nil
	}
	if err := stream.Push("Well,", emit); err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("spoke filler alone %q", got)
	}
	if err := stream.Push(" the capital is.", emit); err != nil {
		t.Fatal(err)
	}
	if err := stream.Flush(emit); err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "Well, the capital is." {
		t.Fatal(got)
	}
	stream = SentenceStream{}
	got = nil
	if err := stream.Push("Yes.", emit); err != nil {
		t.Fatal(err)
	}
	if err := stream.Flush(emit); err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "Yes." {
		t.Fatal(got)
	}
	stream = SentenceStream{}
	got = nil
	if err := stream.Push("Well,", emit); err != nil {
		t.Fatal(err)
	}
	if err := stream.Flush(emit); err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "Well," {
		t.Fatal(got)
	}
}

func TestTrimSilenceAndFade(t *testing.T) {
	pcm := make([]byte, 20)
	binary.LittleEndian.PutUint16(pcm[8:10], 1000)
	binary.LittleEndian.PutUint16(pcm[10:12], 1000)
	got := TrimSilencePCM16(pcm, 200)
	if len(got) != 4 {
		t.Fatalf("trim %d", len(got))
	}
	faded := FadePCM16([]byte{0, 4, 0, 4, 0, 4, 0, 4}, 8)
	if binary.LittleEndian.Uint16(faded[6:8]) != 0 {
		t.Fatalf("fade %#v", faded)
	}
}

func TestDuplexDefaultTurnUsesHumanGap(t *testing.T) {
	if DefaultTurn().SilenceMS != DefaultSilenceMS || DefaultSilenceMS != 500 {
		t.Fatal(DefaultTurn())
	}
	if DuplexDefaultTurn().SilenceMS != DuplexSilenceMS || DuplexSilenceMS != 300 {
		t.Fatal(DuplexDefaultTurn())
	}
	on, turn, err := ParseTurn(map[string]any{"type": "server_vad"})
	if err != nil || !on || turn.SilenceMS != 500 {
		t.Fatal(err, on, turn)
	}
	on, turn, err = ParseDuplexTurn(map[string]any{"type": "server_vad"})
	if err != nil || !on || turn.SilenceMS != 300 {
		t.Fatal(err, on, turn)
	}
	on, turn, err = ParseDuplexTurn(map[string]any{"type": "server_vad", "silence_duration_ms": 500})
	if err != nil || !on || turn.SilenceMS != 500 {
		t.Fatal(err, on, turn)
	}
}

func TestSentenceStreamStartsAtAClause(t *testing.T) {
	var stream SentenceStream
	var got []string
	emit := func(s string) error {
		got = append(got, s)
		return nil
	}
	if err := stream.Push("Yes, I can help.", emit); err != nil {
		t.Fatal(err)
	}
	if err := stream.Flush(emit); err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "Yes,|I can help." {
		t.Fatal(got)
	}
	stream = SentenceStream{}
	got = nil
	if err := stream.Push("The time is 15:49.", emit); err != nil {
		t.Fatal(err)
	}
	if err := stream.Flush(emit); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.Join(got, "|"), "15:|") || !strings.Contains(strings.Join(got, " "), "15:49.") {
		t.Fatalf("clock time split: %q", got)
	}
	stream = SentenceStream{}
	got = nil
	if err := stream.Push("Wait: the cup is here.", emit); err != nil {
		t.Fatal(err)
	}
	if err := stream.Flush(emit); err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "Wait:|the cup is here." {
		t.Fatalf("clause colon: %q", got)
	}
	stream = SentenceStream{}
	got = nil
	long := strings.TrimSpace(strings.Repeat("alpha ", 12))
	if err := stream.Push(long, emit); err != nil {
		t.Fatal(err)
	}
	if err := stream.Flush(emit); err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, " ") != long || len(got) < 2 {
		t.Fatalf("%q", got)
	}
	later := false
	for i, part := range got {
		if i < len(got)-1 && utf8.RuneCountInString(part) > clauseRunes {
			t.Fatalf("%q", part)
		}
		if i > 0 && utf8.RuneCountInString(part) > firstClauseRunes {
			later = true
		}
	}
	if !later {
		t.Fatalf("later stretches stayed at the first-word budget %q", got)
	}
}

func TestWarmHearStartsBeforeCommit(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	started := make(chan struct{})
	var once sync.Once
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			once.Do(func() { close(started) })
			return func(ctx context.Context, wav []byte) (string, error) {
				return "warm", nil
			}, nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("transcriber did not start")
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(1)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.commit"})
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "conversation.item.input_audio_transcription.completed":
			if ev["transcript"] != "warm" {
				t.Fatal(ev)
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func TestNextWhisperStartsBeforeTheReply(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	var mu sync.Mutex
	n := 0
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			mu.Lock()
			n++
			mu.Unlock()
			return func(ctx context.Context, wav []byte) (string, error) {
				return "warm", nil
			}, nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(1)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.commit"})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "response.done" || ev["type"] == "error" {
			if ev["type"] == "error" {
				t.Fatal(ev)
			}
			deadline := time.Now().Add(time.Second)
			for {
				mu.Lock()
				got := n
				mu.Unlock()
				if got >= 2 {
					return
				}
				if time.Now().After(deadline) {
					t.Fatalf("next transcriber was not started, calls %d", got)
				}
				time.Sleep(5 * time.Millisecond)
			}
		}
	}
}

func TestPlaybackDoesNotWaitForTheNextWhisper(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	var mu sync.Mutex
	n := 0
	releaseNext := make(chan struct{})
	spoke := make(chan struct{})
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			mu.Lock()
			n++
			call := n
			mu.Unlock()
			if call > 1 {
				<-releaseNext
			}
			return func(ctx context.Context, wav []byte) (string, error) {
				return "warm", nil
			}, nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			mu.Lock()
			got := n
			mu.Unlock()
			if got > 1 {
				t.Errorf("next transcriber started before audio, calls %d", got)
			}
			close(spoke)
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	go func() {
		for {
			if _, _, err := readFrame(br); err != nil {
				return
			}
		}
	}()
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(1)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.commit"})
	select {
	case <-spoke:
	case <-time.After(time.Second):
		t.Fatal("playback waited for the next transcriber")
	}
	close(releaseNext)
}

func TestWarmHearUsesUpdatedLanguage(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	var mu sync.Mutex
	var langs []string
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			mu.Lock()
			langs = append(langs, language)
			mu.Unlock()
			return func(ctx context.Context, wav []byte) (string, error) {
				return "warm", nil
			}, nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "session.update", "session": map[string]any{
		"audio": map[string]any{"input": map[string]any{"transcription": map[string]any{"language": "fr"}}},
	}})
	ev := readJSON(t, br)
	if ev["type"] != "session.updated" {
		t.Fatal(ev)
	}
	deadline := time.Now().Add(time.Second)
	for {
		mu.Lock()
		got := append([]string(nil), langs...)
		mu.Unlock()
		if len(got) > 0 && got[len(got)-1] == "fr" {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal(got)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestStreamSpeaksFirstSentenceFirst(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	started := make(chan struct{})
	var spokeOne bool
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "hi", nil
		},
		StreamReply: func(ctx context.Context, instructions, user string, prior []Exchange, emit func(string) error) error {
			if err := emit("One."); err != nil {
				return err
			}
			select {
			case <-started:
			case <-ctx.Done():
				return ctx.Err()
			}
			if !spokeOne {
				return fmt.Errorf("second sentence started before the first was speaking")
			}
			return emit("Two.")
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			if text == "One." {
				spokeOne = true
				close(started)
			}
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(1)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.commit"})
	var spoken string
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "response.output_text.delta":
			delta, _ := ev["delta"].(string)
			spoken += delta
		case "response.done":
			if spoken != "One. Two." {
				t.Fatalf("text %q", spoken)
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func TestStreamCancelDropsUnsentSentence(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	started := make(chan struct{})
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "hi", nil
		},
		StreamReply: func(ctx context.Context, instructions, user string, prior []Exchange, emit func(string) error) error {
			if err := emit("One."); err != nil {
				return err
			}
			select {
			case <-started:
			case <-ctx.Done():
				return ctx.Err()
			}
			<-ctx.Done()
			return ctx.Err()
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			if text == "One." {
				close(started)
				<-ctx.Done()
				return Clip{}, ctx.Err()
			}
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(1)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.commit"})
	var spoken string
	cancelled := false
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "response.output_text.delta":
			delta, _ := ev["delta"].(string)
			spoken += delta
			if spoken == "One." && !cancelled {
				sendJSON(t, client, map[string]any{"type": "response.cancel"})
				cancelled = true
			}
		case "response.done":
			resp, _ := ev["response"].(map[string]any)
			status, _ := resp["status"].(string)
			if status != "cancelled" || spoken != "One." {
				t.Fatalf("status %q text %q", status, spoken)
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func TestGateDraftsOnTheFirstQuietChunk(t *testing.T) {
	g := Gate{Turn: DefaultTurn(), Rate: 16000}
	loud := []byte{0xA0, 0x0F}
	step := g.Push(loud, false)
	if !step.Started || step.Draft != nil || step.Commit != nil {
		t.Fatalf("start %+v", step)
	}
	quiet := make([]byte, 160*2)
	step = g.Push(quiet, false)
	if step.Commit != nil || len(step.Draft) != len(loud)+len(quiet) {
		t.Fatalf("draft %d commit %d", len(step.Draft), len(step.Commit))
	}
	step = g.Push(quiet, false)
	if step.Draft != nil || step.Commit != nil {
		t.Fatal("second quiet chunk should keep waiting")
	}
	rest := make([]byte, (8000-320)*2)
	step = g.Push(rest, false)
	if step.Draft != nil || len(step.Commit) == 0 {
		t.Fatal("expected commit without a second draft")
	}
}

func TestQuietStartsTranscription(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	started := make(chan struct{})
	release := make(chan struct{})
	turn := DuplexDefaultTurn()
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Turn:      &turn,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			return func(ctx context.Context, wav []byte) (string, error) {
				close(started)
				select {
				case <-release:
					return "early", nil
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}, nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "input_audio_buffer.speech_started" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 160)})
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("transcription did not start when the audio went quiet")
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 7840)})
	close(release)
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "conversation.item.input_audio_transcription.completed":
			if ev["transcript"] != "early" {
				t.Fatal(ev)
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func TestChatStartsDuringSilence(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	heard := make(chan struct{})
	release := make(chan struct{})
	modelStarted := make(chan struct{})
	synthed := make(chan struct{})
	nextReady := make(chan struct{})
	var mu sync.Mutex
	n := 0
	turn := DuplexDefaultTurn()
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Turn:      &turn,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			mu.Lock()
			n++
			call := n
			mu.Unlock()
			if call > 1 {
				close(nextReady)
				return func(ctx context.Context, wav []byte) (string, error) {
					return "next", nil
				}, nil
			}
			return func(ctx context.Context, wav []byte) (string, error) {
				close(heard)
				select {
				case <-release:
					return "hello", nil
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}, nil
		},
		StreamReply: func(ctx context.Context, instructions, user string, prior []Exchange, emit func(string) error) error {
			if user != "hello" {
				t.Errorf("user %q", user)
			}
			close(modelStarted)
			return emit("Yes.")
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			if text == "Yes." {
				close(synthed)
			}
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "input_audio_buffer.speech_started" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 160)})
	select {
	case <-heard:
	case <-time.After(time.Second):
		t.Fatal("transcription did not start")
	}
	close(release)
	select {
	case <-modelStarted:
	case <-time.After(time.Second):
		t.Fatal("chat did not start during the silence")
	}
	select {
	case <-synthed:
	case <-time.After(time.Second):
		t.Fatal("voice did not synthesize during the silence")
	}
	select {
	case <-nextReady:
	case <-time.After(time.Second):
		t.Fatal("next transcriber waited for the turn to commit")
	}
	_ = client.SetReadDeadline(time.Now().Add(40 * time.Millisecond))
	if _, _, err := readFrame(br); err == nil {
		t.Fatal("audio was sent before the turn committed")
	}
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 7840)})
	var text string
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "response.output_text.delta":
			delta, _ := ev["delta"].(string)
			text += delta
		case "response.done":
			if text != "Yes." {
				t.Fatalf("text %q", text)
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func TestSpeechResumeDropsEarlyTranscript(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	started := make(chan struct{})
	var mu sync.Mutex
	n := 0
	turn := DuplexDefaultTurn()
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Turn:      &turn,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			return func(ctx context.Context, wav []byte) (string, error) {
				mu.Lock()
				n++
				call := n
				mu.Unlock()
				if call == 1 {
					close(started)
					<-ctx.Done()
					return "", ctx.Err()
				}
				return "second", nil
			}, nil
		},
		Reply: func(ctx context.Context, instructions, user string, prior []Exchange) (string, error) {
			return "Yes.", nil
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "input_audio_buffer.speech_started" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 160)})
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("transcription did not start")
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 160)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 7840)})
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "conversation.item.input_audio_transcription.completed":
			if ev["transcript"] != "second" {
				t.Fatal(ev)
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func tonePCM(rate, ms int, amp int16) []byte {
	n := rate * ms / 1000
	if n < 1 {
		n = 1
	}
	out := make([]byte, n*2)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(amp))
	}
	return out
}

func TestSpeakStreamPlaysHold(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(5 * time.Second))
	speakGate := make(chan struct{})
	var releaseSpeak sync.Once
	hold := Clip{MIME: "audio/pcm", Rate: 8000, PCM: tonePCM(8000, 400, 1000)}
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hold:      hold,
		HoldMore:  Clip{MIME: "audio/pcm", Rate: 8000, PCM: tonePCM(8000, 200, 800)},
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "hello", nil
		},
		StreamReply: func(ctx context.Context, instructions, user string, prior []Exchange, emit func(string) error) error {
			return emit("The capital is Paris.")
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			select {
			case <-speakGate:
			case <-ctx.Done():
				return Clip{}, ctx.Err()
			}
			return Clip{MIME: "audio/pcm", Rate: 8000, PCM: tonePCM(8000, 40, 2000)}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.commit"})
	var sawHoldAudio bool
	var text string
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "response.output_audio.delta":
			sawHoldAudio = true
			releaseSpeak.Do(func() { close(speakGate) })
		case "response.output_text.delta":
			delta, _ := ev["delta"].(string)
			text += delta
		case "response.done":
			if !sawHoldAudio {
				t.Fatal("hold audio was not sent on speakStream")
			}
			if text != "The capital is Paris." {
				t.Fatalf("text %q", text)
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func TestReadyOpeningSkipsHold(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	heard := make(chan struct{})
	release := make(chan struct{})
	synthed := make(chan struct{})
	turn := DuplexDefaultTurn()
	hold := Clip{MIME: "audio/pcm", Rate: 8000, PCM: tonePCM(8000, 2000, 1000)}
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Turn:      &turn,
		Hold:      hold,
		HoldMore:  Clip{MIME: "audio/pcm", Rate: 8000, PCM: tonePCM(8000, 200, 1000)},
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			return func(ctx context.Context, wav []byte) (string, error) {
				close(heard)
				select {
				case <-release:
					return "hello", nil
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}, nil
		},
		StreamReply: func(ctx context.Context, instructions, user string, prior []Exchange, emit func(string) error) error {
			return emit("The capital is Paris.")
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			close(synthed)
			return Clip{MIME: "audio/pcm", Rate: 8000, PCM: tonePCM(8000, 40, 2000)}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "input_audio_buffer.speech_started" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 160)})
	select {
	case <-heard:
	case <-time.After(time.Second):
		t.Fatal("transcription did not start")
	}
	close(release)
	select {
	case <-synthed:
	case <-time.After(time.Second):
		t.Fatal("opening was not ready before commit")
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 7840)})
	start := time.Now()
	var text string
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "response.output_text.delta":
			delta, _ := ev["delta"].(string)
			text += delta
		case "response.done":
			if text != "The capital is Paris." {
				t.Fatalf("text %q", text)
			}
			if time.Since(start) > 500*time.Millisecond {
				t.Fatalf("hold played for %s", time.Since(start))
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func TestLateOpeningPlaysHold(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(5 * time.Second))
	heard := make(chan struct{})
	release := make(chan struct{})
	speakGate := make(chan struct{})
	var releaseSpeak sync.Once
	turn := DuplexDefaultTurn()
	hold := Clip{MIME: "audio/pcm", Rate: 8000, PCM: tonePCM(8000, 400, 1000)}
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Turn:      &turn,
		Hold:      hold,
		HoldMore:  Clip{MIME: "audio/pcm", Rate: 8000, PCM: tonePCM(8000, 200, 800)},
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			return func(ctx context.Context, wav []byte) (string, error) {
				close(heard)
				select {
				case <-release:
					return "hello", nil
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}, nil
		},
		StreamReply: func(ctx context.Context, instructions, user string, prior []Exchange, emit func(string) error) error {
			return emit("The capital is Paris.")
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			select {
			case <-speakGate:
			case <-ctx.Done():
				return Clip{}, ctx.Err()
			}
			return Clip{MIME: "audio/pcm", Rate: 8000, PCM: tonePCM(8000, 40, 2000)}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "input_audio_buffer.speech_started" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 160)})
	select {
	case <-heard:
	case <-time.After(time.Second):
		t.Fatal("transcription did not start")
	}
	close(release)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 7840)})
	var sawHoldAudio bool
	var text string
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "response.output_audio.delta":
			if text == "" {
				sawHoldAudio = true
				releaseSpeak.Do(func() { close(speakGate) })
			}
		case "response.output_text.delta":
			delta, _ := ev["delta"].(string)
			text += delta
		case "response.done":
			if !sawHoldAudio {
				t.Fatal("hold audio was not sent")
			}
			if text != "The capital is Paris." {
				t.Fatalf("text %q", text)
			}
			if strings.Contains(strings.ToLower(text), "um") || strings.Contains(text, "One moment") {
				t.Fatalf("hold leaked into history text %q", text)
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func TestBargeDropsHold(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(5 * time.Second))
	heard := make(chan struct{})
	release := make(chan struct{})
	speakGate := make(chan struct{})
	cancelled := make(chan struct{})
	var noteCancel sync.Once
	turn := DuplexDefaultTurn()
	hold := Clip{MIME: "audio/pcm", Rate: 8000, PCM: tonePCM(8000, 800, 1000)}
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Turn:      &turn,
		Hold:      hold,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "cold", nil
		},
		WarmHear: func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error) {
			return func(ctx context.Context, wav []byte) (string, error) {
				close(heard)
				select {
				case <-release:
					return "hello", nil
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}, nil
		},
		StreamReply: func(ctx context.Context, instructions, user string, prior []Exchange, emit func(string) error) error {
			return emit("The capital is Paris.")
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			select {
			case <-speakGate:
				return Clip{MIME: "audio/pcm", Rate: 8000, PCM: tonePCM(8000, 40, 2000)}, nil
			case <-ctx.Done():
				noteCancel.Do(func() { close(cancelled) })
				return Clip{}, ctx.Err()
			}
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "input_audio_buffer.speech_started" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 160)})
	select {
	case <-heard:
	case <-time.After(time.Second):
		t.Fatal("transcription did not start")
	}
	close(release)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 7840)})
	for {
		ev := readJSON(t, br)
		if ev["type"] == "response.output_audio.delta" {
			break
		}
		if ev["type"] == "error" {
			t.Fatal(ev)
		}
	}
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	for {
		ev := readJSON(t, br)
		switch ev["type"] {
		case "response.done":
			status, _ := ev["response"].(map[string]any)["status"].(string)
			if status != "cancelled" {
				t.Fatalf("status %q", status)
			}
			select {
			case <-cancelled:
			case <-time.After(time.Second):
				t.Fatal("speak was not cancelled")
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
}

func sendJSON(t *testing.T, conn net.Conn, ev map[string]any) {
	t.Helper()
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteClientFrame(conn, b); err != nil {
		t.Fatal(err)
	}
}

func TestDuplexStatusStages(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(5 * time.Second))
	vad := Turn{Threshold: 0.02, SilenceMS: 80, PrefixMS: 0}
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Turn:      &vad,
		Stages:    &StageReporter{},
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "hey", nil
		},
		StreamReply: func(ctx context.Context, instructions, user string, prior []Exchange, emit func(string) error) error {
			return emit("Hello.")
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{1, 0, 1, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(4000)})
	saw := map[string]bool{}
	quietSent := false
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		_ = client.SetDeadline(time.Now().Add(time.Second))
		ev := readJSON(t, br)
		switch ev["type"] {
		case "input_audio_buffer.speech_started":
			saw["speech_started"] = true
			if !quietSent {
				quietSent = true
				sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(0, 1600)})
			}
		case "response.status":
			st, _ := ev["status"].(string)
			saw[st] = true
			if st == "silence_started" {
				if ev["silence_duration_ms"] != float64(80) {
					t.Fatalf("silence_duration_ms = %#v", ev["silence_duration_ms"])
				}
			}
		case "conversation.item.input_audio_transcription.started":
			saw["stt_started"] = true
		case "conversation.item.input_audio_transcription.completed":
			saw["stt_done"] = true
		case "response.output_audio.done":
			saw["audio_done_event"] = true
		case "response.done":
			if !saw["silence_started"] || !saw["transcribing"] || !saw["generating"] || !saw["audio_done"] {
				t.Fatalf("missing stages: %#v", saw)
			}
			if !saw["stt_started"] || !saw["stt_done"] || !saw["audio_done_event"] {
				t.Fatalf("missing events: %#v", saw)
			}
			return
		case "error":
			t.Fatal(ev)
		}
	}
	t.Fatalf("timed out stages=%#v", saw)
}

func TestFailSendsModelLoadCode(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(3 * time.Second))
	go Serve(context.Background(), server, nil, Engine{
		Duplex:    true,
		InputRate: 16000,
		Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			return "hey", nil
		},
		StreamReply: func(ctx context.Context, instructions, user string, prior []Exchange, emit func(string) error) error {
			return &Error{Message: "the chat model did not finish loading", Code: "model_load_failed"}
		},
		Speak: func(ctx context.Context, text, voice, language, emotion string) (Clip, error) {
			return Clip{PCM: []byte{0, 0}, Rate: 8000}, nil
		},
	})
	br := bufio.NewReader(client)
	_ = readJSON(t, br)
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.append", "audio": b64pcm(1)})
	sendJSON(t, client, map[string]any{"type": "input_audio_buffer.commit"})
	for {
		ev := readJSON(t, br)
		if ev["type"] != "error" {
			continue
		}
		errObj, _ := ev["error"].(map[string]any)
		if errObj["code"] != "model_load_failed" {
			t.Fatalf("code = %#v", errObj["code"])
		}
		msg, _ := errObj["message"].(string)
		if msg != "the chat model did not finish loading" || strings.Contains(msg, "{") {
			t.Fatalf("message = %q", msg)
		}
		return
	}
}

func readJSON(t *testing.T, r *bufio.Reader) map[string]any {
	t.Helper()
	opcode, payload, err := readFrame(r)
	if err != nil || opcode != 0x1 {
		t.Fatal(err, opcode)
	}
	var ev map[string]any
	if err := json.Unmarshal(payload, &ev); err != nil {
		t.Fatal(err)
	}
	return ev
}

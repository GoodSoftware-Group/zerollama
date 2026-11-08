package main

import (
	"bufio"
	"context"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/server/voicepipe"
)

// fakeRealtime is a llama-server socket on 127.0.0.1. It sends session.created,
// reads the client turn, then replies. barge waits for response.cancel.
func fakeRealtime(t *testing.T, barge bool, transcript string) (host string, seen <-chan []string) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	out := make(chan []string, 1)
	go func() {
		defer close(out)
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		br := bufio.NewReader(conn)
		key := readUpgradeKey(br)
		if key == "" {
			return
		}
		sum := sha1.Sum([]byte(key + wsGUID))
		accept := base64.StdEncoding.EncodeToString(sum[:])
		io.WriteString(conn, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: "+accept+"\r\n\r\n")

		writeServerText(conn, `{"type":"session.created","session":{"audio":{"input":{"format":{"type":"audio/pcm","rate":16000}},"output":{"format":{"type":"audio/pcm","rate":24000}}}}}`)
		_ = conn.SetReadDeadline(time.Now().Add(5 * time.Second))
		var types []string
		sawTools := false
		responses := 0
		replies := 0
		userFollow := false
		toolOut := ""
		for {
			payload, err := readClientText(br)
			if err != nil {
				out <- types
				return
			}
			var ev map[string]any
			if json.Unmarshal(payload, &ev) != nil {
				continue
			}
			kind, _ := ev["type"].(string)
			types = append(types, kind)
			switch kind {
			case "session.update":
				if strings.Contains(string(payload), `"tools"`) {
					sawTools = true
				}
				writeServerText(conn, `{"type":"session.updated","session":{"audio":{"input":{"format":{"type":"audio/pcm","rate":16000}},"output":{"format":{"type":"audio/pcm","rate":24000}}}}}`)
				continue
			case "conversation.item.create":
				if item, ok := ev["item"].(map[string]any); ok && item["type"] == "function_call_output" {
					toolOut, _ = item["output"].(string)
					if item["call_id"] != "call_1" {
						toolOut = ""
					}
				} else if item, ok := ev["item"].(map[string]any); ok && item["type"] == "message" && replies > 0 {
					userFollow = true
				}
			case "input_audio_buffer.append":
				if replies > 0 {
					userFollow = true
				}
			case "input_audio_buffer.commit":
				if replies > 0 {
					userFollow = true
				}
				if _, ok := ev["final"]; ok && transcript != "" {
					writeServerText(conn, `{"type":"conversation.item.input_audio_transcription.completed","transcript":"`+transcript+`"}`)
					out <- types
					return
				}
			case "response.create":
				if userFollow {
					userFollow = false
					writeServerText(conn, `{"type":"response.output_text.delta","delta":"next"}`)
					writeServerText(conn, `{"type":"response.done","response":{"status":"completed"}}`)
					continue
				}
				if sawTools && responses == 0 {
					responses++
					writeServerText(conn, `{"type":"response.function_call_arguments.done","name":"lookup","call_id":"call_1","arguments":"{}"}`)
					writeServerText(conn, `{"type":"response.output_text.delta","delta":"`+transcript+`"}`)
					writeServerText(conn, `{"type":"response.done","response":{"status":"completed"}}`)
					continue
				}
				if responses > 0 {
					if toolOut == "" {
						writeServerText(conn, `{"type":"error","error":{"message":"missing tool output"}}`)
						out <- types
						return
					}
					writeServerText(conn, `{"type":"response.output_text.delta","delta":"sunny"}`)
					writeServerText(conn, `{"type":"response.done","response":{"status":"completed"}}`)
					replies++
					continue
				}
				if barge {
					writeServerText(conn, `{"type":"response.output_text.delta","delta":"hi"}`)
					continue
				}
				pcm := base64.StdEncoding.EncodeToString(pcm16Bytes([]int16{1000, -1000}))
				writeServerText(conn, `{"type":"response.output_text.delta","delta":"`+transcript+`"}`)
				writeServerText(conn, `{"type":"response.output_audio.delta","delta":"`+pcm+`"}`)
				writeServerText(conn, `{"type":"response.done","response":{"status":"completed"}}`)
				replies++
				continue
			case "response.cancel":
				writeServerText(conn, `{"type":"response.done","response":{"status":"cancelled"}}`)
				out <- types
				return
			}
		}
	}()
	return "http://" + ln.Addr().String(), out
}

func readUpgradeKey(br *bufio.Reader) string {
	var key string
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return ""
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			return key
		}
		k, v, ok := strings.Cut(line, ":")
		if ok && strings.EqualFold(strings.TrimSpace(k), "Sec-WebSocket-Key") {
			key = strings.TrimSpace(v)
		}
	}
}

func writeServerText(conn net.Conn, s string) {
	p := []byte(s)
	hdr := []byte{0x81, byte(len(p))}
	if len(p) >= 126 {
		hdr = []byte{0x81, 126, byte(len(p) >> 8), byte(len(p))}
	}
	conn.Write(hdr)
	conn.Write(p)
}

func readClientText(br *bufio.Reader) ([]byte, error) {
	hdr := make([]byte, 2)
	if _, err := io.ReadFull(br, hdr); err != nil {
		return nil, err
	}
	if hdr[0]&0x0f != 0x1 {
		return nil, io.ErrUnexpectedEOF
	}
	n := int(hdr[1] & 0x7f)
	if n == 126 {
		var ext [2]byte
		if _, err := io.ReadFull(br, ext[:]); err != nil {
			return nil, err
		}
		n = int(binary.BigEndian.Uint16(ext[:]))
	}
	var mask [4]byte
	if hdr[1]&0x80 != 0 {
		if _, err := io.ReadFull(br, mask[:]); err != nil {
			return nil, err
		}
	}
	payload := make([]byte, n)
	if _, err := io.ReadFull(br, payload); err != nil {
		return nil, err
	}
	if hdr[1]&0x80 != 0 {
		for i := range payload {
			payload[i] ^= mask[i%4]
		}
	}
	return payload, nil
}

func TestProbeConversationLoopback(t *testing.T) {
	host, seen := fakeRealtime(t, false, "hello")
	dir := t.TempDir()
	out := dir + "/reply.wav"
	report, err := probeRealtime(host, "lfm3-audio", "", "what is this", []string{"audio"}, false, out, []int16{1, -1, 2, -2}, 8000, 1000, 5*time.Second, nil, "", "", nil, 0, "", "", realtimeExtra{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Transcript != "hello" || report.OutputRate != 24000 || report.SessionRate != 16000 || !report.Resampled {
		t.Fatalf("%+v", report)
	}
	if report.AudioBytes != 4 || math.Abs(report.Seconds-2.0/24000) > 1e-12 || report.FirstTextMs < 0 || report.Status != "completed" {
		t.Fatalf("%+v", report)
	}
	raw, err := os.ReadFile(out)
	if err != nil || !strings.HasPrefix(string(raw), "RIFF") {
		t.Fatalf("wav %v %q", err, raw)
	}
	types := <-seen
	joined := strings.Join(types, ",")
	if !strings.Contains(joined, "conversation.item.create") || !strings.Contains(joined, "response.create") {
		t.Fatalf("client events %s", joined)
	}
}

func TestProbeBargeLoopback(t *testing.T) {
	host, seen := fakeRealtime(t, true, "")
	report, err := probeRealtime(host, "lfm3-audio", "", "stop", nil, true, "", nil, 0, 320, 5*time.Second, nil, "", "", nil, 0, "", "", realtimeExtra{})
	if err != nil {
		t.Fatal(err)
	}
	if !report.Cancelled || report.Status != "cancelled" || report.Transcript != "hi" {
		t.Fatalf("%+v", report)
	}
	types := <-seen
	if joined := strings.Join(types, ","); !strings.Contains(joined, "response.cancel") {
		t.Fatalf("client events %s", joined)
	}
}

func TestProbeTranscribeLoopback(t *testing.T) {
	host, seen := fakeRealtime(t, false, "heard")
	report, err := probeRealtime(host, "voxtral-realtime", "transcription", "", nil, false, "", []int16{3, 4}, 16000, 1000, 5*time.Second, nil, "", "", nil, 0, "", "", realtimeExtra{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Transcript != "heard" || report.Resampled || report.SessionRate != 16000 {
		t.Fatalf("%+v", report)
	}
	types := <-seen
	joined := strings.Join(types, ",")
	if strings.Contains(joined, "response.create") || !strings.Contains(joined, "input_audio_buffer.commit") {
		t.Fatalf("client events %s", joined)
	}
}

func TestProbeTranscriptionLanguage(t *testing.T) {
	host, seen := fakeRealtime(t, false, "heard")
	update, err := sessionUpdate("", "", "", "", "EN", "transcription")
	if err != nil {
		t.Fatal(err)
	}
	report, err := probeRealtime(host, "lfm3-audio", "transcription", "", nil, false, "", []int16{3, 4}, 16000, 1000, 5*time.Second, update, "", "", nil, 0, "", "", realtimeExtra{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Transcript != "heard" {
		t.Fatalf("%+v", report)
	}
	got := <-seen
	if len(got) < 2 || got[0] != "session.update" || got[len(got)-1] != "input_audio_buffer.commit" {
		t.Fatalf("events %v", got)
	}
}

func TestProbeTranscribeLanguage(t *testing.T) {
	var lang, prompt, temp, diarize, tinyDiarize, splitOnWord, maxTokens, suppress, noFallback, noTimestamps, printSpecial, detectLanguage, vad, vadThreshold, vadMinSpeech, vadMinSilence, vadMaxSpeech, vadSpeechPad, vadOverlap, tempInc, entropy, logprob, noSpeech, beam, bestOf, carry, maxLen, wordThold, maxContext, audioCtx, offsetT, durationMS, suppressRegex, offsetN, grammar, grammarRule, grammarPenalty, dtw string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		lang = r.FormValue("language")
		prompt = r.FormValue("prompt")
		temp = r.FormValue("temperature")
		diarize = r.FormValue("diarize")
		tinyDiarize = r.FormValue("tinydiarize")
		splitOnWord = r.FormValue("split_on_word")
		maxTokens = r.FormValue("max_tokens")
		suppress = r.FormValue("suppress_nst")
		noFallback = r.FormValue("no_fallback")
		tempInc = r.FormValue("temperature_inc")
		entropy = r.FormValue("entropy_thold")
		logprob = r.FormValue("logprob_thold")
		noSpeech = r.FormValue("no_speech_thold")
		beam = r.FormValue("beam_size")
		bestOf = r.FormValue("best_of")
		carry = r.FormValue("carry_initial_prompt")
		maxLen = r.FormValue("max_len")
		wordThold = r.FormValue("word_thold")
		maxContext = r.FormValue("max_context")
		audioCtx = r.FormValue("audio_ctx")
		offsetT = r.FormValue("offset_t")
		durationMS = r.FormValue("duration_ms")
		suppressRegex = r.FormValue("suppress_regex")
		offsetN = r.FormValue("offset_n")
		noTimestamps = r.FormValue("no_timestamps")
		printSpecial = r.FormValue("print_special")
		detectLanguage = r.FormValue("detect_language")
		dtw = r.FormValue("dtw")
		vad = r.FormValue("vad")
		vadThreshold = r.FormValue("vad_threshold")
		vadMinSpeech = r.FormValue("vad_min_speech_duration_ms")
		vadMinSilence = r.FormValue("vad_min_silence_duration_ms")
		vadMaxSpeech = r.FormValue("vad_max_speech_duration_s")
		vadSpeechPad = r.FormValue("vad_speech_pad_ms")
		vadOverlap = r.FormValue("vad_samples_overlap")
		grammar = r.FormValue("grammar")
		grammarRule = r.FormValue("grammar_rule")
		grammarPenalty = r.FormValue("grammar_penalty")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text":"hola"}`))
	}))
	defer srv.Close()
	report, err := probeTranscribe(srv.URL, "lfm3-audio", "clip.wav", []int16{1, 2}, 16000, "en", "names", "0.2", false, "json", false, false, "", false, true, "32", true, true, "0.20", "2.40", "-1.0", "0.60", "05", "03", "080", "0.20", "0", "064", "01500", "02500", `\[MUSIC\]`, "05", `root ::= "yes"`, "root", "00", "large.v3", "0.50", "0250", "0100", "030", "00", "0.20", true, true, true, true, true, false, true, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if lang != "en" || prompt != "names" || temp != "0.2" || diarize != "true" || maxTokens != "32" || suppress != "true" || noFallback != "true" || noTimestamps != "true" || printSpecial != "true" || detectLanguage != "true" || vad != "true" || vadThreshold != "0.5" || vadMinSpeech != "250" || vadMinSilence != "100" || vadMaxSpeech != "30" || vadSpeechPad != "0" || vadOverlap != "0.2" || tinyDiarize != "" || splitOnWord != "true" || tempInc != "0.2" || entropy != "2.4" || logprob != "-1" || noSpeech != "0.6" || beam != "5" || bestOf != "3" || carry != "true" || maxLen != "80" || wordThold != "0.2" || maxContext != "0" || audioCtx != "64" || offsetT != "1500" || durationMS != "2500" || suppressRegex != `\[MUSIC\]` || offsetN != "5" || grammar != `root ::= "yes"` || grammarRule != "root" || grammarPenalty != "0" || dtw != "large.v3" || report.Transcript != "hola" {
		t.Fatalf("lang %q prompt %q temp %q diarize %q max_tokens %q suppress %q no_fallback %q no_timestamps %q print_special %q detect_language %q vad %q vad_threshold %q vad_min_speech_duration_ms %q vad_min_silence_duration_ms %q vad_max_speech_duration_s %q vad_speech_pad_ms %q vad_samples_overlap %q tinydiarize %q split_on_word %q temperature_inc %q entropy_thold %q logprob_thold %q no_speech_thold %q beam_size %q best_of %q carry_initial_prompt %q max_len %q word_thold %q max_context %q audio_ctx %q offset_t %q duration_ms %q suppress_regex %q offset_n %q grammar %q grammar_rule %q grammar_penalty %q dtw %q %+v", lang, prompt, temp, diarize, maxTokens, suppress, noFallback, noTimestamps, printSpecial, detectLanguage, vad, vadThreshold, vadMinSpeech, vadMinSilence, vadMaxSpeech, vadSpeechPad, vadOverlap, tinyDiarize, splitOnWord, tempInc, entropy, logprob, noSpeech, beam, bestOf, carry, maxLen, wordThold, maxContext, audioCtx, offsetT, durationMS, suppressRegex, offsetN, grammar, grammarRule, grammarPenalty, dtw, report)
	}
}

func TestProbeTranslatePath(t *testing.T) {
	var path, tinyDiarize string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		path = r.URL.Path
		tinyDiarize = r.FormValue("tinydiarize")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text":"hello"}`))
	}))
	defer srv.Close()
	report, err := probeTranscribe(srv.URL, "whisper-base", "clip.wav", []int16{1}, 16000, "es", "", "", true, "json", false, false, "", false, false, "", false, false, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, true, false, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if path != "/v1/audio/translations" || tinyDiarize != "true" || report.Transcript != "hello" || report.Mode != "translate" {
		t.Fatalf("path %q tinydiarize %q %+v", path, tinyDiarize, report)
	}
}

func TestProbeTranscribeFormat(t *testing.T) {
	var format, grains string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		format = r.FormValue("response_format")
		grains = strings.Join(r.Form["timestamp_granularities[]"], ",")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text":"hello there","words":[{"word":"hello"},{"word":"there"}]}`))
	}))
	defer srv.Close()
	dir := t.TempDir()
	out := dir + "/out.json"
	report, err := probeTranscribe(srv.URL, "whisper-base", "clip.wav", []int16{1}, 16000, "", "", "", false, "verbose_json", true, false, out, false, false, "", false, false, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if format != "verbose_json" || grains != "word" || report.Transcript != "hello there" || report.Words != 2 || report.OutFile != out {
		t.Fatalf("format %q grains %q %+v", format, grains, report)
	}
	if _, err := cleanResponseFormat("SRT"); err != nil || mustFormat(t, "vtt") != "vtt" {
		t.Fatal("format")
	}
	if _, err := cleanResponseFormat("mp3"); err == nil {
		t.Fatal("expected reject")
	}
}

func TestProbeTranscribeStream(t *testing.T) {
	var stream string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		stream = r.FormValue("stream")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hel\"}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		_, _ = io.WriteString(w, "data: {\"type\":\"transcript.text.done\",\"text\":\"hello\",\"usage\":{\"type\":\"tokens\",\"total_tokens\":7,\"seconds\":1.25}}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	report, err := probeTranscribe(srv.URL, "parakeet", "clip.wav", []int16{1}, 16000, "", "", "", false, "json", false, false, "", true, false, "", false, false, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if stream != "true" || report.Transcript != "hello" || report.Tokens != 7 || report.Seconds != 1.25 {
		t.Fatalf("stream %q %+v", stream, report)
	}
}

func TestProbeTranscribeLogprobs(t *testing.T) {
	var include string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		include = r.FormValue("include[]")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text":"hi","usage":{"type":"duration","seconds":1.5},"logprobs":[{"token":"hi","logprob":-0.1,"bytes":[104,105]}]}`))
	}))
	defer srv.Close()
	report, err := probeTranscribe(srv.URL, "whisper-base", "clip.wav", []int16{1}, 16000, "", "", "", false, "json", false, true, "", false, false, "", false, false, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if include != "logprobs" || report.Transcript != "hi" || report.Logprobs != 1 || report.Seconds != 1.5 {
		t.Fatalf("include %q %+v", include, report)
	}
}

func mustFormat(t *testing.T, s string) string {
	t.Helper()
	got, err := cleanResponseFormat(s)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func TestProbeSessionUpdate(t *testing.T) {
	host, seen := fakeRealtime(t, false, "ok")
	update, err := sessionUpdate("be brief", `[{"type":"function","name":"lookup","parameters":{"type":"object"}}]`, "required", "16", "", "")
	if err != nil {
		t.Fatal(err)
	}
	report, err := probeRealtime(host, "lfm3-audio", "", "hello", nil, false, "", nil, 0, 320, 5*time.Second, update, "", "", nil, 0, "", "", realtimeExtra{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Transcript != "ok" || len(report.ToolCalls) != 1 || report.ToolCalls[0].Name != "lookup" || report.ToolCalls[0].CallID != "call_1" {
		t.Fatalf("%+v", report)
	}
	got := <-seen
	want := []string{"session.update", "conversation.item.create", "response.create"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("events %v", got)
	}
}

func TestProbeFollowAudio(t *testing.T) {
	host, seen := fakeRealtime(t, false, "ok")
	report, err := probeRealtime(host, "lfm3-audio", "", "hello", nil, false, "", nil, 0, 320, 5*time.Second, nil, "", "", []int16{9, 8}, 16000, "", "", realtimeExtra{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Transcript != "next" {
		t.Fatalf("%+v", report)
	}
	got := <-seen
	want := "conversation.item.create,response.create,input_audio_buffer.append,input_audio_buffer.commit,response.create"
	if strings.Join(got, ",") != want {
		t.Fatalf("events %v", got)
	}
}

func TestProbeFollowTurn(t *testing.T) {
	host, seen := fakeRealtime(t, false, "ok")
	report, err := probeRealtime(host, "lfm3-audio", "", "hello", nil, false, "", nil, 0, 320, 5*time.Second, nil, "", "and then", nil, 0, "", "", realtimeExtra{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Transcript != "next" {
		t.Fatalf("%+v", report)
	}
	got := <-seen
	want := "conversation.item.create,response.create,conversation.item.create,response.create"
	if strings.Join(got, ",") != want {
		t.Fatalf("events %v", got)
	}
}

func TestProbeToolOutput(t *testing.T) {
	host, seen := fakeRealtime(t, false, "ok")
	update, err := sessionUpdate("be brief", `[{"type":"function","name":"lookup","parameters":{"type":"object"}}]`, "required", "16", "", "")
	if err != nil {
		t.Fatal(err)
	}
	report, err := probeRealtime(host, "lfm3-audio", "", "hello", nil, false, "", nil, 0, 320, 5*time.Second, update, "72F", "", nil, 0, "", "", realtimeExtra{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Transcript != "sunny" || len(report.ToolCalls) != 1 || report.ToolCalls[0].CallID != "call_1" {
		t.Fatalf("%+v", report)
	}
	got := <-seen
	want := "session.update,conversation.item.create,response.create,conversation.item.create,response.create"
	if strings.Join(got, ",") != want {
		t.Fatalf("events %v", got)
	}
}

func TestProbeDuplexSocket(t *testing.T) {
	host, _ := serveDuplexProbe(t, duplexStub{})
	extra := realtimeExtra{Pipeline: "duplex", Voice: "piper", ASR: "whisper", Rate: 16000}
	report, err := probeRealtime(host, "chat", "transcription", "", nil, false, "", []int16{1, -1}, 16000, 1000, 5*time.Second, nil, "", "", nil, 0, "", "", extra)
	if err != nil {
		t.Fatal(err)
	}
	if report.Transcript != "heard" || report.Reply != "Yes." || report.AudioBytes != 4 || report.OutputRate != 16000 || report.Status != "completed" {
		t.Fatalf("%+v", report)
	}
}

func TestProbeDuplexBarge(t *testing.T) {
	host, _ := serveDuplexProbe(t, duplexStub{block: true})
	extra := realtimeExtra{Pipeline: "duplex", Voice: "piper", ASR: "whisper", Rate: 16000}
	report, err := probeRealtime(host, "chat", "transcription", "", nil, true, "", []int16{1, -1}, 16000, 1000, 5*time.Second, nil, "", "", nil, 0, "", "", extra)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Cancelled || report.Status != "cancelled" || report.Transcript != "heard" || report.Reply != "Yes." {
		t.Fatalf("%+v", report)
	}
}

func TestProbeDuplexServerVAD(t *testing.T) {
	host, query := serveDuplexProbe(t, duplexStub{})
	extra := realtimeExtra{Pipeline: "duplex", Voice: "piper", ASR: "whisper", Rate: 16000}
	report, err := probeRealtime(host, "chat", "transcription", "", nil, false, "", []int16{4000}, 16000, 1000, 5*time.Second, nil, "", "", nil, 0, "server_vad", "", extra)
	if err != nil {
		t.Fatal(err)
	}
	q := <-query
	if q.Get("turn") != "server_vad" || q.Get("pipeline") != "duplex" || q.Get("intent") != "" {
		t.Fatalf("query %v", q)
	}
	if report.Transcript != "heard" || report.Reply != "Yes." || report.AudioBytes != 4 || report.Status != "completed" {
		t.Fatalf("%+v", report)
	}
}

func TestProbeDuplexRawAudio(t *testing.T) {
	raw := []byte{0x49, 0x44, 0x33, 0x04}
	host, _ := serveDuplexProbe(t, duplexStub{clip: voicepipe.Clip{MIME: "audio/mpeg", Raw: raw}})
	extra := realtimeExtra{Pipeline: "duplex", Voice: "piper", ASR: "whisper", Rate: 16000}
	out := t.TempDir() + "/reply.mp3"
	report, err := probeRealtime(host, "chat", "transcription", "", nil, false, out, []int16{1, -1}, 16000, 1000, 5*time.Second, nil, "", "", nil, 0, "", "", extra)
	if err != nil {
		t.Fatal(err)
	}
	got, rerr := os.ReadFile(out)
	if rerr != nil {
		t.Fatal(rerr)
	}
	if report.AudioMIME != "audio/mpeg" || report.AudioBytes != len(raw) || report.Seconds != 0 || string(got) != string(raw) {
		t.Fatalf("%+v file %q", report, got)
	}
}

type duplexStub struct {
	block bool
	clip  voicepipe.Clip
	hear  func(context.Context, []byte, string) (string, error)
	reply func(context.Context, string, string, []voicepipe.Exchange) (string, error)
}

func TestProbeDuplexFollowRemembers(t *testing.T) {
	var n int
	host, _ := serveDuplexProbe(t, duplexStub{
		hear: func(ctx context.Context, wav []byte, language string) (string, error) {
			n++
			if n == 1 {
				return "heard", nil
			}
			return "again", nil
		},
		reply: func(ctx context.Context, instructions, user string, prior []voicepipe.Exchange) (string, error) {
			if len(prior) == 0 {
				if user != "heard" {
					return "", fmt.Errorf("first user %q", user)
				}
				return "Yes.", nil
			}
			if len(prior) != 1 || prior[0].User != "heard" || prior[0].Assistant != "Yes." || user != "again" {
				return "", fmt.Errorf("prior %+v user %q", prior, user)
			}
			return "Next.", nil
		},
	})
	extra := realtimeExtra{Pipeline: "duplex", Voice: "piper", ASR: "whisper", Rate: 16000}
	report, err := probeRealtime(host, "chat", "transcription", "", nil, false, "", []int16{1, -1}, 16000, 1000, 5*time.Second, nil, "", "", []int16{1, -1}, 16000, "", "", extra)
	if err != nil {
		t.Fatal(err)
	}
	if report.Transcript != "again" || report.Reply != "Next." || report.Status != "completed" {
		t.Fatalf("%+v", report)
	}
}

func serveDuplexProbe(t *testing.T, stub duplexStub) (string, <-chan url.Values) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	query := make(chan url.Values, 1)
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		br := bufio.NewReader(conn)
		req, err := http.ReadRequest(br)
		if err != nil {
			conn.Close()
			return
		}
		query <- req.URL.Query()
		key := req.Header.Get("Sec-WebSocket-Key")
		if key == "" {
			conn.Close()
			return
		}
		sum := sha1.Sum([]byte(key + wsGUID))
		accept := base64.StdEncoding.EncodeToString(sum[:])
		_, _ = io.WriteString(conn, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: "+accept+"\r\n\r\n")
		src := io.Reader(conn)
		if br.Buffered() > 0 {
			src = io.MultiReader(br, conn)
		}
		var turn *voicepipe.Turn
		if req.URL.Query().Get("turn") == "server_vad" {
			d := voicepipe.DefaultTurn()
			turn = &d
		}
		clip := stub.clip
		if clip.PCM == nil && clip.Raw == nil {
			clip = voicepipe.Clip{Rate: 16000, PCM: []byte{1, 0, 2, 0}}
		}
		_ = voicepipe.Serve(context.Background(), conn, src, voicepipe.Engine{
			Duplex:     true,
			InputRate:  16000,
			OutputRate: 16000,
			Turn:       turn,
			Hear: func(ctx context.Context, wav []byte, language string) (string, error) {
				if stub.hear != nil {
					return stub.hear(ctx, wav, language)
				}
				return "heard", nil
			},
			Reply: func(ctx context.Context, instructions, user string, prior []voicepipe.Exchange) (string, error) {
				if stub.reply != nil {
					return stub.reply(ctx, instructions, user, prior)
				}
				return "Yes.", nil
			},
			Speak: func(ctx context.Context, text, voice, language, emotion string) (voicepipe.Clip, error) {
				if stub.block {
					<-ctx.Done()
					return voicepipe.Clip{}, ctx.Err()
				}
				return clip, nil
			},
		})
	}()
	return "http://" + ln.Addr().String(), query
}

func TestProbeSpeechLoopback(t *testing.T) {
	bodyCh := make(chan []byte, 1)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		br := bufio.NewReader(conn)
		req, err := http.ReadRequest(br)
		if err != nil {
			return
		}
		raw, _ := io.ReadAll(req.Body)
		bodyCh <- raw
		wav := []byte("RIFF")
		fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nX-Audio-Seconds: 1.5\r\nContent-Length: %d\r\nConnection: close\r\n\r\n", len(wav))
		_, _ = conn.Write(wav)
	}()
	dir := t.TempDir()
	ref := dir + "/ref.wav"
	if err := os.WriteFile(ref, []byte("ref-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	out := dir + "/speech.wav"
	report, err := probeSpeech("http://"+ln.Addr().String(), "lfm3-audio", "hello", "female", "en", ref, out, false, "", nil, nil, speechSample{}, nil, nil, nil, nil, nil, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if report.AudioBytes != 4 || report.Seconds != 1.5 || report.OutFile != out || report.Status != "audio/wav" {
		t.Fatalf("%+v", report)
	}
	var got map[string]string
	if err := json.Unmarshal(<-bodyCh, &got); err != nil {
		t.Fatal(err)
	}
	if got["lang"] != "en" || got["voice"] != "female" || got["input"] != "hello" || got["speaker_ref"] == "" || got["response_format"] != "" {
		t.Fatalf("%v", got)
	}
}

func TestProbeSpeechStreamLoopback(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &got)
		w.Header().Set("Content-Type", "audio/pcm;rate=24000;encoding=float;bits=32")
		w.Header().Set("Trailer", "X-Audio-Seconds")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("abcd"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(60 * time.Millisecond)
		_, _ = w.Write([]byte("efgh"))
		w.Header().Set("X-Audio-Seconds", "1.25")
	}))
	defer srv.Close()
	seed := int64(7)
	maxTokens := 32
	topK := 8
	topP := 0.9
	repeat := 1.1
	silence := 0.2
	noise := 0.5
	width := 0.8
	speaker := 7
	out := t.TempDir() + "/speech.wav"
	report, err := probeSpeech(srv.URL, "lfm3-audio", "hello", "", "", "", out, true, "pcm", &seed, &maxTokens, speechSample{TopK: &topK, TopP: &topP, RepeatPenalty: &repeat}, &silence, &noise, &width, &speaker, map[string]float64{".": 0.3}, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	saved, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if report.AudioBytes != 8 || report.OutputRate != 24000 || report.Seconds != 1.25 || report.OutFile != out || report.FirstAudioMs <= 0 || report.TotalMs < report.FirstAudioMs {
		t.Fatalf("%+v", report)
	}
	if rate, ok := wavHeaderRate(saved); !ok || rate != 24000 || string(saved[:4]) != "RIFF" {
		t.Fatalf("%+v", report)
	}
	pauses, _ := got["phoneme_silence"].(map[string]any)
	if got["stream"] != true || got["response_format"] != "pcm" || got["seed"] != float64(7) || got["max_new_tokens"] != float64(32) || got["top_k"] != float64(8) || got["top_p"] != 0.9 || got["repeat_penalty"] != 1.1 || got["sentence_silence"] != 0.2 || got["noise_scale"] != 0.5 || got["noise_w"] != 0.8 || got["speaker_id"] != float64(7) || pauses["."] != 0.3 {
		t.Fatalf("body=%v", got)
	}
}

func TestCleanSpeechFormat(t *testing.T) {
	got, err := cleanSpeechFormat(" PCM ")
	if err != nil || got != "pcm" {
		t.Fatal(got, err)
	}
	got, err = cleanSpeechFormat("json")
	if err != nil || got != "" {
		t.Fatal(got, err)
	}
	if _, err := cleanSpeechFormat("srt"); err == nil {
		t.Fatal("expected reject")
	}
}

func TestProbeRejectsTextOnTranscription(t *testing.T) {
	host, seen := fakeRealtime(t, false, "")
	_, err := probeRealtime(host, "voxtral-realtime", "transcription", "hello", nil, false, "", []int16{1}, 16000, 320, 5*time.Second, nil, "", "", nil, 0, "", "", realtimeExtra{})
	if err == nil || !strings.Contains(err.Error(), "text turn") {
		t.Fatal(err)
	}
	<-seen
}

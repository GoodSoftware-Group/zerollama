package wsproxy

import (
	"bufio"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/server/voicepipe"
)

func TestProxyTurnCommits(t *testing.T) {
	for _, kind := range []string{"transcription", "realtime"} {
		t.Run(kind, func(t *testing.T) {
			got := proxyTurnFrames(t, kind, false, func(conn net.Conn, _ <-chan struct{}) {
				sendText(t, conn, pcmEvent(4000, 1))
				sendText(t, conn, pcmEvent(0, 8000))
			})
			want := []string{"input_audio_buffer.append", "input_audio_buffer.append", "input_audio_buffer.commit"}
			if kind == "realtime" {
				want = append(want, "response.create")
			}
			if strings.Join(got, ",") != strings.Join(want, ",") {
				t.Fatalf("frames %v", got)
			}
		})
	}
}

func TestProxyTurnIdleCommitsAfterAQuietTail(t *testing.T) {
	for _, kind := range []string{"transcription", "realtime"} {
		t.Run(kind, func(t *testing.T) {
			got := proxyTurnFrames(t, kind, false, func(conn net.Conn, _ <-chan struct{}) {
				pcm := make([]byte, 22)
				binary.LittleEndian.PutUint16(pcm[:2], 4000)
				b, err := json.Marshal(map[string]any{
					"type":  "input_audio_buffer.append",
					"audio": base64.StdEncoding.EncodeToString(pcm),
				})
				if err != nil {
					t.Fatal(err)
				}
				sendText(t, conn, b)
			})
			want := []string{"input_audio_buffer.append", "input_audio_buffer.commit"}
			if kind == "realtime" {
				want = append(want, "response.create")
			}
			if strings.Join(got, ",") != strings.Join(want, ",") {
				t.Fatalf("frames %v", got)
			}
		})
	}
}

func TestProxyTurnIdleCommitsAfterALoudPacket(t *testing.T) {
	for _, kind := range []string{"transcription", "realtime"} {
		t.Run(kind, func(t *testing.T) {
			got := proxyTurnFrames(t, kind, false, func(conn net.Conn, _ <-chan struct{}) {
				sendText(t, conn, pcmEvent(4000, 1))
			})
			want := []string{"input_audio_buffer.append", "input_audio_buffer.commit"}
			if kind == "realtime" {
				want = append(want, "response.create")
			}
			if strings.Join(got, ",") != strings.Join(want, ",") {
				t.Fatalf("frames %v", got)
			}
		})
	}
}

func TestProxyTurnIgnoresQuiet(t *testing.T) {
	got := proxyTurnFrames(t, "transcription", false, func(conn net.Conn, _ <-chan struct{}) {
		sendText(t, conn, pcmEvent(0, 320))
	})
	if len(got) != 0 {
		t.Fatalf("frames %v", got)
	}
}

func TestProxyTurnLoudBarge(t *testing.T) {
	got := proxyTurnFrames(t, "realtime", true, func(conn net.Conn, reply <-chan struct{}) {
		sendText(t, conn, pcmEvent(4000, 1))
		sendText(t, conn, pcmEvent(0, 8000))
		select {
		case <-reply:
		case <-time.After(2 * time.Second):
			t.Error("response.created was not forwarded")
			return
		}
		sendText(t, conn, pcmEvent(0, 160))
		sendText(t, conn, pcmEvent(4000, 1))
	})
	want := []string{
		"input_audio_buffer.append",
		"input_audio_buffer.append",
		"input_audio_buffer.commit",
		"response.create",
		"response.cancel",
		"input_audio_buffer.append",
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("frames %v", got)
	}
}

func proxyTurnFrames(t *testing.T, kind string, untilCancel bool, speak func(net.Conn, <-chan struct{})) []string {
	t.Helper()
	frames := make(chan string, 8)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			http.Error(w, "hijack", http.StatusInternalServerError)
			return
		}
		conn, rw, err := hj.Hijack()
		if err != nil {
			return
		}
		defer conn.Close()
		accept := AcceptValue(r.Header.Get("Sec-WebSocket-Key"))
		_, _ = io.WriteString(rw, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: "+accept+"\r\n\r\n")
		created := serverTextFrame([]byte(`{"type":"session.created","session":{"type":"` + kind + `","audio":{"input":{"format":{"type":"audio/pcm","rate":16000}}}}}`))
		if _, err := rw.Write(created); err != nil {
			return
		}
		_ = rw.Flush()
		br := bufio.NewReader(io.MultiReader(rw.Reader, conn))
		_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		for {
			_, payload, err := readFrame(br)
			if err != nil {
				close(frames)
				return
			}
			typ := eventType(payload)
			frames <- typ
			if typ == "response.create" {
				_, _ = rw.Write(serverTextFrame([]byte(`{"type":"response.created"}`)))
				_ = rw.Flush()
			}
		}
	}))
	defer backend.Close()
	_, portStr, err := net.SplitHostPort(backend.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		turn := voicepipe.Turn{Threshold: 0.02, SilenceMS: 500, PrefixMS: 0}
		if err := ProxyTurn(w, r, port, "", "", turn); err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
		}
	}))
	defer front.Close()

	conn, err := net.Dial("tcp", front.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(3 * time.Second))
	_, err = io.WriteString(conn, "GET /v1/realtime?model=lfm&turn=server_vad HTTP/1.1\r\n"+
		"Host: "+front.Listener.Addr().String()+"\r\n"+
		"Upgrade: websocket\r\n"+
		"Connection: Upgrade\r\n"+
		"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"+
		"Sec-WebSocket-Version: 13\r\n\r\n")
	if err != nil {
		t.Fatal(err)
	}
	br := bufio.NewReader(conn)
	status, _, err := readHeaders(br)
	if err != nil {
		t.Fatal(err)
	}
	if status != http.StatusSwitchingProtocols {
		t.Fatalf("status %d", status)
	}
	_, created, err := readFrame(br)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(created), `"server_vad"`) || !strings.Contains(string(created), `"rate":16000`) {
		t.Fatalf("created %s", created)
	}
	reply := make(chan struct{})
	go func() {
		for {
			_, payload, err := readFrame(br)
			if err != nil {
				return
			}
			if strings.Contains(string(payload), `"response.created"`) {
				select {
				case <-reply:
				default:
					close(reply)
				}
			}
		}
	}()
	go speak(conn, reply)
	wait := 1500 * time.Millisecond
	if untilCancel || kind == "realtime" {
		wait = 2 * time.Second
	}
	var got []string
	timeout := time.After(wait)
	for {
		select {
		case typ, ok := <-frames:
			if !ok {
				return got
			}
			got = append(got, typ)
			if untilCancel && typ == "response.cancel" {
				select {
				case next := <-frames:
					got = append(got, next)
				case <-time.After(time.Second):
				}
				return got
			}
			if !untilCancel && kind != "realtime" && typ == "input_audio_buffer.commit" {
				return got
			}
			if !untilCancel && kind == "realtime" && typ == "response.create" {
				return got
			}
		case <-timeout:
			return got
		}
	}
}

func sendText(t *testing.T, conn net.Conn, payload []byte) {
	t.Helper()
	if err := writeFrame(conn, 0x1, payload, true); err != nil {
		t.Fatal(err)
	}
}

func pcmEvent(sample int16, n int) []byte {
	if sample != 0 {
		n = 1
	}
	buf := make([]byte, n*2)
	if sample != 0 {
		binary.LittleEndian.PutUint16(buf, uint16(sample))
	}
	b, _ := json.Marshal(map[string]any{
		"type":  "input_audio_buffer.append",
		"audio": base64.StdEncoding.EncodeToString(buf),
	})
	return b
}

func TestSessionUpdateReappliesNeuralThreshold(t *testing.T) {
	st := &turnState{
		on: true,
		gate: voicepipe.Gate{
			Turn: voicepipe.Turn{
				Threshold:      voicepipe.NeuralDefaultThreshold,
				AmplitudeScale: voicepipe.DefaultAmplitude,
			},
			Score: func([]byte, int, float64) (int, error) { return -1, nil },
		},
	}
	backend, peer := net.Pipe()
	defer backend.Close()
	defer peer.Close()
	go io.Copy(io.Discard, peer)
	body, _ := json.Marshal(map[string]any{
		"type": "session.update",
		"session": map[string]any{
			"audio": map[string]any{
				"input": map[string]any{
					"turn_detection": map[string]any{"type": "server_vad"},
				},
			},
		},
	})
	if err := st.clientText(backend, func(byte, []byte) error { return nil }, body); err != nil {
		t.Fatal(err)
	}
	if st.gate.Turn.Threshold != voicepipe.NeuralDefaultThreshold {
		t.Fatalf("threshold=%v", st.gate.Turn.Threshold)
	}
	if st.gate.Turn.AmplitudeScale != voicepipe.DefaultAmplitude {
		t.Fatalf("amp=%v", st.gate.Turn.AmplitudeScale)
	}
}

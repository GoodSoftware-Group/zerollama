package wsproxy

import (
	"bufio"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestAcceptValueRFC6455(t *testing.T) {
	// RFC 6455 section 1.3 example.
	got := AcceptValue("dGhlIHNhbXBsZSBub25jZQ==")
	const want = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
	if got != want {
		t.Fatalf("accept = %q, want %q", got, want)
	}
}

func TestCleanIntent(t *testing.T) {
	if _, err := cleanIntent("transcription"); err != nil {
		t.Fatal(err)
	}
	if _, err := cleanIntent("ok name"); err == nil {
		t.Fatal("expected reject")
	}
}

func TestProxyEcho(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := Validate(r); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if r.URL.Path != "/v1/realtime" || r.URL.Query().Get("intent") != "transcription" || r.URL.Query().Get("language") != "en" {
			http.Error(w, r.URL.String(), http.StatusBadRequest)
			return
		}
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
		_ = rw.Flush()
		created := serverTextFrame([]byte(`{"type":"session.created","session":{"type":"realtime"}}`))
		if _, err := rw.Write(created); err != nil {
			return
		}
		_ = rw.Flush()
		br := bufio.NewReader(io.MultiReader(rw.Reader, conn))
		_, payload, err := readFrame(br)
		if err != nil {
			t.Errorf("read update: %v", err)
			return
		}
		if !strings.Contains(string(payload), `"session.update"`) || !strings.Contains(string(payload), `"transcription"`) || !strings.Contains(string(payload), `"en"`) {
			t.Errorf("update %s", payload)
			return
		}
		updated := serverTextFrame([]byte(`{"type":"session.updated","session":{"type":"transcription"}}`))
		if _, err := rw.Write(updated); err != nil {
			return
		}
		_ = rw.Flush()
		_, _ = io.Copy(conn, br)
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
		if err := Proxy(w, r, port, r.URL.Query().Get("intent"), r.URL.Query().Get("language")); err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
		}
	}))
	defer front.Close()

	conn, err := net.Dial("tcp", front.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(5 * time.Second))
	_, err = io.WriteString(conn, "GET /v1/realtime?model=voxtral&intent=transcription&language=EN HTTP/1.1\r\n"+
		"Host: "+front.Listener.Addr().String()+"\r\n"+
		"Upgrade: websocket\r\n"+
		"Connection: Upgrade\r\n"+
		"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"+
		"Sec-WebSocket-Version: 13\r\n\r\n")
	if err != nil {
		t.Fatal(err)
	}
	br := bufio.NewReader(conn)
	status, hdrs, err := readHeaders(br)
	if err != nil {
		t.Fatal(err)
	}
	if status != http.StatusSwitchingProtocols {
		t.Fatalf("status %d", status)
	}
	if got := hdrs.Get("Sec-WebSocket-Accept"); got != "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=" {
		t.Fatalf("accept %q", got)
	}

	payload := []byte(`{"type":"input_audio_buffer.commit"}`)
	frame := clientTextFrame(payload)
	if _, err := conn.Write(frame); err != nil {
		t.Fatal(err)
	}
	_, created, err := readFrame(br)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(created), `"session.created"`) {
		t.Fatalf("created %s", created)
	}
	_, updated, err := readFrame(br)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(updated), `"session.updated"`) {
		t.Fatalf("updated %s", updated)
	}
	got := make([]byte, len(frame))
	if _, err := io.ReadFull(br, got); err != nil {
		t.Fatal(err)
	}
	if string(got) != string(frame) {
		t.Fatalf("echo mismatch\n got %x\nwant %x", got, frame)
	}
}

func serverTextFrame(payload []byte) []byte {
	return append([]byte{0x81, byte(len(payload))}, payload...)
}

func clientTextFrame(payload []byte) []byte {
	mask := []byte{1, 2, 3, 4}
	frame := []byte{0x81, byte(len(payload)) | 0x80}
	frame = append(frame, mask...)
	for i, b := range payload {
		frame = append(frame, b^mask[i%4])
	}
	return frame
}

func TestValidateRejectsPlainGET(t *testing.T) {
	req, err := http.NewRequest(http.MethodGet, "/v1/realtime", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(req); err == nil {
		t.Fatal("expected error")
	}
}

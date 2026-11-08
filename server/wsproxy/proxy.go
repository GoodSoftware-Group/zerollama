// Package wsproxy bridges a client WebSocket to llama-server's /v1/realtime.
// Frames are copied unchanged. The event schema is llama-server's OpenAI
// realtime subset (session.update, input_audio_buffer.append, commit).
package wsproxy

import (
	"bufio"
	"crypto/rand"
	"crypto/sha1"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/ollama/ollama/server/voicepipe"
)

const wsGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

// Validate reports whether r is a WebSocket upgrade zerollama can accept.
func Validate(r *http.Request) error {
	if r == nil {
		return errors.New("missing request")
	}
	if !headerToken(r.Header, "Upgrade", "websocket") {
		return errors.New("expected websocket upgrade")
	}
	if !headerToken(r.Header, "Connection", "upgrade") {
		return errors.New("expected connection upgrade")
	}
	if strings.TrimSpace(r.Header.Get("Sec-WebSocket-Key")) == "" {
		return errors.New("missing Sec-WebSocket-Key")
	}
	if v := strings.TrimSpace(r.Header.Get("Sec-WebSocket-Version")); v != "" && v != "13" {
		return fmt.Errorf("unsupported Sec-WebSocket-Version %q", v)
	}
	return nil
}

// AcceptValue is the Sec-WebSocket-Accept for key (RFC 6455).
func AcceptValue(key string) string {
	sum := sha1.Sum([]byte(strings.TrimSpace(key) + wsGUID))
	return base64.StdEncoding.EncodeToString(sum[:])
}

// Accept completes the client WebSocket handshake and returns the connection.
// The reader includes any bytes already buffered by the hijack.
func Accept(w http.ResponseWriter, r *http.Request) (net.Conn, io.Reader, error) {
	if err := Validate(r); err != nil {
		return nil, nil, err
	}
	hj, ok := w.(http.Hijacker)
	if !ok {
		return nil, nil, errors.New("response writer cannot hijack")
	}
	client, rw, err := hj.Hijack()
	if err != nil {
		return nil, nil, err
	}
	accept := AcceptValue(r.Header.Get("Sec-WebSocket-Key"))
	var resp strings.Builder
	resp.WriteString("HTTP/1.1 101 Switching Protocols\r\n")
	resp.WriteString("Upgrade: websocket\r\n")
	resp.WriteString("Connection: Upgrade\r\n")
	resp.WriteString("Sec-WebSocket-Accept: " + accept + "\r\n")
	if proto := strings.TrimSpace(r.Header.Get("Sec-WebSocket-Protocol")); proto != "" && !strings.ContainsAny(proto, "\r\n") {
		resp.WriteString("Sec-WebSocket-Protocol: " + proto + "\r\n")
	}
	resp.WriteString("\r\n")
	if _, err := io.WriteString(rw, resp.String()); err != nil {
		client.Close()
		return nil, nil, err
	}
	if err := rw.Flush(); err != nil {
		client.Close()
		return nil, nil, err
	}
	src := io.Reader(client)
	if rw.Reader != nil && rw.Reader.Buffered() > 0 {
		src = io.MultiReader(rw.Reader, client)
	}
	return client, src, nil
}

// Proxy dials llama-server on 127.0.0.1:port, completes both WebSocket
// handshakes, and copies frames until either side closes.
// intent is the llama-server query value (use "transcription" for live ASR).
// language is forwarded on that query and, with a transcription intent, is
// written as session.update before client audio is copied.
// An error is returned only before the client socket is hijacked.
func Proxy(w http.ResponseWriter, r *http.Request, port int, intent, language string) error {
	return proxy(w, r, port, intent, language, nil)
}

// ProxyTurn is Proxy with amplitude turn-taking. After session.created names
// a PCM rate, quiet audio following speech becomes input_audio_buffer.commit.
// A session whose type is realtime also gets response.create. The rate is
// never guessed: until session.created carries one, frames are copied through.
func ProxyTurn(w http.ResponseWriter, r *http.Request, port int, intent, language string, turn voicepipe.Turn) error {
	return proxy(w, r, port, intent, language, &turn)
}

func proxy(w http.ResponseWriter, r *http.Request, port int, intent, language string, turn *voicepipe.Turn) error {
	if err := Validate(r); err != nil {
		return err
	}
	if port <= 0 {
		return errors.New("llama-server has no port")
	}
	intent, err := cleanIntent(intent)
	if err != nil {
		return err
	}
	language, err = cleanLanguage(language)
	if err != nil {
		return err
	}

	backend, backendReader, proto, err := dialBackend(port, intent, language, r.Header.Get("Sec-WebSocket-Protocol"))
	if err != nil {
		return err
	}

	hj, ok := w.(http.Hijacker)
	if !ok {
		backend.Close()
		return errors.New("response writer cannot hijack")
	}
	client, rw, err := hj.Hijack()
	if err != nil {
		backend.Close()
		return err
	}

	accept := AcceptValue(r.Header.Get("Sec-WebSocket-Key"))
	var resp strings.Builder
	resp.WriteString("HTTP/1.1 101 Switching Protocols\r\n")
	resp.WriteString("Upgrade: websocket\r\n")
	resp.WriteString("Connection: Upgrade\r\n")
	resp.WriteString("Sec-WebSocket-Accept: " + accept + "\r\n")
	if proto != "" {
		resp.WriteString("Sec-WebSocket-Protocol: " + proto + "\r\n")
	}
	resp.WriteString("\r\n")
	if _, err := io.WriteString(rw, resp.String()); err != nil {
		client.Close()
		backend.Close()
		return nil
	}
	if err := rw.Flush(); err != nil {
		client.Close()
		backend.Close()
		return nil
	}
	var st *turnState
	if turn != nil {
		st = newTurnState(*turn)
	}
	if intent == "transcription" || language != "" {
		_ = backend.SetDeadline(time.Now().Add(2 * time.Second))
		var note func([]byte) []byte
		if st != nil {
			note = st.note
		}
		if err := primeRealtime(client, backend, backendReader, intent, language, note); err != nil {
			client.Close()
			backend.Close()
			return nil
		}
		_ = backend.SetDeadline(time.Time{})
	}

	clientReader := io.Reader(client)
	if rw.Reader != nil && rw.Reader.Buffered() > 0 {
		clientReader = io.MultiReader(rw.Reader, client)
	}
	if st == nil {
		splice(client, clientReader, backend, backendReader)
	} else {
		spliceTurn(client, clientReader, backend, backendReader, st)
	}
	return nil
}

func cleanIntent(intent string) (string, error) {
	intent = strings.TrimSpace(intent)
	if intent == "" {
		return "", nil
	}
	if len(intent) > 32 {
		return "", errors.New("intent is too long")
	}
	for _, r := range intent {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-' {
			continue
		}
		return "", errors.New("intent must be a single token")
	}
	return intent, nil
}

func cleanLanguage(language string) (string, error) {
	language = strings.TrimSpace(language)
	if language == "" {
		return "", nil
	}
	if len(language) > 32 {
		return "", errors.New("language is too long")
	}
	for _, r := range language {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-' {
			continue
		}
		return "", errors.New("language must be a single token")
	}
	return strings.ToLower(language), nil
}

// primeRealtime tells a conversation socket to transcribe, and which language
// to use, before any client audio is copied. The language query is also the
// value the asr-template stream reads when it opens. A session.update that
// names a different language is rejected there.
func primeRealtime(client, backend net.Conn, backendReader *bufio.Reader, intent, language string, note func([]byte) []byte) error {
	opcode, payload, err := readFrame(backendReader)
	if err != nil {
		return err
	}
	if note != nil && opcode == 0x1 {
		payload = note(payload)
	}
	if err := writeFrame(client, opcode, payload, false); err != nil {
		return err
	}
	if opcode != 0x1 || !strings.Contains(string(payload), `"session.created"`) {
		return nil
	}
	update, err := json.Marshal(map[string]any{
		"type":    "session.update",
		"session": primeSession(intent, language),
	})
	if err != nil {
		return err
	}
	if err := writeFrame(backend, 0x1, update, true); err != nil {
		return err
	}
	for i := 0; i < 4; i++ {
		opcode, payload, err = readFrame(backendReader)
		if err != nil {
			return err
		}
		if note != nil && opcode == 0x1 {
			payload = note(payload)
		}
		if err := writeFrame(client, opcode, payload, false); err != nil {
			return err
		}
		if opcode == 0x1 && strings.Contains(string(payload), `"session.updated"`) {
			return nil
		}
	}
	return nil
}

func primeSession(intent, language string) map[string]any {
	session := map[string]any{}
	if intent == "transcription" {
		session["type"] = "transcription"
	}
	if language != "" {
		session["audio"] = map[string]any{
			"input": map[string]any{
				"transcription": map[string]any{"language": language},
			},
		}
	}
	return session
}

func readFrame(r *bufio.Reader) (byte, []byte, error) {
	b0, err := r.ReadByte()
	if err != nil {
		return 0, nil, err
	}
	b1, err := r.ReadByte()
	if err != nil {
		return 0, nil, err
	}
	opcode := b0 & 0x0f
	n := int(b1 & 0x7f)
	switch n {
	case 126:
		var ext [2]byte
		if _, err := io.ReadFull(r, ext[:]); err != nil {
			return 0, nil, err
		}
		n = int(ext[0])<<8 | int(ext[1])
	case 127:
		return 0, nil, errors.New("realtime frame is too large")
	}
	var mask [4]byte
	masked := b1&0x80 != 0
	if masked {
		if _, err := io.ReadFull(r, mask[:]); err != nil {
			return 0, nil, err
		}
	}
	payload := make([]byte, n)
	if _, err := io.ReadFull(r, payload); err != nil {
		return 0, nil, err
	}
	if masked {
		for i := range payload {
			payload[i] ^= mask[i%4]
		}
	}
	return opcode, payload, nil
}

func writeFrame(w io.Writer, opcode byte, payload []byte, mask bool) error {
	if len(payload) > 65535 {
		return errors.New("realtime frame is too large")
	}
	header := []byte{0x80 | opcode}
	if len(payload) < 126 {
		b := byte(len(payload))
		if mask {
			b |= 0x80
		}
		header = append(header, b)
	} else {
		b := byte(126)
		if mask {
			b |= 0x80
		}
		n := len(payload)
		header = append(header, b, byte(n>>8), byte(n))
	}
	if mask {
		var key [4]byte
		if _, err := rand.Read(key[:]); err != nil {
			return err
		}
		header = append(header, key[:]...)
		masked := make([]byte, len(payload))
		for i := range payload {
			masked[i] = payload[i] ^ key[i%4]
		}
		payload = masked
	}
	if _, err := w.Write(header); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

func dialBackend(port int, intent, language, clientProto string) (net.Conn, *bufio.Reader, string, error) {
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	conn, err := net.DialTimeout("tcp", addr, 10*time.Second)
	if err != nil {
		return nil, nil, "", fmt.Errorf("llama-server realtime: %w", err)
	}
	q := url.Values{}
	if intent != "" {
		q.Set("intent", intent)
	}
	if language != "" {
		q.Set("language", language)
	}
	path := "/v1/realtime"
	if enc := q.Encode(); enc != "" {
		path += "?" + enc
	}
	key := make([]byte, 16)
	if _, err := rand.Read(key); err != nil {
		conn.Close()
		return nil, nil, "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "GET %s HTTP/1.1\r\n", path)
	fmt.Fprintf(&b, "Host: %s\r\n", addr)
	b.WriteString("Upgrade: websocket\r\n")
	b.WriteString("Connection: Upgrade\r\n")
	fmt.Fprintf(&b, "Sec-WebSocket-Key: %s\r\n", base64.StdEncoding.EncodeToString(key))
	b.WriteString("Sec-WebSocket-Version: 13\r\n")
	if p := strings.TrimSpace(clientProto); p != "" {
		if strings.ContainsAny(p, "\r\n") {
			conn.Close()
			return nil, nil, "", errors.New("invalid Sec-WebSocket-Protocol")
		}
		fmt.Fprintf(&b, "Sec-WebSocket-Protocol: %s\r\n", p)
	}
	b.WriteString("\r\n")
	if _, err := io.WriteString(conn, b.String()); err != nil {
		conn.Close()
		return nil, nil, "", err
	}
	br := bufio.NewReader(conn)
	status, hdrs, err := readHeaders(br)
	if err != nil {
		conn.Close()
		return nil, nil, "", fmt.Errorf("llama-server realtime: %w", err)
	}
	if status != http.StatusSwitchingProtocols {
		conn.Close()
		return nil, nil, "", fmt.Errorf("llama-server realtime: backend status %d", status)
	}
	return conn, br, strings.TrimSpace(hdrs.Get("Sec-WebSocket-Protocol")), nil
}

func readHeaders(r *bufio.Reader) (int, http.Header, error) {
	line, err := r.ReadString('\n')
	if err != nil {
		return 0, nil, err
	}
	var status int
	if _, err := fmt.Sscanf(line, "HTTP/1.1 %d", &status); err != nil {
		return 0, nil, fmt.Errorf("bad status line %q", strings.TrimSpace(line))
	}
	h := make(http.Header)
	for {
		line, err = r.ReadString('\n')
		if err != nil {
			return 0, nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			return status, h, nil
		}
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		h.Add(strings.TrimSpace(k), strings.TrimSpace(v))
	}
}

func splice(client net.Conn, clientReader io.Reader, backend net.Conn, backendReader io.Reader) {
	var once sync.Once
	closeBoth := func() {
		once.Do(func() {
			client.Close()
			backend.Close()
		})
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, _ = io.Copy(backend, clientReader)
		closeBoth()
	}()
	go func() {
		defer wg.Done()
		_, _ = io.Copy(client, backendReader)
		closeBoth()
	}()
	wg.Wait()
}

func headerToken(h http.Header, key, want string) bool {
	for _, part := range strings.Split(h.Get(key), ",") {
		if strings.EqualFold(strings.TrimSpace(part), want) {
			return true
		}
	}
	return false
}

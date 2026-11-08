package main

import (
	"bufio"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"net/url"
	"strings"
)

const wsGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

// wsConn is a minimal RFC 6455 client. The server sends unmasked frames.
// The client masks every data frame. One text message per frame is enough
// for llama-server realtime events.
type wsConn struct {
	conn net.Conn
	br   *bufio.Reader
	mask int
}

func dialWebSocket(rawURL string) (*wsConn, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, err
	}
	host := u.Host
	if !strings.Contains(host, ":") {
		if u.Scheme == "wss" || u.Scheme == "https" {
			host += ":443"
		} else {
			host += ":80"
		}
	}
	conn, err := net.Dial("tcp", host)
	if err != nil {
		return nil, err
	}
	key := base64.StdEncoding.EncodeToString([]byte("zerollama-realtime-probe"))
	path := u.RequestURI()
	if path == "" {
		path = "/"
	}
	req := fmt.Sprintf("GET %s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n", path, u.Host, key)
	if _, err := io.WriteString(conn, req); err != nil {
		conn.Close()
		return nil, err
	}
	br := bufio.NewReader(conn)
	status, err := br.ReadString('\n')
	if err != nil {
		conn.Close()
		return nil, err
	}
	if !strings.Contains(status, " 101 ") {
		conn.Close()
		return nil, fmt.Errorf("websocket upgrade: %s", strings.TrimSpace(status))
	}
	var accept string
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			conn.Close()
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		k, v, ok := strings.Cut(line, ":")
		if ok && strings.EqualFold(strings.TrimSpace(k), "Sec-WebSocket-Accept") {
			accept = strings.TrimSpace(v)
		}
	}
	sum := sha1.Sum([]byte(key + wsGUID))
	want := base64.StdEncoding.EncodeToString(sum[:])
	if accept != want {
		conn.Close()
		return nil, fmt.Errorf("websocket accept %q", accept)
	}
	return &wsConn{conn: conn, br: br}, nil
}

func (w *wsConn) Close() error { return w.conn.Close() }

func (w *wsConn) WriteText(payload []byte) error {
	w.mask++
	var mask [4]byte
	binary.BigEndian.PutUint32(mask[:], uint32(w.mask))
	header := []byte{0x81}
	n := len(payload)
	switch {
	case n < 126:
		header = append(header, byte(n)|0x80)
	case n <= 65535:
		header = append(header, 126|0x80, byte(n>>8), byte(n))
	default:
		return fmt.Errorf("realtime event is %d bytes", n)
	}
	header = append(header, mask[:]...)
	masked := make([]byte, n)
	for i := range payload {
		masked[i] = payload[i] ^ mask[i%4]
	}
	if _, err := w.conn.Write(header); err != nil {
		return err
	}
	_, err := w.conn.Write(masked)
	return err
}

func (w *wsConn) ReadText() (string, error) {
	for {
		b0, err := w.br.ReadByte()
		if err != nil {
			return "", err
		}
		b1, err := w.br.ReadByte()
		if err != nil {
			return "", err
		}
		opcode := b0 & 0x0f
		if b1&0x80 != 0 {
			return "", fmt.Errorf("server frame was masked")
		}
		n := int(b1 & 0x7f)
		switch n {
		case 126:
			var ext [2]byte
			if _, err := io.ReadFull(w.br, ext[:]); err != nil {
				return "", err
			}
			n = int(binary.BigEndian.Uint16(ext[:]))
		case 127:
			return "", fmt.Errorf("realtime frame is too large")
		}
		payload := make([]byte, n)
		if _, err := io.ReadFull(w.br, payload); err != nil {
			return "", err
		}
		switch opcode {
		case 0x1:
			return string(payload), nil
		case 0x8:
			return "", io.EOF
		case 0x9:
			// pong
			pong := append([]byte{0x8a, byte(len(payload))}, payload...)
			if _, err := w.conn.Write(pong); err != nil {
				return "", err
			}
		}
	}
}

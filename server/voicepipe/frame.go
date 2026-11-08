package voicepipe

import (
	"bufio"
	"crypto/rand"
	"errors"
	"io"
)

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
	if n > 0 {
		if _, err := io.ReadFull(r, payload); err != nil {
			return 0, nil, err
		}
	}
	if masked {
		for i := range payload {
			payload[i] ^= mask[i%4]
		}
	}
	return opcode, payload, nil
}

func writeFrame(w io.Writer, opcode byte, payload []byte) error {
	if len(payload) > 65535 {
		return errors.New("realtime frame is too large")
	}
	header := []byte{0x80 | opcode}
	if len(payload) < 126 {
		header = append(header, byte(len(payload)))
	} else {
		header = append(header, 126, byte(len(payload)>>8), byte(len(payload)))
	}
	if _, err := w.Write(header); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

// WriteClientFrame writes a masked client text frame. Tests and probes use it.
func WriteClientFrame(w io.Writer, payload []byte) error {
	if len(payload) > 65535 {
		return errors.New("realtime frame is too large")
	}
	var key [4]byte
	if _, err := rand.Read(key[:]); err != nil {
		return err
	}
	header := []byte{0x81}
	if len(payload) < 126 {
		header = append(header, 0x80|byte(len(payload)))
	} else {
		header = append(header, 0x80|126, byte(len(payload)>>8), byte(len(payload)))
	}
	header = append(header, key[:]...)
	masked := make([]byte, len(payload))
	for i := range payload {
		masked[i] = payload[i] ^ key[i%4]
	}
	if _, err := w.Write(header); err != nil {
		return err
	}
	_, err := w.Write(masked)
	return err
}

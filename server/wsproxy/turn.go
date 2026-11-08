package wsproxy

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"io"
	"net"
	"sync"
	"time"

	"github.com/ollama/ollama/server/modality/neuralvad"
	"github.com/ollama/ollama/server/voicepipe"
)

// turnState commits a proxied llama-server socket when PCM goes quiet.
// The sample rate and the session type come from session.created.
// A packet that ends quiet starts the clock for the silence still missing.
// A packet that ends loud starts the full silence. If the sender stops,
// that remainder commits the turn.
type turnState struct {
	mu          sync.Mutex
	writeMu     sync.Mutex
	on          bool
	replying    bool
	kind        string
	gate        voicepipe.Gate
	idle        *time.Timer
	idleGen     int
	backend     net.Conn
	writeClient func(byte, []byte) error
}

func newTurnState(turn voicepipe.Turn) *turnState {
	gate := voicepipe.Gate{Turn: turn, Max: 32 << 20}
	if sid, score := neuralvad.Attach(); score != nil {
		turn.ApplyNeuralThreshold()
		gate.Turn = turn
		gate.Score = score
		gate.OnReset = func() { neuralvad.Reset(sid) }
	}
	return &turnState{
		on:   true,
		gate: gate,
	}
}

func (st *turnState) note(payload []byte) []byte {
	st.observe(payload)
	return st.annotate(payload)
}

func (st *turnState) observe(payload []byte) {
	typ, kind, rate := sessionBits(payload)
	st.mu.Lock()
	defer st.mu.Unlock()
	switch typ {
	case "session.created", "session.updated":
		if rate > 0 {
			st.gate.Rate = rate
		}
		if kind != "" {
			st.kind = kind
		}
	case "response.created":
		st.replying = true
	case "response.done", "response.cancelled", "response.canceled":
		st.replying = false
	}
}

func (st *turnState) annotate(payload []byte) []byte {
	typ, _, _ := sessionBits(payload)
	if typ != "session.created" && typ != "session.updated" {
		return payload
	}
	var ev map[string]any
	if json.Unmarshal(payload, &ev) != nil {
		return payload
	}
	sess, _ := ev["session"].(map[string]any)
	if sess == nil {
		return payload
	}
	audio, _ := sess["audio"].(map[string]any)
	if audio == nil {
		audio = map[string]any{}
		sess["audio"] = audio
	}
	in, _ := audio["input"].(map[string]any)
	if in == nil {
		in = map[string]any{}
		audio["input"] = in
	}
	st.mu.Lock()
	on := st.on
	turn := st.gate.Turn
	st.mu.Unlock()
	if on {
		in["turn_detection"] = map[string]any{
			"type":                "server_vad",
			"threshold":           turn.Threshold,
			"silence_duration_ms": turn.SilenceMS,
			"prefix_padding_ms":   turn.PrefixMS,
		}
	} else {
		in["turn_detection"] = nil
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return payload
	}
	return b
}

func spliceTurn(client net.Conn, clientReader io.Reader, backend net.Conn, backendReader *bufio.Reader, st *turnState) {
	var once sync.Once
	closeBoth := func() {
		once.Do(func() {
			st.stopIdle()
			client.Close()
			backend.Close()
		})
	}
	var clientMu sync.Mutex
	writeClient := func(op byte, p []byte) error {
		clientMu.Lock()
		defer clientMu.Unlock()
		return writeFrame(client, op, p, false)
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		defer closeBoth()
		br := bufio.NewReader(clientReader)
		for {
			op, payload, err := readFrame(br)
			if err != nil {
				return
			}
			if op != 0x1 {
				if err := writeFrame(backend, op, payload, true); err != nil {
					return
				}
				if op == 0x8 {
					return
				}
				continue
			}
			if err := st.clientText(backend, writeClient, payload); err != nil {
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		defer closeBoth()
		for {
			op, payload, err := readFrame(backendReader)
			if err != nil {
				return
			}
			if op == 0x1 {
				payload = st.note(payload)
			}
			if err := writeClient(op, payload); err != nil {
				return
			}
			if op == 0x8 {
				return
			}
		}
	}()
	wg.Wait()
}

func (st *turnState) clientText(backend net.Conn, writeClient func(byte, []byte) error, payload []byte) error {
	if has, v := turnValue(payload); has {
		on, turn, err := voicepipe.ParseTurn(v)
		if err != nil {
			return writeClient(0x1, errorEvent(err))
		}
		st.writeMu.Lock()
		st.mu.Lock()
		if on && st.gate.Score != nil {
			turn.ApplyNeuralThreshold()
		}
		st.on = on
		st.gate.Turn = turn
		st.gate.Reset()
		st.stopIdleLocked()
		st.mu.Unlock()
		st.writeMu.Unlock()
	}
	switch eventType(payload) {
	case "input_audio_buffer.append":
		return st.appendAudio(backend, writeClient, payload)
	case "input_audio_buffer.commit":
		return st.forwardCommit(backend, payload)
	case "input_audio_buffer.clear":
		st.writeMu.Lock()
		st.mu.Lock()
		st.gate.Reset()
		st.stopIdleLocked()
		st.mu.Unlock()
		st.writeMu.Unlock()
	case "response.cancel":
		st.mu.Lock()
		st.replying = false
		st.mu.Unlock()
	}
	return writeFrame(backend, 0x1, payload, true)
}

func (st *turnState) appendAudio(backend net.Conn, writeClient func(byte, []byte) error, payload []byte) error {
	var ev map[string]any
	if json.Unmarshal(payload, &ev) != nil {
		return writeFrame(backend, 0x1, payload, true)
	}
	raw, _ := ev["audio"].(string)
	pcm, err := base64.StdEncoding.DecodeString(raw)
	if err != nil || len(pcm) == 0 {
		return writeFrame(backend, 0x1, payload, true)
	}
	st.writeMu.Lock()
	defer st.writeMu.Unlock()
	st.mu.Lock()
	if !st.on || st.gate.Rate <= 0 {
		st.mu.Unlock()
		return writeFrame(backend, 0x1, payload, true)
	}
	step := st.gate.Push(pcm, st.replying)
	if step.Barge {
		st.replying = false
	}
	kind := st.kind
	st.mu.Unlock()
	if step.Err != nil {
		return writeClient(0x1, errorEvent(step.Err))
	}
	if step.Drop {
		return nil
	}
	if step.Started {
		if err := writeClient(0x1, []byte(`{"type":"input_audio_buffer.speech_started"}`)); err != nil {
			return err
		}
	}
	if step.Barge {
		if err := writeFrame(backend, 0x1, []byte(`{"type":"response.cancel"}`), true); err != nil {
			return err
		}
	}
	for _, p := range appendEvents(step.Lead) {
		if err := writeFrame(backend, 0x1, p, true); err != nil {
			return err
		}
	}
	if err := writeFrame(backend, 0x1, payload, true); err != nil {
		return err
	}
	if step.Commit == nil {
		st.armIdle(backend, writeClient)
		return nil
	}
	st.stopIdle()
	return st.commitTurn(backend, writeClient, kind)
}

// armIdle commits an open utterance if no more audio arrives. The wait is
// the silence that is not already in the buffer.
func (st *turnState) armIdle(backend net.Conn, writeClient func(byte, []byte) error) {
	st.mu.Lock()
	defer st.mu.Unlock()
	remain, ok := st.gate.SilenceRemainMS()
	if !st.on || !ok {
		st.stopIdleLocked()
		return
	}
	st.idleGen++
	gen := st.idleGen
	st.backend = backend
	st.writeClient = writeClient
	if st.idle != nil {
		st.idle.Stop()
	}
	st.idle = time.AfterFunc(time.Duration(remain)*time.Millisecond, func() {
		st.fireIdle(gen)
	})
}

func (st *turnState) fireIdle(gen int) {
	st.writeMu.Lock()
	defer st.writeMu.Unlock()
	st.mu.Lock()
	if gen != st.idleGen || !st.on || !st.gate.Heard() {
		st.mu.Unlock()
		return
	}
	pcm := st.gate.Take()
	st.stopIdleLocked()
	kind := st.kind
	backend := st.backend
	writeClient := st.writeClient
	st.mu.Unlock()
	if len(pcm) < 2 || backend == nil || writeClient == nil {
		return
	}
	_ = st.commitTurn(backend, writeClient, kind)
}

func (st *turnState) commitTurn(backend net.Conn, writeClient func(byte, []byte) error, kind string) error {
	if err := writeClient(0x1, []byte(`{"type":"input_audio_buffer.speech_stopped"}`)); err != nil {
		return err
	}
	if err := writeFrame(backend, 0x1, []byte(`{"type":"input_audio_buffer.commit"}`), true); err != nil {
		return err
	}
	if kind == "realtime" {
		return writeFrame(backend, 0x1, []byte(`{"type":"response.create"}`), true)
	}
	return nil
}

func (st *turnState) stopIdle() {
	st.mu.Lock()
	st.stopIdleLocked()
	st.mu.Unlock()
}

func (st *turnState) stopIdleLocked() {
	st.idleGen++
	if st.idle != nil {
		st.idle.Stop()
		st.idle = nil
	}
}

func (st *turnState) forwardCommit(backend net.Conn, payload []byte) error {
	st.writeMu.Lock()
	defer st.writeMu.Unlock()
	st.mu.Lock()
	unsent := st.gate.Unsent()
	st.gate.Reset()
	st.stopIdleLocked()
	st.mu.Unlock()
	for _, p := range appendEvents(unsent) {
		if err := writeFrame(backend, 0x1, p, true); err != nil {
			return err
		}
	}
	return writeFrame(backend, 0x1, payload, true)
}

func appendEvents(pcm []byte) [][]byte {
	const chunk = 24 * 1024
	if len(pcm) == 0 {
		return nil
	}
	var out [][]byte
	for off := 0; off < len(pcm); {
		end := off + chunk
		if end > len(pcm) {
			end = len(pcm)
		} else if end%2 != 0 {
			end--
		}
		if end <= off {
			break
		}
		b, err := json.Marshal(map[string]any{
			"type":  "input_audio_buffer.append",
			"audio": base64.StdEncoding.EncodeToString(pcm[off:end]),
		})
		if err != nil {
			return out
		}
		out = append(out, b)
		off = end
	}
	return out
}

func errorEvent(err error) []byte {
	b, marshalErr := json.Marshal(map[string]any{
		"type":  "error",
		"error": map[string]any{"message": err.Error()},
	})
	if marshalErr != nil {
		return []byte(`{"type":"error","error":{"message":"turn detection failed"}}`)
	}
	return b
}

func eventType(payload []byte) string {
	var ev map[string]any
	if json.Unmarshal(payload, &ev) != nil {
		return ""
	}
	s, _ := ev["type"].(string)
	return s
}

func turnValue(payload []byte) (bool, any) {
	var ev map[string]any
	if json.Unmarshal(payload, &ev) != nil {
		return false, nil
	}
	typ, _ := ev["type"].(string)
	if typ != "session.update" && typ != "transcription_session.update" {
		return false, nil
	}
	sess, _ := ev["session"].(map[string]any)
	audio, _ := sess["audio"].(map[string]any)
	in, _ := audio["input"].(map[string]any)
	if in == nil {
		return false, nil
	}
	v, ok := in["turn_detection"]
	return ok, v
}

func sessionBits(payload []byte) (typ, kind string, rate int) {
	var ev map[string]any
	if json.Unmarshal(payload, &ev) != nil {
		return "", "", 0
	}
	typ, _ = ev["type"].(string)
	sess, _ := ev["session"].(map[string]any)
	if sess == nil {
		return typ, "", 0
	}
	kind, _ = sess["type"].(string)
	audio, _ := sess["audio"].(map[string]any)
	in, _ := audio["input"].(map[string]any)
	format, _ := in["format"].(map[string]any)
	switch n := format["rate"].(type) {
	case float64:
		if n > 0 && n <= 192000 && n == float64(int(n)) {
			rate = int(n)
		}
	}
	return typ, kind, rate
}

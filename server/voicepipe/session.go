// Package voicepipe is the Go realtime voice socket.
// A speech session speaks text. A duplex session transcribes a committed
// clip, asks the chat model for a reply, then speaks that reply in sentences.
// That reply includes up to eight earlier turns from this socket.
// The first words are spoken as soon as the model finishes them — a comma,
// semicolon, colon, or about 16 characters. A later stretch waits for about
// 48 characters. Later text keeps generating while that audio plays.
// The completion stops after 48 tokens.
// Earlier turns are prefilled while the next person is speaking.
// Each clause is sent, then spoken.
// When whisper-server 1.9.1 or newer is installed, one CPU process takes
// every clip for the call. Otherwise Whisper starts when the socket opens,
// and the socket keeps reading while that load runs. The next CLI process
// starts when synthesis finishes, and the clip is written on commit. That
// clip keeps about 200 ms after the last loud sample.
// With server_vad, that transcription starts when the audio first goes quiet,
// and the chat model starts when that transcript is ready. The voice
// synthesizes during that wait, and Piper forwards each PCM chunk as it is
// ready. The audio is sent when the turn commits. When the first clause is
// not ready at commit, a cached hesitation (then one "One moment.") holds
// the floor until that opening arrives; a ready opening has no preamble.
// When the CLI is used, the next Whisper starts when that synthesis finishes,
// still during the silence when the voice is already done.
// response.cancel stops the reply. With turn detection off, any new input
// audio also stops it.
// With server_vad, only a loud chunk stops the reply. A quiet stretch after
// speech commits the clip. Quiet samples at the end of a speech packet count.
// If audio stops after a loud packet, the full silence commits the turn.
// On this socket, transcription starts about 100 ms into that wait.
// A transcriber that is still loading is kept, and the clip is written to
// it when that load finishes.
// Cancelling the reply stops the chat completion that is still
// running. Sentences that have not been sent are dropped, including from later turns.
package voicepipe

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"time"
)

const maxPCM = 32 << 20

// maxHistory is how many finished turns a duplex socket keeps for the next reply.
const maxHistory = 8

// ReplyTokens is the longest duplex completion. A short budget leaves the
// cores free for the voice once the first words are out.
const ReplyTokens = 48

// MaxNumCtx is the largest context a duplex reply may request. Gallery
// defaults (32k+) force a VRAM num_ctx binary search that re-parses the GGUF
// on every turn and starves first_token while the GPU sits idle.
const MaxNumCtx = 8192

// ReplyPredict is num_predict for one duplex reply. A positive request below
// ReplyTokens is kept. Anything else uses ReplyTokens.
func ReplyPredict(n int) int {
	if n > 0 && n < ReplyTokens {
		return n
	}
	return ReplyTokens
}

// ReplyNumCtx caps merged num_ctx for one duplex reply. Zero or oversized
// values become MaxNumCtx; a positive request at or below the cap is kept.
func ReplyNumCtx(n int) int {
	if n > 0 && n <= MaxNumCtx {
		return n
	}
	return MaxNumCtx
}

// Exchange is one duplex turn. User is the heard text or the typed item.
// Assistant is the reply text already sent. Later sentences are added as they go out.
type Exchange struct {
	User      string
	Assistant string
}

// Engine is the models behind one socket. Hear and Reply are set for duplex.
// Speak turns one sentence into audio. Language, Voice, Emotion, and
// Instructions are the values session.update can replace. Emotion is the
// Kokoro delivery word. Piper leaves it unused.
type Engine struct {
	Duplex       bool
	InputRate    int
	OutputRate   int
	Language     string
	Voice        string
	Emotion      string
	Instructions string
	Turn         *Turn
	// SpeechScore is an optional neural last-speech index. Nil uses amplitude.
	SpeechScore SpeechScorer
	// OnTurnReset clears neural state when an utterance ends or a barge starts.
	OnTurnReset func()
	Hear func(ctx context.Context, wav []byte, language string) (string, error)
	// HearEx is Hear plus the STT backend that produced the text (qwen3-asr,
	// whisper, whisper_fallback, llama-asr). Prefer this for duplex so the
	// socket can surface asr_backend on transcription.completed.
	HearEx func(ctx context.Context, wav []byte, language string) (text, backend string, err error)
	// WarmHear starts the transcriber when the socket opens, and again after
	// each transcript, so the model can load before the next clip is ready.
	// The returned function reads the finished WAV. Hear is used when
	// WarmHear is unset or not ready yet.
	WarmHear func(ctx context.Context, language string) (func(context.Context, []byte) (string, error), error)
	// WarmHearEx is WarmHear with a backend name on each finished clip.
	WarmHearEx func(ctx context.Context, language string) (func(context.Context, []byte) (text, backend string, err error), error)
	// ASR is the duplex asr= model tag. ASRBackend is the configured family
	// (qwen3-asr, whisper, llama-asr) advertised on session.created.
	ASR        string
	ASRBackend string
	Reply    func(ctx context.Context, instructions, user string, prior []Exchange) (string, error)
	// StreamReply speaks each sentence as the model writes it. Generation continues
	// while that sentence's audio plays. When it is set, Reply is not called.
	StreamReply func(ctx context.Context, instructions, user string, prior []Exchange, emit func(sentence string) error) error
	Speak       func(ctx context.Context, text, voice, language, emotion string) (Clip, error)
	// SpeakMore emits audio as the voice produces it. The first bytes of a
	// clause do not wait for the rest of that clause. When it is set, Speak
	// is not called.
	SpeakMore func(ctx context.Context, text, voice, language, emotion string, emit func(Clip) error) error
	// WarmReply prefills the instructions and earlier turns. The next reply
	// then spends its time on the new utterance. It runs when speech starts
	// and again after a completed reply.
	WarmReply func(ctx context.Context, instructions string, prior []Exchange) error
	// Hold is optional PCM played after commit when the model's opening is
	// not ready. HoldMore plays once after Hold finishes if the opening is
	// still missing. Neither is stored in history. Leading and trailing
	// silence is stripped on playback. PrepareHolds may fill these after
	// session.created so a cold voice warm does not block the handshake.
	Hold         Clip
	HoldMore     Clip
	PrepareHolds func(ctx context.Context) (hold, more Clip)
	// Stages reports duplex milestones (silence, STT, generating, first
	// token, audio done) as response.status. Serve wires it to the socket.
	Stages *StageReporter
}

// Serve reads client events and writes reply events until the socket closes.
// src is the client byte stream. It may differ from conn when the handshake
// buffered bytes. A nil src reads from conn.
func Serve(ctx context.Context, conn io.WriteCloser, src io.Reader, eng Engine) error {
	if eng.Speak == nil && eng.SpeakMore == nil {
		return errors.New("realtime voice has no speaker")
	}
	if eng.Duplex && ((eng.Hear == nil && eng.HearEx == nil) || eng.InputRate <= 0 || (eng.Reply == nil && eng.StreamReply == nil)) {
		return errors.New("duplex requires a transcriber, a chat model, and an input rate")
	}
	defer conn.Close()
	if src == nil {
		r, ok := conn.(io.Reader)
		if !ok {
			return errors.New("realtime voice has no reader")
		}
		src = r
	}
	s := &session{ctx: ctx, w: conn, eng: eng, lang: eng.Language, voice: eng.Voice, emotion: strings.TrimSpace(eng.Emotion), inst: eng.Instructions, kick: make(chan struct{}), hold: eng.Hold, holdMore: eng.HoldMore}
	defer s.dropEar()
	defer s.dropGuess()
	defer s.dropWarm()
	defer s.stopIdle()
	defer s.stopSpeculate()
	if eng.Stages != nil {
		eng.Stages.Set(s.reportStatus)
		defer eng.Stages.Set(nil)
	}
	s.gate.Rate = eng.InputRate
	s.gate.Max = maxPCM
	if eng.Turn != nil {
		s.vad = true
		s.vadTurn = *eng.Turn
		s.gate.Turn = *eng.Turn
		s.gate.Score = eng.SpeechScore
		s.gate.OnReset = eng.OnTurnReset
	}
	if err := s.send(s.created()); err != nil {
		return err
	}
	if eng.PrepareHolds != nil {
		go s.loadHolds()
	}
	s.warmEar()
	br := bufio.NewReader(src)
	for {
		opcode, payload, err := readFrame(br)
		if err != nil {
			return nil
		}
		switch opcode {
		case 0x8:
			return nil
		case 0x9:
			_ = writeFrame(conn, 0xA, payload)
		case 0x1:
			s.handle(payload)
		}
	}
}

type session struct {
	ctx         context.Context
	w           io.Writer
	eng         Engine
	mu          sync.Mutex
	pcm         []byte
	text        string
	lang        string
	inst        string
	voice       string
	emotion     string
	turn        context.CancelFunc
	gen         int
	replyGen    int
	vad         bool
	vadTurn     Turn
	gate        Gate
	history     []Exchange
	earGen      int
	earPending  bool
	earCancel   context.CancelFunc
	earFinish   earFinishFn
	guessCh     chan guessResult
	guessCancel func()
	early       *earlyReply
	idle        *time.Timer
	idleGen     int
	spec        *time.Timer
	specGen     int
	heldPCM     []byte
	speculating bool
	earWaiters  int
	kick        chan struct{}
	warmKey     string
	warmCancel  context.CancelFunc
	warmDone    bool
	hold             Clip
	holdMore         Clip
	silenceAnnounced bool
	sttAnnounced     bool
}

// earlyReply is a chat completion started during the silence. Sentences are
// synthesized as they arrive. Cancel drops them without sending audio.
type earlyReply struct {
	ctx       context.Context
	cancel    context.CancelFunc
	user      string
	voice     string
	lang      string
	emotion   string
	sentences chan string
	spoken    chan spoken
	done      chan error
}

type spoken struct {
	text string
	clip Clip
	err  error
}

type guessResult struct {
	text    string
	backend string
	err     error
}

// earFinishFn transcribes one finished WAV and names the STT backend used.
type earFinishFn func(context.Context, []byte) (text, backend string, err error)

func (s *session) created() map[string]any {
	in := map[string]any{"type": "audio/pcm"}
	if s.eng.InputRate > 0 {
		in["rate"] = s.eng.InputRate
	}
	out := map[string]any{"type": "audio/pcm"}
	if s.eng.OutputRate > 0 {
		out["rate"] = s.eng.OutputRate
	}
	kind := "realtime"
	if !s.eng.Duplex {
		kind = "speech"
	}
	session := map[string]any{
		"type": kind,
		"audio": map[string]any{
			"input": map[string]any{
				"format":         in,
				"turn_detection": s.turnJSON(),
			},
			"output": map[string]any{"format": out},
		},
	}
	if s.eng.Duplex {
		if tag := strings.TrimSpace(s.eng.ASR); tag != "" {
			session["asr"] = tag
		}
		if backend := strings.TrimSpace(s.eng.ASRBackend); backend != "" {
			session["asr_backend"] = backend
		}
	}
	return map[string]any{
		"type":    "session.created",
		"session": session,
	}
}

func (s *session) handle(payload []byte) {
	var ev map[string]any
	if err := json.Unmarshal(payload, &ev); err != nil {
		s.fail(errors.New("invalid JSON"))
		return
	}
	switch ev["type"] {
	case "session.update":
		s.onUpdate(ev)
	case "input_audio_buffer.append":
		s.onAppend(ev)
	case "input_audio_buffer.commit":
		s.onCommit()
	case "conversation.item.create":
		s.onItem(ev)
	case "response.create":
		s.onCreate()
	case "response.cancel":
		s.cancel()
	default:
		s.fail(errors.New("unknown realtime event"))
	}
}

func (s *session) onUpdate(ev map[string]any) {
	sess, _ := ev["session"].(map[string]any)
	s.mu.Lock()
	prevLang := s.lang
	resetEar := false
	if v, ok := sess["instructions"].(string); ok {
		s.inst = v
	}
	if v, ok := sess["voice"].(string); ok {
		s.voice = strings.TrimSpace(v)
	}
	if v, ok := sess["emotion"].(string); ok {
		s.emotion = strings.TrimSpace(v)
	}
	if audio, ok := sess["audio"].(map[string]any); ok {
		if in, ok := audio["input"].(map[string]any); ok {
			if tr, ok := in["transcription"].(map[string]any); ok {
				if lang, ok := tr["language"].(string); ok {
					s.lang = strings.TrimSpace(lang)
					resetEar = s.lang != prevLang
				}
			}
			if td, ok := in["turn_detection"]; ok {
				if !s.eng.Duplex {
					s.mu.Unlock()
					s.fail(errors.New("turn_detection requires a duplex socket"))
					return
				}
				on, turn, err := ParseDuplexTurn(td)
				if err == nil && on && s.gate.Score != nil {
					turn.ApplyNeuralThreshold()
				}
				if err != nil {
					s.mu.Unlock()
					s.fail(err)
					return
				}
				s.vad = on
				s.vadTurn = turn
				s.gate.Turn = turn
				s.gate.Reset()
				s.pcm = nil
				s.stopSpeculateLocked()
				s.stopIdleLocked()
				resetEar = true
			}
		}
	}
	lang, inst, voice, emotion := s.lang, s.inst, s.voice, s.emotion
	td := s.turnJSON()
	s.mu.Unlock()
	if resetEar {
		s.dropEar()
		s.warmEar()
	}
	_ = s.send(map[string]any{"type": "session.updated", "session": map[string]any{
		"instructions": inst,
		"voice":        voice,
		"emotion":      emotion,
		"audio": map[string]any{"input": map[string]any{
			"transcription":  map[string]any{"language": lang},
			"turn_detection": td,
		}},
	}})
}

func (s *session) turnJSON() any {
	if !s.vad {
		return nil
	}
	return map[string]any{
		"type":                "server_vad",
		"threshold":           s.vadTurn.Threshold,
		"silence_duration_ms": s.vadTurn.SilenceMS,
		"prefix_padding_ms":   s.vadTurn.PrefixMS,
	}
}

func (s *session) onAppend(ev map[string]any) {
	raw, _ := ev["audio"].(string)
	pcm, err := base64.StdEncoding.DecodeString(raw)
	if err != nil {
		s.fail(errors.New("invalid base64 audio"))
		return
	}
	if len(pcm) == 0 {
		return
	}
	if !s.eng.Duplex || !s.vadEnabled() {
		s.cancel()
		if err := s.buffer(pcm); err != nil {
			s.fail(err)
			return
		}
		if s.eng.Duplex {
			s.warmEar()
			s.warmReply()
		}
		return
	}
	s.mu.Lock()
	step := s.gate.Push(pcm, s.turn != nil)
	s.mu.Unlock()
	if step.Err != nil {
		s.fail(step.Err)
		return
	}
	if step.Drop {
		return
	}
	if step.Barge {
		s.cancel()
	}
	resumed := !step.Drop && !step.Started && Loud(pcm, s.vadAmplitude())
	if step.Started || resumed {
		s.clearSilenceAnnounce()
		s.dropGuess()
	}
	if step.Started {
		s.warmEar()
		s.warmReply()
		_ = s.send(map[string]any{"type": "input_audio_buffer.speech_started"})
	} else if resumed {
		s.warmEar()
	}
	if step.Draft != nil {
		s.announceSilence()
		s.speculate(step.Draft)
	}
	if step.Commit != nil {
		s.stopSpeculate()
		s.stopIdle()
		s.announceSilence()
		_ = s.send(map[string]any{"type": "input_audio_buffer.speech_stopped"})
		s.begin(step.Commit)
		return
	}
	if step.Quiet {
		s.announceSilence()
		s.stopSpeculate()
	} else {
		s.armSpeculate()
	}
	s.armIdle()
}

func (s *session) stopIdle() {
	s.mu.Lock()
	s.stopIdleLocked()
	s.mu.Unlock()
}

func (s *session) stopIdleLocked() {
	s.idleGen++
	if s.idle != nil {
		s.idle.Stop()
		s.idle = nil
	}
}

// armIdle commits an open utterance if no more audio arrives. The wait is
// the silence that is not already in the buffer.
func (s *session) armIdle() {
	s.mu.Lock()
	defer s.mu.Unlock()
	remain, ok := s.gate.SilenceRemainMS()
	if !s.vad || !ok {
		s.stopIdleLocked()
		return
	}
	s.idleGen++
	gen := s.idleGen
	if s.idle != nil {
		s.idle.Stop()
	}
	s.idle = time.AfterFunc(time.Duration(remain)*time.Millisecond, func() {
		s.fireIdle(gen)
	})
}

func (s *session) fireIdle(gen int) {
	s.mu.Lock()
	if gen != s.idleGen || !s.gate.heard {
		s.mu.Unlock()
		return
	}
	pcm := s.gate.Take()
	s.stopIdleLocked()
	s.stopSpeculateLocked()
	s.mu.Unlock()
	if len(pcm) < 2 {
		return
	}
	s.announceSilence()
	_ = s.send(map[string]any{"type": "input_audio_buffer.speech_stopped"})
	s.begin(pcm)
}

func (s *session) stopSpeculate() {
	s.mu.Lock()
	s.stopSpeculateLocked()
	s.mu.Unlock()
}

func (s *session) stopSpeculateLocked() {
	s.specGen++
	if s.spec != nil {
		s.spec.Stop()
		s.spec = nil
	}
}

// armSpeculate starts transcription during the silence after a loud packet.
// Another packet cancels it. A quiet packet transcribes immediately instead.
func (s *session) armSpeculate() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.vad || !s.gate.Heard() || s.vadTurn.SilenceMS <= 0 {
		s.stopSpeculateLocked()
		return
	}
	s.specGen++
	gen := s.specGen
	if s.spec != nil {
		s.spec.Stop()
	}
	s.spec = time.AfterFunc(loudSpeculateMS*time.Millisecond, func() {
		s.fireSpeculate(gen)
	})
}

func (s *session) fireSpeculate(gen int) {
	s.mu.Lock()
	if gen != s.specGen || !s.gate.Heard() || s.gate.quiet != 0 {
		s.mu.Unlock()
		return
	}
	pcm := s.gate.Peek()
	s.mu.Unlock()
	if len(pcm) < 2 {
		return
	}
	s.speculate(pcm)
}

func (s *session) vadEnabled() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.vad
}

func (s *session) buffer(pcm []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.pcm)+len(pcm) > maxPCM {
		s.pcm = nil
		return errors.New("input audio is too large")
	}
	s.pcm = append(s.pcm, pcm...)
	return nil
}

func (s *session) onCommit() {
	s.stopSpeculate()
	if !s.eng.Duplex {
		s.fail(errors.New("this socket speaks text; send conversation.item.create"))
		return
	}
	s.mu.Lock()
	var pcm []byte
	if s.vad {
		pcm = s.gate.Take()
	} else {
		pcm = append([]byte(nil), s.pcm...)
		s.pcm = nil
	}
	s.mu.Unlock()
	if len(pcm) < 2 {
		s.fail(errors.New("commit needs audio"))
		return
	}
	s.begin(pcm)
}

// warmNextEar loads the next transcriber without blocking the caller.
// Call it after the first audio byte is sent, so the load does not
// compete with the sentence being synthesized.
func (s *session) warmNextEar() {
	if s.ctx.Err() != nil {
		return
	}
	go s.warmEar()
}

func (s *session) warmEar() {
	s.mu.Lock()
	if (s.eng.WarmHearEx == nil && s.eng.WarmHear == nil) || s.earFinish != nil || s.earPending {
		s.mu.Unlock()
		return
	}
	s.earPending = true
	s.earGen++
	gen := s.earGen
	lang := s.lang
	ex := s.eng.WarmHearEx
	legacy := s.eng.WarmHear
	backend := s.eng.ASRBackend
	s.mu.Unlock()
	var startWarm func(context.Context, string) (earFinishFn, error)
	if ex != nil {
		startWarm = func(ctx context.Context, language string) (earFinishFn, error) {
			return ex(ctx, language)
		}
	} else {
		startWarm = func(ctx context.Context, language string) (earFinishFn, error) {
			fin, err := legacy(ctx, language)
			if err != nil || fin == nil {
				return nil, err
			}
			return func(ctx context.Context, wav []byte) (string, string, error) {
				text, err := fin(ctx, wav)
				return text, backend, err
			}, nil
		}
	}
	go s.completeEar(gen, lang, startWarm)
}

func (s *session) completeEar(gen int, lang string, fn func(context.Context, string) (earFinishFn, error)) {
	ctx, cancel := context.WithCancel(s.ctx)
	s.mu.Lock()
	if s.earGen != gen {
		s.mu.Unlock()
		cancel()
		return
	}
	s.earCancel = cancel
	s.mu.Unlock()
	finish, err := fn(ctx, lang)
	s.mu.Lock()
	if s.earPending && s.earGen == gen {
		s.earPending = false
	}
	if err != nil || finish == nil || s.earGen != gen {
		cancel()
		s.broadcastLocked()
		s.mu.Unlock()
		return
	}
	s.earCancel = cancel
	s.earFinish = finish
	var held []byte
	if s.earWaiters > 0 {
		s.heldPCM = nil
	} else if len(s.heldPCM) >= 2 {
		held = s.heldPCM
		s.heldPCM = nil
		s.speculating = true
	}
	s.broadcastLocked()
	s.mu.Unlock()
	if held != nil {
		s.speculate(held)
	}
}

func (s *session) dropEar() {
	s.mu.Lock()
	s.earGen++
	s.earPending = false
	s.heldPCM = nil
	cancel := s.earCancel
	s.earCancel = nil
	s.earFinish = nil
	s.broadcastLocked()
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (s *session) forEar(pcm []byte) []byte {
	return TrimQuietTail(pcm, s.eng.InputRate, s.vadAmplitude(), hearTailMS)
}

// vadAmplitude is the PCM cutoff for trim and resume peeks. Never the neural
// probability cutoff, even when Threshold was rewritten for Silero.
func (s *session) vadAmplitude() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.vadTurn.Amp()
}

func (s *session) speculate(pcm []byte) {
	s.mu.Lock()
	if s.earFinish == nil && s.earPending {
		s.heldPCM = append([]byte(nil), pcm...)
		s.speculating = false
		s.broadcastLocked()
		s.mu.Unlock()
		return
	}
	s.speculating = true
	s.mu.Unlock()
	finish, earCancel := s.takeEar()
	if finish == nil {
		if earCancel != nil {
			earCancel()
		}
		s.mu.Lock()
		s.speculating = false
		s.broadcastLocked()
		s.mu.Unlock()
		return
	}
	wav, err := WrapPCM16(s.eng.InputRate, s.forEar(pcm))
	if err != nil {
		if earCancel != nil {
			earCancel()
		}
		s.mu.Lock()
		s.speculating = false
		s.broadcastLocked()
		s.mu.Unlock()
		return
	}
	s.announceTranscribing()
	ctx, cancel := context.WithCancel(s.ctx)
	ch := make(chan guessResult, 1)
	go func() {
		text, backend, err := finish(ctx, wav)
		if err == nil {
			err = ctx.Err()
		}
		var start func()
		s.mu.Lock()
		if err == nil && s.guessCh == ch {
			start = s.prepareEarlyLocked(text)
		}
		s.mu.Unlock()
		if start != nil {
			start()
		}
		ch <- guessResult{text: text, backend: backend, err: err}
	}()
	var once sync.Once
	stop := func() {
		once.Do(func() {
			cancel()
			if earCancel != nil {
				earCancel()
			}
		})
	}
	s.mu.Lock()
	s.guessCh = ch
	s.guessCancel = stop
	s.speculating = false
	s.broadcastLocked()
	s.mu.Unlock()
}

func (s *session) dropGuess() {
	s.mu.Lock()
	cancel := s.guessCancel
	s.guessCancel = nil
	s.guessCh = nil
	s.heldPCM = nil
	early := s.early
	s.early = nil
	s.broadcastLocked()
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if early != nil {
		early.cancel()
	}
}

// prepareEarlyLocked starts the chat model for a transcript that is ready
// before the silence ends. The caller runs the returned function outside the lock.
func (s *session) prepareEarlyLocked(user string) func() {
	user = strings.TrimSpace(user)
	if user == "" || s.eng.StreamReply == nil || s.early != nil {
		return nil
	}
	inst, voice, lang, emotion := s.inst, s.voice, s.lang, s.emotion
	prior := append([]Exchange(nil), s.history...)
	ctx, cancel := context.WithCancel(s.ctx)
	er := &earlyReply{
		ctx:       ctx,
		cancel:    cancel,
		user:      user,
		voice:     voice,
		lang:      lang,
		emotion:   emotion,
		sentences: make(chan string, 32),
		spoken:    make(chan spoken, 128),
		done:      make(chan error, 1),
	}
	s.early = er
	reply := s.eng.StreamReply
	stages := s.eng.Stages
	return func() {
		go s.synthesizeEarly(er, voice, lang, emotion)
		stages.Report(StageGenerating, nil)
		err := reply(ctx, inst, user, prior, func(sentence string) error {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case er.sentences <- sentence:
				return nil
			}
		})
		close(er.sentences)
		er.done <- err
	}
}

// synthesizeEarly speaks each sentence as the model emits it. Playback waits
// until the turn commits. When synthesis finishes, the next transcriber
// starts, including while the silence timer is still running.
func (s *session) synthesizeEarly(er *earlyReply, voice, lang, emotion string) {
	defer s.warmNextEar()
	defer close(er.spoken)
	if s.eng.Speak == nil && s.eng.SpeakMore == nil {
		for range er.sentences {
		}
		return
	}
	for part := range er.sentences {
		part = strings.TrimSpace(part)
		if part == "" || er.ctx.Err() != nil {
			continue
		}
		textSent := false
		err := s.speakOut(er.ctx, part, voice, lang, emotion, func(clip Clip) error {
			item := spoken{clip: clip}
			if !textSent {
				item.text = part
				textSent = true
			}
			select {
			case <-er.ctx.Done():
				return er.ctx.Err()
			case er.spoken <- item:
				return nil
			}
		})
		if err != nil {
			select {
			case <-er.ctx.Done():
			case er.spoken <- spoken{err: err}:
			}
			for range er.sentences {
			}
			return
		}
	}
}

// speakOut emits one sentence. SpeakMore forwards each chunk as it is
// synthesized. Speak emits the finished clip.
func (s *session) speakOut(ctx context.Context, text, voice, lang, emotion string, emit func(Clip) error) error {
	if s.eng.SpeakMore != nil {
		return s.eng.SpeakMore(ctx, text, voice, lang, emotion, emit)
	}
	clip, err := s.eng.Speak(ctx, text, voice, lang, emotion)
	if err != nil {
		return err
	}
	if emit == nil {
		return nil
	}
	return emit(clip)
}

func (s *session) takeEarly() *earlyReply {
	s.mu.Lock()
	defer s.mu.Unlock()
	er := s.early
	s.early = nil
	return er
}

func (s *session) takeGuess() (chan guessResult, func()) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ch, cancel := s.guessCh, s.guessCancel
	s.guessCh = nil
	s.guessCancel = nil
	return ch, cancel
}

func (s *session) takeEar() (earFinishFn, context.CancelFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()
	finish, cancel := s.earFinish, s.earCancel
	s.earFinish = nil
	s.earCancel = nil
	return finish, cancel
}

func (s *session) broadcastLocked() {
	if s.kick == nil {
		s.kick = make(chan struct{})
		return
	}
	close(s.kick)
	s.kick = make(chan struct{})
}

// waitForEar blocks while a transcriber is loading or a clip is queued for it.
// A ready guess is preferred. Otherwise the loaded transcriber is returned.
func (s *session) waitForEar(ctx context.Context) (chan guessResult, func(), earFinishFn, context.CancelFunc, bool) {
	s.mu.Lock()
	for {
		if s.guessCh != nil {
			ch, cancel := s.guessCh, s.guessCancel
			s.guessCh, s.guessCancel = nil, nil
			s.mu.Unlock()
			return ch, cancel, nil, nil, false
		}
		if !s.earPending && !s.speculating && len(s.heldPCM) == 0 {
			finish, cancel := s.earFinish, s.earCancel
			s.earFinish, s.earCancel = nil, nil
			s.mu.Unlock()
			return nil, nil, finish, cancel, false
		}
		s.earWaiters++
		if s.kick == nil {
			s.kick = make(chan struct{})
		}
		kick := s.kick
		s.mu.Unlock()
		var cancelled bool
		select {
		case <-ctx.Done():
			cancelled = true
		case <-kick:
		}
		s.mu.Lock()
		s.earWaiters--
		if cancelled {
			s.mu.Unlock()
			return nil, nil, nil, nil, true
		}
	}
}

func (s *session) begin(pcm []byte) {
	s.mu.Lock()
	lang := s.lang
	s.mu.Unlock()
	guess, guessCancel := s.takeGuess()
	var finish earFinishFn
	var earCancel context.CancelFunc
	if guess == nil {
		finish, earCancel = s.takeEar()
	}
	wav, err := WrapPCM16(s.eng.InputRate, s.forEar(pcm))
	if err != nil && guess == nil {
		if earCancel != nil {
			earCancel()
		}
		s.fail(err)
		return
	}
	s.start(func(ctx context.Context) error {
		var text, backend string
		var err error
		if guess == nil && finish == nil {
			var stop bool
			guess, guessCancel, finish, earCancel, stop = s.waitForEar(ctx)
			if stop {
				return ctx.Err()
			}
		}
		if guess == nil {
			s.announceTranscribing()
		}
		if guess != nil {
			select {
			case <-ctx.Done():
				if guessCancel != nil {
					guessCancel()
				}
				return ctx.Err()
			case res := <-guess:
				text, backend, err = res.text, res.backend, res.err
			}
			if guessCancel != nil {
				guessCancel()
			}
			if err != nil && ctx.Err() == nil && wav != nil {
				text, backend, err = s.hearOnce(ctx, wav, lang)
			}
		} else if finish != nil {
			text, backend, err = finish(ctx, wav)
			if earCancel != nil {
				earCancel()
			}
		} else {
			text, backend, err = s.hearOnce(ctx, wav, lang)
		}
		if err != nil {
			if early := s.takeEarly(); early != nil {
				early.cancel()
			}
			s.warmNextEar()
			return err
		}
		text = strings.TrimSpace(text)
		ev := map[string]any{
			"type":       "conversation.item.input_audio_transcription.completed",
			"transcript": text,
		}
		if backend == "" {
			backend = strings.TrimSpace(s.eng.ASRBackend)
		}
		if backend != "" {
			ev["asr_backend"] = backend
		}
		if tag := strings.TrimSpace(s.eng.ASR); tag != "" {
			ev["asr"] = tag
		}
		if err := s.send(ev); err != nil {
			s.warmNextEar()
			return err
		}
		if early := s.takeEarly(); early != nil {
			return s.playEarly(ctx, early)
		}
		return s.speakReply(ctx, text)
	})
}

func (s *session) hearOnce(ctx context.Context, wav []byte, lang string) (text, backend string, err error) {
	if s.eng.HearEx != nil {
		return s.eng.HearEx(ctx, wav, lang)
	}
	if s.eng.Hear != nil {
		text, err = s.eng.Hear(ctx, wav, lang)
		return text, s.eng.ASRBackend, err
	}
	return "", "", errors.New("duplex has no transcriber")
}

func (s *session) onItem(ev map[string]any) {
	item, _ := ev["item"].(map[string]any)
	text := itemText(item)
	if text == "" {
		s.fail(errors.New("conversation item needs text"))
		return
	}
	s.mu.Lock()
	s.text = text
	if v, ok := item["emotion"].(string); ok {
		s.emotion = strings.TrimSpace(v)
	}
	s.mu.Unlock()
}

func itemText(item map[string]any) string {
	if item == nil {
		return ""
	}
	if text, ok := item["text"].(string); ok && strings.TrimSpace(text) != "" {
		return strings.TrimSpace(text)
	}
	content, _ := item["content"].([]any)
	var b strings.Builder
	for _, part := range content {
		m, _ := part.(map[string]any)
		if m == nil {
			continue
		}
		if text, ok := m["text"].(string); ok {
			b.WriteString(text)
		}
	}
	return strings.TrimSpace(b.String())
}

func (s *session) onCreate() {
	s.mu.Lock()
	text := s.text
	s.mu.Unlock()
	if strings.TrimSpace(text) == "" {
		s.fail(errors.New("response.create needs a conversation item"))
		return
	}
	s.start(func(ctx context.Context) error {
		return s.speakReply(ctx, text)
	})
}

func (s *session) speakReply(ctx context.Context, user string) error {
	s.mu.Lock()
	inst, voice, lang, emotion := s.inst, s.voice, s.lang, s.emotion
	prior := append([]Exchange(nil), s.history...)
	s.mu.Unlock()
	if s.eng.Duplex && s.eng.StreamReply != nil {
		return s.speakStream(ctx, user, inst, voice, lang, emotion, prior)
	}
	reply := user
	if s.eng.Duplex {
		var err error
		reply, err = s.eng.Reply(ctx, inst, user, prior)
		if err != nil {
			s.warmNextEar()
			return err
		}
		if err := ctx.Err(); err != nil {
			s.warmNextEar()
			return err
		}
	}
	reply = strings.TrimSpace(reply)
	if err := s.send(map[string]any{"type": "response.created"}); err != nil {
		s.warmNextEar()
		return err
	}
	s.mu.Lock()
	s.replyGen = s.gen
	s.mu.Unlock()
	parts := Sentences(reply)
	if len(parts) == 0 {
		if s.eng.Duplex {
			s.remember(user, "")
		}
		s.warmNextEar()
		return s.finishTurn("completed")
	}
	sent := false
	defer func() {
		if !sent {
			s.warmNextEar()
		}
	}()
	for i, part := range parts {
		if err := ctx.Err(); err != nil {
			return err
		}
		delta := part
		if i > 0 {
			delta = " " + part
		}
		if err := s.send(map[string]any{"type": "response.output_text.delta", "delta": delta}); err != nil {
			return err
		}
		if s.eng.Duplex {
			if i == 0 {
				s.remember(user, part)
			} else {
				s.extend(part)
			}
		}
		if err := s.speakOut(ctx, part, voice, lang, emotion, func(clip Clip) error {
			if err := s.emitClip(clip); err != nil {
				return err
			}
			if !sent {
				sent = true
				s.warmNextEar()
			}
			return nil
		}); err != nil {
			return err
		}
	}
	if err := s.send(map[string]any{"type": "response.output_audio.done"}); err != nil {
		return err
	}
	s.reportStatus(StageAudioDone, nil)
	return s.finishTurn("completed")
}

// playEarly speaks a reply that was generated during the silence.
func (s *session) playEarly(ctx context.Context, er *earlyReply) error {
	defer er.cancel()
	go func() {
		select {
		case <-ctx.Done():
			er.cancel()
		case <-er.ctx.Done():
		}
	}()
	if err := s.send(map[string]any{"type": "response.created"}); err != nil {
		return err
	}
	s.mu.Lock()
	s.replyGen = s.gen
	s.mu.Unlock()
	playErr := s.playSpoken(ctx, er.user, er.spoken)
	streamErr := <-er.done
	if playErr != nil {
		return playErr
	}
	if streamErr != nil && ctx.Err() == nil {
		return streamErr
	}
	return nil
}

const holdChunkMS = 20
const holdSilenceAmp int16 = 200

func (s *session) loadHolds() {
	if s.eng.PrepareHolds == nil {
		return
	}
	hold, more := s.eng.PrepareHolds(s.ctx)
	s.mu.Lock()
	if !clipEmpty(hold) {
		s.hold = hold
	}
	if !clipEmpty(more) {
		s.holdMore = more
	}
	s.mu.Unlock()
}

func (s *session) holds() (hold, more Clip) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.hold, s.holdMore
}

func (s *session) playSpoken(ctx context.Context, user string, clips <-chan spoken) error {
	hold, _ := s.holds()
	if !clipEmpty(hold) {
		return s.playSpokenWithHold(ctx, user, clips)
	}
	return s.playSpokenDirect(ctx, user, clips)
}

// speakStream plays each sentence as StreamReply emits it. The model keeps
// writing while the sentence's audio plays. A cancel drops sentences still
// inside the model. The same hold path as playEarly covers a late opening.
func (s *session) speakStream(ctx context.Context, user, inst, voice, lang, emotion string, prior []Exchange) error {
	if err := s.send(map[string]any{"type": "response.created"}); err != nil {
		return err
	}
	s.mu.Lock()
	s.replyGen = s.gen
	s.mu.Unlock()
	sentences := make(chan string, 32)
	spokenCh := make(chan spoken, 128)
	done := make(chan error, 1)
	go func() {
		defer close(spokenCh)
		for part := range sentences {
			part = strings.TrimSpace(part)
			if part == "" || ctx.Err() != nil {
				continue
			}
			// Text goes out before synthesis finishes so a cancel can land
			// while Speak is still running. Hold waits for PCM, not text.
			select {
			case <-ctx.Done():
				return
			case spokenCh <- spoken{text: part}:
			}
			err := s.speakOut(ctx, part, voice, lang, emotion, func(clip Clip) error {
				select {
				case <-ctx.Done():
					return ctx.Err()
				case spokenCh <- spoken{clip: clip}:
					return nil
				}
			})
			if err != nil {
				select {
				case <-ctx.Done():
				case spokenCh <- spoken{err: err}:
				}
				for range sentences {
				}
				return
			}
		}
	}()
	go func() {
		s.eng.Stages.Report(StageGenerating, nil)
		err := s.eng.StreamReply(ctx, inst, user, prior, func(sentence string) error {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case sentences <- sentence:
				return nil
			}
		})
		close(sentences)
		done <- err
	}()
	playErr := s.playSpoken(ctx, user, spokenCh)
	streamErr := <-done
	if playErr != nil {
		return playErr
	}
	if streamErr != nil && ctx.Err() == nil {
		return streamErr
	}
	return nil
}

func (s *session) playSpokenWithHold(ctx context.Context, user string, clips <-chan spoken) error {
	n := 0
	sent := false
	defer func() {
		if !sent {
			s.warmNextEar()
		}
	}()
	var playErr error
	emitItem := func(item spoken) {
		if playErr != nil || ctx.Err() != nil {
			if playErr == nil {
				playErr = ctx.Err()
			}
			return
		}
		if item.err != nil {
			playErr = item.err
			return
		}
		part := strings.TrimSpace(item.text)
		if part == "" && clipEmpty(item.clip) {
			return
		}
		if part != "" {
			delta := part
			if n > 0 {
				delta = " " + part
			}
			if err := s.send(map[string]any{"type": "response.output_text.delta", "delta": delta}); err != nil {
				playErr = err
				return
			}
			if n == 0 {
				s.remember(user, part)
			} else {
				s.extend(part)
			}
			n++
		}
		if !clipEmpty(item.clip) {
			if err := s.emitClip(item.clip); err != nil {
				playErr = err
				return
			}
			sent = true
		}
	}
	onText := func(item spoken) error {
		if playErr != nil {
			return playErr
		}
		emitItem(item)
		return playErr
	}
	first, err := s.takeOpeningOrHold(ctx, clips, onText)
	if err != nil {
		return err
	}
	if first != nil {
		emitItem(*first)
	}
	for item := range clips {
		emitItem(item)
	}
	if playErr != nil {
		return playErr
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if n == 0 && !sent {
		s.remember(user, "")
		return s.finishTurn("completed")
	}
	if err := s.send(map[string]any{"type": "response.output_audio.done"}); err != nil {
		return err
	}
	s.reportStatus(StageAudioDone, nil)
	return s.finishTurn("completed")
}

// takeOpeningOrHold returns the first model PCM when it is already ready.
// Otherwise it plays Hold, then HoldMore once, until that audio arrives.
// Text-only items are forwarded with onText and do not end the hold.
func (s *session) takeOpeningOrHold(ctx context.Context, clips <-chan spoken, onText func(spoken) error) (*spoken, error) {
	if item, err, ready := s.pollOpening(ctx, clips, true, onText); ready {
		return item, err
	}
	hold, more := s.holds()
	opening, err := s.emitHoldUntil(ctx, hold, clips, onText)
	if err != nil || opening != nil {
		return opening, err
	}
	if clipEmpty(more) {
		item, err, _ := s.pollOpening(ctx, clips, false, onText)
		return item, err
	}
	return s.emitHoldUntil(ctx, more, clips, onText)
}

// emitHoldUntil streams hold PCM in short chunks. When opening PCM arrives it
// fades about 20 ms of the remaining hold and returns that opening. The hold
// is not history. Text-only items call onText and keep the hold going.
func (s *session) emitHoldUntil(ctx context.Context, hold Clip, clips <-chan spoken, onText func(spoken) error) (*spoken, error) {
	pcm := TrimSilencePCM16(hold.PCM, holdSilenceAmp)
	if len(pcm) < 2 || hold.Rate <= 0 {
		item, err, ready := s.pollOpening(ctx, clips, true, onText)
		if ready {
			return item, err
		}
		return nil, nil
	}
	chunkBytes := hold.Rate * 2 * holdChunkMS / 1000
	if chunkBytes < 2 {
		chunkBytes = 2
	}
	chunkBytes -= chunkBytes % 2
	mime := hold.MIME
	if mime == "" {
		mime = "audio/pcm"
	}
	for off := 0; off < len(pcm); {
		if item, err, handled := s.pollOpening(ctx, clips, true, onText); handled {
			if err != nil || item == nil {
				return item, err
			}
			fadeEnd := off + chunkBytes
			if fadeEnd > len(pcm) {
				fadeEnd = len(pcm) - len(pcm)%2
			}
			if fadeEnd > off {
				if err := s.emitClip(Clip{MIME: mime, Rate: hold.Rate, PCM: FadePCM16(pcm[off:fadeEnd], fadeEnd-off)}); err != nil {
					return nil, err
				}
			}
			return item, nil
		}
		end := off + chunkBytes
		if end > len(pcm) {
			end = len(pcm)
			end -= end % 2
		}
		if end <= off {
			break
		}
		if err := s.emitClip(Clip{MIME: mime, Rate: hold.Rate, PCM: append([]byte(nil), pcm[off:end]...)}); err != nil {
			return nil, err
		}
		off = end
		timer := time.NewTimer(time.Duration(holdChunkMS) * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case item, ok := <-clips:
			timer.Stop()
			if !ok {
				return nil, nil
			}
			if item.err != nil {
				return nil, item.err
			}
			if !openingAudio(item) {
				if onText != nil {
					if err := onText(item); err != nil {
						return nil, err
					}
				}
				continue
			}
			fadeEnd := off + chunkBytes
			if fadeEnd > len(pcm) {
				fadeEnd = len(pcm) - len(pcm)%2
			}
			if fadeEnd > off {
				if err := s.emitClip(Clip{MIME: mime, Rate: hold.Rate, PCM: FadePCM16(pcm[off:fadeEnd], fadeEnd-off)}); err != nil {
					return nil, err
				}
			}
			return &item, nil
		case <-timer.C:
		}
	}
	return nil, nil
}

func openingAudio(item spoken) bool {
	return item.err != nil || !clipEmpty(item.clip)
}

// pollOpening reads opening PCM from clips. Text-only items call onText and
// are not treated as ready. When nonblock is true an empty channel returns
// handled=false.
func (s *session) pollOpening(ctx context.Context, clips <-chan spoken, nonblock bool, onText func(spoken) error) (*spoken, error, bool) {
	for {
		if nonblock {
			select {
			case <-ctx.Done():
				return nil, ctx.Err(), true
			case item, ok := <-clips:
				if !ok {
					return nil, nil, true
				}
				if item.err != nil {
					return nil, item.err, true
				}
				if !openingAudio(item) {
					if onText != nil {
						if err := onText(item); err != nil {
							return nil, err, true
						}
					}
					return nil, nil, false
				}
				return &item, nil, true
			default:
				return nil, nil, false
			}
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err(), true
		case item, ok := <-clips:
			if !ok {
				return nil, nil, true
			}
			if item.err != nil {
				return nil, item.err, true
			}
			if !openingAudio(item) {
				if onText != nil {
					if err := onText(item); err != nil {
						return nil, err, true
					}
				}
				continue
			}
			return &item, nil, true
		}
	}
}

func (s *session) playSpokenDirect(ctx context.Context, user string, clips <-chan spoken) error {
	n := 0
	sent := false
	defer func() {
		if !sent {
			s.warmNextEar()
		}
	}()
	var playErr error
	for item := range clips {
		if playErr != nil || ctx.Err() != nil {
			if playErr == nil {
				playErr = ctx.Err()
			}
			continue
		}
		if item.err != nil {
			playErr = item.err
			continue
		}
		part := strings.TrimSpace(item.text)
		if part == "" && clipEmpty(item.clip) {
			continue
		}
		if part != "" {
			delta := part
			if n > 0 {
				delta = " " + part
			}
			if err := s.send(map[string]any{"type": "response.output_text.delta", "delta": delta}); err != nil {
				playErr = err
				continue
			}
			if n == 0 {
				s.remember(user, part)
			} else {
				s.extend(part)
			}
		}
		n++
		if err := s.emitClip(item.clip); err != nil {
			playErr = err
			continue
		}
		sent = true
	}
	if playErr != nil {
		return playErr
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if n == 0 {
		s.remember(user, "")
		return s.finishTurn("completed")
	}
	if err := s.send(map[string]any{"type": "response.output_audio.done"}); err != nil {
		return err
	}
	s.reportStatus(StageAudioDone, nil)
	return s.finishTurn("completed")
}

func (s *session) finishTurn(status string) error {
	if status == "completed" {
		s.mu.Lock()
		s.replyGen = 0
		s.mu.Unlock()
		s.warmReply()
	}
	return s.done(status)
}

// warmReply prefills the current instructions and history. The same prefix
// is left running, so a later speech start does not cancel it.
func (s *session) warmReply() {
	if s.eng.WarmReply == nil || !s.eng.Duplex {
		return
	}
	s.mu.Lock()
	inst := s.inst
	prior := append([]Exchange(nil), s.history...)
	key := prefixKey(inst, prior)
	if key == s.warmKey && (s.warmCancel != nil || s.warmDone) {
		s.mu.Unlock()
		return
	}
	if s.warmCancel != nil {
		s.warmCancel()
	}
	ctx, cancel := context.WithCancel(s.ctx)
	s.warmCancel = cancel
	s.warmKey = key
	s.warmDone = false
	s.mu.Unlock()
	go func() {
		err := s.eng.WarmReply(ctx, inst, prior)
		s.mu.Lock()
		if s.warmKey == key {
			s.warmCancel = nil
			s.warmDone = err == nil && ctx.Err() == nil
		}
		s.mu.Unlock()
	}()
}

func (s *session) dropWarm() {
	s.mu.Lock()
	cancel := s.warmCancel
	s.warmCancel = nil
	s.warmDone = false
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func prefixKey(inst string, prior []Exchange) string {
	var b strings.Builder
	b.WriteString(inst)
	b.WriteByte(0)
	for _, ex := range prior {
		b.WriteString(ex.User)
		b.WriteByte(0)
		b.WriteString(ex.Assistant)
		b.WriteByte(0)
	}
	return b.String()
}

func (s *session) remember(user, assistant string) {
	user = strings.TrimSpace(user)
	assistant = strings.TrimSpace(assistant)
	if user == "" && assistant == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.history = append(s.history, Exchange{User: user, Assistant: assistant})
	if len(s.history) > maxHistory {
		s.history = append([]Exchange(nil), s.history[len(s.history)-maxHistory:]...)
	}
}

func (s *session) extend(sentence string) {
	sentence = strings.TrimSpace(sentence)
	if sentence == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.history) == 0 {
		return
	}
	last := &s.history[len(s.history)-1]
	if last.Assistant != "" {
		last.Assistant += " "
	}
	last.Assistant += sentence
}

func (s *session) done(status string) error {
	return s.send(map[string]any{"type": "response.done", "response": map[string]any{"status": status}})
}

func clipEmpty(clip Clip) bool {
	return len(clip.PCM) == 0 && len(clip.Raw) == 0
}

func (s *session) emitClip(clip Clip) error {
	mime := clip.MIME
	payload := clip.PCM
	if mime == "" || mime == "audio/pcm" {
		mime = "audio/pcm"
		payload = clip.PCM
	} else {
		payload = clip.Raw
	}
	rate := clip.Rate
	const chunk = 24 * 1024
	if len(payload) == 0 {
		return nil
	}
	for off := 0; off < len(payload); off += chunk {
		end := off + chunk
		if end > len(payload) {
			end = len(payload)
		}
		ev := map[string]any{
			"type":  "response.output_audio.delta",
			"delta": base64.StdEncoding.EncodeToString(payload[off:end]),
			"mime":  mime,
		}
		if rate > 0 {
			ev["rate"] = rate
		}
		if err := s.send(ev); err != nil {
			return err
		}
	}
	return nil
}

func (s *session) start(fn func(context.Context) error) {
	s.cancel()
	ctx, cancel := context.WithCancel(s.ctx)
	s.mu.Lock()
	s.gen++
	gen := s.gen
	s.turn = cancel
	s.mu.Unlock()
	go func() {
		err := fn(ctx)
		s.mu.Lock()
		if s.gen == gen {
			s.turn = nil
		}
		cancelled := ctx.Err() != nil && s.replyGen == gen
		if s.replyGen == gen {
			s.replyGen = 0
		}
		s.mu.Unlock()
		if cancelled {
			_ = s.done("cancelled")
		}
		if err != nil && ctx.Err() == nil {
			s.fail(err)
		}
	}()
}

func (s *session) cancel() {
	s.mu.Lock()
	cancel := s.turn
	s.turn = nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (s *session) fail(err error) {
	if err == nil {
		return
	}
	payload := map[string]any{"message": err.Error()}
	var coded *Error
	if errors.As(err, &coded) && coded != nil && coded.Code != "" {
		payload["code"] = coded.Code
		payload["message"] = coded.Message
	}
	_ = s.send(map[string]any{"type": "error", "error": payload})
}

// reportStatus sends response.status for duplex milestones. Extra keys are
// copied onto the event (silence_duration_ms, etc).
func (s *session) reportStatus(status string, extra map[string]any) {
	if status == "" {
		return
	}
	ev := map[string]any{"type": "response.status", "status": status}
	for k, v := range extra {
		if k == "" || k == "type" || k == "status" {
			continue
		}
		ev[k] = v
	}
	_ = s.send(ev)
}

func (s *session) announceSilence() {
	s.mu.Lock()
	if s.silenceAnnounced || !s.vad {
		s.mu.Unlock()
		return
	}
	s.silenceAnnounced = true
	ms := s.vadTurn.SilenceMS
	s.mu.Unlock()
	s.reportStatus(StageSilenceStarted, map[string]any{"silence_duration_ms": ms})
}

func (s *session) clearSilenceAnnounce() {
	s.mu.Lock()
	s.silenceAnnounced = false
	s.sttAnnounced = false
	s.mu.Unlock()
}

func (s *session) announceTranscribing() {
	s.mu.Lock()
	if s.sttAnnounced {
		s.mu.Unlock()
		return
	}
	s.sttAnnounced = true
	s.mu.Unlock()
	_ = s.send(map[string]any{"type": "conversation.item.input_audio_transcription.started"})
	s.reportStatus(StageTranscribing, nil)
}

func (s *session) send(ev map[string]any) error {
	b, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return writeFrame(s.w, 0x1, b)
}

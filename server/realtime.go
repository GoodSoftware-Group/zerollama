package server

import (
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/server/voicepipe"
	"github.com/ollama/ollama/server/wsproxy"
	"github.com/ollama/ollama/types/model"
)

// RealtimeHandler serves GET /v1/realtime. A streaming llama-server model is
// proxied. A speech model speaks text on this socket. pipeline=duplex
// transcribes a committed clip, asks the chat model named by model, and
// speaks the reply with the voice query. turn=server_vad commits on PCM
// silence. On this llama-server socket the sample rate is session.created's
// rate, and a conversation session also gets response.create. response.cancel
// stops the reply.
func (s *Server) RealtimeHandler(c *gin.Context) {
	if err := wsproxy.Validate(c.Request); err != nil {
		c.JSON(http.StatusUpgradeRequired, gin.H{"error": err.Error()})
		return
	}
	modelName := strings.TrimSpace(c.Query("model"))
	if modelName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model query parameter is required"})
		return
	}
	served, err := applyModelAlias(c, modelName)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	modelRef, err := parseAndValidateModelRef(served)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.JSON(http.StatusBadRequest, gin.H{"error": "realtime is a local llama-server socket"})
		return
	}
	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "model not found"})
		return
	}
	m, err := GetModel(name.String())
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "model not found"})
		return
	}
	turnOn, turn, err := parseTurnQuery(c.Query("turn"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if strings.TrimSpace(c.Query("pipeline")) == "duplex" {
		s.serveDuplex(c, name.String(), m)
		return
	}
	role := modelMTMDRole(m)
	if msg := model.RealtimeBlock(role); msg != "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": msg})
		return
	}
	if hostSpeechOnly(m) || role.SpeechOnly() {
		s.serveSpeech(c, name.String(), m)
		return
	}
	if hostTranscribeOnly(m) || modelUsesLlamaASR(m) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "this model transcribes on POST /v1/audio/transcriptions"})
		return
	}

	// The request context stays live until this handler returns. The scheduler
	// holds the runner for that whole socket, then applies keep_alive.
	ctx := llm.WithRealtimeLoad(c.Request.Context())
	runner, _, _, _, releaseQoS, err := s.scheduleRunner(ctx, name.String(), nil, nil, nil, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, name.String(), err)
		return
	}
	defer releaseQoS()

	if _, ok := runner.(llm.LlamaASRTranscriber); !ok || runner.GetPort() <= 0 {
		c.JSON(http.StatusConflict, gin.H{"error": "realtime requires llama-server; unload this model and retry (ZEROLLAMA_LLAMA_SERVER must not be 0)"})
		return
	}

	if turnOn {
		err = wsproxy.ProxyTurn(c.Writer, c.Request, runner.GetPort(), c.Query("intent"), c.Query("language"), turn)
	} else {
		err = wsproxy.Proxy(c.Writer, c.Request, runner.GetPort(), c.Query("intent"), c.Query("language"))
	}
	if err != nil {
		if !c.Writer.Written() {
			c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		}
	}
}

func parseTurnQuery(q string) (bool, voicepipe.Turn, error) {
	switch strings.ToLower(strings.TrimSpace(q)) {
	case "", "none":
		return false, voicepipe.Turn{}, nil
	case "server_vad":
		return true, voicepipe.DefaultTurn(), nil
	default:
		return false, voicepipe.Turn{}, errors.New("turn must be server_vad or none")
	}
}

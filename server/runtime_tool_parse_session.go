package server

import (
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/tools"
)

const (
	toolParseSessionTTL    = 10 * time.Minute
	maxToolParseSessions   = 256
)

type toolParseSession struct {
	method         string
	builtinParser  parsers.Parser
	templateParser *tools.Parser
	lastUsed       time.Time
}

type toolParseSessionRegistry struct {
	mu       sync.Mutex
	sessions map[string]*toolParseSession
}

var toolParseSessions toolParseSessionRegistry

func init() {
	toolParseSessions.sessions = make(map[string]*toolParseSession)
}

func newToolParseSessionID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

func (r *toolParseSessionRegistry) sweep() {
	cutoff := time.Now().Add(-toolParseSessionTTL)
	for id, sess := range r.sessions {
		if sess.lastUsed.Before(cutoff) {
			delete(r.sessions, id)
		}
	}
}

func (r *toolParseSessionRegistry) open(
	m *Model,
	toolsList api.Tools,
	msgs []api.Message,
	think *api.ThinkValue,
) (string, string, error) {
	processedTools, builtinParser, hasBuiltin := prepareToolsForRender(m, toolsList, msgs, think)
	sess := &toolParseSession{lastUsed: time.Now()}
	method := ""
	if builtinParser != nil && hasBuiltin {
		sess.builtinParser = builtinParser
		method = strings.TrimSpace(m.Config.Parser)
	} else if len(processedTools) > 0 && m.Template != nil && m.Template.Template != nil {
		sess.templateParser = tools.NewParser(m.Template.Template, processedTools)
		method = "template"
	}
	if sess.builtinParser == nil && sess.templateParser == nil {
		return "", "", errNoToolParser
	}
	sess.method = method
	id := newToolParseSessionID()
	r.mu.Lock()
	r.sweep()
	if len(r.sessions) >= maxToolParseSessions {
		r.mu.Unlock()
		return "", "", errTooManyToolParseSessions
	}
	r.sessions[id] = sess
	r.mu.Unlock()
	return id, method, nil
}

var (
	errNoToolParser              = &parseSessionError{msg: "no tool parser available for model"}
	errTooManyToolParseSessions  = &parseSessionError{msg: "too many tool parse sessions"}
)

type parseSessionError struct {
	msg string
}

func (e *parseSessionError) Error() string {
	return e.msg
}

func (r *toolParseSessionRegistry) close(id string) {
	r.mu.Lock()
	delete(r.sessions, id)
	r.mu.Unlock()
}

func (r *toolParseSessionRegistry) add(
	id string,
	content string,
	done bool,
) (string, string, []api.ToolCall, string, error) {
	r.mu.Lock()
	sess, ok := r.sessions[id]
	if !ok {
		r.mu.Unlock()
		return "", "", nil, "", errSessionNotFound
	}
	sess.lastUsed = time.Now()
	r.mu.Unlock()

	if sess.builtinParser != nil {
		c, th, tc, err := sess.builtinParser.Add(content, done)
		if err != nil {
			return "", "", nil, sess.method, err
		}
		if done {
			r.close(id)
		}
		return c, th, tc, sess.method, nil
	}
	if sess.templateParser != nil {
		tc, c := sess.templateParser.Add(content)
		if done {
			if len(tc) == 0 {
				if tail := sess.templateParser.Content(); tail != "" {
					c = tail
				}
			}
			r.close(id)
		}
		return c, "", tc, sess.method, nil
	}
	return content, "", nil, "", nil
}

var errSessionNotFound = &parseSessionError{msg: "tool parse session not found"}

type openToolParseSessionRequest struct {
	Model    string          `json:"model"`
	Messages []api.Message   `json:"messages"`
	Tools    api.Tools       `json:"tools"`
	Think    *api.ThinkValue `json:"think"`
}

type toolParseSessionChunkRequest struct {
	SessionID string `json:"session_id"`
	Content   string `json:"content"`
	Done      bool   `json:"done"`
}

func toolParseResponse(
	content, thinking string,
	toolCalls []api.ToolCall,
	method string,
) gin.H {
	out := gin.H{
		"content": content,
		"method":  method,
	}
	if thinking != "" {
		out["thinking"] = thinking
	}
	if len(toolCalls) > 0 {
		for i := range toolCalls {
			if toolCalls[i].ID == "" {
				toolCalls[i].ID = toolCallId()
			}
		}
		out["tool_calls"] = toolCalls
	}
	return out
}

// OpenToolParseSessionHandler starts a stateful tool-output parse session (streaming Q4).
func (s *Server) OpenToolParseSessionHandler(c *gin.Context) {
	_ = s
	var req openToolParseSessionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	modelName := strings.TrimSpace(req.Model)
	if modelName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model required"})
		return
	}
	m, err := GetModel(modelName)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	id, method, err := toolParseSessions.open(m, req.Tools, req.Messages, req.Think)
	if err != nil {
		if err == errNoToolParser {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if err == errTooManyToolParseSessions {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	tag := "{"
	if m.Template != nil && m.Template.Template != nil {
		tag = tools.TemplateToolTag(m.Template.Template)
	}
	c.JSON(http.StatusOK, gin.H{
		"session_id": id,
		"method":     method,
		"tool_tag":   tag,
	})
}

// ToolParseSessionChunkHandler feeds a stream chunk into a parse session.
func (s *Server) ToolParseSessionChunkHandler(c *gin.Context) {
	_ = s
	var req toolParseSessionChunkRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if strings.TrimSpace(req.SessionID) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "session_id required"})
		return
	}
	content, thinking, toolCalls, method, err := toolParseSessions.add(
		req.SessionID,
		req.Content,
		req.Done,
	)
	if err != nil {
		if err == errSessionNotFound {
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, toolParseResponse(content, thinking, toolCalls, method))
}

// CloseToolParseSessionHandler drops a parse session without a final chunk.
func (s *Server) CloseToolParseSessionHandler(c *gin.Context) {
	var req struct {
		SessionID string `json:"session_id"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if strings.TrimSpace(req.SessionID) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "session_id required"})
		return
	}
	toolParseSessions.close(req.SessionID)
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

package server

import (
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/fleet"
)

// AssignHoldRegistry tracks F5 soft holds (jti → expiry).
// Why not a real scheduler slot: docs forbid long quote windows; this only bumps
// reported queue_depth until TTL or the agent presents the token on chat/generate.
type AssignHoldRegistry struct {
	mu    sync.Mutex
	holds map[string]time.Time
}

func newAssignHoldRegistry() *AssignHoldRegistry {
	return &AssignHoldRegistry{holds: make(map[string]time.Time)}
}

func (r *AssignHoldRegistry) Register(jti string, exp time.Time) {
	if r == nil || jti == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.holds == nil {
		r.holds = make(map[string]time.Time)
	}
	r.holds[jti] = exp.UTC()
}

// Consume removes a hold (agent started inference). Returns true if it existed.
func (r *AssignHoldRegistry) Consume(jti string) bool {
	if r == nil || jti == "" {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	_, ok := r.holds[jti]
	delete(r.holds, jti)
	return ok
}

// ActiveCount prunes expired holds and returns the live count.
func (r *AssignHoldRegistry) ActiveCount(now time.Time) int {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	now = now.UTC()
	for jti, exp := range r.holds {
		if !now.Before(exp) {
			delete(r.holds, jti)
		}
	}
	return len(r.holds)
}

func (s *Server) ensureAssignHolds() *AssignHoldRegistry {
	if s == nil {
		return nil
	}
	if s.assignHolds == nil {
		s.assignHolds = newAssignHoldRegistry()
	}
	return s.assignHolds
}

// AssignHoldHandler registers a soft hold from a valid F5 token (fleet push or agent).
func (s *Server) AssignHoldHandler(c *gin.Context) {
	if !fleet.AssignTokenEnabled() {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{"error": "assignment tokens disabled"})
		return
	}
	var req fleet.AssignHoldRequest
	if err := c.ShouldBindJSON(&req); err != nil || strings.TrimSpace(req.Token) == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "token required"})
		return
	}
	claims, err := fleet.ParseAssignToken(req.Token, time.Now())
	if err != nil {
		status := http.StatusUnauthorized
		if errors.Is(err, fleet.ErrAssignTokenExpired) {
			status = http.StatusConflict
		}
		c.AbortWithStatusJSON(status, gin.H{"error": err.Error()})
		return
	}
	s.ensureAssignHolds().Register(claims.JTI, claims.ExpiresAt)
	c.JSON(http.StatusOK, gin.H{
		"status":     "held",
		"jti":        claims.JTI,
		"expires_at": claims.ExpiresAt.UTC().Format(time.RFC3339),
		"node_id":    claims.NodeID,
		"model":      claims.Model,
	})
}

// assignmentTokenMiddleware validates optional X-Zerollama-Assignment-Token and consumes the hold.
func (s *Server) assignmentTokenMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		tok := strings.TrimSpace(c.GetHeader(fleet.AssignmentTokenHeader))
		if tok == "" {
			c.Next()
			return
		}
		if !fleet.AssignTokenEnabled() {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "assignment tokens disabled on node"})
			return
		}
		claims, err := fleet.ParseAssignToken(tok, time.Now())
		if err != nil {
			status := http.StatusUnauthorized
			if errors.Is(err, fleet.ErrAssignTokenExpired) {
				status = http.StatusConflict
			}
			c.AbortWithStatusJSON(status, gin.H{"error": err.Error()})
			return
		}
		s.ensureAssignHolds().Consume(claims.JTI)
		c.Set("assignment_jti", claims.JTI)
		c.Set("assignment_model", claims.Model)
		c.Next()
	}
}

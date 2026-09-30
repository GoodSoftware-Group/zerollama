package server

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

// MultiDecodeHandler serves POST /v1/multidecode (forest decode via llama-server --multidecode).
//
// WHY dedicated: not chat, not batch chat — harnesses send tokens/pos/parent (or
// parent_node_ids for sticky multi-step). See docs/multidecode-llama-cpp.md.
func (s *Server) MultiDecodeHandler(c *gin.Context) {
	var req api.MultiDecodeRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if strings.TrimSpace(req.Model) == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	if len(req.Tokens) == 0 || len(req.Pos) == 0 || len(req.Leaves) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "tokens, pos, and leaves are required"})
		return
	}
	if len(req.Parent) == 0 && len(req.ParentNodeIDs) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "parent or parent_node_ids required"})
		return
	}

	if served, err := applyModelAlias(c, req.Model); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	} else {
		req.Model = served
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.AbortWithStatusJSON(http.StatusNotImplemented, gin.H{"error": "multidecode is not available for cloud models"})
		return
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	r, _, _, _, releaseQoS, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	defer releaseQoS()

	md, ok := r.(llm.MultiDecoder)
	if !ok {
		c.AbortWithStatusJSON(http.StatusNotImplemented, gin.H{"error": "model runner does not support multidecode"})
		return
	}

	llmReq := llm.MultiDecodeRequest{
		Model:         req.Model,
		Tokens:        req.Tokens,
		Pos:           req.Pos,
		Parent:        req.Parent,
		NodeIDs:       req.NodeIDs,
		ParentNodeIDs: req.ParentNodeIDs,
		Leaves:        req.Leaves,
		ReturnLogits:  req.ReturnLogits,
		Clear:         req.Clear,
		NPredict:      req.NPredict,
	}
	resp, err := md.MultiDecode(c.Request.Context(), llmReq)
	if err != nil {
		var se api.StatusError
		if errors.As(err, &se) {
			c.AbortWithStatusJSON(se.StatusCode, gin.H{"error": se.ErrorMessage})
			return
		}
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	out := api.MultiDecodeResponse{
		Results: make([]api.MultiDecodeLeafResult, len(resp.Results)),
		NodeIDs: resp.NodeIDs,
	}
	for i, row := range resp.Results {
		out.Results[i] = api.MultiDecodeLeafResult{
			Leaf:   row.Leaf,
			Token:  row.Token,
			NodeID: row.NodeID,
			Logits: row.Logits,
		}
	}
	c.JSON(http.StatusOK, out)
}

package server

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

const proposeLoadMaxBatch = 8

// ProposeLoadHandler implements POST /api/propose-load — batch can-load with honest co-residency.
// WHY: Decide needs a multi-model plan API; when 2+ distinct runtime GGUFs appear we set
// serialize_required instead of implying co-resident warmth (Python is single-GGUF).
func (s *Server) ProposeLoadHandler(c *gin.Context) {
	var req api.ProposeLoadRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if len(req.Models) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "models is required"})
		return
	}
	if len(req.Models) > proposeLoadMaxBatch {
		c.JSON(http.StatusBadRequest, gin.H{"error": "models batch exceeds max of 8"})
		return
	}

	ctx := c.Request.Context()
	results := make([]api.CanLoadResponse, 0, len(req.Models))
	runtimeGGUFKeys := make([]string, 0)
	var warm api.ProcessResponse
	loadOrder := make([]string, 0, len(req.Models))
	evict := make([]string, 0)
	allCanWithoutEvict := true
	confExact, confHeur := 0, 0

	for _, item := range req.Models {
		item.Model = strings.TrimSpace(item.Model)
		if item.Model == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "each model entry requires model"})
			return
		}
		resp := s.evaluateCanLoad(ctx, item)
		results = append(results, resp)
		warm = resp.Warm
		if resp.AlreadyLoaded {
			loadOrder = append([]string{item.Model}, loadOrder...)
		} else {
			loadOrder = append(loadOrder, item.Model)
		}
		if resp.NeedsEviction {
			allCanWithoutEvict = false
			if resp.EvictionReason != "" {
				evict = append(evict, item.Model+":"+resp.EvictionReason)
			} else {
				evict = append(evict, item.Model)
			}
		}
		if !resp.CanLoad {
			allCanWithoutEvict = false
		}
		switch resp.Confidence {
		case canLoadConfidenceExact:
			confExact++
		default:
			confHeur++
		}
		if resp.Backend == "runtime" {
			gguf := ""
			if est, ok := resp.VramEstimate["gguf"].(string); ok {
				gguf = est
			}
			if gguf == "" && item.Options != nil {
				if g, ok := item.Options["gguf"].(string); ok {
					gguf = strings.TrimSpace(g)
				}
			}
			if gguf == "" {
				opts := runtimeProxyOptions(item.Model, 0, false, item.Options)
				if g, ok := opts["gguf"].(string); ok {
					gguf = strings.TrimSpace(g)
				}
			}
			key := gguf
			if key == "" {
				key = "runtime:" + item.Model
			}
			dup := false
			for _, existing := range runtimeGGUFKeys {
				if ggufPathsEqual(existing, key) || existing == key {
					dup = true
					break
				}
			}
			if !dup {
				runtimeGGUFKeys = append(runtimeGGUFKeys, key)
			}
		}
	}

	coResident := len(runtimeGGUFKeys) <= 1
	serialize := !coResident
	fits := allCanWithoutEvict && coResident
	conf := canLoadConfidenceHeuristic
	notes := ""
	if confExact > 0 && confHeur == 0 {
		conf = canLoadConfidenceExact
	} else if confExact > 0 && confHeur > 0 {
		conf = "mixed"
	}
	if serialize {
		notes = "runtime_single_resident: batch has multiple distinct runtime GGUFs; serialize interviews"
		fits = false
	} else if !allCanWithoutEvict {
		notes = "one or more models need eviction or cannot load"
	}

	plan := api.ProposeLoadPlan{
		FitsWithoutEviction: fits,
		CoResident:          coResident,
		SerializeRequired:   serialize,
		LoadOrder:           loadOrder,
		EvictCandidates:     evict,
		Confidence:          conf,
		Notes:               notes,
	}
	for _, r := range results {
		if r.DeviceCount > plan.DeviceCount {
			plan.DeviceCount = r.DeviceCount
		}
		if r.TensorParallel > plan.TensorParallel {
			plan.TensorParallel = r.TensorParallel
			plan.SplitMode = r.SplitMode
			plan.TensorSplit = r.TensorSplit
			plan.MainGPU = r.MainGPU
		}
	}

	c.JSON(http.StatusOK, api.ProposeLoadResponse{
		Models: results,
		Warm:   warm,
		Plan:   plan,
	})
}

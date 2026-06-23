package server

import (
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func TestRecordInferenceMultimodalEstimate(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	meta := &inferenceAccessMeta{route: "/api/chat", start: time.Now()}
	c.Set(inferenceAccessLogKey, meta)

	recordInferenceMultimodalEstimate(c, 768, 1536, 0)
	if meta.imageTokens != 768 || meta.videoTokens != 1536 || meta.audioTokens != 0 {
		t.Fatalf("meta=%+v", meta)
	}

	recordInferenceMultimodalEstimate(c, 0, 0, 0)
	if meta.imageTokens != 768 {
		t.Fatal("zero estimate should not clear prior values")
	}
}

func TestRecordInferencePaddedLayout(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	meta := &inferenceAccessMeta{route: "/api/chat", start: time.Now()}
	c.Set(inferenceAccessLogKey, meta)

	recordInferencePaddedLayout(c, 900, "qwen3vl_hf_skip_placeholders")
	if meta.paddedInputIDsLen != 900 {
		t.Fatalf("padded_input_ids_len=%d want 900", meta.paddedInputIDsLen)
	}
	if meta.paddedLayoutConsume != "qwen3vl_hf_skip_placeholders" {
		t.Fatalf("consume=%q", meta.paddedLayoutConsume)
	}
	recordInferencePaddedLayout(c, 0, "")
	if meta.paddedInputIDsLen != 900 {
		t.Fatal("zero should not clear prior padded layout len")
	}
}

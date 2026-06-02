package trainingworker

import (
	"context"
	"errors"
	"testing"
)

func TestTrainSubmitJobRespectsSubmitGuard(t *testing.T) {
	c := &Client{}
	c.SetInferenceSubmitGuard(func(context.Context) error {
		return errors.New("inference busy")
	})
	_, err := c.trainSubmitJob(context.Background(), "train", []byte("{}"))
	if err == nil || err.Error() != "inference busy" {
		t.Fatalf("err=%v", err)
	}
}

package server

import "github.com/ollama/ollama/x/trainprio"

// TrainingPriority controls idle-wait and defer-queue behavior on submit.
// Defined type (not alias) so methods stay in server/; logic lives in x/trainprio
// for CGO-free host units.
type TrainingPriority trainprio.Priority

const (
	TrainingPriorityNormal TrainingPriority = TrainingPriority(trainprio.Normal)
	TrainingPriorityLow    TrainingPriority = TrainingPriority(trainprio.Low)
	TrainingPriorityHigh   TrainingPriority = TrainingPriority(trainprio.High)
)

func (p TrainingPriority) String() string {
	return trainprio.Priority(p).String()
}

func (p TrainingPriority) bypassesIdleWait() bool {
	return trainprio.Priority(p).BypassesIdleWait()
}

func (p TrainingPriority) prefersDeferQueue() bool {
	return trainprio.Priority(p).PrefersDeferQueue()
}

func parseTrainingPriority(raw string) (TrainingPriority, error) {
	p, err := trainprio.Parse(raw)
	return TrainingPriority(p), err
}

func resolveTrainingSubmitPriority(priority, slo string) (TrainingPriority, error) {
	p, err := trainprio.Resolve(priority, slo)
	return TrainingPriority(p), err
}

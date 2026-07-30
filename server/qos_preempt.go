package server

import (
	"errors"
	"fmt"
)

// qosDeferAbortError wraps ctx cancel/deadline while waiting behind higher-priority QoS.
// WHY: Hermes needs preempted_reason to distinguish "waited behind interactive" from
// generic client cancel or queue-full busy — without mid-stream hard preemption.
type qosDeferAbortError struct {
	cause  error
	policy string
}

func (e *qosDeferAbortError) Error() string {
	if e == nil || e.cause == nil {
		return "qos deferred"
	}
	if e.policy == "" {
		return e.cause.Error()
	}
	return fmt.Sprintf("%s (preempted_reason=%s)", e.cause.Error(), e.policy)
}

func (e *qosDeferAbortError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

func wrapQoSDeferAbort(err error, policy string) error {
	if err == nil {
		return nil
	}
	if policy == "" {
		return err
	}
	return &qosDeferAbortError{cause: err, policy: policy}
}

func preemptedReasonFromErr(err error) string {
	var e *qosDeferAbortError
	if errors.As(err, &e) && e != nil {
		return e.policy
	}
	return ""
}

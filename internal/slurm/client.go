// Package slurm talks to a real Slurm installation through its CLI.
//
// Why CLI and not libslurm: Slurm is GPL. sbatch / scontrol / scancel keep the
// cluster's slurm.conf, MUNGE identity, and accounting. This package does not
// embed slurmctld and does not speak the binary RPC.
package slurm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

var (
	// ErrBadRequest is a caller or operator error (missing walltime, illegal field).
	ErrBadRequest = errors.New("invalid slurm submit")
	// ErrRejected is sbatch/scancel refusing the request. The message is Slurm's.
	ErrRejected = errors.New("slurm rejected request")
	// ErrUnavailable means the Slurm binary is not on PATH.
	ErrUnavailable = errors.New("slurm command unavailable")
	// ErrNotFound means the job is not in scontrol, squeue, or sacct.
	ErrNotFound = errors.New("slurm job not found")
)

// Request is the subset of sbatch flags zerollama forwards unchanged.
// T6 priority is not copied onto --qos; the cluster's QoS stays the site's.
type Request struct {
	Partition   string `json:"partition,omitempty"`
	Account     string `json:"account,omitempty"`
	QOS         string `json:"qos,omitempty"`
	Time        string `json:"time,omitempty"`
	GPUs        int    `json:"gpus,omitempty"`
	CPUsPerTask int    `json:"cpus_per_task,omitempty"`
	Mem         string `json:"mem,omitempty"`
	Constraint  string `json:"constraint,omitempty"`
	Dependency  string `json:"dependency,omitempty"`
	UID         string `json:"uid,omitempty"`
	JobName     string `json:"-"`
	WorkDir     string `json:"-"`
	ScriptPath  string `json:"-"`
	Output      string `json:"-"`
	Error       string `json:"-"`
}

// View is one job as reported by scontrol, squeue, or sacct.
type View struct {
	JobID     string
	State     string // Slurm token, e.g. PENDING, RUNNING, COMPLETED
	Partition string
	TimeLeft  string
	Reason    string
}

// Mapped is the train-job vocabulary plus the design's state word.
type Mapped struct {
	// Status matches embedded training jobs: pending, running, completed, failed, cancelled.
	Status string
	// State is the design word: waiting, running, done, error, cancelled.
	State string
}

// Runner executes a Slurm CLI. Tests substitute a fake; production uses ExecRunner.
type Runner interface {
	Run(ctx context.Context, name string, args ...string) (stdout, stderr string, err error)
}

// ExecRunner shells out to the real binaries.
type ExecRunner struct{}

func (ExecRunner) Run(ctx context.Context, name string, args ...string) (string, string, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	return out.String(), errb.String(), err
}

// Client is a queue-client transport over sbatch.
type Client struct {
	Runner   Runner
	Sbatch   string
	Scontrol string
	Squeue   string
	Sacct    string
	Scancel  string
}

// NewExecClient returns a client that runs the Slurm commands on PATH.
func NewExecClient() *Client {
	return &Client{Runner: ExecRunner{}}
}

func (c *Client) run(ctx context.Context, name string, args ...string) (string, string, error) {
	r := c.Runner
	if r == nil {
		r = ExecRunner{}
	}
	stdout, stderr, err := r.Run(ctx, name, args...)
	if err != nil && errors.Is(err, exec.ErrNotFound) {
		return stdout, stderr, fmt.Errorf("%w: %s", ErrUnavailable, name)
	}
	return stdout, stderr, err
}

func (c *Client) bin(set, fallback string) string {
	if strings.TrimSpace(set) != "" {
		return strings.TrimSpace(set)
	}
	return fallback
}

// Submit runs sbatch --parsable and returns the numeric job id (cluster suffix stripped).
func (c *Client) Submit(ctx context.Context, spec Request) (string, error) {
	if strings.TrimSpace(spec.ScriptPath) == "" {
		return "", fmt.Errorf("%w: script path required", ErrBadRequest)
	}
	args := sbatchArgs(spec)
	stdout, stderr, err := c.run(ctx, c.bin(c.Sbatch, "sbatch"), args...)
	if err != nil {
		if errors.Is(err, ErrUnavailable) {
			return "", err
		}
		msg := strings.TrimSpace(stderr)
		if msg == "" {
			msg = strings.TrimSpace(stdout)
		}
		if msg == "" {
			msg = err.Error()
		}
		return "", fmt.Errorf("%w: %s", ErrRejected, firstLine(msg))
	}
	id := ParseParsable(stdout)
	if id == "" {
		return "", fmt.Errorf("%w: sbatch returned no job id", ErrRejected)
	}
	return id, nil
}

// Job reads state. scontrol --json first, then squeue, then sacct -X for jobs
// that have already left the queue.
func (c *Client) Job(ctx context.Context, id string) (View, error) {
	id = strings.TrimSpace(id)
	if id == "" {
		return View{}, fmt.Errorf("%w: empty job id", ErrBadRequest)
	}
	stdout, _, err := c.run(ctx, c.bin(c.Scontrol, "scontrol"), "show", "job", id, "--json")
	if err == nil {
		if v, ok := ParseScontrolJSON(stdout, id); ok {
			return v, nil
		}
	} else if errors.Is(err, ErrUnavailable) {
		return View{}, err
	}
	stdout, _, err = c.run(ctx, c.bin(c.Squeue, "squeue"), "-h", "-j", id, "-o", "%T|%P|%L")
	if err == nil {
		if v, ok := ParseSqueueLine(stdout, id); ok {
			return v, nil
		}
	} else if errors.Is(err, ErrUnavailable) {
		return View{}, err
	}
	stdout, _, err = c.run(ctx, c.bin(c.Sacct, "sacct"), "-X", "-n", "-P", "-j", id, "--format=State,Partition,Elapsed")
	if err != nil {
		if errors.Is(err, ErrUnavailable) {
			return View{}, err
		}
		return View{}, fmt.Errorf("%w: %s", ErrNotFound, id)
	}
	if v, ok := ParseSacctLine(stdout, id); ok {
		return v, nil
	}
	return View{}, fmt.Errorf("%w: %s", ErrNotFound, id)
}

// Cancel runs scancel. Callers skip this once the job is already terminal.
func (c *Client) Cancel(ctx context.Context, id string) error {
	id = strings.TrimSpace(id)
	if id == "" {
		return fmt.Errorf("%w: empty job id", ErrBadRequest)
	}
	_, stderr, err := c.run(ctx, c.bin(c.Scancel, "scancel"), id)
	if err != nil {
		if errors.Is(err, ErrUnavailable) {
			return err
		}
		msg := strings.TrimSpace(stderr)
		if msg == "" {
			msg = err.Error()
		}
		return fmt.Errorf("%w: %s", ErrRejected, firstLine(msg))
	}
	return nil
}

func sbatchArgs(spec Request) []string {
	args := []string{"--parsable"}
	if spec.JobName != "" {
		args = append(args, "--job-name="+spec.JobName)
	}
	if spec.WorkDir != "" {
		args = append(args, "--chdir="+spec.WorkDir)
	}
	if spec.Output != "" {
		args = append(args, "--output="+spec.Output)
	}
	if spec.Error != "" {
		args = append(args, "--error="+spec.Error)
	}
	if spec.Partition != "" {
		args = append(args, "--partition="+spec.Partition)
	}
	if spec.Account != "" {
		args = append(args, "--account="+spec.Account)
	}
	if spec.QOS != "" {
		args = append(args, "--qos="+spec.QOS)
	}
	if spec.Time != "" {
		args = append(args, "--time="+spec.Time)
	}
	if spec.GPUs > 0 {
		args = append(args, "--gpus="+strconv.Itoa(spec.GPUs))
	}
	if spec.CPUsPerTask > 0 {
		args = append(args, "--cpus-per-task="+strconv.Itoa(spec.CPUsPerTask))
	}
	if spec.Mem != "" {
		args = append(args, "--mem="+spec.Mem)
	}
	if spec.Constraint != "" {
		args = append(args, "--constraint="+spec.Constraint)
	}
	if spec.Dependency != "" {
		args = append(args, "--dependency="+spec.Dependency)
	}
	if spec.UID != "" {
		args = append(args, "--uid="+spec.UID)
	}
	args = append(args, spec.ScriptPath)
	return args
}

// ParseParsable reads `sbatch --parsable` output: "48212" or "48212;cluster".
func ParseParsable(stdout string) string {
	line := firstLine(stdout)
	if line == "" {
		return ""
	}
	id, _, _ := strings.Cut(line, ";")
	id = strings.TrimSpace(id)
	if id == "" || strings.ContainsAny(id, " \t") {
		return ""
	}
	return id
}

// ParseScontrolJSON reads `scontrol show job --json`.
func ParseScontrolJSON(stdout, wantID string) (View, bool) {
	raw := stdout
	if i := strings.Index(stdout, "{"); i >= 0 {
		raw = stdout[i:]
	}
	var doc struct {
		Jobs []map[string]any `json:"jobs"`
	}
	if err := json.Unmarshal([]byte(raw), &doc); err != nil || len(doc.Jobs) == 0 {
		return View{}, false
	}
	job := doc.Jobs[0]
	for _, j := range doc.Jobs {
		if jsonJobID(j["job_id"]) == wantID {
			job = j
			break
		}
	}
	state := jsonJobState(job["job_state"])
	if state == "" {
		return View{}, false
	}
	v := View{
		JobID:     jsonJobID(job["job_id"]),
		State:     state,
		Partition: jsonString(job["partition"]),
		Reason:    cleanReason(jsonString(job["state_reason"])),
		TimeLeft:  jsonTimeLeft(job["time_left"]),
	}
	if v.JobID == "" {
		v.JobID = wantID
	}
	return v, true
}

// ParseSqueueLine reads `squeue -h -o %T|%P|%L` (one job).
func ParseSqueueLine(stdout, id string) (View, bool) {
	line := firstLine(stdout)
	if line == "" {
		return View{}, false
	}
	parts := strings.Split(line, "|")
	if len(parts) < 1 || strings.TrimSpace(parts[0]) == "" {
		return View{}, false
	}
	v := View{JobID: id, State: strings.TrimSpace(parts[0])}
	if len(parts) > 1 {
		v.Partition = strings.TrimSpace(parts[1])
	}
	if len(parts) > 2 {
		v.TimeLeft = strings.TrimSpace(parts[2])
	}
	return v, true
}

// ParseSacctLine reads `sacct -X -n -P --format=State,Partition,Elapsed`.
func ParseSacctLine(stdout, id string) (View, bool) {
	line := firstLine(stdout)
	if line == "" {
		return View{}, false
	}
	parts := strings.Split(line, "|")
	state := strings.TrimSpace(parts[0])
	if state == "" {
		return View{}, false
	}
	v := View{JobID: id, State: state}
	if len(parts) > 1 {
		v.Partition = strings.TrimSpace(parts[1])
	}
	if len(parts) > 2 {
		v.TimeLeft = strings.TrimSpace(parts[2])
	}
	return v, true
}

// MapState converts a Slurm job state onto the training job status and the
// design's state word. Compound forms ("CANCELLED by 1000") use the first token.
func MapState(slurmState string) Mapped {
	token := strings.ToUpper(strings.TrimSpace(slurmState))
	if i := strings.IndexAny(token, " +"); i >= 0 {
		token = token[:i]
	}
	switch token {
	case "PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED":
		return Mapped{Status: "pending", State: "waiting"}
	case "RUNNING", "COMPLETING":
		return Mapped{Status: "running", State: "running"}
	case "COMPLETED":
		return Mapped{Status: "completed", State: "done"}
	case "CANCELLED":
		return Mapped{Status: "cancelled", State: "cancelled"}
	case "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "REVOKED":
		return Mapped{Status: "failed", State: "error"}
	case "PREEMPTED":
		// A preempted job can be requeued. Marking it failed stops polling, so
		// the requeue would never show up. Keep it pending until Slurm says
		// PENDING, RUNNING, or a real terminal state.
		return Mapped{Status: "pending", State: "waiting"}
	default:
		if token == "" {
			return Mapped{Status: "pending", State: "waiting"}
		}
		return Mapped{Status: "pending", State: "waiting"}
	}
}

var (
	reToken = regexp.MustCompile(`^[A-Za-z0-9_@.,-]+$`)
	reCons  = regexp.MustCompile(`^[A-Za-z0-9_@.,&|-]+$`)
	reTime  = regexp.MustCompile(`^([0-9]+-)?[0-9]{1,2}:[0-9]{2}(:[0-9]{2})?$|^[0-9]+$`)
	reMem   = regexp.MustCompile(`(?i)^[0-9]+[KMGT]?$`)
	reDep   = regexp.MustCompile(`^[A-Za-z0-9_:]+$`)
	reUID   = regexp.MustCompile(`^[0-9]+$`)
	rePy    = regexp.MustCompile(`^[A-Za-z0-9_./-]+$`)
)

// Validate checks fields that would otherwise be passed straight to sbatch.
// Walltime is required: sites with a partition default set ZEROLLAMA_SLURM_TIME
// (or the request time) so a missing clock does not become UNLIMITED by accident.
func Validate(spec Request) error {
	if err := optional(spec.Partition, reToken, "partition"); err != nil {
		return err
	}
	if err := optional(spec.Account, reToken, "account"); err != nil {
		return err
	}
	if err := optional(spec.QOS, reToken, "qos"); err != nil {
		return err
	}
	if err := optional(spec.Constraint, reCons, "constraint"); err != nil {
		return err
	}
	if strings.TrimSpace(spec.Time) == "" {
		return fmt.Errorf("%w: time is required (request slurm.time or ZEROLLAMA_SLURM_TIME)", ErrBadRequest)
	}
	if !reTime.MatchString(spec.Time) {
		return fmt.Errorf("%w: time %q", ErrBadRequest, spec.Time)
	}
	if spec.GPUs < 0 {
		return fmt.Errorf("%w: gpus", ErrBadRequest)
	}
	if spec.CPUsPerTask < 0 {
		return fmt.Errorf("%w: cpus_per_task", ErrBadRequest)
	}
	if err := optional(spec.Mem, reMem, "mem"); err != nil {
		return err
	}
	if spec.Dependency != "" {
		if !reDep.MatchString(spec.Dependency) {
			return fmt.Errorf("%w: dependency %q", ErrBadRequest, spec.Dependency)
		}
		if spec.Dependency != "singleton" && !strings.Contains(spec.Dependency, ":") {
			return fmt.Errorf("%w: dependency %q", ErrBadRequest, spec.Dependency)
		}
	}
	if spec.UID != "" && !reUID.MatchString(spec.UID) {
		return fmt.Errorf("%w: uid %q", ErrBadRequest, spec.UID)
	}
	return nil
}

// ValidatePython accepts an interpreter path with no shell metacharacters.
func ValidatePython(bin string) error {
	if bin == "" {
		return nil
	}
	if !rePy.MatchString(bin) {
		return fmt.Errorf("%w: python %q", ErrBadRequest, bin)
	}
	return nil
}

func optional(v string, re *regexp.Regexp, name string) error {
	if v == "" {
		return nil
	}
	if strings.HasPrefix(v, "-") || !re.MatchString(v) {
		return fmt.Errorf("%w: %s %q", ErrBadRequest, name, v)
	}
	return nil
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	line, _, _ := strings.Cut(s, "\n")
	return strings.TrimSpace(line)
}

func jsonJobID(v any) string {
	switch n := v.(type) {
	case string:
		return strings.TrimSpace(n)
	case float64:
		return strconv.FormatInt(int64(n), 10)
	case json.Number:
		return n.String()
	default:
		return ""
	}
}

func jsonString(v any) string {
	s, _ := v.(string)
	return strings.TrimSpace(s)
}

func jsonJobState(v any) string {
	switch n := v.(type) {
	case string:
		return strings.TrimSpace(n)
	case []any:
		if len(n) == 0 {
			return ""
		}
		s, _ := n[0].(string)
		return strings.TrimSpace(s)
	default:
		return ""
	}
}

func jsonTimeLeft(v any) string {
	switch n := v.(type) {
	case string:
		return strings.TrimSpace(n)
	case map[string]any:
		if inf, _ := n["infinite"].(bool); inf {
			return "UNLIMITED"
		}
		num, ok := n["number"]
		if !ok {
			return ""
		}
		var sec int64
		switch x := num.(type) {
		case float64:
			sec = int64(x)
		case json.Number:
			sec, _ = x.Int64()
		default:
			return ""
		}
		if sec < 0 {
			return ""
		}
		h := sec / 3600
		m := (sec % 3600) / 60
		s := sec % 60
		return fmt.Sprintf("%d:%02d:%02d", h, m, s)
	default:
		return ""
	}
}

func cleanReason(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "none", "null":
		return ""
	default:
		return s
	}
}

package slurm

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestParseParsable(t *testing.T) {
	if got := ParseParsable("48212\n"); got != "48212" {
		t.Fatalf("id=%q", got)
	}
	if got := ParseParsable("48212;campus\n"); got != "48212" {
		t.Fatalf("cluster id=%q", got)
	}
	if got := ParseParsable("not a job\n"); got != "" {
		t.Fatalf("expected empty, got %q", got)
	}
}

func TestParseScontrolAndMap(t *testing.T) {
	raw := `{
	  "jobs": [{
	    "job_id": 48212,
	    "job_state": ["PENDING"],
	    "partition": "gpu",
	    "state_reason": "None",
	    "time_left": {"set": true, "infinite": false, "number": 3661}
	  }]
	}`
	v, ok := ParseScontrolJSON(raw, "48212")
	if !ok {
		t.Fatal("expected parse")
	}
	if v.State != "PENDING" || v.Partition != "gpu" || v.TimeLeft != "1:01:01" {
		t.Fatalf("%+v", v)
	}
	if v.Reason != "" {
		t.Fatalf("reason=%q", v.Reason)
	}
	m := MapState(v.State)
	if m.Status != "pending" || m.State != "waiting" {
		t.Fatalf("%+v", m)
	}
	for _, tc := range []struct {
		in, status, state string
	}{
		{"RUNNING", "running", "running"},
		{"COMPLETING", "running", "running"},
		{"COMPLETED", "completed", "done"},
		{"CANCELLED by 1000", "cancelled", "cancelled"},
		{"TIMEOUT", "failed", "error"},
		{"OUT_OF_MEMORY", "failed", "error"},
		{"NODE_FAIL", "failed", "error"},
		{"SUSPENDED", "pending", "waiting"},
		{"PREEMPTED", "pending", "waiting"},
	} {
		got := MapState(tc.in)
		if got.Status != tc.status || got.State != tc.state {
			t.Fatalf("%s → %+v", tc.in, got)
		}
	}
}

func TestSqueueAndSacct(t *testing.T) {
	v, ok := ParseSqueueLine("RUNNING|gpu|3:59:00\n", "9")
	if !ok || v.State != "RUNNING" || v.Partition != "gpu" || v.TimeLeft != "3:59:00" {
		t.Fatalf("%+v %v", v, ok)
	}
	v, ok = ParseSacctLine("COMPLETED|gpu|00:01:02\n", "9")
	if !ok || v.State != "COMPLETED" {
		t.Fatalf("%+v", v)
	}
}

func TestValidate(t *testing.T) {
	ok := Request{Partition: "gpu", Account: "lab", QOS: "normal", Time: "04:00:00", GPUs: 1, Mem: "32G", Constraint: "a100", Dependency: "afterok:48001"}
	if err := Validate(ok); err != nil {
		t.Fatal(err)
	}
	if err := Validate(Request{}); err == nil || !strings.Contains(err.Error(), "time") {
		t.Fatalf("missing time: %v", err)
	}
	bad := ok
	bad.Partition = "gpu;rm"
	if err := Validate(bad); err == nil {
		t.Fatal("expected partition reject")
	}
	bad = ok
	bad.Dependency = "afterok"
	if err := Validate(bad); err == nil {
		t.Fatal("expected dependency reject")
	}
	bad = ok
	bad.UID = "root"
	if err := Validate(bad); err == nil {
		t.Fatal("expected uid reject")
	}
}

func TestSbatchArgsNoUIDUnlessSet(t *testing.T) {
	args := sbatchArgs(Request{Time: "1:00:00", Partition: "gpu", GPUs: 1, ScriptPath: "/tmp/job.sh"})
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--parsable") || !strings.Contains(joined, "--gpus=1") || !strings.Contains(joined, "--partition=gpu") {
		t.Fatalf("%s", joined)
	}
	if strings.Contains(joined, "--uid") || strings.Contains(joined, "--qos") {
		t.Fatalf("unexpected flags: %s", joined)
	}
	with := sbatchArgs(Request{Time: "1:00:00", UID: "1000", ScriptPath: "/tmp/job.sh"})
	if !strings.Contains(strings.Join(with, " "), "--uid=1000") {
		t.Fatalf("%v", with)
	}
}

func TestWriteBundleUsesTrainingEntry(t *testing.T) {
	dir := t.TempDir()
	path, err := WriteBundle(dir, "/srv/zerollama", "python3", "train", []byte(`{"model_name":"m"}`))
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if !strings.Contains(s, "training.process_training_request") {
		t.Fatal(s)
	}
	if strings.Contains(s, "zerollama serve") {
		t.Fatal("script must not start serve")
	}
	if _, err := WriteBundle(dir, "/srv/zerollama", "python3;id", "train", nil); err == nil {
		t.Fatal("expected python reject")
	}
}

type fakeRunner struct {
	out map[string]string
	err map[string]error
	got [][]string
}

func (f *fakeRunner) Run(_ context.Context, name string, args ...string) (string, string, error) {
	f.got = append(f.got, append([]string{name}, args...))
	if err := f.err[name]; err != nil {
		return "", "Invalid partition name specified", err
	}
	return f.out[name], "", nil
}

func TestSubmitAndJobUseCLI(t *testing.T) {
	f := &fakeRunner{out: map[string]string{
		"sbatch":   "48212;campus\n",
		"scontrol": `{"jobs":[{"job_id":48212,"job_state":["RUNNING"],"partition":"gpu","time_left":{"number":10}}]}`,
	}}
	c := &Client{Runner: f}
	id, err := c.Submit(context.Background(), Request{Time: "01:00:00", ScriptPath: "/tmp/job.sh", Partition: "gpu"})
	if err != nil || id != "48212" {
		t.Fatalf("submit %q %v", id, err)
	}
	if !strings.Contains(strings.Join(f.got[0], " "), "--parsable") {
		t.Fatalf("%v", f.got[0])
	}
	v, err := c.Job(context.Background(), id)
	if err != nil || v.State != "RUNNING" || v.TimeLeft != "0:00:10" {
		t.Fatalf("%+v %v", v, err)
	}
	f.err = map[string]error{"sbatch": errExit(1)}
	_, err = c.Submit(context.Background(), Request{Time: "01:00:00", ScriptPath: "/tmp/job.sh"})
	if err == nil || !strings.Contains(err.Error(), "Invalid partition") {
		t.Fatalf("%v", err)
	}
}

type errExit int

func (e errExit) Error() string { return "exit" }

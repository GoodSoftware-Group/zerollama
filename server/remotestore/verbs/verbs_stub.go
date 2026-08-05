//go:build !rdma

package verbs

import (
	"errors"
	"os"
	"unsafe"
)

// Endpoint is a placeholder when built without RDMA.
type Endpoint struct {
	QPN, PSN uint32
	LID      uint16
	GID      string
	GIDIndex int
	Port     uint8
	MTU      int
	Device   string
}

type Device struct{}
type QP struct{}
type MR struct{}

func OpenFirst() (*Device, error) { return nil, errors.New("verbs: built without -tags rdma") }
func Available() bool             { return false }
func (d *Device) Close()          {}
func (d *Device) CreateQP() (*QP, error) {
	return nil, errors.New("verbs: built without -tags rdma")
}
func (d *Device) ProbeCap() (string, string, int, int, uint16) { return "", "", 0, 0, 0 }
func (d *Device) AllocMR(size int, src []byte, remoteRead bool) (*MR, error) {
	return nil, errors.New("verbs: built without -tags rdma")
}
func (d *Device) RegFileRange(f *os.File, offset, length int64, remoteRead bool) (*MR, error) {
	return nil, errors.New("verbs: built without -tags rdma")
}
func (m *MR) Bytes() []byte { return nil }
func (d *Device) RegMRAddr(addr unsafe.Pointer, length int, remoteRead bool) (*MR, error) {
	return nil, errors.New("verbs: built without -tags rdma")
}
func (q *QP) Endpoint() Endpoint { return Endpoint{} }
func (q *QP) Close()             {}
func (q *QP) Connect(Endpoint) error {
	return errors.New("verbs: built without -tags rdma")
}
func (q *QP) ReadRemote(*MR, int, uint64, uint32, int) error {
	return errors.New("verbs: built without -tags rdma")
}
func (q *QP) ReadRemotePipeline(*MR, uint64, uint32, int, int, int) error {
	return errors.New("verbs: built without -tags rdma")
}
func (m *MR) Close()           {}
func (m *MR) Addr() uint64     { return 0 }
func (m *MR) RKey() uint32     { return 0 }
func (m *MR) Len() int         { return 0 }
func FormatGID([]byte) string  { return "" }
func ParseGID(string) ([16]byte, error) {
	return [16]byte{}, errors.New("verbs: built without -tags rdma")
}

//go:build rdma

// Package verbs is a thin cgo wrapper around libibverbs for RC QP + RDMA READ.
//
// Why: storaged needs one-sided READ of content-addressed blob ranges. Control
// plane stays HMAC HTTP; this package only owns HCA/QP/MR/CQ.
package verbs

/*
#cgo pkg-config: libibverbs
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <infiniband/verbs.h>

static void gid_to_bytes(union ibv_gid *g, unsigned char *out) {
	memcpy(out, g->raw, 16);
}
static void bytes_to_gid(const unsigned char *in, union ibv_gid *g) {
	memcpy(g->raw, in, 16);
}

// Avoid cgo port_attr compatibility macros by wrapping query_port.
static int zl_query_port(struct ibv_context *ctx, uint8_t port, struct ibv_port_attr *attr) {
	return ibv_query_port(ctx, port, attr);
}

static int zl_post_rdma_read(struct ibv_qp *qp, uint64_t laddr, uint32_t length, uint32_t lkey,
                             uint64_t raddr, uint32_t rkey) {
	struct ibv_sge sge = {
		.addr   = laddr,
		.length = length,
		.lkey   = lkey,
	};
	struct ibv_send_wr wr = {0}, *bad = NULL;
	wr.opcode = IBV_WR_RDMA_READ;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.wr.rdma.remote_addr = raddr;
	wr.wr.rdma.rkey = rkey;
	return ibv_post_send(qp, &wr, &bad);
}

static int zl_modify_qp_init(struct ibv_qp *qp, uint8_t port) {
	struct ibv_qp_attr a = {0};
	a.qp_state = IBV_QPS_INIT;
	a.port_num = port;
	a.pkey_index = 0;
	a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
	return ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

static int zl_modify_qp_rtr(struct ibv_qp *qp, uint8_t port, enum ibv_mtu mtu,
                            uint32_t dest_qpn, uint32_t rq_psn,
                            int use_lid, uint16_t dlid, uint8_t sgid_index,
                            const unsigned char *dgid) {
	struct ibv_qp_attr a = {0};
	a.qp_state = IBV_QPS_RTR;
	a.path_mtu = mtu;
	a.dest_qp_num = dest_qpn;
	a.rq_psn = rq_psn;
	a.max_dest_rd_atomic = 1;
	a.min_rnr_timer = 12;
	a.ah_attr.port_num = port;
	if (use_lid) {
		a.ah_attr.is_global = 0;
		a.ah_attr.dlid = dlid;
		a.ah_attr.sl = 0;
		a.ah_attr.src_path_bits = 0;
	} else {
		a.ah_attr.is_global = 1;
		a.ah_attr.dlid = dlid;
		a.ah_attr.grh.hop_limit = 1;
		a.ah_attr.grh.sgid_index = sgid_index;
		memcpy(a.ah_attr.grh.dgid.raw, dgid, 16);
	}
	return ibv_modify_qp(qp, &a,
		IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
		IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

static int zl_modify_qp_rts(struct ibv_qp *qp, uint32_t sq_psn) {
	struct ibv_qp_attr a = {0};
	a.qp_state = IBV_QPS_RTS;
	a.timeout = 14;
	a.retry_cnt = 7;
	a.rnr_retry = 7;
	a.sq_psn = sq_psn;
	a.max_rd_atomic = 1;
	return ibv_modify_qp(qp, &a,
		IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
		IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

static int zl_errno(void) { return errno; }

static void *zl_malloc(size_t n) { return malloc(n); }
static void zl_free(void *p) { free(p); }
*/
import "C"

import (
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
	"unsafe"
)

// Endpoint is the TCP-exchanged QP addressing info.
type Endpoint struct {
	QPN      uint32 `json:"qpn"`
	PSN      uint32 `json:"psn"`
	LID      uint16 `json:"lid"`
	GID      string `json:"gid"`
	GIDIndex int    `json:"gid_index"`
	Port     uint8  `json:"port"`
	MTU      int    `json:"mtu"`
	Device   string `json:"device,omitempty"`
}

// Device is an opened HCA + protection domain shared by sessions.
type Device struct {
	ctx    *C.struct_ibv_context
	pd     *C.struct_ibv_pd
	name   string
	port   uint8
	lid    uint16
	gidIdx int
	gid    [16]byte
	mtu    C.enum_ibv_mtu
	linkIB bool
}

// OpenFirst opens a usable IB/RoCE device with an active port.
// Preference: ZEROLLAMA_RDMA_DEVICE (exact name) → InfiniBand → any RoCE/Ethernet.
// Why: mlx4_0 (RoCE on bond) often sorts first but peers on pure IB (mlx4_1) need matching link layer.
func OpenFirst() (*Device, error) {
	var n C.int
	list := C.ibv_get_device_list(&n)
	if list == nil || n == 0 {
		return nil, errors.New("verbs: no RDMA devices")
	}
	defer C.ibv_free_device_list(list)

	devs := unsafe.Slice(list, int(n))
	want := os.Getenv("ZEROLLAMA_RDMA_DEVICE")

	var last error
	var firstRoCE *Device
	for i := 0; i < int(n); i++ {
		d, err := openDevice(devs[i], 1)
		if err != nil {
			last = err
			continue
		}
		if want != "" {
			if d.name == want {
				return d, nil
			}
			d.Close()
			continue
		}
		if d.linkIB {
			if firstRoCE != nil {
				firstRoCE.Close()
			}
			return d, nil
		}
		if firstRoCE == nil {
			firstRoCE = d
		} else {
			d.Close()
		}
	}
	if want != "" {
		if last == nil {
			last = fmt.Errorf("verbs: device %q not found/usable", want)
		}
		return nil, last
	}
	if firstRoCE != nil {
		return firstRoCE, nil
	}
	if last == nil {
		last = errors.New("verbs: no active RDMA port")
	}
	return nil, last
}

func openDevice(dev *C.struct_ibv_device, port uint8) (*Device, error) {
	ctx := C.ibv_open_device(dev)
	if ctx == nil {
		return nil, errors.New("ibv_open_device failed")
	}
	var pattr C.struct_ibv_port_attr
	if C.zl_query_port(ctx, C.uint8_t(port), &pattr) != 0 {
		C.ibv_close_device(ctx)
		return nil, errors.New("ibv_query_port failed")
	}
	if pattr.state != C.IBV_PORT_ACTIVE {
		C.ibv_close_device(ctx)
		return nil, errors.New("port not active")
	}
	pd := C.ibv_alloc_pd(ctx)
	if pd == nil {
		C.ibv_close_device(ctx)
		return nil, errors.New("ibv_alloc_pd failed")
	}

	gidIdx := 0
	var gid C.union_ibv_gid
	if C.ibv_query_gid(ctx, C.uint8_t(port), C.int(gidIdx), &gid) != 0 {
		C.ibv_dealloc_pd(pd)
		C.ibv_close_device(ctx)
		return nil, errors.New("ibv_query_gid failed")
	}
	var raw [16]byte
	C.gid_to_bytes(&gid, (*C.uchar)(unsafe.Pointer(&raw[0])))
	if isZeroGID(raw[:]) {
		C.ibv_dealloc_pd(pd)
		C.ibv_close_device(ctx)
		return nil, errors.New("zero GID")
	}

	name := C.GoString(C.ibv_get_device_name(dev))
	d := &Device{
		ctx:    ctx,
		pd:     pd,
		name:   name,
		port:   port,
		lid:    uint16(pattr.lid),
		gidIdx: gidIdx,
		gid:    raw,
		mtu:    pattr.active_mtu,
		linkIB: pattr.link_layer == C.IBV_LINK_LAYER_INFINIBAND,
	}
	runtime.SetFinalizer(d, func(x *Device) { x.Close() })
	return d, nil
}

func isZeroGID(b []byte) bool {
	for _, c := range b {
		if c != 0 {
			return false
		}
	}
	return true
}

// Close releases the HCA.
func (d *Device) Close() {
	if d == nil {
		return
	}
	if d.pd != nil {
		C.ibv_dealloc_pd(d.pd)
		d.pd = nil
	}
	if d.ctx != nil {
		C.ibv_close_device(d.ctx)
		d.ctx = nil
	}
}

// LocalEndpoint returns this device's addressing for TCP exchange.
func (d *Device) LocalEndpoint(qpn, psn uint32) Endpoint {
	return Endpoint{
		QPN: qpn, PSN: psn, LID: d.lid, GID: FormatGID(d.gid[:]),
		GIDIndex: d.gidIdx, Port: d.port, MTU: int(d.mtu), Device: d.name,
	}
}

// ProbeCap returns capability fields for /v1/capability.
func (d *Device) ProbeCap() (device, gid string, gidIndex int, port int, lid uint16) {
	return d.name, FormatGID(d.gid[:]), d.gidIdx, int(d.port), d.lid
}

// FormatGID renders 16 bytes as fe80:0000:… style.
func FormatGID(b []byte) string {
	if len(b) != 16 {
		return hex.EncodeToString(b)
	}
	parts := make([]string, 8)
	for i := 0; i < 8; i++ {
		parts[i] = fmt.Sprintf("%04x", binary.BigEndian.Uint16(b[i*2:]))
	}
	return strings.Join(parts, ":")
}

// ParseGID accepts fe80:0000:… or 32 hex chars.
func ParseGID(s string) ([16]byte, error) {
	var out [16]byte
	s = strings.TrimSpace(s)
	hexOnly := strings.ReplaceAll(s, ":", "")
	if len(hexOnly) != 32 {
		return out, fmt.Errorf("bad gid %q", s)
	}
	b, err := hex.DecodeString(hexOnly)
	if err != nil || len(b) != 16 {
		return out, fmt.Errorf("bad gid: %w", err)
	}
	copy(out[:], b)
	return out, nil
}

// QP is a Reliable Connection queue pair.
type QP struct {
	dev *Device
	qp  *C.struct_ibv_qp
	cq  *C.struct_ibv_cq
	qpn uint32
	psn uint32
	mu  sync.Mutex
}

// CreateQP allocates a RC QP (RESET). Call Connect after peer exchange.
func (d *Device) CreateQP() (*QP, error) {
	cq := C.ibv_create_cq(d.ctx, 128, nil, nil, 0)
	if cq == nil {
		return nil, errors.New("ibv_create_cq failed")
	}
	var init C.struct_ibv_qp_init_attr
	init.send_cq = cq
	init.recv_cq = cq
	init.qp_type = C.IBV_QPT_RC
	init.cap.max_send_wr = 64
	init.cap.max_recv_wr = 4
	init.cap.max_send_sge = 1
	init.cap.max_recv_sge = 1

	qp := C.ibv_create_qp(d.pd, &init)
	if qp == nil {
		C.ibv_destroy_cq(cq)
		return nil, errors.New("ibv_create_qp failed")
	}
	qpn := uint32(qp.qp_num)
	psn := qpn & 0xffffff
	q := &QP{dev: d, qp: qp, cq: cq, qpn: qpn, psn: psn}
	runtime.SetFinalizer(q, func(x *QP) { x.Close() })
	return q, nil
}

// Endpoint returns local addressing for this QP.
func (q *QP) Endpoint() Endpoint { return q.dev.LocalEndpoint(q.qpn, q.psn) }

// Close destroys the QP.
func (q *QP) Close() {
	if q == nil {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.qp != nil {
		C.ibv_destroy_qp(q.qp)
		q.qp = nil
	}
	if q.cq != nil {
		C.ibv_destroy_cq(q.cq)
		q.cq = nil
	}
}

// Connect brings the QP to RTS against the remote endpoint.
func (q *QP) Connect(remote Endpoint) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.qp == nil {
		return errors.New("qp closed")
	}
	rgid, err := ParseGID(remote.GID)
	if err != nil {
		return err
	}
	if C.zl_modify_qp_init(q.qp, C.uint8_t(q.dev.port)) != 0 {
		return errors.New("qp INIT failed")
	}

	mtu := q.dev.mtu
	if remote.MTU > 0 && C.enum_ibv_mtu(remote.MTU) < mtu {
		mtu = C.enum_ibv_mtu(remote.MTU)
	}
	useLID := 0
	if q.dev.linkIB && remote.LID != 0 && q.dev.lid != 0 {
		useLID = 1
	}
	if C.zl_modify_qp_rtr(q.qp, C.uint8_t(q.dev.port), mtu,
		C.uint32_t(remote.QPN), C.uint32_t(remote.PSN),
		C.int(useLID), C.uint16_t(remote.LID), C.uint8_t(q.dev.gidIdx),
		(*C.uchar)(unsafe.Pointer(&rgid[0]))) != 0 {
		return fmt.Errorf("qp RTR failed (lid=%d remote_lid=%d use_lid=%d)", q.dev.lid, remote.LID, useLID)
	}
	if C.zl_modify_qp_rts(q.qp, C.uint32_t(q.psn)) != 0 {
		return errors.New("qp RTS failed")
	}
	return nil
}

// MR is a registered memory region.
type MR struct {
	mr     *C.struct_ibv_mr
	addr   uintptr
	len    int
	lkey   uint32
	rkey   uint32
	owned  unsafe.Pointer // C malloc to free on Close; nil if external
}

// AllocMR allocates C memory, optionally copies src, and registers it.
// Why C heap: Go slices can move; ibv_reg_mr requires stable physical pages.
func (d *Device) AllocMR(size int, src []byte, remoteRead bool) (*MR, error) {
	if size <= 0 {
		return nil, errors.New("bad size")
	}
	ptr := C.zl_malloc(C.size_t(size))
	if ptr == nil {
		return nil, errors.New("malloc failed")
	}
	if len(src) > 0 {
		n := len(src)
		if n > size {
			n = size
		}
		C.memcpy(ptr, unsafe.Pointer(&src[0]), C.size_t(n))
	} else {
		C.memset(ptr, 0, C.size_t(size))
	}
	flags := C.IBV_ACCESS_LOCAL_WRITE
	if remoteRead {
		flags |= C.IBV_ACCESS_REMOTE_READ
	}
	mr := C.ibv_reg_mr(d.pd, ptr, C.size_t(size), C.int(flags))
	if mr == nil {
		C.zl_free(ptr)
		return nil, fmt.Errorf("ibv_reg_mr failed errno=%d", int(C.zl_errno()))
	}
	return &MR{mr: mr, addr: uintptr(ptr), len: size, lkey: uint32(mr.lkey), rkey: uint32(mr.rkey), owned: ptr}, nil
}

// RegMR registers buf for local write (+ remote read when remoteRead).
// Prefer AllocMR for buffers that outlive a single syscall.
func (d *Device) RegMR(buf []byte, remoteRead bool) (*MR, error) {
	if len(buf) == 0 {
		return nil, errors.New("empty buffer")
	}
	return d.RegMRAddr(unsafe.Pointer(&buf[0]), len(buf), remoteRead)
}

// RegMRAddr registers an arbitrary address (e.g. mmap) for remote read.
func (d *Device) RegMRAddr(addr unsafe.Pointer, length int, remoteRead bool) (*MR, error) {
	if length <= 0 || addr == nil {
		return nil, errors.New("bad mr addr")
	}
	flags := C.IBV_ACCESS_LOCAL_WRITE
	if remoteRead {
		flags |= C.IBV_ACCESS_REMOTE_READ
	}
	mr := C.ibv_reg_mr(d.pd, addr, C.size_t(length), C.int(flags))
	if mr == nil {
		return nil, fmt.Errorf("ibv_reg_mr failed errno=%d", int(C.zl_errno()))
	}
	return &MR{mr: mr, addr: uintptr(addr), len: length, lkey: uint32(mr.lkey), rkey: uint32(mr.rkey)}, nil
}

func (m *MR) Close() {
	if m == nil {
		return
	}
	if m.mr != nil {
		C.ibv_dereg_mr(m.mr)
		m.mr = nil
	}
	if m.owned != nil {
		C.zl_free(m.owned)
		m.owned = nil
	}
}

// Bytes returns a Go view of owned C memory (valid until Close).
func (m *MR) Bytes() []byte {
	if m == nil || m.addr == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(m.addr)), m.len)
}

func (m *MR) Addr() uint64 { return uint64(m.addr) }
func (m *MR) RKey() uint32 { return m.rkey }
func (m *MR) LKey() uint32 { return m.lkey }
func (m *MR) Len() int     { return m.len }

// ReadRemote posts RDMA READ of remote [raddr,raddr+n) into local MR at localOff.
func (q *QP) ReadRemote(local *MR, localOff int, raddr uint64, rkey uint32, n int) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.qp == nil {
		return errors.New("qp closed")
	}
	if n <= 0 || localOff+n > local.len {
		return errors.New("bad read length")
	}
	if C.zl_post_rdma_read(q.qp,
		C.uint64_t(uint64(local.addr)+uint64(localOff)), C.uint32_t(n), C.uint32_t(local.lkey),
		C.uint64_t(raddr), C.uint32_t(rkey)) != 0 {
		return errors.New("ibv_post_send RDMA_READ failed")
	}
	return q.pollCQ(30 * time.Second)
}

func (q *QP) pollCQ(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var wc C.struct_ibv_wc
	for {
		n := C.ibv_poll_cq(q.cq, 1, &wc)
		if n < 0 {
			return errors.New("ibv_poll_cq error")
		}
		if n > 0 {
			if wc.status != C.IBV_WC_SUCCESS {
				return fmt.Errorf("wc status=%d", int(wc.status))
			}
			return nil
		}
		if time.Now().After(deadline) {
			return errors.New("cq poll timeout")
		}
		runtime.Gosched()
	}
}

// Available reports whether any RDMA device can be opened.
func Available() bool {
	d, err := OpenFirst()
	if err != nil {
		return false
	}
	d.Close()
	return true
}

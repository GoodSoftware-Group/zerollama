//go:build !rdma

package storaged

import (
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/server/remotestore"
)

type rdmaServer struct{}

func (s *Server) initRDMA() {}

func (s *Server) rdmaCapability() *remotestore.RDMACap {
	ents, err := os.ReadDir("/sys/class/infiniband")
	if err != nil {
		return nil
	}
	for _, dev := range ents {
		gidPath := fmt.Sprintf("/sys/class/infiniband/%s/ports/1/gids/0", dev.Name())
		b, err := os.ReadFile(gidPath)
		if err != nil {
			continue
		}
		gid := strings.TrimSpace(string(b))
		if gid == "" || gid == "0000:0000:0000:0000:0000:0000:0000:0000" {
			continue
		}
		var lid uint16
		if lb, err := os.ReadFile(fmt.Sprintf("/sys/class/infiniband/%s/ports/1/lid", dev.Name())); err == nil {
			var v uint64
			fmt.Sscanf(string(lb), "0x%x", &v)
			lid = uint16(v)
		}
		return &remotestore.RDMACap{Device: dev.Name(), GID: gid, Port: 1, LID: lid, Verbs: false}
	}
	return nil
}

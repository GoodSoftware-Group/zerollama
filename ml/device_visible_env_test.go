package ml

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestVisibleDeviceEnvTokenCUDAOrdinal(t *testing.T) {
	require.Equal(t, "0", visibleDeviceEnvToken(DeviceInfo{DeviceID: DeviceID{Library: "CUDA", ID: "CUDA0"}}))
	require.Equal(t, "1", visibleDeviceEnvToken(DeviceInfo{DeviceID: DeviceID{Library: "CUDA", ID: "CUDA1"}}))
	require.Equal(t, "GPU-uuid", visibleDeviceEnvToken(DeviceInfo{DeviceID: DeviceID{Library: "CUDA", ID: "GPU-uuid"}}))
}

func TestGetDevicesEnvMapsCUDANameToOrdinal(t *testing.T) {
	env := GetDevicesEnv([]DeviceInfo{{
		DeviceID:    DeviceID{Library: "CUDA", ID: "CUDA0"},
		Description: "NVIDIA GeForce RTX 5080",
	}})
	require.Equal(t, "0", env["CUDA_VISIBLE_DEVICES"])
}

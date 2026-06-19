package latents

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

const magic = "ZLAT"

// SaveBin writes float32 latent tensor data to a binary file.
func SaveBin(path string, shape []int32, data []float32) error {
	if len(shape) == 0 || len(shape) > 4 {
		return fmt.Errorf("invalid latent shape %v", shape)
	}
	expected := int32(1)
	for _, d := range shape {
		expected *= d
	}
	if int(expected) != len(data) {
		return fmt.Errorf("latent data length %d != shape product %d", len(data), expected)
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := io.WriteString(f, magic); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, int32(len(shape))); err != nil {
		return err
	}
	padded := make([]int32, 4)
	copy(padded, shape)
	if err := binary.Write(f, binary.LittleEndian, padded); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, int32(len(data))); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, data); err != nil {
		return err
	}
	return nil
}

// LoadBin reads a latent tensor written by SaveBin.
func LoadBin(path string) (*mlx.Array, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	hdr := make([]byte, 4)
	if _, err := io.ReadFull(f, hdr); err != nil {
		return nil, err
	}
	if string(hdr) != magic {
		return nil, fmt.Errorf("invalid latent file magic %q", hdr)
	}

	var ndim int32
	if err := binary.Read(f, binary.LittleEndian, &ndim); err != nil {
		return nil, err
	}
	if ndim <= 0 || ndim > 4 {
		return nil, fmt.Errorf("invalid latent ndim %d", ndim)
	}
	padded := make([]int32, 4)
	if err := binary.Read(f, binary.LittleEndian, &padded); err != nil {
		return nil, err
	}
	shape := padded[:ndim]

	var count int32
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	if count <= 0 {
		return nil, fmt.Errorf("empty latent data")
	}
	data := make([]float32, count)
	if err := binary.Read(f, binary.LittleEndian, &data); err != nil {
		return nil, err
	}

	arr := mlx.NewArray(data, shape)
	mlx.Untrack(arr)
	mlx.Eval(arr)
	return arr, nil
}

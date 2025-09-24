package kernels

import (
	"embed"
)

//go:embed *.ptx
var Kernels embed.FS

func GetKernel(name string) ([]byte, error) {
	if len(name) < 4 || name[len(name)-4:] != ".ptx" {
		name = name + ".ptx"
	}
	return Kernels.ReadFile(name)
}

// ListKernels returns all available kernel names
func ListKernels() ([]string, error) {
	entries, err := Kernels.ReadDir(".")
	if err != nil {
		return nil, err
	}

	var kernels []string
	for _, entry := range entries {
		if !entry.IsDir() && len(entry.Name()) > 4 && entry.Name()[len(entry.Name())-4:] == ".ptx" {
			kernels = append(kernels, entry.Name()[:len(entry.Name())-4])
		}
	}
	return kernels, nil
}

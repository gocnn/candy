package kernels

import "embed"

//go:embed *.ptx
var Kernels embed.FS

// GetKernel returns the content of a specific PTX kernel file
func GetKernel(name string) ([]byte, error) {
	return Kernels.ReadFile(name + ".ptx")
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
			// Remove .ptx extension
			kernels = append(kernels, entry.Name()[:len(entry.Name())-4])
		}
	}
	return kernels, nil
}

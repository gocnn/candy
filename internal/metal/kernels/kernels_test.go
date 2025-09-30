package kernels

import (
	"testing"
)

func TestKernelsEmbedded(t *testing.T) {
	kernels := AllKernels()
	if len(kernels) == 0 {
		t.Fatal("No kernels found")
	}

	t.Logf("Found kernels: %v", kernels)

	data, err := GetKernel(kernels[0])
	if err != nil {
		t.Errorf("GetKernel failed: %v", err)
	}

	t.Logf("Kernel %s size: %d bytes", kernels[0], len(data))
	t.Logf("Kernel %s preview: %s", kernels[0], string(data[:min(200, len(data))]))

	if len(data) == 0 {
		t.Error("Kernel data is empty")
	}
}

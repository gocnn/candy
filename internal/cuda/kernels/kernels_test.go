package kernels

import (
	"testing"
)

func TestKernelsEmbedded(t *testing.T) {
	kernels, err := ListKernels()
	if err != nil {
		t.Fatalf("ListKernels failed: %v", err)
	}

	if len(kernels) == 0 {
		t.Fatal("No kernels found")
	}

	t.Logf("Found kernels: %v", kernels)

	// Test reading first kernel
	data, err := GetKernel(kernels[0])
	if err != nil {
		t.Errorf("GetKernel failed: %v", err)
	}

	// t.Logf("Kernel %s size: %d bytes", kernels[0], len(data))
	// t.Logf("Kernel %s preview: %s", kernels[0], string(data[:min(200, len(data))]))

	if len(data) == 0 {
		t.Error("Kernel data is empty")
	}
}

func TestGetKernelWithSuffix(t *testing.T) {
	kernels, err := ListKernels()
	if err != nil {
		t.Fatalf("ListKernels failed: %v", err)
	}

	if len(kernels) == 0 {
		t.Skip("No kernels found for testing")
	}

	kernelName := kernels[0]

	// Test without .ptx suffix
	data1, err1 := GetKernel(kernelName)
	if err1 != nil {
		t.Errorf("GetKernel without suffix failed: %v", err1)
	}

	// Test with .ptx suffix
	data2, err2 := GetKernel(kernelName + ".ptx")
	if err2 != nil {
		t.Errorf("GetKernel with suffix failed: %v", err2)
	}

	// Both should return the same data
	if len(data1) != len(data2) {
		t.Errorf("Data length mismatch: %d vs %d", len(data1), len(data2))
	}

	t.Logf("Successfully tested kernel '%s' with and without .ptx suffix", kernelName)
}

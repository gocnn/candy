package kernels_test

import (
	"slices"
	"testing"

	"github.com/gocnn/spark/internal/cpu/kernels"
)

// Arithmetic tests

func TestBAddF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{5, 7, 9}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{15}},
		{4, []float32{-1, -2, 3, 4}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{0, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.BAddF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBAddF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{5, 7, 9}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{15}},
		{4, []float64{-1, -2, 3, 4}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{0, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.BAddF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBAddStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{5, 7, 9}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{4, 0, 5, 0, 6}, make([]float32, 3), []float32{5, 7, 9}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30, 40, 50, 60}, make([]float32, 6), []float32{11, 22, 33, 44, 55, 66}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 40, 20, 50, 30, 60}, make([]float32, 6), []float32{11, 22, 33, 44, 55, 66}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{11, 22, 33, 44, 55, 66, 77, 88}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{11, 22, 33, 44, 55, 66, 77, 88}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BAddStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBAddStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{5, 7, 9}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{4, 0, 5, 0, 6}, make([]float64, 3), []float64{5, 7, 9}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 20, 30, 40, 50, 60}, make([]float64, 6), []float64{11, 22, 33, 44, 55, 66}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 40, 20, 50, 30, 60}, make([]float64, 6), []float64{11, 22, 33, 44, 55, 66}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{11, 22, 33, 44, 55, 66, 77, 88}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{11, 22, 33, 44, 55, 66, 77, 88}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BAddStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBSubF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{-3, -3, -3}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{-5}},
		{4, []float32{-1, -2, 3, 4}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{-2, -4, 6, 8}},
	}

	for _, tt := range tests {
		kernels.BSubF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBSubF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{-3, -3, -3}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{-5}},
		{4, []float64{-1, -2, 3, 4}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{-2, -4, 6, 8}},
	}

	for _, tt := range tests {
		kernels.BSubF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBSubStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{-3, -3, -3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{4, 0, 5, 0, 6}, make([]float32, 3), []float32{-3, -3, -3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30, 40, 50, 60}, make([]float32, 6), []float32{-9, -18, -27, -36, -45, -54}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 40, 20, 50, 30, 60}, make([]float32, 6), []float32{-9, -18, -27, -36, -45, -54}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{-9, -18, -27, -36, -45, -54, -63, -72}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{-9, -18, -27, -36, -45, -54, -63, -72}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BSubStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBSubStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{-3, -3, -3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{4, 0, 5, 0, 6}, make([]float64, 3), []float64{-3, -3, -3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 20, 30, 40, 50, 60}, make([]float64, 6), []float64{-9, -18, -27, -36, -45, -54}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 40, 20, 50, 30, 60}, make([]float64, 6), []float64{-9, -18, -27, -36, -45, -54}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{-9, -18, -27, -36, -45, -54, -63, -72}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{-9, -18, -27, -36, -45, -54, -63, -72}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BSubStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{4, 10, 18}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{50}},
		{4, []float32{-1, -2, 3, 4}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{-1, -4, -9, -16}},
	}

	for _, tt := range tests {
		kernels.BMulF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{4, 10, 18}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{50}},
		{4, []float64{-1, -2, 3, 4}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{-1, -4, -9, -16}},
	}

	for _, tt := range tests {
		kernels.BMulF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{4, 10, 18}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{4, 0, 5, 0, 6}, make([]float32, 3), []float32{4, 10, 18}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30, 40, 50, 60}, make([]float32, 6), []float32{10, 40, 90, 160, 250, 360}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 40, 20, 50, 30, 60}, make([]float32, 6), []float32{10, 40, 90, 160, 250, 360}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{10, 40, 90, 160, 250, 360, 490, 640}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{10, 40, 90, 160, 250, 360, 490, 640}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMulStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{4, 10, 18}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{4, 0, 5, 0, 6}, make([]float64, 3), []float64{4, 10, 18}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 20, 30, 40, 50, 60}, make([]float64, 6), []float64{10, 40, 90, 160, 250, 360}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 40, 20, 50, 30, 60}, make([]float64, 6), []float64{10, 40, 90, 160, 250, 360}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{10, 40, 90, 160, 250, 360, 490, 640}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{10, 40, 90, 160, 250, 360, 490, 640}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMulStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{4, 10, 18}, []float32{4, 5, 6}, make([]float32, 3), []float32{1, 2, 3}},
		{0, nil, nil, nil, nil},
		{1, []float32{50}, []float32{10}, []float32{0}, []float32{5}},
		{4, []float32{-1, -4, -9, -16}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{-1, -2, 3, 4}},
	}

	for _, tt := range tests {
		kernels.BDivF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{4, 10, 18}, []float64{4, 5, 6}, make([]float64, 3), []float64{1, 2, 3}},
		{0, nil, nil, nil, nil},
		{1, []float64{50}, []float64{10}, []float64{0}, []float64{5}},
		{4, []float64{-1, -4, -9, -16}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{-1, -2, 3, 4}},
	}

	for _, tt := range tests {
		kernels.BDivF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{4, 10, 18}, []float32{4, 5, 6}, make([]float32, 3), []float32{1, 2, 3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{4, 10, 18}, []float32{4, 0, 5, 0, 6}, make([]float32, 3), []float32{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 10, 10, 10, 10, 10}, make([]float32, 6), []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 10, 10, 10, 10, 10}, make([]float32, 6), []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{10, 40, 90, 160, 250, 360, 490, 640}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{10, 90, 40, 160, 250, 490, 360, 640}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BDivStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{4, 10, 18}, []float64{4, 5, 6}, make([]float64, 3), []float64{1, 2, 3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{4, 10, 18}, []float64{4, 0, 5, 0, 6}, make([]float64, 3), []float64{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 10, 10, 10, 10, 10}, make([]float64, 6), []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 10, 10, 10, 10, 10}, make([]float64, 6), []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{10, 40, 90, 160, 250, 360, 490, 640}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{10, 90, 40, 160, 250, 490, 360, 640}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BDivStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{1, 5, 3}, []float32{4, 2, 6}, make([]float32, 3), []float32{4, 5, 6}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{10}},
		{4, []float32{-1, -2, 3, 4}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{1, 2, 3, 4}},
	}

	for _, tt := range tests {
		kernels.BMaximumF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{1, 5, 3}, []float64{4, 2, 6}, make([]float64, 3), []float64{4, 5, 6}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{10}},
		{4, []float64{-1, -2, 3, 4}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{1, 2, 3, 4}},
	}

	for _, tt := range tests {
		kernels.BMaximumF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 5, 3}, []float32{4, 2, 6}, make([]float32, 3), []float32{4, 5, 6}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 5, 3}, []float32{4, 0, 2, 0, 6}, make([]float32, 3), []float32{4, 5, 6}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30, 40, 50, 60}, make([]float32, 6), []float32{10, 20, 30, 40, 50, 60}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 40, 20, 50, 30, 60}, make([]float32, 6), []float32{10, 20, 30, 40, 50, 60}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{10, 20, 30, 40, 50, 60, 70, 80}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{10, 20, 30, 40, 50, 60, 70, 80}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMaximumStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 5, 3}, []float64{4, 2, 6}, make([]float64, 3), []float64{4, 5, 6}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 5, 3}, []float64{4, 0, 2, 0, 6}, make([]float64, 3), []float64{4, 5, 6}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 20, 30, 40, 50, 60}, make([]float64, 6), []float64{10, 20, 30, 40, 50, 60}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 40, 20, 50, 30, 60}, make([]float64, 6), []float64{10, 20, 30, 40, 50, 60}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{10, 20, 30, 40, 50, 60, 70, 80}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{10, 20, 30, 40, 50, 60, 70, 80}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMaximumStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{1, 5, 3}, []float32{4, 2, 6}, make([]float32, 3), []float32{1, 2, 3}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{5}},
		{4, []float32{-1, -2, 3, 4}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{-1, -2, -3, -4}},
	}

	for _, tt := range tests {
		kernels.BMinimumF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{1, 5, 3}, []float64{4, 2, 6}, make([]float64, 3), []float64{1, 2, 3}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{5}},
		{4, []float64{-1, -2, 3, 4}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{-1, -2, -3, -4}},
	}

	for _, tt := range tests {
		kernels.BMinimumF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 5, 3}, []float32{4, 2, 6}, make([]float32, 3), []float32{1, 2, 3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 5, 3}, []float32{4, 0, 2, 0, 6}, make([]float32, 3), []float32{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30, 40, 50, 60}, make([]float32, 6), []float32{1, 2, 3, 4, 5, 6}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 40, 20, 50, 30, 60}, make([]float32, 6), []float32{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMinimumStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 5, 3}, []float64{4, 2, 6}, make([]float64, 3), []float64{1, 2, 3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 5, 3}, []float64{4, 0, 2, 0, 6}, make([]float64, 3), []float64{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 20, 30, 40, 50, 60}, make([]float64, 6), []float64{1, 2, 3, 4, 5, 6}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 40, 20, 50, 30, 60}, make([]float64, 6), []float64{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMinimumStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// Comparison tests

// TestEqF32F32 tests y = (x1 == x2) ? 1 : 0 for float32 with float32 output
func TestEqF32F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []float32
		want   []float32
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{5}, []float32{0}, []float32{1}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]float32, 4), []float32{0, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.EqF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqU8F32 tests y = (x1 == x2) ? 1 : 0 for float32 with uint8 output
func TestEqU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{5}, []uint8{0}, []uint8{1}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.EqU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqF64F64 tests y = (x1 == x2) ? 1 : 0 for float64 with float64 output
func TestEqF64F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []float64
		want   []float64
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{5}, []float64{0}, []float64{1}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]float64, 4), []float64{0, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.EqF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqU8F64 tests y = (x1 == x2) ? 1 : 0 for float64 with uint8 output
func TestEqU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{5}, []uint8{0}, []uint8{1}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.EqU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqStridedF32F32 tests y = (x1 == x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestEqStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{0, 1, 0, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{0, 1, 0, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 0, 0, 1, 0, 0, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 0, 0, 1, 0, 0, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.EqStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqStridedU8F32 tests y = (x1 == x2) ? 1 : 0 for float32 with strided memory and uint8 output
func TestEqStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 0, 1, 0, 0, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 0, 1, 0, 0, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.EqStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqStridedF64F64 tests y = (x1 == x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestEqStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{0, 1, 0, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{0, 1, 0, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 0, 0, 1, 0, 0, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 0, 0, 1, 0, 0, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.EqStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqStridedU8F64 tests y = (x1 == x2) ? 1 : 0 for float64 with strided memory and uint8 output
func TestEqStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 0, 1, 0, 0, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 0, 1, 0, 0, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.EqStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeF32F32 tests y = (x1 != x2) ? 1 : 0 for float32 with float32 output
func TestNeF32F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []float32
		want   []float32
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{5}, []float32{0}, []float32{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]float32, 4), []float32{1, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.NeF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeU8F32 tests y = (x1 != x2) ? 1 : 0 for float32 with uint8 output
func TestNeU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{5}, []uint8{0}, []uint8{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.NeU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeF64F64 tests y = (x1 != x2) ? 1 : 0 for float64 with float64 output
func TestNeF64F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []float64
		want   []float64
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{5}, []float64{0}, []float64{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]float64, 4), []float64{1, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.NeF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeU8F64 tests y = (x1 != x2) ? 1 : 0 for float64 with uint8 output
func TestNeU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{5}, []uint8{0}, []uint8{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.NeU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeStridedF32F32 tests y = (x1 != x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestNeStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{1, 0, 1, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{1, 0, 1, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 1, 1, 0, 1, 1, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 1, 1, 0, 1, 1, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.NeStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeStridedU8F32 tests y = (x1 != x2) ? 1 : 0 for float32 with strided memory and uint8 output
func TestNeStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 1, 0, 1, 1, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 1, 0, 1, 1, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.NeStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeStridedF64F64 tests y = (x1 != x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestNeStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{1, 0, 1, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{1, 0, 1, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 1, 1, 0, 1, 1, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 1, 1, 0, 1, 1, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.NeStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeStridedU8F64 tests y = (x1 != x2) ? 1 : 0 for float64 with strided memory and uint8 output
func TestNeStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 1, 0, 1, 1, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 1, 0, 1, 1, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.NeStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtF32F32 tests y = (x1 < x2) ? 1 : 0 for float32 with float32 output
func TestLtF32F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []float32
		want   []float32
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{1}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]float32, 4), []float32{1, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.LtF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtU8F32 tests y = (x1 < x2) ? 1 : 0 for float32 with uint8 output
func TestLtU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []uint8{0}, []uint8{1}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.LtU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtF64F64 tests y = (x1 < x2) ? 1 : 0 for float64 with float64 output
func TestLtF64F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []float64
		want   []float64
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{1}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]float64, 4), []float64{1, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.LtF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtU8F64 tests y = (x1 < x2) ? 1 : 0 for float64 with uint8 output
func TestLtU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []uint8{0}, []uint8{1}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.LtU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtStridedF32F32 tests y = (x1 < x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestLtStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{1, 0, 1, 0, 0, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{1, 0, 1, 0, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 1, 0, 0, 1, 1, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 1, 0, 0, 1, 1, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LtStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtStridedU8F32 tests y = (x1 < x2) ? 1 : 0 for float32 with strided memory and uint8 output
func TestLtStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 0, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 0, 0, 1, 1, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 0, 0, 1, 1, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LtStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtStridedF64F64 tests y = (x1 < x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestLtStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{1, 0, 1, 0, 0, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{1, 0, 1, 0, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 1, 0, 0, 1, 1, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 1, 0, 0, 1, 1, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LtStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtStridedU8F64 tests y = (x1 < x2) ? 1 : 0 for float64 with strided memory and uint8 output
func TestLtStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 0, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 0, 0, 1, 1, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 0, 0, 1, 1, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LtStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeF32F32 tests y = (x1 <= x2) ? 1 : 0 for float32 with float32 output
func TestLeF32F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []float32
		want   []float32
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{1, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{1}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]float32, 4), []float32{1, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.LeF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeU8F32 tests y = (x1 <= x2) ? 1 : 0 for float32 with uint8 output
func TestLeU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []uint8{0}, []uint8{1}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.LeU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeF64F64 tests y = (x1 <= x2) ? 1 : 0 for float64 with float64 output
func TestLeF64F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []float64
		want   []float64
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{1, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{1}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]float64, 4), []float64{1, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.LeF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeU8F64 tests y = (x1 <= x2) ? 1 : 0 for float64 with uint8 output
func TestLeU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []uint8{0}, []uint8{1}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.LeU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeStridedF32F32 tests y = (x1 <= x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestLeStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{1, 1, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{1, 1, 1, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 1, 0, 1, 1, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 1, 0, 1, 1, 1, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LeStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeStridedU8F32 tests y = (x1 <= x2) ? 1 : 0 for float32 with strided memory and uint8 output
func TestLeStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{1, 1, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{1, 1, 1, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 1, 0, 1, 1, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 1, 0, 1, 1, 1, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LeStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeStridedF64F64 tests y = (x1 <= x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestLeStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{1, 1, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{1, 1, 1, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 1, 0, 1, 1, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 1, 0, 1, 1, 1, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LeStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeStridedU8F64 tests y = (x1 <= x2) ? 1 : 0 for float64 with strided memory and uint8 output
func TestLeStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{1, 1, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{1, 1, 1, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 1, 0, 1, 1, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 1, 0, 1, 1, 1, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LeStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtF32F32 tests y = (x1 > x2) ? 1 : 0 for float32 with float32 output
func TestGtF32F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []float32
		want   []float32
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]float32, 4), []float32{0, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.GtF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtU8F32 tests y = (x1 > x2) ? 1 : 0 for float32 with uint8 output
func TestGtU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []uint8{0}, []uint8{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.GtU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtF64F64 tests y = (x1 > x2) ? 1 : 0 for float64 with float64 output
func TestGtF64F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []float64
		want   []float64
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]float64, 4), []float64{0, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.GtF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtU8F64 tests y = (x1 > x2) ? 1 : 0 for float64 with uint8 output
func TestGtU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []uint8{0}, []uint8{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.GtU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtStridedF32F32 tests y = (x1 > x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestGtStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{0, 0, 0, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{0, 0, 0, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 0, 1, 0, 0, 0, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 0, 1, 0, 0, 0, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GtStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtStridedU8F32 tests y = (x1 > x2) ? 1 : 0 for float32 with strided memory and uint8 output
func TestGtStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 0, 0, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 0, 0, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 0, 1, 0, 0, 0, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 0, 1, 0, 0, 0, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GtStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtStridedF64F64 tests y = (x1 > x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestGtStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{0, 0, 0, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{0, 0, 0, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 0, 1, 0, 0, 0, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 0, 1, 0, 0, 0, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GtStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtStridedU8F64 tests y = (x1 > x2) ? 1 : 0 for float64 with strided memory and uint8 output
func TestGtStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 0, 0, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 0, 0, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 0, 1, 0, 0, 0, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 0, 1, 0, 0, 0, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GtStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeF32F32 tests y = (x1 >= x2) ? 1 : 0 for float32 with float32 output
func TestGeF32F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []float32
		want   []float32
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]float32, 4), []float32{0, 1, 1, 1}},
	}

	for _, tt := range tests {
		kernels.GeF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeU8F32 tests y = (x1 >= x2) ? 1 : 0 for float32 with uint8 output
func TestGeU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []uint8{0}, []uint8{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 1, 1, 1}},
	}

	for _, tt := range tests {
		kernels.GeU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeF64F64 tests y = (x1 >= x2) ? 1 : 0 for float64 with float64 output
func TestGeF64F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []float64
		want   []float64
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]float64, 4), []float64{0, 1, 1, 1}},
	}

	for _, tt := range tests {
		kernels.GeF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeU8F64 tests y = (x1 >= x2) ? 1 : 0 for float64 with uint8 output
func TestGeU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []uint8{0}, []uint8{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 1, 1, 1}},
	}

	for _, tt := range tests {
		kernels.GeU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeStridedF32F32 tests y = (x1 >= x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestGeStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{0, 1, 0, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{0, 1, 0, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 0, 1, 1, 0, 0, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 0, 1, 1, 0, 0, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GeStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeStridedU8F32 tests y = (x1 >= x2) ? 1 : 0 for float32 with strided memory and uint8 output
func TestGeStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 1, 1, 0, 0, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 1, 1, 0, 0, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GeStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeStridedF64F64 tests y = (x1 >= x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestGeStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{0, 1, 0, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{0, 1, 0, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 0, 1, 1, 0, 0, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 0, 1, 1, 0, 0, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GeStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeStridedU8F64 tests y = (x1 >= x2) ? 1 : 0 for float64 with strided memory and uint8 output
func TestGeStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 1, 1, 0, 0, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 1, 1, 0, 0, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GeStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

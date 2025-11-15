package kernels_test

import (
	"math"
	"slices"
	"testing"

	"github.com/gocnn/candy/tensor/internal/cpu/kernels"
)

func TestCastF32F32(t *testing.T) {
	tests := []struct {
		numel int
		x, y  []float32
		want  []float32
	}{
		{3, []float32{1.1, 2.2, 3.3}, make([]float32, 3), []float32{1.1, 2.2, 3.3}},
		{0, nil, nil, nil},
		{1, []float32{0}, []float32{0}, []float32{0}},
		{4, []float32{-1.5, 0, 2.5, 3}, make([]float32, 4), []float32{-1.5, 0, 2.5, 3}},
	}

	for _, tt := range tests {
		kernels.CastF32F32(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastF32F64(t *testing.T) {
	tests := []struct {
		numel int
		x     []float32
		y     []float64
		want  []float64
	}{
		{3, []float32{1.1, 2.2, 3.3}, make([]float64, 3), []float64{1.1, 2.2, 3.3}},
		{0, nil, nil, nil},
		{1, []float32{0}, []float64{0}, []float64{0}},
		{4, []float32{-1.5, 0, 2.5, 3}, make([]float64, 4), []float64{-1.5, 0, 2.5, 3}},
	}

	for _, tt := range tests {
		kernels.CastF32F64(tt.numel, tt.x, tt.y)
		if !slices.EqualFunc(tt.y, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastF32U8(t *testing.T) {
	tests := []struct {
		numel int
		x     []float32
		y     []uint8
		want  []uint8
	}{
		{3, []float32{1, 2, 3}, make([]uint8, 3), []uint8{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []float32{0}, []uint8{0}, []uint8{0}},
		{4, []float32{0, 1.5, 255, 256}, make([]uint8, 4), []uint8{0, 1, 255, 0}}, // Clamp/overflow behavior
	}

	for _, tt := range tests {
		kernels.CastF32U8(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastF32U32(t *testing.T) {
	tests := []struct {
		numel int
		x     []float32
		y     []uint32
		want  []uint32
	}{
		{3, []float32{1, 2, 3}, make([]uint32, 3), []uint32{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []float32{0}, []uint32{0}, []uint32{0}},
		{4, []float32{0, 1.5, 16777216, 4294967296}, make([]uint32, 4), []uint32{0, 1, 16777216, 0}}, // Clamp/overflow
	}

	for _, tt := range tests {
		kernels.CastF32U32(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastF32I64(t *testing.T) {
	tests := []struct {
		numel int
		x     []float32
		y     []int64
		want  []int64
	}{
		{3, []float32{1, 2, 3}, make([]int64, 3), []int64{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []float32{0}, []int64{0}, []int64{0}},
		{4, []float32{-1.5, 0, 2.5, -3}, make([]int64, 4), []int64{-1, 0, 2, -3}},
	}

	for _, tt := range tests {
		kernels.CastF32I64(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// Strided cast from float32

func TestCastStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x, y         []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []float32{1.1, 2.2, 3.3}, make([]float32, 3), []float32{1.1, 2.2, 3.3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []float32{1.1, 0, 2.2, 0, 3.3}, make([]float32, 3), []float32{1.1, 2.2, 3.3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, make([]float32, 6), []float32{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []float32{1, 4, 2, 5, 3, 6}, make([]float32, 6), []float32{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.CastStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedF32F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []float32
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []float32{1.1, 2.2, 3.3}, make([]float64, 3), []float64{1.1, 2.2, 3.3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []float32{1.1, 0, 2.2, 0, 3.3}, make([]float64, 3), []float64{1.1, 2.2, 3.3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, make([]float64, 6), []float64{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []float32{1, 4, 2, 5, 3, 6}, make([]float64, 6), []float64{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.CastStridedF32F64(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.EqualFunc(tt.y, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedF32U8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []float32{1, 2, 3}, make([]uint8, 3), []uint8{1, 2, 3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []float32{1, 0, 2, 0, 3}, make([]uint8, 3), []uint8{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, make([]uint8, 6), []uint8{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []float32{1, 4, 2, 5, 3, 6}, make([]uint8, 6), []uint8{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, make([]uint8, 8), []uint8{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, make([]uint8, 8), []uint8{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.CastStridedF32U8(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedF32U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []float32
		y            []uint32
		want         []uint32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []float32{1, 2, 3}, make([]uint32, 3), []uint32{1, 2, 3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []float32{1, 0, 2, 0, 3}, make([]uint32, 3), []uint32{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, make([]uint32, 6), []uint32{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []float32{1, 4, 2, 5, 3, 6}, make([]uint32, 6), []uint32{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, make([]uint32, 8), []uint32{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, make([]uint32, 8), []uint32{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.CastStridedF32U32(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedF32I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []float32
		y            []int64
		want         []int64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []float32{1, 2, 3}, make([]int64, 3), []int64{1, 2, 3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []float32{1, 0, 2, 0, 3}, make([]int64, 3), []int64{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, make([]int64, 6), []int64{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []float32{1, 4, 2, 5, 3, 6}, make([]int64, 6), []int64{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, make([]int64, 8), []int64{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, make([]int64, 8), []int64{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.CastStridedF32I64(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// Cast from float64

func TestCastF64F32(t *testing.T) {
	tests := []struct {
		numel int
		x     []float64
		y     []float32
		want  []float32
	}{
		{3, []float64{1.1, 2.2, 3.3}, make([]float32, 3), []float32{1.1, 2.2, 3.3}},
		{0, nil, nil, nil},
		{1, []float64{0}, []float32{0}, []float32{0}},
		{4, []float64{-1.5, 0, 2.5, 3}, make([]float32, 4), []float32{-1.5, 0, 2.5, 3}},
	}

	for _, tt := range tests {
		kernels.CastF64F32(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastF64F64(t *testing.T) {
	tests := []struct {
		numel int
		x, y  []float64
		want  []float64
	}{
		{3, []float64{1.1, 2.2, 3.3}, make([]float64, 3), []float64{1.1, 2.2, 3.3}},
		{0, nil, nil, nil},
		{1, []float64{0}, []float64{0}, []float64{0}},
		{4, []float64{-1.5, 0, 2.5, 3}, make([]float64, 4), []float64{-1.5, 0, 2.5, 3}},
	}

	for _, tt := range tests {
		kernels.CastF64F64(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastF64U8(t *testing.T) {
	tests := []struct {
		numel int
		x     []float64
		y     []uint8
		want  []uint8
	}{
		{3, []float64{1, 2, 3}, make([]uint8, 3), []uint8{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []float64{0}, []uint8{0}, []uint8{0}},
		{4, []float64{0, 1.5, 255, 256}, make([]uint8, 4), []uint8{0, 1, 255, 0}},
	}

	for _, tt := range tests {
		kernels.CastF64U8(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastF64U32(t *testing.T) {
	tests := []struct {
		numel int
		x     []float64
		y     []uint32
		want  []uint32
	}{
		{3, []float64{1, 2, 3}, make([]uint32, 3), []uint32{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []float64{0}, []uint32{0}, []uint32{0}},
		{4, []float64{0, 1.5, 4294967295, 4294967296}, make([]uint32, 4), []uint32{0, 1, 4294967295, 0}},
	}

	for _, tt := range tests {
		kernels.CastF64U32(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastF64I64(t *testing.T) {
	tests := []struct {
		numel int
		x     []float64
		y     []int64
		want  []int64
	}{
		{3, []float64{1, 2, 3}, make([]int64, 3), []int64{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []float64{0}, []int64{0}, []int64{0}},
		{4, []float64{-1.5, 0, 2.5, -3}, make([]int64, 4), []int64{-1, 0, 2, -3}},
	}

	for _, tt := range tests {
		kernels.CastF64I64(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// Strided cast from float64

func TestCastStridedF64F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []float64
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []float64{1.1, 2.2, 3.3}, make([]float32, 3), []float32{1.1, 2.2, 3.3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []float64{1.1, 0, 2.2, 0, 3.3}, make([]float32, 3), []float32{1.1, 2.2, 3.3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, make([]float32, 6), []float32{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []float64{1, 4, 2, 5, 3, 6}, make([]float32, 6), []float32{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.CastStridedF64F32(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x, y         []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []float64{1.1, 2.2, 3.3}, make([]float64, 3), []float64{1.1, 2.2, 3.3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []float64{1.1, 0, 2.2, 0, 3.3}, make([]float64, 3), []float64{1.1, 2.2, 3.3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, make([]float64, 6), []float64{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []float64{1, 4, 2, 5, 3, 6}, make([]float64, 6), []float64{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.CastStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedF64U8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []float64{1, 2, 3}, make([]uint8, 3), []uint8{1, 2, 3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []float64{1, 0, 2, 0, 3}, make([]uint8, 3), []uint8{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, make([]uint8, 6), []uint8{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []float64{1, 4, 2, 5, 3, 6}, make([]uint8, 6), []uint8{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, make([]uint8, 8), []uint8{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, make([]uint8, 8), []uint8{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.CastStridedF64U8(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedF64U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []float64
		y            []uint32
		want         []uint32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []float64{1, 2, 3}, make([]uint32, 3), []uint32{1, 2, 3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []float64{1, 0, 2, 0, 3}, make([]uint32, 3), []uint32{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, make([]uint32, 6), []uint32{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []float64{1, 4, 2, 5, 3, 6}, make([]uint32, 6), []uint32{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, make([]uint32, 8), []uint32{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, make([]uint32, 8), []uint32{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.CastStridedF64U32(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedF64I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []float64
		y            []int64
		want         []int64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []float64{1, 2, 3}, make([]int64, 3), []int64{1, 2, 3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []float64{1, 0, 2, 0, 3}, make([]int64, 3), []int64{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, make([]int64, 6), []int64{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []float64{1, 4, 2, 5, 3, 6}, make([]int64, 6), []int64{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, make([]int64, 8), []int64{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, make([]int64, 8), []int64{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.CastStridedF64I64(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastU8F32(t *testing.T) {
	tests := []struct {
		numel int
		x     []uint8
		y     []float32
		want  []float32
	}{
		{3, []uint8{106, 71, 188}, make([]float32, 3), []float32{106, 71, 188}},
		{0, nil, nil, nil},
		{1, []uint8{0}, []float32{0}, []float32{0}},
		{4, []uint8{102, 179, 92, 14}, make([]float32, 4), []float32{102, 179, 92, 14}},
	}

	for _, tt := range tests {
		kernels.CastU8F32(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastU8F64(t *testing.T) {
	tests := []struct {
		numel int
		x     []uint8
		y     []float64
		want  []float64
	}{
		{3, []uint8{106, 71, 188}, make([]float64, 3), []float64{106, 71, 188}},
		{0, nil, nil, nil},
		{1, []uint8{0}, []float64{0}, []float64{0}},
		{4, []uint8{102, 179, 92, 14}, make([]float64, 4), []float64{102, 179, 92, 14}},
	}

	for _, tt := range tests {
		kernels.CastU8F64(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastU8U8(t *testing.T) {
	tests := []struct {
		numel int
		x     []uint8
		y     []uint8
		want  []uint8
	}{
		{3, []uint8{106, 71, 188}, make([]uint8, 3), []uint8{106, 71, 188}},
		{0, nil, nil, nil},
		{1, []uint8{0}, []uint8{0}, []uint8{0}},
		{4, []uint8{102, 179, 92, 14}, make([]uint8, 4), []uint8{102, 179, 92, 14}},
	}

	for _, tt := range tests {
		kernels.CastU8U8(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastU8U32(t *testing.T) {
	tests := []struct {
		numel int
		x     []uint8
		y     []uint32
		want  []uint32
	}{
		{3, []uint8{106, 71, 188}, make([]uint32, 3), []uint32{106, 71, 188}},
		{0, nil, nil, nil},
		{1, []uint8{0}, []uint32{0}, []uint32{0}},
		{4, []uint8{102, 179, 92, 14}, make([]uint32, 4), []uint32{102, 179, 92, 14}},
	}

	for _, tt := range tests {
		kernels.CastU8U32(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastU8I64(t *testing.T) {
	tests := []struct {
		numel int
		x     []uint8
		y     []int64
		want  []int64
	}{
		{3, []uint8{106, 71, 188}, make([]int64, 3), []int64{106, 71, 188}},
		{0, nil, nil, nil},
		{1, []uint8{0}, []int64{0}, []int64{0}},
		{4, []uint8{102, 179, 92, 14}, make([]int64, 4), []int64{102, 179, 92, 14}},
	}

	for _, tt := range tests {
		kernels.CastU8I64(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// Strided cast from uint8
func TestCastStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims             int
		dims, stridesX, stridesY []int
		x                        []uint8
		y                        []float32
		want                     []float32
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []uint8{102, 179, 92}, make([]float32, 3), []float32{102.0, 179.0, 92.0}},
		{3, 1, []int{3}, []int{2}, []int{1}, []uint8{68, 0, 64, 0, 255}, make([]float32, 3), []float32{68.0, 64.0, 255.0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []uint8{20, 163, 241, 173, 59, 131}, make([]float32, 6), []float32{20.0, 163.0, 241.0, 173.0, 59.0, 131.0}},
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []uint8{203, 124, 158, 32, 131, 95}, make([]float32, 6), []float32{203.0, 158.0, 131.0, 124.0, 32.0, 95.0}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{189, 69, 40, 116, 186, 147, 146, 203}, make([]float32, 8), []float32{189.0, 69.0, 40.0, 116.0, 186.0, 147.0, 146.0, 203.0}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []uint8{135, 71, 134, 8, 72, 80, 179, 23}, make([]float32, 8), []float32{135.0, 134.0, 71.0, 8.0, 72.0, 179.0, 80.0, 23.0}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestCastStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims             int
		dims, stridesX, stridesY []int
		x                        []uint8
		y                        []float64
		want                     []float64
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []uint8{102, 179, 92}, make([]float64, 3), []float64{102.0, 179.0, 92.0}},
		{3, 1, []int{3}, []int{2}, []int{1}, []uint8{68, 0, 64, 0, 255}, make([]float64, 3), []float64{68.0, 64.0, 255.0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []uint8{20, 163, 241, 173, 59, 131}, make([]float64, 6), []float64{20.0, 163.0, 241.0, 173.0, 59.0, 131.0}},
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []uint8{203, 124, 158, 32, 131, 95}, make([]float64, 6), []float64{203.0, 158.0, 131.0, 124.0, 32.0, 95.0}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{189, 69, 40, 116, 186, 147, 146, 203}, make([]float64, 8), []float64{189.0, 69.0, 40.0, 116.0, 186.0, 147.0, 146.0, 203.0}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []uint8{135, 71, 134, 8, 72, 80, 179, 23}, make([]float64, 8), []float64{135.0, 134.0, 71.0, 8.0, 72.0, 179.0, 80.0, 23.0}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestCastStridedU8U8(t *testing.T) {
	tests := []struct {
		numel, ndims             int
		dims, stridesX, stridesY []int
		x, y                     []uint8
		want                     []uint8
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []uint8{102, 179, 92}, make([]uint8, 3), []uint8{102, 179, 92}},
		{3, 1, []int{3}, []int{2}, []int{1}, []uint8{68, 0, 64, 0, 255}, make([]uint8, 3), []uint8{68, 64, 255}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []uint8{20, 163, 241, 173, 59, 131}, make([]uint8, 6), []uint8{20, 163, 241, 173, 59, 131}},
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []uint8{203, 124, 158, 32, 131, 95}, make([]uint8, 6), []uint8{203, 158, 131, 124, 32, 95}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{189, 69, 40, 116, 186, 147, 146, 203}, make([]uint8, 8), []uint8{189, 69, 40, 116, 186, 147, 146, 203}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []uint8{135, 71, 134, 8, 72, 80, 179, 23}, make([]uint8, 8), []uint8{135, 134, 71, 8, 72, 179, 80, 23}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedU8U8(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestCastStridedU8U32(t *testing.T) {
	tests := []struct {
		numel, ndims             int
		dims, stridesX, stridesY []int
		x                        []uint8
		y                        []uint32
		want                     []uint32
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []uint8{102, 179, 92}, make([]uint32, 3), []uint32{102, 179, 92}},
		{3, 1, []int{3}, []int{2}, []int{1}, []uint8{68, 0, 64, 0, 255}, make([]uint32, 3), []uint32{68, 64, 255}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []uint8{20, 163, 241, 173, 59, 131}, make([]uint32, 6), []uint32{20, 163, 241, 173, 59, 131}},
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []uint8{203, 124, 158, 32, 131, 95}, make([]uint32, 6), []uint32{203, 158, 131, 124, 32, 95}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{189, 69, 40, 116, 186, 147, 146, 203}, make([]uint32, 8), []uint32{189, 69, 40, 116, 186, 147, 146, 203}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []uint8{135, 71, 134, 8, 72, 80, 179, 23}, make([]uint32, 8), []uint32{135, 134, 71, 8, 72, 179, 80, 23}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedU8U32(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestCastStridedU8I64(t *testing.T) {
	tests := []struct {
		numel, ndims             int
		dims, stridesX, stridesY []int
		x                        []uint8
		y                        []int64
		want                     []int64
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []uint8{102, 179, 92}, make([]int64, 3), []int64{102, 179, 92}},
		{3, 1, []int{3}, []int{2}, []int{1}, []uint8{68, 0, 64, 0, 255}, make([]int64, 3), []int64{68, 64, 255}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []uint8{20, 163, 241, 173, 59, 131}, make([]int64, 6), []int64{20, 163, 241, 173, 59, 131}},
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []uint8{203, 124, 158, 32, 131, 95}, make([]int64, 6), []int64{203, 158, 131, 124, 32, 95}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{189, 69, 40, 116, 186, 147, 146, 203}, make([]int64, 8), []int64{189, 69, 40, 116, 186, 147, 146, 203}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []uint8{135, 71, 134, 8, 72, 80, 179, 23}, make([]int64, 8), []int64{135, 134, 71, 8, 72, 179, 80, 23}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedU8I64(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastU32F32(t *testing.T) {
	tests := []struct {
		numel int
		x     []uint32
		y     []float32
		want  []float32
	}{
		{3, []uint32{1, 2, 3}, make([]float32, 3), []float32{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []uint32{0}, make([]float32, 1), []float32{0}},
		{5, []uint32{0, 1, 16777216, 16777217, 4294967295}, make([]float32, 5), []float32{0, 1, 16777216, 16777216, 4294967296}},
	}
	for _, tt := range tests {
		kernels.CastU32F32(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastU32F64(t *testing.T) {
	tests := []struct {
		numel int
		x     []uint32
		y     []float64
		want  []float64
	}{
		{3, []uint32{1, 2, 3}, make([]float64, 3), []float64{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []uint32{0}, make([]float64, 1), []float64{0}},
		{5, []uint32{0, 1, 16777216, 16777217, 4294967295}, make([]float64, 5), []float64{0, 1, 16777216, 16777217, 4294967295}},
	}
	for _, tt := range tests {
		kernels.CastU32F64(tt.numel, tt.x, tt.y)
		if !slices.EqualFunc(tt.y, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastU32U8(t *testing.T) {
	tests := []struct {
		numel int
		x     []uint32
		y     []uint8
		want  []uint8
	}{
		{3, []uint32{1, 2, 3}, make([]uint8, 3), []uint8{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []uint32{0}, make([]uint8, 1), []uint8{0}},
		{5, []uint32{0, 1, 255, 256, 257}, make([]uint8, 5), []uint8{0, 1, 255, 0, 1}},
	}
	for _, tt := range tests {
		kernels.CastU32U8(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastU32U32(t *testing.T) {
	tests := []struct {
		numel int
		x     []uint32
		y     []uint32
		want  []uint32
	}{
		{3, []uint32{1, 2, 3}, make([]uint32, 3), []uint32{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []uint32{0}, make([]uint32, 1), []uint32{0}},
		{5, []uint32{0, 1, 16777216, 16777217, 4294967295}, make([]uint32, 5), []uint32{0, 1, 16777216, 16777217, 4294967295}},
	}
	for _, tt := range tests {
		kernels.CastU32U32(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastU32I64(t *testing.T) {
	tests := []struct {
		numel int
		x     []uint32
		y     []int64
		want  []int64
	}{
		{3, []uint32{1, 2, 3}, make([]int64, 3), []int64{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []uint32{0}, make([]int64, 1), []int64{0}},
		{5, []uint32{0, 1, 16777216, 16777217, 4294967295}, make([]int64, 5), []int64{0, 1, 16777216, 16777217, 4294967295}},
	}
	for _, tt := range tests {
		kernels.CastU32I64(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// Cast from uint32
func TestCastStridedU32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []uint32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []uint32{1, 2, 3}, make([]float32, 3), []float32{1, 2, 3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []uint32{1, 0, 2, 0, 3}, make([]float32, 3), []float32{1, 2, 3}},
		// 1D strided y
		{3, 1, []int{3}, []int{1}, []int{2}, []uint32{1, 2, 3}, make([]float32, 5), []float32{1, 0, 2, 0, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []uint32{1, 2, 3, 4, 5, 6}, make([]float32, 6), []float32{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []uint32{1, 4, 2, 5, 3, 6}, make([]float32, 6), []float32{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1, 2, 3, 4, 5, 6, 7, 8}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []uint32{1, 3, 2, 4, 5, 7, 6, 8}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		// Special values for casting behavior
		{4, 1, []int{4}, []int{1}, []int{1}, []uint32{255, 256, 16777217, 4294967295}, make([]float32, 4), []float32{255, 256, 16777216, 4294967296}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedU32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedU32F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []uint32
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []uint32{1, 2, 3}, make([]float64, 3), []float64{1, 2, 3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []uint32{1, 0, 2, 0, 3}, make([]float64, 3), []float64{1, 2, 3}},
		// 1D strided y
		{3, 1, []int{3}, []int{1}, []int{2}, []uint32{1, 2, 3}, make([]float64, 5), []float64{1, 0, 2, 0, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []uint32{1, 2, 3, 4, 5, 6}, make([]float64, 6), []float64{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []uint32{1, 4, 2, 5, 3, 6}, make([]float64, 6), []float64{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1, 2, 3, 4, 5, 6, 7, 8}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []uint32{1, 3, 2, 4, 5, 7, 6, 8}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		// Special values for casting behavior
		{4, 1, []int{4}, []int{1}, []int{1}, []uint32{255, 256, 16777217, 4294967295}, make([]float64, 4), []float64{255, 256, 16777217, 4294967295}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedU32F64(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedU32U8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []uint32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []uint32{1, 2, 3}, make([]uint8, 3), []uint8{1, 2, 3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []uint32{1, 0, 2, 0, 3}, make([]uint8, 3), []uint8{1, 2, 3}},
		// 1D strided y
		{3, 1, []int{3}, []int{1}, []int{2}, []uint32{1, 2, 3}, make([]uint8, 5), []uint8{1, 0, 2, 0, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []uint32{1, 2, 3, 4, 5, 6}, make([]uint8, 6), []uint8{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []uint32{1, 4, 2, 5, 3, 6}, make([]uint8, 6), []uint8{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1, 2, 3, 4, 5, 6, 7, 8}, make([]uint8, 8), []uint8{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []uint32{1, 3, 2, 4, 5, 7, 6, 8}, make([]uint8, 8), []uint8{1, 2, 3, 4, 5, 6, 7, 8}},
		// Special values for casting behavior
		{4, 1, []int{4}, []int{1}, []int{1}, []uint32{255, 256, 16777217, 4294967295}, make([]uint8, 4), []uint8{255, 0, 1, 255}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedU32U8(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedU32U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []uint32
		y            []uint32
		want         []uint32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []uint32{1, 2, 3}, make([]uint32, 3), []uint32{1, 2, 3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []uint32{1, 0, 2, 0, 3}, make([]uint32, 3), []uint32{1, 2, 3}},
		// 1D strided y
		{3, 1, []int{3}, []int{1}, []int{2}, []uint32{1, 2, 3}, make([]uint32, 5), []uint32{1, 0, 2, 0, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []uint32{1, 2, 3, 4, 5, 6}, make([]uint32, 6), []uint32{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []uint32{1, 4, 2, 5, 3, 6}, make([]uint32, 6), []uint32{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1, 2, 3, 4, 5, 6, 7, 8}, make([]uint32, 8), []uint32{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []uint32{1, 3, 2, 4, 5, 7, 6, 8}, make([]uint32, 8), []uint32{1, 2, 3, 4, 5, 6, 7, 8}},
		// Special values for casting behavior (no change since same type)
		{4, 1, []int{4}, []int{1}, []int{1}, []uint32{255, 256, 16777217, 4294967295}, make([]uint32, 4), []uint32{255, 256, 16777217, 4294967295}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedU32U32(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastStridedU32I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX     []int
		stridesY     []int
		x            []uint32
		y            []int64
		want         []int64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []uint32{1, 2, 3}, make([]int64, 3), []int64{1, 2, 3}},
		// 1D strided x
		{3, 1, []int{3}, []int{2}, []int{1}, []uint32{1, 0, 2, 0, 3}, make([]int64, 3), []int64{1, 2, 3}},
		// 1D strided y
		{3, 1, []int{3}, []int{1}, []int{2}, []uint32{1, 2, 3}, make([]int64, 5), []int64{1, 0, 2, 0, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []uint32{1, 2, 3, 4, 5, 6}, make([]int64, 6), []int64{1, 2, 3, 4, 5, 6}},
		// 2D strided x (transposed)
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []uint32{1, 4, 2, 5, 3, 6}, make([]int64, 6), []int64{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1, 2, 3, 4, 5, 6, 7, 8}, make([]int64, 8), []int64{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D strided x (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []uint32{1, 3, 2, 4, 5, 7, 6, 8}, make([]int64, 8), []int64{1, 2, 3, 4, 5, 6, 7, 8}},
		// Special values for casting behavior
		{4, 1, []int{4}, []int{1}, []int{1}, []uint32{255, 256, 16777217, 4294967295}, make([]int64, 4), []int64{255, 256, 16777217, 4294967295}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedU32I64(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastI64F32(t *testing.T) {
	tests := []struct {
		numel int
		x     []int64
		y     []float32
		want  []float32
	}{
		{3, []int64{1, 2, 3}, make([]float32, 3), []float32{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []int64{0}, []float32{0}, []float32{0}},
		{4, []int64{-1, 0, 16777217, -16777217}, make([]float32, 4), []float32{-1, 0, 16777216, -16777216}},
	}

	for _, tt := range tests {
		kernels.CastI64F32(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastI64F64(t *testing.T) {
	tests := []struct {
		numel int
		x     []int64
		y     []float64
		want  []float64
	}{
		{3, []int64{1, 2, 3}, make([]float64, 3), []float64{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []int64{0}, []float64{0}, []float64{0}},
		{4, []int64{-1, 0, 9007199254740993, -9007199254740993}, make([]float64, 4), []float64{-1, 0, 9007199254740992, -9007199254740992}},
	}

	for _, tt := range tests {
		kernels.CastI64F64(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastI64U8(t *testing.T) {
	tests := []struct {
		numel int
		x     []int64
		y     []uint8
		want  []uint8
	}{
		{3, []int64{1, 2, 3}, make([]uint8, 3), []uint8{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []int64{0}, []uint8{0}, []uint8{0}},
		{4, []int64{0, -1, 255, 256}, make([]uint8, 4), []uint8{0, 255, 255, 0}}, // Wrap around behavior
	}

	for _, tt := range tests {
		kernels.CastI64U8(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastI64U32(t *testing.T) {
	tests := []struct {
		numel int
		x     []int64
		y     []uint32
		want  []uint32
	}{
		{3, []int64{1, 2, 3}, make([]uint32, 3), []uint32{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []int64{0}, []uint32{0}, []uint32{0}},
		{4, []int64{0, -1, 4294967295, 4294967296}, make([]uint32, 4), []uint32{0, 4294967295, 4294967295, 0}}, // Wrap around behavior
	}

	for _, tt := range tests {
		kernels.CastI64U32(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestCastI64I64(t *testing.T) {
	tests := []struct {
		numel int
		x     []int64
		y     []int64
		want  []int64
	}{
		{3, []int64{1, 2, 3}, make([]int64, 3), []int64{1, 2, 3}},
		{0, nil, nil, nil},
		{1, []int64{0}, []int64{0}, []int64{0}},
		{4, []int64{-1, 0, 2, -3}, make([]int64, 4), []int64{-1, 0, 2, -3}},
	}

	for _, tt := range tests {
		kernels.CastI64I64(tt.numel, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// Strided cast from int64
func TestCastStridedI64F32(t *testing.T) {
	tests := []struct {
		numel, ndims             int
		dims, stridesX, stridesY []int
		x                        []int64
		y                        []float32
		want                     []float32
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []int64{42, 167, -224}, make([]float32, 3), []float32{42.0, 167.0, -224.0}},
		{3, 1, []int{3}, []int{2}, []int{1}, []int64{-86, 0, 126, 0, 35}, make([]float32, 3), []float32{-86.0, 126.0, 35.0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int64{-80, 224, 250, -187, -122, -86}, make([]float32, 6), []float32{-80.0, 224.0, 250.0, -187.0, -122.0, -86.0}},
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []int64{110, -128, 54, 215, -69, -5}, make([]float32, 6), []float32{110.0, 54.0, -69.0, -128.0, 215.0, -5.0}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{267, -194, 49, 176, -227, 11, -1, -287}, make([]float32, 8), []float32{267.0, -194.0, 49.0, 176.0, -227.0, 11.0, -1.0, -287.0}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int64{41, 187, 69, -281, 72, 175, 180, 229}, make([]float32, 8), []float32{41.0, 69.0, 187.0, -281.0, 72.0, 180.0, 175.0, 229.0}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedI64F32(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestCastStridedI64F64(t *testing.T) {
	tests := []struct {
		numel, ndims             int
		dims, stridesX, stridesY []int
		x                        []int64
		y                        []float64
		want                     []float64
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []int64{233, 64, 239}, make([]float64, 3), []float64{233.0, 64.0, 239.0}},
		{3, 1, []int{3}, []int{2}, []int{1}, []int64{-124, 0, -68, 0, -90}, make([]float64, 3), []float64{-124.0, -68.0, -90.0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int64{86, -78, 277, 219, 207, 23}, make([]float64, 6), []float64{86.0, -78.0, 277.0, 219.0, 207.0, 23.0}},
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []int64{-157, -223, -206, -30, -207, 9}, make([]float64, 6), []float64{-157.0, -206.0, -207.0, -223.0, -30.0, 9.0}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{70, 39, -14, 199, 115, 284, 278, -192}, make([]float64, 8), []float64{70.0, 39.0, -14.0, 199.0, 115.0, 284.0, 278.0, -192.0}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int64{-34, -160, -270, -140, -30, -177, -139, 220}, make([]float64, 8), []float64{-34.0, -270.0, -160.0, -140.0, -30.0, -139.0, -177.0, 220.0}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedI64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.EqualFunc(tt.y, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestCastStridedI64U8(t *testing.T) {
	tests := []struct {
		numel, ndims             int
		dims, stridesX, stridesY []int
		x                        []int64
		y                        []uint8
		want                     []uint8
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []int64{-89, -239, 77}, make([]uint8, 3), []uint8{167, 17, 77}},
		{3, 1, []int{3}, []int{2}, []int{1}, []int64{189, 0, -116, 0, 253}, make([]uint8, 3), []uint8{189, 140, 253}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int64{-152, -91, 183, 7, -142, 191}, make([]uint8, 6), []uint8{104, 165, 183, 7, 114, 191}},
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []int64{114, 3, 47, 82, 192, 246}, make([]uint8, 6), []uint8{114, 47, 192, 3, 82, 246}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{289, 228, -145, 133, -173, 47, 165, -111}, make([]uint8, 8), []uint8{33, 228, 111, 133, 83, 47, 165, 145}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int64{241, 201, 61, 95, 161, 120, 26, 109}, make([]uint8, 8), []uint8{241, 61, 201, 95, 161, 26, 120, 109}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedI64U8(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestCastStridedI64U32(t *testing.T) {
	tests := []struct {
		numel, ndims             int
		dims, stridesX, stridesY []int
		x                        []int64
		y                        []uint32
		want                     []uint32
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []int64{237, -25, -83}, make([]uint32, 3), []uint32{237, 4294967271, 4294967213}},
		{3, 1, []int{3}, []int{2}, []int{1}, []int64{21, 0, 105, 0, 7}, make([]uint32, 3), []uint32{21, 105, 7}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int64{145, 138, 195, 234, 291, -279}, make([]uint32, 6), []uint32{145, 138, 195, 234, 291, 4294967017}},
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []int64{290, -71, 239, -189, 20, 168}, make([]uint32, 6), []uint32{290, 239, 20, 4294967225, 4294967107, 168}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-61, 136, -283, 146, -200, -21, 165, 92}, make([]uint32, 8), []uint32{4294967235, 136, 4294967013, 146, 4294967096, 4294967275, 165, 92}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int64{39, 247, -49, -232, 236, 106, -80, 208}, make([]uint32, 8), []uint32{39, 4294967247, 247, 4294967064, 236, 4294967216, 106, 208}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedI64U32(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestCastStridedI64I64(t *testing.T) {
	tests := []struct {
		numel, ndims             int
		dims, stridesX, stridesY []int
		x, y                     []int64
		want                     []int64
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []int64{-124, 148, 210}, make([]int64, 3), []int64{-124, 148, 210}},
		{3, 1, []int{3}, []int{2}, []int{1}, []int64{196, 0, -61, 0, -300}, make([]int64, 3), []int64{196, -61, -300}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int64{-145, -181, 235, -118, -130, -82}, make([]int64, 6), []int64{-145, -181, 235, -118, -130, -82}},
		{6, 2, []int{2, 3}, []int{1, 2}, []int{3, 1}, []int64{268, 182, -293, -204, 22, -2}, make([]int64, 6), []int64{268, -293, 22, 182, -204, -2}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-168, 277, 66, 253, -300, -17, 266, -220}, make([]int64, 8), []int64{-168, 277, 66, 253, -300, -17, 266, -220}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int64{163, -153, -107, -172, -61, -22, -198, -175}, make([]int64, 8), []int64{163, -107, -153, -172, -61, -198, -22, -175}},
		{0, 0, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.CastStridedI64I64(tt.numel, tt.ndims, tt.dims, tt.stridesX, tt.stridesY, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

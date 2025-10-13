package kernels_test

import (
	"math"
	"slices"
	"testing"

	"github.com/gocnn/spark/internal/cpu/kernels"
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

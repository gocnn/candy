package kernels_test

import (
	"slices"
	"testing"

	"github.com/gocnn/spark/internal/cpu/kernels"
)

func TestAffineF32(t *testing.T) {
	tests := []struct {
		numel int
		a, b  float32
		x, y  []float32
		want  []float32
	}{
		{3, 2, 1, []float32{1, 2, 3}, make([]float32, 3), []float32{3, 5, 7}},
		{0, 0, 0, nil, nil, nil},
		{1, 0, 0, []float32{5}, []float32{0}, []float32{0}},
		{2, 0.5, 1, []float32{1, 2}, make([]float32, 2), []float32{1.5, 2}},
		{4, -1, 0.5, []float32{0.5, 1.5, 2.5, 3.5}, make([]float32, 4), []float32{0, -1, -2, -3}},
	}

	for i, tt := range tests {
		kernels.AffineF32(tt.numel, tt.a, tt.b, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("test %d: got %v, want %v", i, tt.y, tt.want)
		}
	}
}

func TestAffineF64(t *testing.T) {
	tests := []struct {
		numel int
		a, b  float64
		x, y  []float64
		want  []float64
	}{
		{3, 2, 1, []float64{1, 2, 3}, make([]float64, 3), []float64{3, 5, 7}},
		{0, 0, 0, nil, nil, nil},
		{1, 0, 0, []float64{5}, []float64{0}, []float64{0}},
		{2, 0.5, 1, []float64{1, 2}, make([]float64, 2), []float64{1.5, 2}},
		{4, -1, 0.5, []float64{0.5, 1.5, 2.5, 3.5}, make([]float64, 4), []float64{0, -1, -2, -3}},
	}

	for i, tt := range tests {
		kernels.AffineF64(tt.numel, tt.a, tt.b, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("test %d: got %v, want %v", i, tt.y, tt.want)
		}
	}
}

func TestAffineU8(t *testing.T) {
	tests := []struct {
		numel int
		a, b  uint8
		x, y  []uint8
		want  []uint8
	}{
		{3, 2, 1, []uint8{1, 2, 3}, make([]uint8, 3), []uint8{3, 5, 7}},
		{0, 0, 0, nil, nil, nil},
		{1, 0, 0, []uint8{5}, []uint8{0}, []uint8{0}},
		{2, 2, 1, []uint8{1, 2}, make([]uint8, 2), []uint8{3, 5}},
		{4, 1, 2, []uint8{1, 2, 3, 4}, make([]uint8, 4), []uint8{3, 4, 5, 6}},
	}

	for i, tt := range tests {
		kernels.AffineU8(tt.numel, tt.a, tt.b, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("test %d: got %v, want %v", i, tt.y, tt.want)
		}
	}
}

func TestAffineU32(t *testing.T) {
	tests := []struct {
		numel int
		a, b  uint32
		x, y  []uint32
		want  []uint32
	}{
		{3, 2, 1, []uint32{1, 2, 3}, make([]uint32, 3), []uint32{3, 5, 7}},
		{0, 0, 0, nil, nil, nil},
		{1, 0, 0, []uint32{5}, []uint32{0}, []uint32{0}},
		{2, 2, 1, []uint32{1, 2}, make([]uint32, 2), []uint32{3, 5}},
		{4, 1, 2, []uint32{1, 2, 3, 4}, make([]uint32, 4), []uint32{3, 4, 5, 6}},
	}

	for i, tt := range tests {
		kernels.AffineU32(tt.numel, tt.a, tt.b, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("test %d: got %v, want %v", i, tt.y, tt.want)
		}
	}
}

func TestAffineI64(t *testing.T) {
	tests := []struct {
		numel int
		a, b  int64
		x, y  []int64
		want  []int64
	}{
		{3, 2, 1, []int64{1, 2, 3}, make([]int64, 3), []int64{3, 5, 7}},
		{0, 0, 0, nil, nil, nil},
		{1, 0, 0, []int64{5}, []int64{0}, []int64{0}},
		{2, 2, 1, []int64{1, 2}, make([]int64, 2), []int64{3, 5}},
		{4, -1, 2, []int64{1, 2, 3, 4}, make([]int64, 4), []int64{1, 0, -1, -2}},
	}

	for i, tt := range tests {
		kernels.AffineI64(tt.numel, tt.a, tt.b, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("test %d: got %v, want %v", i, tt.y, tt.want)
		}
	}
}

func TestAffineStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims  int
		dims, strides []int
		a, b          float32
		x, y          []float32
		want          []float32
	}{
		{3, 1, []int{3}, []int{1}, 2, 1, []float32{1, 2, 3}, make([]float32, 3), []float32{3, 5, 7}},
		{3, 1, []int{3}, []int{2}, 2, 1, []float32{1, 0, 2, 0, 3}, make([]float32, 3), []float32{3, 5, 7}},
		{6, 2, []int{2, 3}, []int{3, 1}, 2, 1, []float32{1, 2, 3, 4, 5, 6}, make([]float32, 6), []float32{3, 5, 7, 9, 11, 13}},
		{6, 2, []int{2, 3}, []int{1, 2}, 2, 1, []float32{1, 2, 3, 4, 5, 6}, make([]float32, 6), []float32{3, 7, 11, 5, 9, 13}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, 2, 1, []float32{1, 2, 3, 4, 5, 6, 7, 8}, make([]float32, 8), []float32{3, 5, 7, 9, 11, 13, 15, 17}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, 2, 1, []float32{1, 2, 3, 4, 5, 6, 7, 8}, make([]float32, 8), []float32{3, 7, 5, 9, 11, 15, 13, 17}},
		{0, 0, nil, nil, 0, 0, nil, nil, nil},
	}

	for i, tt := range tests {
		kernels.AffineStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.a, tt.b, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("test %d: got %v, want %v", i, tt.y, tt.want)
		}
	}
}

func TestAffineStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims  int
		dims, strides []int
		a, b          float64
		x, y          []float64
		want          []float64
	}{
		{3, 1, []int{3}, []int{1}, 2, 1, []float64{1, 2, 3}, make([]float64, 3), []float64{3, 5, 7}},
		{3, 1, []int{3}, []int{2}, 2, 1, []float64{1, 0, 2, 0, 3}, make([]float64, 3), []float64{3, 5, 7}},
		{6, 2, []int{2, 3}, []int{3, 1}, 2, 1, []float64{1, 2, 3, 4, 5, 6}, make([]float64, 6), []float64{3, 5, 7, 9, 11, 13}},
		{6, 2, []int{2, 3}, []int{1, 2}, 2, 1, []float64{1, 2, 3, 4, 5, 6}, make([]float64, 6), []float64{3, 7, 11, 5, 9, 13}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, 2, 1, []float64{1, 2, 3, 4, 5, 6, 7, 8}, make([]float64, 8), []float64{3, 5, 7, 9, 11, 13, 15, 17}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, 2, 1, []float64{1, 2, 3, 4, 5, 6, 7, 8}, make([]float64, 8), []float64{3, 7, 5, 9, 11, 15, 13, 17}},
		{0, 0, nil, nil, 0, 0, nil, nil, nil},
	}

	for i, tt := range tests {
		kernels.AffineStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.a, tt.b, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("test %d: got %v, want %v", i, tt.y, tt.want)
		}
	}
}

func TestAffineStridedU8(t *testing.T) {
	tests := []struct {
		numel, ndims  int
		dims, strides []int
		a, b          uint8
		x, y          []uint8
		want          []uint8
	}{
		{3, 1, []int{3}, []int{1}, 2, 1, []uint8{1, 2, 3}, make([]uint8, 3), []uint8{3, 5, 7}},
		{3, 1, []int{3}, []int{2}, 2, 1, []uint8{1, 0, 2, 0, 3}, make([]uint8, 3), []uint8{3, 5, 7}},
		{6, 2, []int{2, 3}, []int{3, 1}, 2, 1, []uint8{1, 2, 3, 4, 5, 6}, make([]uint8, 6), []uint8{3, 5, 7, 9, 11, 13}},
		{6, 2, []int{2, 3}, []int{1, 2}, 2, 1, []uint8{1, 2, 3, 4, 5, 6}, make([]uint8, 6), []uint8{3, 7, 11, 5, 9, 13}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, 2, 1, []uint8{1, 2, 3, 4, 5, 6, 7, 8}, make([]uint8, 8), []uint8{3, 5, 7, 9, 11, 13, 15, 17}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, 2, 1, []uint8{1, 2, 3, 4, 5, 6, 7, 8}, make([]uint8, 8), []uint8{3, 7, 5, 9, 11, 15, 13, 17}},
		{0, 0, nil, nil, 0, 0, nil, nil, nil},
	}

	for i, tt := range tests {
		kernels.AffineStridedU8(tt.numel, tt.ndims, tt.dims, tt.strides, tt.a, tt.b, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("test %d: got %v, want %v", i, tt.y, tt.want)
		}
	}
}

func TestAffineStridedU32(t *testing.T) {
	tests := []struct {
		numel, ndims  int
		dims, strides []int
		a, b          uint32
		x, y          []uint32
		want          []uint32
	}{
		{3, 1, []int{3}, []int{1}, 2, 1, []uint32{1, 2, 3}, make([]uint32, 3), []uint32{3, 5, 7}},
		{3, 1, []int{3}, []int{2}, 2, 1, []uint32{1, 0, 2, 0, 3}, make([]uint32, 3), []uint32{3, 5, 7}},
		{6, 2, []int{2, 3}, []int{3, 1}, 2, 1, []uint32{1, 2, 3, 4, 5, 6}, make([]uint32, 6), []uint32{3, 5, 7, 9, 11, 13}},
		{6, 2, []int{2, 3}, []int{1, 2}, 2, 1, []uint32{1, 2, 3, 4, 5, 6}, make([]uint32, 6), []uint32{3, 7, 11, 5, 9, 13}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, 2, 1, []uint32{1, 2, 3, 4, 5, 6, 7, 8}, make([]uint32, 8), []uint32{3, 5, 7, 9, 11, 13, 15, 17}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, 2, 1, []uint32{1, 2, 3, 4, 5, 6, 7, 8}, make([]uint32, 8), []uint32{3, 7, 5, 9, 11, 15, 13, 17}},
		{0, 0, nil, nil, 0, 0, nil, nil, nil},
	}

	for i, tt := range tests {
		kernels.AffineStridedU32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.a, tt.b, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("test %d: got %v, want %v", i, tt.y, tt.want)
		}
	}
}

func TestAffineStridedI64(t *testing.T) {
	tests := []struct {
		numel, ndims  int
		dims, strides []int
		a, b          int64
		x, y          []int64
		want          []int64
	}{
		{3, 1, []int{3}, []int{1}, 2, 1, []int64{1, 2, 3}, make([]int64, 3), []int64{3, 5, 7}},
		{3, 1, []int{3}, []int{2}, 2, 1, []int64{1, 0, 2, 0, 3}, make([]int64, 3), []int64{3, 5, 7}},
		{6, 2, []int{2, 3}, []int{3, 1}, 2, 1, []int64{1, 2, 3, 4, 5, 6}, make([]int64, 6), []int64{3, 5, 7, 9, 11, 13}},
		{6, 2, []int{2, 3}, []int{1, 2}, 2, 1, []int64{1, 2, 3, 4, 5, 6}, make([]int64, 6), []int64{3, 7, 11, 5, 9, 13}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, 2, 1, []int64{1, 2, 3, 4, 5, 6, 7, 8}, make([]int64, 8), []int64{3, 5, 7, 9, 11, 13, 15, 17}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, 2, 1, []int64{1, 2, 3, 4, 5, 6, 7, 8}, make([]int64, 8), []int64{3, 7, 5, 9, 11, 15, 13, 17}},
		{0, 0, nil, nil, 0, 0, nil, nil, nil},
	}

	for i, tt := range tests {
		kernels.AffineStridedI64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.a, tt.b, tt.x, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("test %d: got %v, want %v", i, tt.y, tt.want)
		}
	}
}

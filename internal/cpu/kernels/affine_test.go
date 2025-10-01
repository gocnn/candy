package kernels_test

import (
	"reflect"
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

	for _, tt := range tests {
		kernels.AffineF32(tt.numel, tt.a, tt.b, tt.x, tt.y)
		if !reflect.DeepEqual(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
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

	for _, tt := range tests {
		kernels.AffineF64(tt.numel, tt.a, tt.b, tt.x, tt.y)
		if !reflect.DeepEqual(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestAffineStridedF32(t *testing.T) {
	tests := []struct {
		numel, numDims int
		dims, strides  []int
		a, b           float32
		x, y           []float32
		want           []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, 2, 1, []float32{1, 2, 3}, make([]float32, 3), []float32{3, 5, 7}},
		// 1D strided
		{3, 1, []int{3}, []int{2}, 2, 1, []float32{1, 0, 2, 0, 3}, make([]float32, 3), []float32{3, 5, 7}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, 2, 1, []float32{1, 2, 3, 4, 5, 6}, make([]float32, 6), []float32{3, 5, 7, 9, 11, 13}},
		// 2D strided (transposed view)
		{6, 2, []int{2, 3}, []int{1, 2}, 2, 1, []float32{1, 2, 3, 4, 5, 6}, make([]float32, 6), []float32{3, 7, 11, 5, 9, 13}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, 2, 1, []float32{1, 2, 3, 4, 5, 6, 7, 8}, make([]float32, 8), []float32{3, 5, 7, 9, 11, 13, 15, 17}},
		// 3D strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, 2, 1, []float32{1, 2, 3, 4, 5, 6, 7, 8}, make([]float32, 8), []float32{3, 7, 5, 9, 11, 15, 13, 17}},
		{0, 0, nil, nil, 0, 0, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.AffineStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.a, tt.b, tt.x, tt.y)
		if !reflect.DeepEqual(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestAffineStridedF64(t *testing.T) {
	tests := []struct {
		numel, numDims int
		dims, strides  []int
		a, b           float64
		x, y           []float64
		want           []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, 2, 1, []float64{1, 2, 3}, make([]float64, 3), []float64{3, 5, 7}},
		// 1D strided
		{3, 1, []int{3}, []int{2}, 2, 1, []float64{1, 0, 2, 0, 3}, make([]float64, 3), []float64{3, 5, 7}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, 2, 1, []float64{1, 2, 3, 4, 5, 6}, make([]float64, 6), []float64{3, 5, 7, 9, 11, 13}},
		// 2D strided (transposed view)
		{6, 2, []int{2, 3}, []int{1, 2}, 2, 1, []float64{1, 2, 3, 4, 5, 6}, make([]float64, 6), []float64{3, 7, 11, 5, 9, 13}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, 2, 1, []float64{1, 2, 3, 4, 5, 6, 7, 8}, make([]float64, 8), []float64{3, 5, 7, 9, 11, 13, 15, 17}},
		// 3D strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, 2, 1, []float64{1, 2, 3, 4, 5, 6, 7, 8}, make([]float64, 8), []float64{3, 7, 5, 9, 11, 15, 13, 17}},
		{0, 0, nil, nil, 0, 0, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.AffineStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.a, tt.b, tt.x, tt.y)
		if !reflect.DeepEqual(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

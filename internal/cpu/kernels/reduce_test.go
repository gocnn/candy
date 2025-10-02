package kernels_test

import (
	"math"
	"slices"
	"testing"

	"github.com/gocnn/spark/internal/cpu/kernels"
)

func TestFastSumF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x3 contiguous",
			numel:   6,
			numDims: 2,
			dims:    []int{2, 3},
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{6, 15},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			src:     []float32{42},
			want:    []float32{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float32{-1, -2, 3, 4},
			want:    []float32{-3, 7},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastSumF32(tt.numel, tt.numDims, tt.dims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastSumF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x3 contiguous",
			numel:   6,
			numDims: 2,
			dims:    []int{2, 3},
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{6, 15},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			src:     []float64{42},
			want:    []float64{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float64{-1, -2, 3, 4},
			want:    []float64{-3, 7},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastSumF64(tt.numel, tt.numDims, tt.dims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastSumStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{1, 3, 2, 4},
			want:    []float32{3, 7},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float32{1, 2, 3, 4},
			want:    []float32{3, 7},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{42},
			want:    []float32{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{-1, 3, -2, 4},
			want:    []float32{-3, 7},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastSumStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastSumStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{1, 3, 2, 4},
			want:    []float64{3, 7},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float64{1, 2, 3, 4},
			want:    []float64{3, 7},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{42},
			want:    []float64{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{-1, 3, -2, 4},
			want:    []float64{-3, 7},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastSumStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastMinF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x3 contiguous",
			numel:   6,
			numDims: 2,
			dims:    []int{2, 3},
			src:     []float32{3, 1, 2, 6, 5, 4},
			want:    []float32{1, 4},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			src:     []float32{42},
			want:    []float32{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float32{-1, -2, 3, 4},
			want:    []float32{-2, 3},
		},
		{
			name:    "With Inf",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float32{float32(math.Inf(1)), -2, 3, float32(math.Inf(-1))},
			want:    []float32{-2, float32(math.Inf(-1))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastMinF32(tt.numel, tt.numDims, tt.dims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool {
				return math.Abs(float64(a-b)) < 1e-6 || (math.IsInf(float64(a), -1) && math.IsInf(float64(b), -1))
			}) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastMinF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x3 contiguous",
			numel:   6,
			numDims: 2,
			dims:    []int{2, 3},
			src:     []float64{3, 1, 2, 6, 5, 4},
			want:    []float64{1, 4},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			src:     []float64{42},
			want:    []float64{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float64{-1, -2, 3, 4},
			want:    []float64{-2, 3},
		},
		{
			name:    "With Inf",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float64{math.Inf(1), -2, 3, math.Inf(-1)},
			want:    []float64{-2, math.Inf(-1)},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastMinF64(tt.numel, tt.numDims, tt.dims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 || (math.IsInf(a, -1) && math.IsInf(b, -1)) }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastMinStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{3, 1, 2, 4},
			want:    []float32{2, 1},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float32{3, 1, 2, 4},
			want:    []float32{1, 2},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{42},
			want:    []float32{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{-1, 3, -2, 4},
			want:    []float32{-2, 3},
		},
		{
			name:    "With Inf",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{float32(math.Inf(1)), -2, 3, float32(math.Inf(-1))},
			want:    []float32{3, float32(math.Inf(-1))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastMinStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool {
				return math.Abs(float64(a-b)) < 1e-6 || (math.IsInf(float64(a), -1) && math.IsInf(float64(b), -1))
			}) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastMinStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{3, 1, 2, 4},
			want:    []float64{2, 1},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float64{3, 1, 2, 4},
			want:    []float64{1, 2},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{42},
			want:    []float64{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{-1, 3, -2, 4},
			want:    []float64{-2, 3},
		},
		{
			name:    "With Inf",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{math.Inf(1), -2, 3, math.Inf(-1)},
			want:    []float64{3, math.Inf(-1)},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastMinStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 || (math.IsInf(a, -1) && math.IsInf(b, -1)) }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastMaxF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x3 contiguous",
			numel:   6,
			numDims: 2,
			dims:    []int{2, 3},
			src:     []float32{3, 1, 2, 6, 5, 4},
			want:    []float32{3, 6},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			src:     []float32{42},
			want:    []float32{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float32{-1, -2, 3, 4},
			want:    []float32{-1, 4},
		},
		{
			name:    "With Inf",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float32{float32(math.Inf(-1)), 2, -3, float32(math.Inf(1))},
			want:    []float32{2, float32(math.Inf(1))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastMaxF32(tt.numel, tt.numDims, tt.dims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool {
				return math.Abs(float64(a-b)) < 1e-6 || (math.IsInf(float64(a), 1) && math.IsInf(float64(b), 1))
			}) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastMaxF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x3 contiguous",
			numel:   6,
			numDims: 2,
			dims:    []int{2, 3},
			src:     []float64{3, 1, 2, 6, 5, 4},
			want:    []float64{3, 6},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			src:     []float64{42},
			want:    []float64{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float64{-1, -2, 3, 4},
			want:    []float64{-1, 4},
		},
		{
			name:    "With Inf",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float64{math.Inf(-1), 2, -3, math.Inf(1)},
			want:    []float64{2, math.Inf(1)},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastMaxF64(tt.numel, tt.numDims, tt.dims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 || (math.IsInf(a, 1) && math.IsInf(b, 1)) }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastMaxStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{3, 1, 2, 4},
			want:    []float32{3, 4},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float32{3, 1, 2, 4},
			want:    []float32{3, 4},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{42},
			want:    []float32{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{-1, 3, -2, 4},
			want:    []float32{-1, 4},
		},
		{
			name:    "With Inf",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{float32(math.Inf(-1)), 2, -3, float32(math.Inf(1))},
			want:    []float32{-3, float32(math.Inf(1))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastMaxStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool {
				return math.Abs(float64(a-b)) < 1e-6 || (math.IsInf(float64(a), 1) && math.IsInf(float64(b), 1))
			}) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastMaxStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{3, 1, 2, 4},
			want:    []float64{3, 4},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float64{3, 1, 2, 4},
			want:    []float64{3, 4},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{42},
			want:    []float64{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{-1, 3, -2, 4},
			want:    []float64{-1, 4},
		},
		{
			name:    "With Inf",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{math.Inf(-1), 2, -3, math.Inf(1)},
			want:    []float64{-3, math.Inf(1)},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastMaxStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 || (math.IsInf(a, 1) && math.IsInf(b, 1)) }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgminF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		src     []float32
		want    []uint32
	}{
		{
			name:    "2x3 contiguous",
			numel:   6,
			numDims: 2,
			dims:    []int{2, 3},
			src:     []float32{3, 1, 2, 6, 5, 4},
			want:    []uint32{1, 2},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			src:     []float32{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float32{1, 1, 3, 2},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			src:     []float32{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgminF32(tt.numel, tt.numDims, tt.dims, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgminF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		src     []float64
		want    []uint32
	}{
		{
			name:    "2x3 contiguous",
			numel:   6,
			numDims: 2,
			dims:    []int{2, 3},
			src:     []float64{3, 1, 2, 6, 5, 4},
			want:    []uint32{1, 2},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			src:     []float64{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float64{1, 1, 3, 2},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			src:     []float64{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgminF64(tt.numel, tt.numDims, tt.dims, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgminStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		src     []float32
		want    []uint32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{3, 1, 2, 4},
			want:    []uint32{1, 0},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float32{3, 1, 2, 4},
			want:    []uint32{1, 0},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{1, 3, 1, 2},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgminStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgminStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		src     []float64
		want    []uint32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{3, 1, 2, 4},
			want:    []uint32{1, 0},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float64{3, 1, 2, 4},
			want:    []uint32{1, 0},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{1, 3, 1, 2},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgminStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgmaxF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		src     []float32
		want    []uint32
	}{
		{
			name:    "2x3 contiguous",
			numel:   6,
			numDims: 2,
			dims:    []int{2, 3},
			src:     []float32{3, 1, 2, 6, 5, 4},
			want:    []uint32{0, 0},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			src:     []float32{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float32{3, 3, 2, 4},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			src:     []float32{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgmaxF32(tt.numel, tt.numDims, tt.dims, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgmaxF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		src     []float64
		want    []uint32
	}{
		{
			name:    "2x3 contiguous",
			numel:   6,
			numDims: 2,
			dims:    []int{2, 3},
			src:     []float64{3, 1, 2, 6, 5, 4},
			want:    []uint32{0, 0},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			src:     []float64{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			src:     []float64{3, 3, 2, 4},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			src:     []float64{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgmaxF64(tt.numel, tt.numDims, tt.dims, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgmaxStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		src     []float32
		want    []uint32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{3, 1, 2, 4},
			want:    []uint32{0, 1},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float32{3, 1, 2, 4},
			want:    []uint32{0, 1},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{3, 1, 3, 2},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgmaxStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgmaxStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		src     []float64
		want    []uint32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{3, 1, 2, 4},
			want:    []uint32{0, 1},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float64{3, 1, 2, 4},
			want:    []uint32{0, 1},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{3, 1, 3, 2},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgmaxStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestSoftmaxF32(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float32
		want  []float32
	}{
		{
			name:  "2x2",
			ncols: 2,
			src:   []float32{1, 3, 2, 4},
			want:  []float32{0.11920292, 0.880797, 0.11920292, 0.880797},
		},
		{
			name:  "1x3",
			ncols: 3,
			src:   []float32{3, 1, 2},
			want:  []float32{0.66524096, 0.09003057, 0.24472847},
		},
		{
			name:  "With Inf",
			ncols: 2,
			src:   []float32{float32(math.Inf(1)), 0, 0, float32(math.Inf(-1))},
			want:  []float32{float32(math.NaN()), float32(math.NaN()), 1, 0},
		},
		{
			name:  "Empty",
			ncols: 1,
			src:   []float32{},
			want:  []float32{},
		},
		{
			name:  "Single row zero sum",
			ncols: 2,
			src:   []float32{0, 0},
			want:  []float32{0.5, 0.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.src))
			kernels.SoftmaxF32(tt.ncols, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool {
				if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
					return true
				}
				if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
					return false
				}
				return math.Abs(float64(a)-float64(b)) < 1e-5
			}) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestSoftmaxF64(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float64
		want  []float64
	}{
		{
			name:  "2x2",
			ncols: 2,
			src:   []float64{1, 3, 2, 4},
			want:  []float64{0.11920292, 0.880797, 0.11920292, 0.880797},
		},
		{
			name:  "1x3",
			ncols: 3,
			src:   []float64{3, 1, 2},
			want:  []float64{0.66524096, 0.09003057, 0.24472847},
		},
		{
			name:  "With Inf",
			ncols: 2,
			src:   []float64{math.Inf(1), 0, 0, math.Inf(-1)},
			want:  []float64{math.NaN(), math.NaN(), 1, 0},
		},
		{
			name:  "Empty",
			ncols: 1,
			src:   []float64{},
			want:  []float64{},
		},
		{
			name:  "Single row zero sum",
			ncols: 2,
			src:   []float64{0, 0},
			want:  []float64{0.5, 0.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.src))
			kernels.SoftmaxF64(tt.ncols, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool {
				if math.IsNaN(a) && math.IsNaN(b) {
					return true
				}
				if math.IsNaN(a) || math.IsNaN(b) {
					return false
				}
				return math.Abs(a-b) < 1e-5
			}) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestRopeF32(t *testing.T) {
	tests := []struct {
		name    string
		bh      int
		td      int
		d       int
		strideB int
		src     []float32
		cos     []float32
		sin     []float32
		want    []float32
	}{
		{
			name:    "bh=1 td=4 d=4 strideB=0",
			bh:      1,
			td:      4,
			d:       4,
			strideB: 0,
			src:     []float32{1, 2, 3, 4},
			cos:     []float32{1, 0, 1, 0},
			sin:     []float32{0, 1, 0, 1},
			want:    []float32{1, -4, 3, 2},
		},
		{
			name:    "bh=2 td=4 d=4 strideB=4",
			bh:      2,
			td:      4,
			d:       4,
			strideB: 4,
			src:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			cos:     []float32{1, 0, 0.5, 0.5},
			sin:     []float32{0, 1, 0.5, 0.5},
			want:    []float32{1, -4, 3, 2, -1, -1, 6, 7},
		},
		{
			name:    "Empty",
			bh:      0,
			td:      0,
			d:       0,
			strideB: 0,
			src:     []float32{},
			cos:     []float32{},
			sin:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.src))
			copy(dst, tt.src)
			kernels.RopeF32(tt.bh, tt.td, tt.d, tt.strideB, tt.src, tt.cos, tt.sin, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestRopeF64(t *testing.T) {
	tests := []struct {
		name    string
		bh      int
		td      int
		d       int
		strideB int
		src     []float64
		cos     []float64
		sin     []float64
		want    []float64
	}{
		{
			name:    "bh=1 td=4 d=4 strideB=0",
			bh:      1,
			td:      4,
			d:       4,
			strideB: 0,
			src:     []float64{1, 2, 3, 4},
			cos:     []float64{1, 0, 1, 0},
			sin:     []float64{0, 1, 0, 1},
			want:    []float64{1, -4, 3, 2},
		},
		{
			name:    "bh=2 td=4 d=4 strideB=4",
			bh:      2,
			td:      4,
			d:       4,
			strideB: 4,
			src:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
			cos:     []float64{1, 0, 0.5, 0.5},
			sin:     []float64{0, 1, 0.5, 0.5},
			want:    []float64{1, -4, 3, 2, -1, -1, 6, 7},
		},
		{
			name:    "Empty",
			bh:      0,
			td:      0,
			d:       0,
			strideB: 0,
			src:     []float64{},
			cos:     []float64{},
			sin:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.src))
			copy(dst, tt.src)
			kernels.RopeF64(tt.bh, tt.td, tt.d, tt.strideB, tt.src, tt.cos, tt.sin, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestRopeIF32(t *testing.T) {
	tests := []struct {
		name    string
		bh      int
		td      int
		strideB int
		src     []float32
		cos     []float32
		sin     []float32
		want    []float32
	}{
		{
			name:    "bh=1 td=4 strideB=0",
			bh:      1,
			td:      4,
			strideB: 0,
			src:     []float32{1, 3, 2, 4},
			cos:     []float32{1, 0},
			sin:     []float32{0, 1},
			want:    []float32{1, 3, -4, 2},
		},
		{
			name:    "bh=2 td=4 strideB=4",
			bh:      2,
			td:      4,
			strideB: 4,
			src:     []float32{1, 3, 2, 4, 5, 7, 6, 8},
			cos:     []float32{1, 0, 0.5, 0.5},
			sin:     []float32{0, 1, 0.5, 0.5},
			want:    []float32{1, 3, -4, 2, -1, 6, -1, 7},
		},
		{
			name:    "Empty",
			bh:      0,
			td:      0,
			strideB: 0,
			src:     []float32{},
			cos:     []float32{},
			sin:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.src))
			copy(dst, tt.src)
			kernels.RopeIF32(tt.bh, tt.td, tt.strideB, tt.src, tt.cos, tt.sin, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestRopeIF64(t *testing.T) {
	tests := []struct {
		name    string
		bh      int
		td      int
		strideB int
		src     []float64
		cos     []float64
		sin     []float64
		want    []float64
	}{
		{
			name:    "bh=1 td=4 strideB=0",
			bh:      1,
			td:      4,
			strideB: 0,
			src:     []float64{1, 3, 2, 4},
			cos:     []float64{1, 0},
			sin:     []float64{0, 1},
			want:    []float64{1, 3, -4, 2},
		},
		{
			name:    "bh=2 td=4 strideB=4",
			bh:      2,
			td:      4,
			strideB: 4,
			src:     []float64{1, 3, 2, 4, 5, 7, 6, 8},
			cos:     []float64{1, 0, 0.5, 0.5},
			sin:     []float64{0, 1, 0.5, 0.5},
			want:    []float64{1, 3, -4, 2, -1, 6, -1, 7},
		},
		{
			name:    "Empty",
			bh:      0,
			td:      0,
			strideB: 0,
			src:     []float64{},
			cos:     []float64{},
			sin:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.src))
			copy(dst, tt.src)
			kernels.RopeIF64(tt.bh, tt.td, tt.strideB, tt.src, tt.cos, tt.sin, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestRopeThdF32(t *testing.T) {
	tests := []struct {
		name    string
		b       int
		t       int
		h       int
		d       int
		strideB int
		src     []float32
		cos     []float32
		sin     []float32
		want    []float32
	}{
		{
			name:    "b=1 t=1 h=1 d=4 strideB=0",
			b:       1,
			t:       1,
			h:       1,
			d:       4,
			strideB: 0,
			src:     []float32{1, 2, 3, 4},
			cos:     []float32{1, 0},
			sin:     []float32{0, 1},
			want:    []float32{1, -4, 3, 2},
		},
		{
			name:    "b=1 t=2 h=1 d=4 strideB=0",
			b:       1,
			t:       2,
			h:       1,
			d:       4,
			strideB: 0,
			src:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			cos:     []float32{1, 0, 0.5, 0.5},
			sin:     []float32{0, 1, 0.5, 0.5},
			want:    []float32{1, -4, 3, 2, -1, -1, 6, 7},
		},
		{
			name:    "Empty",
			b:       0,
			t:       0,
			h:       0,
			d:       0,
			strideB: 0,
			src:     []float32{},
			cos:     []float32{},
			sin:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.src))
			copy(dst, tt.src)
			kernels.RopeThdF32(tt.b, tt.t, tt.h, tt.d, tt.strideB, tt.src, tt.cos, tt.sin, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestRopeThdF64(t *testing.T) {
	tests := []struct {
		name    string
		b       int
		t       int
		h       int
		d       int
		strideB int
		src     []float64
		cos     []float64
		sin     []float64
		want    []float64
	}{
		{
			name:    "b=1 t=1 h=1 d=4 strideB=0",
			b:       1,
			t:       1,
			h:       1,
			d:       4,
			strideB: 0,
			src:     []float64{1, 2, 3, 4},
			cos:     []float64{1, 0},
			sin:     []float64{0, 1},
			want:    []float64{1, -4, 3, 2},
		},
		{
			name:    "b=1 t=2 h=1 d=4 strideB=0",
			b:       1,
			t:       2,
			h:       1,
			d:       4,
			strideB: 0,
			src:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
			cos:     []float64{1, 0, 0.5, 0.5},
			sin:     []float64{0, 1, 0.5, 0.5},
			want:    []float64{1, -4, 3, 2, -1, -1, 6, 7},
		},
		{
			name:    "Empty",
			b:       0,
			t:       0,
			h:       0,
			d:       0,
			strideB: 0,
			src:     []float64{},
			cos:     []float64{},
			sin:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.src))
			copy(dst, tt.src)
			kernels.RopeThdF64(tt.b, tt.t, tt.h, tt.d, tt.strideB, tt.src, tt.cos, tt.sin, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

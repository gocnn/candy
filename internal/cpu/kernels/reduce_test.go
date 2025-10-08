package kernels_test

import (
	"math"
	"slices"
	"testing"

	"github.com/gocnn/spark/internal/cpu/kernels"
)

func TestFastSumF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		src   []float32
		want  []float32
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float32{1, 2, 3, 4, 5, 6},
			want:  []float32{6, 15},
		},
		{
			name:  "1x1 single element",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float32{42},
			want:  []float32{42},
		},
		{
			name:  "Negative values",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float32{-1, -2, 3, 4},
			want:  []float32{-3, 7},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 1,
			dims:  []int{1},
			src:   []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastSumF32(tt.numel, tt.ndims, tt.dims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastSumF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		src   []float64
		want  []float64
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float64{1, 2, 3, 4, 5, 6},
			want:  []float64{6, 15},
		},
		{
			name:  "1x1 single element",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float64{42},
			want:  []float64{42},
		},
		{
			name:  "Negative values",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float64{-1, -2, 3, 4},
			want:  []float64{-3, 7},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 1,
			dims:  []int{1},
			src:   []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastSumF64(tt.numel, tt.ndims, tt.dims, tt.src, dst)
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
		ndims   int
		dims    []int
		strides []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{1, 3, 2, 4},
			want:    []float32{3, 7},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float32{1, 2, 3, 4},
			want:    []float32{3, 7},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{42},
			want:    []float32{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{-1, 3, -2, 4},
			want:    []float32{-3, 7},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastSumStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
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
		ndims   int
		dims    []int
		strides []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{1, 3, 2, 4},
			want:    []float64{3, 7},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float64{1, 2, 3, 4},
			want:    []float64{3, 7},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{42},
			want:    []float64{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{-1, 3, -2, 4},
			want:    []float64{-3, 7},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastSumStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastMinF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		src   []float32
		want  []float32
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float32{3, 1, 2, 6, 5, 4},
			want:  []float32{1, 4},
		},
		{
			name:  "1x1 single element",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float32{42},
			want:  []float32{42},
		},
		{
			name:  "Negative values",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float32{-1, -2, 3, 4},
			want:  []float32{-2, 3},
		},
		{
			name:  "With Inf",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float32{float32(math.Inf(1)), -2, 3, float32(math.Inf(-1))},
			want:  []float32{-2, float32(math.Inf(-1))},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 1,
			dims:  []int{1},
			src:   []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastMinF32(tt.numel, tt.ndims, tt.dims, tt.src, dst)
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
		name  string
		numel int
		ndims int
		dims  []int
		src   []float64
		want  []float64
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float64{3, 1, 2, 6, 5, 4},
			want:  []float64{1, 4},
		},
		{
			name:  "1x1 single element",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float64{42},
			want:  []float64{42},
		},
		{
			name:  "Negative values",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float64{-1, -2, 3, 4},
			want:  []float64{-2, 3},
		},
		{
			name:  "With Inf",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float64{math.Inf(1), -2, 3, math.Inf(-1)},
			want:  []float64{-2, math.Inf(-1)},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 1,
			dims:  []int{1},
			src:   []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastMinF64(tt.numel, tt.ndims, tt.dims, tt.src, dst)
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
		ndims   int
		dims    []int
		strides []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{3, 1, 2, 4},
			want:    []float32{2, 1},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float32{3, 1, 2, 4},
			want:    []float32{1, 2},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{42},
			want:    []float32{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{-1, 3, -2, 4},
			want:    []float32{-2, 3},
		},
		{
			name:    "With Inf",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{float32(math.Inf(1)), -2, 3, float32(math.Inf(-1))},
			want:    []float32{3, float32(math.Inf(-1))},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastMinStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
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
		ndims   int
		dims    []int
		strides []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{3, 1, 2, 4},
			want:    []float64{2, 1},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float64{3, 1, 2, 4},
			want:    []float64{1, 2},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{42},
			want:    []float64{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{-1, 3, -2, 4},
			want:    []float64{-2, 3},
		},
		{
			name:    "With Inf",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{math.Inf(1), -2, 3, math.Inf(-1)},
			want:    []float64{3, math.Inf(-1)},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastMinStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 || (math.IsInf(a, -1) && math.IsInf(b, -1)) }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastMaxF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		src   []float32
		want  []float32
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float32{3, 1, 2, 6, 5, 4},
			want:  []float32{3, 6},
		},
		{
			name:  "1x1 single element",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float32{42},
			want:  []float32{42},
		},
		{
			name:  "Negative values",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float32{-1, -2, 3, 4},
			want:  []float32{-1, 4},
		},
		{
			name:  "With Inf",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float32{float32(math.Inf(-1)), 2, -3, float32(math.Inf(1))},
			want:  []float32{2, float32(math.Inf(1))},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 1,
			dims:  []int{1},
			src:   []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastMaxF32(tt.numel, tt.ndims, tt.dims, tt.src, dst)
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
		name  string
		numel int
		ndims int
		dims  []int
		src   []float64
		want  []float64
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float64{3, 1, 2, 6, 5, 4},
			want:  []float64{3, 6},
		},
		{
			name:  "1x1 single element",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float64{42},
			want:  []float64{42},
		},
		{
			name:  "Negative values",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float64{-1, -2, 3, 4},
			want:  []float64{-1, 4},
		},
		{
			name:  "With Inf",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float64{math.Inf(-1), 2, -3, math.Inf(1)},
			want:  []float64{2, math.Inf(1)},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 1,
			dims:  []int{1},
			src:   []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastMaxF64(tt.numel, tt.ndims, tt.dims, tt.src, dst)
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
		ndims   int
		dims    []int
		strides []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{3, 1, 2, 4},
			want:    []float32{3, 4},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float32{3, 1, 2, 4},
			want:    []float32{3, 4},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{42},
			want:    []float32{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{-1, 3, -2, 4},
			want:    []float32{-1, 4},
		},
		{
			name:    "With Inf",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{float32(math.Inf(-1)), 2, -3, float32(math.Inf(1))},
			want:    []float32{-3, float32(math.Inf(1))},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.FastMaxStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
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
		ndims   int
		dims    []int
		strides []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{3, 1, 2, 4},
			want:    []float64{3, 4},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float64{3, 1, 2, 4},
			want:    []float64{3, 4},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{42},
			want:    []float64{42},
		},
		{
			name:    "Negative values",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{-1, 3, -2, 4},
			want:    []float64{-1, 4},
		},
		{
			name:    "With Inf",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{math.Inf(-1), 2, -3, math.Inf(1)},
			want:    []float64{-3, math.Inf(1)},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.FastMaxStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 || (math.IsInf(a, 1) && math.IsInf(b, 1)) }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgminF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		src   []float32
		want  []uint32
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float32{3, 1, 2, 6, 5, 4},
			want:  []uint32{1, 2},
		},
		{
			name:  "1x1 single element",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float32{42},
			want:  []uint32{0},
		},
		{
			name:  "Ties, take first",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float32{1, 1, 3, 2},
			want:  []uint32{0, 1},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 1,
			dims:  []int{1},
			src:   []float32{},
			want:  []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgminF32(tt.numel, tt.ndims, tt.dims, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgminF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		src   []float64
		want  []uint32
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float64{3, 1, 2, 6, 5, 4},
			want:  []uint32{1, 2},
		},
		{
			name:  "1x1 single element",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float64{42},
			want:  []uint32{0},
		},
		{
			name:  "Ties, take first",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float64{1, 1, 3, 2},
			want:  []uint32{0, 1},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 1,
			dims:  []int{1},
			src:   []float64{},
			want:  []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgminF64(tt.numel, tt.ndims, tt.dims, tt.src, dst)
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
		ndims   int
		dims    []int
		strides []int
		src     []float32
		want    []uint32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{3, 1, 2, 4},
			want:    []uint32{1, 0},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float32{3, 1, 2, 4},
			want:    []uint32{1, 0},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{1, 3, 1, 2},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgminStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
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
		ndims   int
		dims    []int
		strides []int
		src     []float64
		want    []uint32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{3, 1, 2, 4},
			want:    []uint32{1, 0},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float64{3, 1, 2, 4},
			want:    []uint32{1, 0},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{1, 3, 1, 2},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgminStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgmaxF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		src   []float32
		want  []uint32
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float32{3, 1, 2, 6, 5, 4},
			want:  []uint32{0, 0},
		},
		{
			name:  "1x1 single element",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float32{42},
			want:  []uint32{0},
		},
		{
			name:  "Ties, take first",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float32{3, 3, 2, 4},
			want:  []uint32{0, 1},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 1,
			dims:  []int{1},
			src:   []float32{},
			want:  []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgmaxF32(tt.numel, tt.ndims, tt.dims, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastArgmaxF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		src   []float64
		want  []uint32
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float64{3, 1, 2, 6, 5, 4},
			want:  []uint32{0, 0},
		},
		{
			name:  "1x1 single element",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float64{42},
			want:  []uint32{0},
		},
		{
			name:  "Ties, take first",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float64{3, 3, 2, 4},
			want:  []uint32{0, 1},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 1,
			dims:  []int{1},
			src:   []float64{},
			want:  []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgmaxF64(tt.numel, tt.ndims, tt.dims, tt.src, dst)
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
		ndims   int
		dims    []int
		strides []int
		src     []float32
		want    []uint32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{3, 1, 2, 4},
			want:    []uint32{0, 1},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float32{3, 1, 2, 4},
			want:    []uint32{0, 1},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{3, 1, 3, 2},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgmaxStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
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
		ndims   int
		dims    []int
		strides []int
		src     []float64
		want    []uint32
	}{
		{
			name:    "2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{3, 1, 2, 4},
			want:    []uint32{0, 1},
		},
		{
			name:    "Contiguous fallback",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			src:     []float64{3, 1, 2, 4},
			want:    []uint32{0, 1},
		},
		{
			name:    "1x1 single element",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{42},
			want:    []uint32{0},
		},
		{
			name:    "Ties, take first",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{3, 1, 3, 2},
			want:    []uint32{0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{},
			want:    []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint32, len(tt.want))
			kernels.FastArgmaxStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestSumF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		sumDims []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x3 sum last",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			sumDims: []int{1},
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{6, 15},
		},
		{
			name:    "2x3 sum first",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			sumDims: []int{0},
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{5, 7, 9},
		},
		{
			name:    "2x3 sum all",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			sumDims: []int{0, 1},
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{21},
		},
		{
			name:    "1x1 single",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			sumDims: []int{0},
			src:     []float32{1},
			want:    []float32{1},
		},
		{
			name:    "negative 2x2 sum last",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			sumDims: []int{1},
			src:     []float32{-1, -2, 3, 4},
			want:    []float32{-3, 7},
		},
		{
			name:    "2x3x4 sum middle",
			numel:   24,
			ndims:   3,
			dims:    []int{2, 3, 4},
			sumDims: []int{1},
			src:     []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
			want:    []float32{15, 18, 21, 24, 51, 54, 57, 60},
		},
		{
			name:    "2x3x4 sum multiple",
			numel:   24,
			ndims:   3,
			dims:    []int{2, 3, 4},
			sumDims: []int{0, 2},
			src:     []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
			want:    []float32{68, 100, 132},
		},
		{
			name:    "Empty no reduction",
			numel:   0,
			ndims:   0,
			dims:    []int{},
			sumDims: []int{},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.SumF32(tt.numel, tt.ndims, tt.dims, tt.sumDims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestSumF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		sumDims []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x3 sum last",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			sumDims: []int{1},
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{6, 15},
		},
		{
			name:    "2x3 sum first",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			sumDims: []int{0},
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{5, 7, 9},
		},
		{
			name:    "2x3 sum all",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			sumDims: []int{0, 1},
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{21},
		},
		{
			name:    "1x1 single",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			sumDims: []int{0},
			src:     []float64{1},
			want:    []float64{1},
		},
		{
			name:    "negative 2x2 sum last",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			sumDims: []int{1},
			src:     []float64{-1, -2, 3, 4},
			want:    []float64{-3, 7},
		},
		{
			name:    "2x3x4 sum middle",
			numel:   24,
			ndims:   3,
			dims:    []int{2, 3, 4},
			sumDims: []int{1},
			src:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
			want:    []float64{15, 18, 21, 24, 51, 54, 57, 60},
		},
		{
			name:    "2x3x4 sum multiple",
			numel:   24,
			ndims:   3,
			dims:    []int{2, 3, 4},
			sumDims: []int{0, 2},
			src:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
			want:    []float64{68, 100, 132},
		},
		{
			name:    "Empty no reduction",
			numel:   0,
			ndims:   0,
			dims:    []int{},
			sumDims: []int{},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.SumF64(tt.numel, tt.ndims, tt.dims, tt.sumDims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestSumStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		sumDims []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x3 sum last contiguous fallback",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			strides: []int{3, 1},
			sumDims: []int{1},
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{6, 15},
		},
		{
			name:    "3x2 sum last strided",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			sumDims: []int{1},
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{5, 7, 9},
		},
		{
			name:    "3x2 sum first strided",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			sumDims: []int{0},
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{6, 15},
		},
		{
			name:    "3x2 sum all strided",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			sumDims: []int{0, 1},
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{21},
		},
		{
			name:    "negative 2x2 sum last strided",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			sumDims: []int{1},
			src:     []float32{-1, 3, -2, 4},
			want:    []float32{-3, 7},
		},
		{
			name:    "Empty no reduction",
			numel:   0,
			ndims:   0,
			dims:    []int{},
			strides: []int{},
			sumDims: []int{},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			kernels.SumStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.sumDims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestSumStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		sumDims []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x3 sum last contiguous fallback",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			strides: []int{3, 1},
			sumDims: []int{1},
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{6, 15},
		},
		{
			name:    "3x2 sum last strided",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			sumDims: []int{1},
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{5, 7, 9},
		},
		{
			name:    "3x2 sum first strided",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			sumDims: []int{0},
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{6, 15},
		},
		{
			name:    "3x2 sum all strided",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			sumDims: []int{0, 1},
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{21},
		},
		{
			name:    "negative 2x2 sum last strided",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			sumDims: []int{1},
			src:     []float64{-1, 3, -2, 4},
			want:    []float64{-3, 7},
		},
		{
			name:    "Empty no reduction",
			numel:   0,
			ndims:   0,
			dims:    []int{},
			strides: []int{},
			sumDims: []int{},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			kernels.SumStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.sumDims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastSoftmaxF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		src   []float32
		want  []float32
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float32{1, 2, 3, 4, 5, 6},
			want:  []float32{0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094},
		},
		{
			name:  "1x1 single",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float32{42},
			want:  []float32{1},
		},
		{
			name:  "Negative values 2x2",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float32{-1, -2, 3, 4},
			want:  []float32{0.7310586, 0.26894143, 0.26894143, 0.7310586},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 2,
			dims:  []int{0, 3},
			src:   []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.FastSoftmaxF32(tt.numel, tt.ndims, tt.dims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastSoftmaxF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		src   []float64
		want  []float64
	}{
		{
			name:  "2x3 contiguous",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			src:   []float64{1, 2, 3, 4, 5, 6},
			want:  []float64{0.09003057317038046, 0.24472847105479767, 0.6652409557748219, 0.09003057317038046, 0.24472847105479767, 0.6652409557748219},
		},
		{
			name:  "1x1 single",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			src:   []float64{42},
			want:  []float64{1},
		},
		{
			name:  "Negative values 2x2",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			src:   []float64{-1, -2, 3, 4},
			want:  []float64{0.7310585786300049, 0.2689414213699951, 0.2689414213699951, 0.7310585786300049},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 2,
			dims:  []int{0, 3},
			src:   []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.FastSoftmaxF64(tt.numel, tt.ndims, tt.dims, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastSoftmaxStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		src     []float32
		want    []float32
	}{
		{
			name:    "2x3 contiguous fallback",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			strides: []int{3, 1},
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094},
		},
		{
			name:    "3x2 strided (transposed)",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{0.047425874, 0.047425874, 0.047425874, 0.95257413, 0.95257413, 0.95257413},
		},
		{
			name:    "Negative 2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float32{-1, 3, -2, 4},
			want:    []float32{0.7310585786300049, 0.26894142136999516, 0.26894142136999516, 0.7310585786300049},
		},
		{
			name:    "1x1 single",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float32{42},
			want:    []float32{1},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   2,
			dims:    []int{0, 3},
			strides: []int{3, 1},
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.FastSoftmaxStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastSoftmaxStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		src     []float64
		want    []float64
	}{
		{
			name:    "2x3 contiguous fallback",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			strides: []int{3, 1},
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{0.09003057317038046, 0.24472847105479767, 0.6652409557748219, 0.09003057317038046, 0.24472847105479767, 0.6652409557748219},
		},
		{
			name:    "3x2 strided (transposed)",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{0.047425874, 0.047425874, 0.047425874, 0.95257413, 0.95257413, 0.95257413},
		},
		{
			name:    "Negative 2x2 strided (transposed)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			src:     []float64{-1, 3, -2, 4},
			want:    []float64{0.7310585786300049, 0.26894142136999516, 0.26894142136999516, 0.7310585786300049},
		},
		{
			name:    "1x1 single",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			src:     []float64{42},
			want:    []float64{1},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   2,
			dims:    []int{0, 3},
			strides: []int{3, 1},
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.FastSoftmaxStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastRmsNormF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		eps   float32
		alpha []float32
		src   []float32
		want  []float32
	}{
		{
			name:  "2x3 contiguous no alpha",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			eps:   1e-5,
			alpha: nil,
			src:   []float32{1, 2, 3, 4, 5, 6},
			want:  []float32{0.4629095494747162, 0.9258190989494324, 1.3887286186218262, 0.7895419001579285, 0.9869273900985718, 1.1843128204345703},
		},
		{
			name:  "2x3 contiguous with alpha",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			eps:   1e-5,
			alpha: []float32{0.5, 0.5, 0.5},
			src:   []float32{1, 2, 3, 4, 5, 6},
			want:  []float32{0.2314547747373581, 0.4629095494747162, 0.6943643093109131, 0.39477095007896423, 0.4934636950492859, 0.5921564102172852},
		},
		{
			name:  "1x1 single",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			eps:   1e-5,
			alpha: nil,
			src:   []float32{42},
			want:  []float32{1.0},
		},
		{
			name:  "Negative values 2x2",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			eps:   1e-5,
			alpha: nil,
			src:   []float32{-1, -2, 3, 4},
			want:  []float32{-0.6324542760848999, -1.2649085521697998, 0.8485277891159058, 1.1313704252243042},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.FastRmsNormF32(tt.numel, tt.ndims, tt.dims, tt.eps, tt.alpha, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastRmsNormF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		eps   float64
		alpha []float64
		src   []float64
		want  []float64
	}{
		{
			name:  "2x3 contiguous no alpha",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			eps:   1e-5,
			alpha: nil,
			src:   []float64{1, 2, 3, 4, 5, 6},
			want:  []float64{0.4629095494747162, 0.9258190989494324, 1.3887286186218262, 0.7895419001579285, 0.9869273900985718, 1.1843128204345703},
		},
		{
			name:  "2x3 contiguous with alpha",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			eps:   1e-5,
			alpha: []float64{0.5, 0.5, 0.5},
			src:   []float64{1, 2, 3, 4, 5, 6},
			want:  []float64{0.2314547747373581, 0.4629095494747162, 0.6943643093109131, 0.39477095007896423, 0.4934636950492859, 0.5921564102172852},
		},
		{
			name:  "1x1 single",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			eps:   1e-5,
			alpha: nil,
			src:   []float64{42},
			want:  []float64{1.0},
		},
		{
			name:  "Negative values 2x2",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			eps:   1e-5,
			alpha: nil,
			src:   []float64{-1, -2, 3, 4},
			want:  []float64{-0.6324542760848999, -1.2649085521697998, 0.8485277891159058, 1.1313704252243042},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.FastRmsNormF64(tt.numel, tt.ndims, tt.dims, tt.eps, tt.alpha, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastRmsNormStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		eps     float32
		alpha   []float32
		src     []float32
		want    []float32
	}{
		{
			name:    "2x3 contiguous fallback no alpha",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			strides: []int{3, 1},
			eps:     1e-5,
			alpha:   nil,
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{0.4629095494747162, 0.9258190989494324, 1.3887286186218262, 0.7895419001579285, 0.9869273900985718, 1.1843128204345703},
		},
		{
			name:    "3x2 strided (transposed) no alpha",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			eps:     1e-5,
			alpha:   nil,
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{0.3429969847202301, 0.5252255797386169, 0.6324554085731506, 1.3719879388809204, 1.3130638599395752, 1.2649108171463013},
		},
		{
			name:    "Negative 2x2 strided (transposed) no alpha",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			eps:     1e-5,
			alpha:   nil,
			src:     []float32{-1, -2, 3, 4},
			want:    []float32{-0.44721317291259766, -0.6324552297592163, 1.341639518737793, 1.2649104595184326},
		},
		{
			name:    "1x1 single",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			eps:     1e-5,
			alpha:   nil,
			src:     []float32{42},
			want:    []float32{1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.FastRmsNormStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.eps, tt.alpha, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastRmsNormStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		eps     float64
		alpha   []float64
		src     []float64
		want    []float64
	}{
		{
			name:    "2x3 contiguous fallback no alpha",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			strides: []int{3, 1},
			eps:     1e-5,
			alpha:   nil,
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{0.4629095494747162, 0.9258190989494324, 1.3887286186218262, 0.7895419001579285, 0.9869273900985718, 1.1843128204345703},
		},
		{
			name:    "3x2 strided (transposed) no alpha",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			eps:     1e-5,
			alpha:   nil,
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{0.3429969847202301, 0.5252255797386169, 0.6324554085731506, 1.3719879388809204, 1.3130638599395752, 1.2649108171463013},
		},
		{
			name:    "Negative 2x2 strided (transposed) no alpha",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			eps:     1e-5,
			alpha:   nil,
			src:     []float64{-1, -2, 3, 4},
			want:    []float64{-0.44721317291259766, -0.6324552297592163, 1.341639518737793, 1.2649104595184326},
		},
		{
			name:    "1x1 single",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			eps:     1e-5,
			alpha:   nil,
			src:     []float64{42},
			want:    []float64{1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.FastRmsNormStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.eps, tt.alpha, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastLayerNormF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		eps   float32
		alpha []float32
		beta  []float32
		src   []float32
		want  []float32
	}{
		{
			name:  "2x3 contiguous no alpha beta",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			eps:   1e-5,
			alpha: nil,
			beta:  nil,
			src:   []float32{1, 2, 3, 4, 5, 6},
			want:  []float32{-1.2247357, 0, 1.2247357, -1.2247357, 0, 1.2247357},
		},
		{
			name:  "2x3 contiguous with alpha beta",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			eps:   1e-5,
			alpha: []float32{0.5, 0.5, 0.5},
			beta:  []float32{1, 1, 1},
			src:   []float32{1, 2, 3, 4, 5, 6},
			want:  []float32{0.38763216, 1, 1.6123678, 0.38763216, 1, 1.6123678},
		},
		{
			name:  "1x1 single",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			eps:   1e-5,
			alpha: nil,
			beta:  nil,
			src:   []float32{42},
			want:  []float32{0},
		},
		{
			name:  "Negative values 2x2",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			eps:   1e-5,
			alpha: nil,
			beta:  nil,
			src:   []float32{-1, -2, 3, 4},
			want:  []float32{0.99998, -0.99998, -0.99998, 0.99998},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 2,
			dims:  []int{0, 3},
			eps:   1e-5,
			alpha: nil,
			beta:  nil,
			src:   []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.FastLayerNormF32(tt.numel, tt.ndims, tt.dims, tt.eps, tt.alpha, tt.beta, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastLayerNormF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dims  []int
		eps   float64
		alpha []float64
		beta  []float64
		src   []float64
		want  []float64
	}{
		{
			name:  "2x3 contiguous no alpha beta",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			eps:   1e-5,
			alpha: nil,
			beta:  nil,
			src:   []float64{1, 2, 3, 4, 5, 6},
			want:  []float64{-1.2247356859083902, 0.0, 1.2247356859083902, -1.2247356859083902, 0.0, 1.2247356859083902},
		},
		{
			name:  "2x3 contiguous with alpha beta",
			numel: 6,
			ndims: 2,
			dims:  []int{2, 3},
			eps:   1e-5,
			alpha: []float64{0.5, 0.5, 0.5},
			beta:  []float64{1, 1, 1},
			src:   []float64{1, 2, 3, 4, 5, 6},
			want:  []float64{0.3876321570458049, 1.0, 1.6123678429541952, 0.3876321570458049, 1.0, 1.6123678429541952},
		},
		{
			name:  "1x1 single",
			numel: 1,
			ndims: 1,
			dims:  []int{1},
			eps:   1e-5,
			alpha: nil,
			beta:  nil,
			src:   []float64{42},
			want:  []float64{0.0},
		},
		{
			name:  "Negative values 2x2",
			numel: 4,
			ndims: 2,
			dims:  []int{2, 2},
			eps:   1e-5,
			alpha: nil,
			beta:  nil,
			src:   []float64{-1, -2, 3, 4},
			want:  []float64{0.9999800005999799, -0.9999800005999799, -0.9999800005999799, 0.9999800005999799},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 2,
			dims:  []int{0, 3},
			eps:   1e-5,
			alpha: nil,
			beta:  nil,
			src:   []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.FastLayerNormF64(tt.numel, tt.ndims, tt.dims, tt.eps, tt.alpha, tt.beta, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastLayerNormStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		eps     float32
		alpha   []float32
		beta    []float32
		src     []float32
		want    []float32
	}{
		{
			name:    "2x3 contiguous fallback no alpha beta",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			strides: []int{3, 1},
			eps:     1e-5,
			alpha:   nil,
			beta:    nil,
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{-1.2247357, 0, 1.2247357, -1.2247357, 0, 1.2247357},
		},
		{
			name:    "3x2 strided (transposed) no alpha beta",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			eps:     1e-5,
			alpha:   nil,
			beta:    nil,
			src:     []float32{1, 2, 3, 4, 5, 6},
			want:    []float32{-0.9999978, -0.9999978, -0.9999978, 0.9999978, 0.9999978, 0.9999978},
		},
		{
			name:    "Negative 2x2 strided (transposed) no alpha beta",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			eps:     1e-5,
			alpha:   nil,
			beta:    nil,
			src:     []float32{-1, 3, -2, 4},
			want:    []float32{0.99998, -0.99998, -0.99998, 0.99998},
		},
		{
			name:    "1x1 single",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			eps:     1e-5,
			alpha:   nil,
			beta:    nil,
			src:     []float32{42},
			want:    []float32{0},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   2,
			dims:    []int{0, 3},
			strides: []int{3, 1},
			eps:     1e-5,
			alpha:   nil,
			beta:    nil,
			src:     []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.FastLayerNormStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.eps, tt.alpha, tt.beta, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFastLayerNormStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		eps     float64
		alpha   []float64
		beta    []float64
		src     []float64
		want    []float64
	}{
		{
			name:    "2x3 contiguous fallback no alpha beta",
			numel:   6,
			ndims:   2,
			dims:    []int{2, 3},
			strides: []int{3, 1},
			eps:     1e-5,
			alpha:   nil,
			beta:    nil,
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{-1.2247356859083902, 0.0, 1.2247356859083902, -1.2247356859083902, 0.0, 1.2247356859083902},
		},
		{
			name:    "3x2 strided (transposed) no alpha beta",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{1, 3},
			eps:     1e-5,
			alpha:   nil,
			beta:    nil,
			src:     []float64{1, 2, 3, 4, 5, 6},
			want:    []float64{-0.9999977777851852, -0.9999977777851852, -0.9999977777851852, 0.9999977777851852, 0.9999977777851852, 0.9999977777851852},
		},
		{
			name:    "Negative 2x2 strided (transposed) no alpha beta",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			eps:     1e-5,
			alpha:   nil,
			beta:    nil,
			src:     []float64{-1, 3, -2, 4},
			want:    []float64{0.9999800005999799, -0.9999800005999799, -0.9999800005999799, 0.9999800005999799},
		},
		{
			name:    "1x1 single",
			numel:   1,
			ndims:   1,
			dims:    []int{1},
			strides: []int{1},
			eps:     1e-5,
			alpha:   nil,
			beta:    nil,
			src:     []float64{42},
			want:    []float64{0.0},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   2,
			dims:    []int{0, 3},
			strides: []int{3, 1},
			eps:     1e-5,
			alpha:   nil,
			beta:    nil,
			src:     []float64{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.FastLayerNormStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.eps, tt.alpha, tt.beta, tt.src, dst)
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

func TestRopeIStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		ndims   int
		dims    []int
		strides []int
		bh      int
		td      int
		strideB int
		src     []float32
		cos     []float32
		sin     []float32
		want    []float32
	}{
		{
			name:    "Contiguous fallback bh=2 td=4 strideB=0",
			ndims:   2,
			dims:    []int{2, 4},
			strides: []int{4, 1},
			bh:      2,
			td:      4,
			strideB: 0,
			src:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			cos:     []float32{0.5, 0.6},
			sin:     []float32{0.7, 0.8},
			want:    []float32{-0.9, 1.7, -1.4, 4.8, -1.7, 6.5, -2.2, 10.4},
		},
		{
			name:    "Strided bh=2 td=4 strideB=0",
			ndims:   2,
			dims:    []int{4, 2},
			strides: []int{1, 4},
			bh:      2,
			td:      4,
			strideB: 0,
			src:     []float32{1, 3, 5, 7, 2, 4, 6, 8},
			cos:     []float32{0.5, 0.6},
			sin:     []float32{0.7, 0.8},
			want:    []float32{-0.9, -1.4, -1.7, -2.2, 1.7, 4.8, 6.5, 10.4},
		},
		{
			name:    "bh=1 td=2 strideB=0 negative",
			ndims:   2,
			dims:    []int{1, 2},
			strides: []int{2, 1},
			bh:      1,
			td:      2,
			strideB: 0,
			src:     []float32{-1, -2},
			cos:     []float32{0.9},
			sin:     []float32{0.1},
			want:    []float32{-0.7, -1.9},
		},
		{
			name:    "Empty bh=0 td=4 strideB=0",
			ndims:   2,
			dims:    []int{0, 4},
			strides: []int{4, 1},
			bh:      0,
			td:      4,
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
			kernels.RopeIStridedF32(tt.ndims, tt.dims, tt.strides, tt.bh, tt.td, tt.strideB, tt.src, tt.cos, tt.sin, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestRopeIStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		ndims   int
		dims    []int
		strides []int
		bh      int
		td      int
		strideB int
		src     []float64
		cos     []float64
		sin     []float64
		want    []float64
	}{
		{
			name:    "Contiguous fallback bh=2 td=4 strideB=0",
			ndims:   2,
			dims:    []int{2, 4},
			strides: []int{4, 1},
			bh:      2,
			td:      4,
			strideB: 0,
			src:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
			cos:     []float64{0.5, 0.6},
			sin:     []float64{0.7, 0.8},
			want:    []float64{-0.9, 1.7, -1.4, 4.8, -1.7, 6.5, -2.2, 10.4},
		},
		{
			name:    "Strided bh=2 td=4 strideB=0",
			ndims:   2,
			dims:    []int{4, 2},
			strides: []int{1, 4},
			bh:      2,
			td:      4,
			strideB: 0,
			src:     []float64{1, 3, 5, 7, 2, 4, 6, 8},
			cos:     []float64{0.5, 0.6},
			sin:     []float64{0.7, 0.8},
			want:    []float64{-0.9, -1.4, -1.7, -2.2, 1.7, 4.8, 6.5, 10.4},
		},
		{
			name:    "bh=1 td=2 strideB=0 negative",
			ndims:   2,
			dims:    []int{1, 2},
			strides: []int{2, 1},
			bh:      1,
			td:      2,
			strideB: 0,
			src:     []float64{-1, -2},
			cos:     []float64{0.9},
			sin:     []float64{0.1},
			want:    []float64{-0.7, -1.9},
		},
		{
			name:    "Empty bh=0 td=4 strideB=0",
			ndims:   2,
			dims:    []int{0, 4},
			strides: []int{4, 1},
			bh:      0,
			td:      4,
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
			kernels.RopeIStridedF64(tt.ndims, tt.dims, tt.strides, tt.bh, tt.td, tt.strideB, tt.src, tt.cos, tt.sin, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
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

func TestRopeStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		ndims   int
		dims    []int
		strides []int
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
			name:    "Contiguous fallback bh=2 td=4 d=2 strideB=0",
			ndims:   2,
			dims:    []int{2, 4},
			strides: []int{4, 1},
			bh:      2,
			td:      4,
			d:       2,
			strideB: 0,
			src:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			cos:     []float32{0.5, 0.6},
			sin:     []float32{0.7, 0.8},
			want:    []float32{-0.9, 1.7, -1.4, 4.8, -1.7, 6.5, -2.2, 10.4},
		},
		{
			name:    "Strided bh=2 td=4 d=2 strideB=0",
			ndims:   2,
			dims:    []int{4, 2},
			strides: []int{1, 4},
			bh:      2,
			td:      4,
			d:       2,
			strideB: 0,
			src:     []float32{1, 5, 2, 6, 3, 7, 4, 8},
			cos:     []float32{0.5, 0.6},
			sin:     []float32{0.7, 0.8},
			want:    []float32{-1.6, -2.6, -1.8, -2.8, 2.2, 8.2, 3.4, 9.6},
		},
		{
			name:    "Empty bh=0 td=4 d=2 strideB=0",
			ndims:   2,
			dims:    []int{0, 4},
			strides: []int{4, 1},
			bh:      0,
			td:      4,
			d:       2,
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
			kernels.RopeStridedF32(tt.ndims, tt.dims, tt.strides, tt.bh, tt.td, tt.d, tt.strideB, tt.src, tt.cos, tt.sin, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestRopeStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		ndims   int
		dims    []int
		strides []int
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
			name:    "Contiguous fallback bh=2 td=4 d=2 strideB=0",
			ndims:   2,
			dims:    []int{2, 4},
			strides: []int{4, 1},
			bh:      2,
			td:      4,
			d:       2,
			strideB: 0,
			src:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
			cos:     []float64{0.5, 0.6},
			sin:     []float64{0.7, 0.8},
			want:    []float64{-0.9, 1.7, -1.4, 4.8, -1.7, 6.5, -2.2, 10.4},
		},
		{
			name:    "Strided bh=2 td=4 d=2 strideB=0",
			ndims:   2,
			dims:    []int{4, 2},
			strides: []int{1, 4},
			bh:      2,
			td:      4,
			d:       2,
			strideB: 0,
			src:     []float64{1, 5, 2, 6, 3, 7, 4, 8},
			cos:     []float64{0.5, 0.6},
			sin:     []float64{0.7, 0.8},
			want:    []float64{-1.6, -2.6, -1.8, -2.8, 2.2, 8.2, 3.4, 9.6},
		},
		{
			name:    "Empty bh=0 td=4 d=2 strideB=0",
			ndims:   2,
			dims:    []int{0, 4},
			strides: []int{4, 1},
			bh:      0,
			td:      4,
			d:       2,
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
			kernels.RopeStridedF64(tt.ndims, tt.dims, tt.strides, tt.bh, tt.td, tt.d, tt.strideB, tt.src, tt.cos, tt.sin, dst)
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

func TestRopeThdStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		ndims   int
		dims    []int
		strides []int
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
			name:    "Contiguous fallback b=1 t=2 h=2 d=2 strideB=0",
			ndims:   4,
			dims:    []int{1, 2, 2, 2},
			strides: []int{8, 4, 2, 1},
			b:       1,
			t:       2,
			h:       2,
			d:       2,
			strideB: 0,
			src:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			cos:     []float32{0.5, 0.6},
			sin:     []float32{0.7, 0.8},
			want:    []float32{-0.9, 1.7, -1.3, 4.1, -1.8, 7.6, -2.2, 10.4},
		},
		{
			name:    "Strided b=1 t=2 h=2 d=2 strideB=0",
			ndims:   4,
			dims:    []int{2, 2, 2, 1},
			strides: []int{4, 2, 1, 8},
			b:       1,
			t:       2,
			h:       2,
			d:       2,
			strideB: 0,
			src:     []float32{1, 3, 5, 7, 2, 4, 6, 8},
			cos:     []float32{0.5, 0.6},
			sin:     []float32{0.7, 0.8},
			want:    []float32{-1.6, 2.2, -2.4, 7, -2, 4, -2.8, 9.6},
		},
		{
			name:    "Empty b=0 t=2 h=2 d=2 strideB=0",
			ndims:   4,
			dims:    []int{0, 2, 2, 2},
			strides: []int{8, 4, 2, 1},
			b:       0,
			t:       2,
			h:       2,
			d:       2,
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
			kernels.RopeThdStridedF32(tt.ndims, tt.dims, tt.strides, tt.b, tt.t, tt.h, tt.d, tt.strideB, tt.src, tt.cos, tt.sin, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestRopeThdStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		ndims   int
		dims    []int
		strides []int
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
			name:    "Contiguous fallback b=1 t=2 h=2 d=2 strideB=0",
			ndims:   4,
			dims:    []int{1, 2, 2, 2},
			strides: []int{8, 4, 2, 1},
			b:       1,
			t:       2,
			h:       2,
			d:       2,
			strideB: 0,
			src:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
			cos:     []float64{0.5, 0.6},
			sin:     []float64{0.7, 0.8},
			want:    []float64{-0.9, 1.7, -1.3, 4.1, -1.8, 7.6, -2.2, 10.4},
		},
		{
			name:    "Strided b=1 t=2 h=2 d=2 strideB=0",
			ndims:   4,
			dims:    []int{2, 2, 2, 1},
			strides: []int{4, 2, 1, 8},
			b:       1,
			t:       2,
			h:       2,
			d:       2,
			strideB: 0,
			src:     []float64{1, 3, 5, 7, 2, 4, 6, 8},
			cos:     []float64{0.5, 0.6},
			sin:     []float64{0.7, 0.8},
			want:    []float64{-1.6, 2.2, -2.4, 7, -2, 4, -2.8, 9.6},
		},
		{
			name:    "Empty b=0 t=2 h=2 d=2 strideB=0",
			ndims:   4,
			dims:    []int{0, 2, 2, 2},
			strides: []int{8, 4, 2, 1},
			b:       0,
			t:       2,
			h:       2,
			d:       2,
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
			kernels.RopeThdStridedF64(tt.ndims, tt.dims, tt.strides, tt.b, tt.t, tt.h, tt.d, tt.strideB, tt.src, tt.cos, tt.sin, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

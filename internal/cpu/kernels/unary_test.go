package kernels_test

import (
	"math"
	"testing"

	"slices"

	"github.com/gocnn/spark/internal/cpu/kernels"
)

func TestUCopyStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   5,
			numDims: 1,
			dims:    []int{5},
			strides: []int{1},
			inp:     []float64{1, 2, 3, 4, 5},
			out:     make([]float64, 5),
			want:    []float64{1, 2, 3, 4, 5},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			inp:     []float64{1, 2, 3, 4},
			out:     make([]float64, 4),
			want:    []float64{1, 2, 3, 4},
		},
		{
			name:    "In-place no-op",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{10, 20, 30},
			want:    []float64{10, 20, 30},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UCopyStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUNegF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic negation",
			numel: 5,
			inp:   []float32{1, -2, 3, -4, 5},
			out:   make([]float32, 5),
			want:  []float32{-1, 2, -3, 4, -5},
		},
		{
			name:  "In-place negation",
			numel: 3,
			inp:   nil,
			out:   []float32{10, -20, 30},
			want:  []float32{-10, 20, -30},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UNegF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUNegStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   5,
			numDims: 1,
			dims:    []int{5},
			strides: []int{1},
			inp:     []float32{1, -2, 3, -4, 5},
			out:     make([]float32, 5),
			want:    []float32{-1, 2, -3, 4, -5},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			inp:     []float32{1, -2, 3, -4},
			out:     make([]float32, 4),
			want:    []float32{-1, 2, -3, 4},
		},
		{
			name:    "In-place negation",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{10, -20, 30},
			want:    []float32{-10, 20, -30},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UNegStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUNegF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic negation",
			numel: 5,
			inp:   []float64{1, -2, 3, -4, 5},
			out:   make([]float64, 5),
			want:  []float64{-1, 2, -3, 4, -5},
		},
		{
			name:  "In-place negation",
			numel: 3,
			inp:   nil,
			out:   []float64{10, -20, 30},
			want:  []float64{-10, 20, -30},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UNegF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUNegStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   5,
			numDims: 1,
			dims:    []int{5},
			strides: []int{1},
			inp:     []float64{1, -2, 3, -4, 5},
			out:     make([]float64, 5),
			want:    []float64{-1, 2, -3, 4, -5},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			inp:     []float64{1, -2, 3, -4},
			out:     make([]float64, 4),
			want:    []float64{-1, 2, -3, 4},
		},
		{
			name:    "In-place negation",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{10, -20, 30},
			want:    []float64{-10, 20, -30},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UNegStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestURecipF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic reciprocal",
			numel: 5,
			inp:   []float32{1, 2, 4, 5, 10},
			out:   make([]float32, 5),
			want:  []float32{1, 0.5, 0.25, 0.2, 0.1},
		},
		{
			name:  "In-place reciprocal",
			numel: 3,
			inp:   nil,
			out:   []float32{10, 20, 4},
			want:  []float32{0.1, 0.05, 0.25},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.URecipF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestURecipStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   5,
			numDims: 1,
			dims:    []int{5},
			strides: []int{1},
			inp:     []float32{1, 2, 4, 5, 10},
			out:     make([]float32, 5),
			want:    []float32{1, 0.5, 0.25, 0.2, 0.1},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			inp:     []float32{1, 2, 4, 5},
			out:     make([]float32, 4),
			want:    []float32{1, 0.5, 0.25, 0.2},
		},
		{
			name:    "In-place reciprocal",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{10, 20, 4},
			want:    []float32{0.1, 0.05, 0.25},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.URecipStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestURecipF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic reciprocal",
			numel: 5,
			inp:   []float64{1, 2, 4, 5, 10},
			out:   make([]float64, 5),
			want:  []float64{1, 0.5, 0.25, 0.2, 0.1},
		},
		{
			name:  "In-place reciprocal",
			numel: 3,
			inp:   nil,
			out:   []float64{10, 20, 4},
			want:  []float64{0.1, 0.05, 0.25},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.URecipF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestURecipStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   5,
			numDims: 1,
			dims:    []int{5},
			strides: []int{1},
			inp:     []float64{1, 2, 4, 5, 10},
			out:     make([]float64, 5),
			want:    []float64{1, 0.5, 0.25, 0.2, 0.1},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			inp:     []float64{1, 2, 4, 5},
			out:     make([]float64, 4),
			want:    []float64{1, 0.5, 0.25, 0.2},
		},
		{
			name:    "In-place reciprocal",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{10, 20, 4},
			want:    []float64{0.1, 0.05, 0.25},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.URecipStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUExpF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic exp",
			numel: 3,
			inp:   []float32{0, 1, -1},
			out:   make([]float32, 3),
			want:  []float32{1, float32(math.E), float32(1 / math.E)},
		},
		{
			name:  "In-place exp",
			numel: 3,
			inp:   nil,
			out:   []float32{0, 1, -1},
			want:  []float32{1, float32(math.E), float32(1 / math.E)},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UExpF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUExpStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{0, 1, -1},
			out:     make([]float32, 3),
			want:    []float32{1, float32(math.E), float32(1 / math.E)},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			inp:     []float32{0, 1, 0, -1},
			out:     make([]float32, 4),
			want:    []float32{1, float32(math.E), 1, float32(1 / math.E)},
		},
		{
			name:    "In-place exp",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{0, 1, -1},
			want:    []float32{1, float32(math.E), float32(1 / math.E)},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UExpStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUExpF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic exp",
			numel: 3,
			inp:   []float64{0, 1, -1},
			out:   make([]float64, 3),
			want:  []float64{1, math.E, 1 / math.E},
		},
		{
			name:  "In-place exp",
			numel: 3,
			inp:   nil,
			out:   []float64{0, 1, -1},
			want:  []float64{1, math.E, 1 / math.E},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UExpF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUExpStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{0, 1, -1},
			out:     make([]float64, 3),
			want:    []float64{1, math.E, 1 / math.E},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   4,
			numDims: 2,
			dims:    []int{2, 2},
			strides: []int{2, 1},
			inp:     []float64{0, 1, 0, -1},
			out:     make([]float64, 4),
			want:    []float64{1, math.E, 1, 1 / math.E},
		},
		{
			name:    "In-place exp",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{0, 1, -1},
			want:    []float64{1, math.E, 1 / math.E},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UExpStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestULogF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic log",
			numel: 3,
			inp:   []float32{1, float32(math.E), float32(math.E * math.E)},
			out:   make([]float32, 3),
			want:  []float32{0, 1, 2},
		},
		{
			name:  "In-place log",
			numel: 3,
			inp:   nil,
			out:   []float32{1, float32(math.E), float32(math.E * math.E)},
			want:  []float32{0, 1, 2},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.ULogF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestULogStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{1, float32(math.E), float32(math.E * math.E)},
			out:     make([]float32, 3),
			want:    []float32{0, 1, 2},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{1, 0, float32(math.E), 0},
			out:     make([]float32, 2),
			want:    []float32{0, 1},
		},
		{
			name:    "In-place log",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{1, float32(math.E), float32(math.E * math.E)},
			want:    []float32{0, 1, 2},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.ULogStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestULogF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic log",
			numel: 3,
			inp:   []float64{1, math.E, math.E * math.E},
			out:   make([]float64, 3),
			want:  []float64{0, 1, 2},
		},
		{
			name:  "In-place log",
			numel: 3,
			inp:   nil,
			out:   []float64{1, math.E, math.E * math.E},
			want:  []float64{0, 1, 2},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.ULogF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestULogStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{1, math.E, math.E * math.E},
			out:     make([]float64, 3),
			want:    []float64{0, 1, 2},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{1, 0, math.E, 0},
			out:     make([]float64, 2),
			want:    []float64{0, 1},
		},
		{
			name:    "In-place log",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{1, math.E, math.E * math.E},
			want:    []float64{0, 1, 2},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.ULogStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSinF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic sin",
			numel: 3,
			inp:   []float32{0, float32(math.Pi / 2), float32(math.Pi)},
			out:   make([]float32, 3),
			want:  []float32{0, 1, 0},
		},
		{
			name:  "In-place sin",
			numel: 3,
			inp:   nil,
			out:   []float32{0, float32(math.Pi / 2), float32(math.Pi)},
			want:  []float32{0, 1, 0},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USinF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSinStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{0, float32(math.Pi / 2), float32(math.Pi)},
			out:     make([]float32, 3),
			want:    []float32{0, 1, 0},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{0, 0, float32(math.Pi / 2), 0},
			out:     make([]float32, 2),
			want:    []float32{0, 1},
		},
		{
			name:    "In-place sin",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{0, float32(math.Pi / 2), float32(math.Pi)},
			want:    []float32{0, 1, 0},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USinStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSinF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic sin",
			numel: 3,
			inp:   []float64{0, math.Pi / 2, math.Pi},
			out:   make([]float64, 3),
			want:  []float64{0, 1, 0},
		},
		{
			name:  "In-place sin",
			numel: 3,
			inp:   nil,
			out:   []float64{0, math.Pi / 2, math.Pi},
			want:  []float64{0, 1, 0},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USinF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSinStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{0, math.Pi / 2, math.Pi},
			out:     make([]float64, 3),
			want:    []float64{0, 1, 0},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{0, 0, math.Pi / 2, 0},
			out:     make([]float64, 2),
			want:    []float64{0, 1},
		},
		{
			name:    "In-place sin",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{0, math.Pi / 2, math.Pi},
			want:    []float64{0, 1, 0},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USinStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUCosF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic cos",
			numel: 3,
			inp:   []float32{0, float32(math.Pi / 2), float32(math.Pi)},
			out:   make([]float32, 3),
			want:  []float32{1, 0, -1},
		},
		{
			name:  "In-place cos",
			numel: 3,
			inp:   nil,
			out:   []float32{0, float32(math.Pi / 2), float32(math.Pi)},
			want:  []float32{1, 0, -1},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UCosF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUCosStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{0, float32(math.Pi / 2), float32(math.Pi)},
			out:     make([]float32, 3),
			want:    []float32{1, 0, -1},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{0, 0, float32(math.Pi / 2), 0},
			out:     make([]float32, 2),
			want:    []float32{1, 0},
		},
		{
			name:    "In-place cos",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{0, float32(math.Pi / 2), float32(math.Pi)},
			want:    []float32{1, 0, -1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UCosStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUCosF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic cos",
			numel: 3,
			inp:   []float64{0, math.Pi / 2, math.Pi},
			out:   make([]float64, 3),
			want:  []float64{1, 0, -1},
		},
		{
			name:  "In-place cos",
			numel: 3,
			inp:   nil,
			out:   []float64{0, math.Pi / 2, math.Pi},
			want:  []float64{1, 0, -1},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UCosF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUCosStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{0, math.Pi / 2, math.Pi},
			out:     make([]float64, 3),
			want:    []float64{1, 0, -1},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{0, 0, math.Pi / 2, 0},
			out:     make([]float64, 2),
			want:    []float64{1, 0},
		},
		{
			name:    "In-place cos",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{0, math.Pi / 2, math.Pi},
			want:    []float64{1, 0, -1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UCosStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUTanhF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic tanh",
			numel: 3,
			inp:   []float32{0, 1, -1},
			out:   make([]float32, 3),
			want:  []float32{0, float32(math.Tanh(1)), float32(math.Tanh(-1))},
		},
		{
			name:  "In-place tanh",
			numel: 3,
			inp:   nil,
			out:   []float32{0, 1, -1},
			want:  []float32{0, float32(math.Tanh(1)), float32(math.Tanh(-1))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UTanhF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUTanhStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{0, 1, -1},
			out:     make([]float32, 3),
			want:    []float32{0, float32(math.Tanh(1)), float32(math.Tanh(-1))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{0, 0, 1, 0},
			out:     make([]float32, 2),
			want:    []float32{0, float32(math.Tanh(1))},
		},
		{
			name:    "In-place tanh",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{0, 1, -1},
			want:    []float32{0, float32(math.Tanh(1)), float32(math.Tanh(-1))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UTanhStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUTanhF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic tanh",
			numel: 3,
			inp:   []float64{0, 1, -1},
			out:   make([]float64, 3),
			want:  []float64{0, math.Tanh(1), math.Tanh(-1)},
		},
		{
			name:  "In-place tanh",
			numel: 3,
			inp:   nil,
			out:   []float64{0, 1, -1},
			want:  []float64{0, math.Tanh(1), math.Tanh(-1)},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UTanhF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUTanhStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{0, 1, -1},
			out:     make([]float64, 3),
			want:    []float64{0, math.Tanh(1), math.Tanh(-1)},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{0, 0, 1, 0},
			out:     make([]float64, 2),
			want:    []float64{0, math.Tanh(1)},
		},
		{
			name:    "In-place tanh",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{0, 1, -1},
			want:    []float64{0, math.Tanh(1), math.Tanh(-1)},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UTanhStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUErfF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic erf",
			numel: 3,
			inp:   []float32{0, 1, -1},
			out:   make([]float32, 3),
			want:  []float32{0, float32(math.Erf(1)), float32(math.Erf(-1))},
		},
		{
			name:  "In-place erf",
			numel: 3,
			inp:   nil,
			out:   []float32{0, 1, -1},
			want:  []float32{0, float32(math.Erf(1)), float32(math.Erf(-1))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UErfF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUErfStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{0, 1, -1},
			out:     make([]float32, 3),
			want:    []float32{0, float32(math.Erf(1)), float32(math.Erf(-1))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{0, 0, 1, 0},
			out:     make([]float32, 2),
			want:    []float32{0, float32(math.Erf(1))},
		},
		{
			name:    "In-place erf",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{0, 1, -1},
			want:    []float32{0, float32(math.Erf(1)), float32(math.Erf(-1))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UErfStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUErfF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic erf",
			numel: 3,
			inp:   []float64{0, 1, -1},
			out:   make([]float64, 3),
			want:  []float64{0, math.Erf(1), math.Erf(-1)},
		},
		{
			name:  "In-place erf",
			numel: 3,
			inp:   nil,
			out:   []float64{0, 1, -1},
			want:  []float64{0, math.Erf(1), math.Erf(-1)},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UErfF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUErfStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{0, 1, -1},
			out:     make([]float64, 3),
			want:    []float64{0, math.Erf(1), math.Erf(-1)},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{0, 0, 1, 0},
			out:     make([]float64, 2),
			want:    []float64{0, math.Erf(1)},
		},
		{
			name:    "In-place erf",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{0, 1, -1},
			want:    []float64{0, math.Erf(1), math.Erf(-1)},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UErfStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUCeilF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic ceil",
			numel: 4,
			inp:   []float32{1.1, 2.9, -1.1, -2.9},
			out:   make([]float32, 4),
			want:  []float32{2, 3, -1, -2},
		},
		{
			name:  "In-place ceil",
			numel: 4,
			inp:   nil,
			out:   []float32{1.1, 2.9, -1.1, -2.9},
			want:  []float32{2, 3, -1, -2},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UCeilF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUCeilStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float32{1.1, 2.9, -1.1, -2.9},
			out:     make([]float32, 4),
			want:    []float32{2, 3, -1, -2},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{1.1, 0, 2.9, 0},
			out:     make([]float32, 2),
			want:    []float32{2, 3},
		},
		{
			name:    "In-place ceil",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float32{1.1, 2.9, -1.1, -2.9},
			want:    []float32{2, 3, -1, -2},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UCeilStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUCeilF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic ceil",
			numel: 4,
			inp:   []float64{1.1, 2.9, -1.1, -2.9},
			out:   make([]float64, 4),
			want:  []float64{2, 3, -1, -2},
		},
		{
			name:  "In-place ceil",
			numel: 4,
			inp:   nil,
			out:   []float64{1.1, 2.9, -1.1, -2.9},
			want:  []float64{2, 3, -1, -2},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UCeilF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUCeilStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float64{1.1, 2.9, -1.1, -2.9},
			out:     make([]float64, 4),
			want:    []float64{2, 3, -1, -2},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{1.1, 0, 2.9, 0},
			out:     make([]float64, 2),
			want:    []float64{2, 3},
		},
		{
			name:    "In-place ceil",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float64{1.1, 2.9, -1.1, -2.9},
			want:    []float64{2, 3, -1, -2},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UCeilStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUFloorF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic floor",
			numel: 4,
			inp:   []float32{1.1, 2.9, -1.1, -2.9},
			out:   make([]float32, 4),
			want:  []float32{1, 2, -2, -3},
		},
		{
			name:  "In-place floor",
			numel: 4,
			inp:   nil,
			out:   []float32{1.1, 2.9, -1.1, -2.9},
			want:  []float32{1, 2, -2, -3},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UFloorF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUFloorStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float32{1.1, 2.9, -1.1, -2.9},
			out:     make([]float32, 4),
			want:    []float32{1, 2, -2, -3},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{1.1, 0, 2.9, 0},
			out:     make([]float32, 2),
			want:    []float32{1, 2},
		},
		{
			name:    "In-place floor",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float32{1.1, 2.9, -1.1, -2.9},
			want:    []float32{1, 2, -2, -3},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UFloorStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUFloorF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic floor",
			numel: 4,
			inp:   []float64{1.1, 2.9, -1.1, -2.9},
			out:   make([]float64, 4),
			want:  []float64{1, 2, -2, -3},
		},
		{
			name:  "In-place floor",
			numel: 4,
			inp:   nil,
			out:   []float64{1.1, 2.9, -1.1, -2.9},
			want:  []float64{1, 2, -2, -3},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UFloorF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUFloorStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float64{1.1, 2.9, -1.1, -2.9},
			out:     make([]float64, 4),
			want:    []float64{1, 2, -2, -3},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{1.1, 0, 2.9, 0},
			out:     make([]float64, 2),
			want:    []float64{1, 2},
		},
		{
			name:    "In-place floor",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float64{1.1, 2.9, -1.1, -2.9},
			want:    []float64{1, 2, -2, -3},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UFloorStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestURoundF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic round",
			numel: 4,
			inp:   []float32{1.1, 2.6, -1.1, -2.6},
			out:   make([]float32, 4),
			want:  []float32{1, 3, -1, -3},
		},
		{
			name:  "In-place round",
			numel: 4,
			inp:   nil,
			out:   []float32{1.1, 2.6, -1.1, -2.6},
			want:  []float32{1, 3, -1, -3},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.URoundF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestURoundStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float32{1.1, 2.6, -1.1, -2.6},
			out:     make([]float32, 4),
			want:    []float32{1, 3, -1, -3},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{1.1, 0, 2.6, 0},
			out:     make([]float32, 2),
			want:    []float32{1, 3},
		},
		{
			name:    "In-place round",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float32{1.1, 2.6, -1.1, -2.6},
			want:    []float32{1, 3, -1, -3},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.URoundStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestURoundF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic round",
			numel: 4,
			inp:   []float64{1.1, 2.6, -1.1, -2.6},
			out:   make([]float64, 4),
			want:  []float64{1, 3, -1, -3},
		},
		{
			name:  "In-place round",
			numel: 4,
			inp:   nil,
			out:   []float64{1.1, 2.6, -1.1, -2.6},
			want:  []float64{1, 3, -1, -3},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.URoundF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestURoundStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float64{1.1, 2.6, -1.1, -2.6},
			out:     make([]float64, 4),
			want:    []float64{1, 3, -1, -3},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{1.1, 0, 2.6, 0},
			out:     make([]float64, 2),
			want:    []float64{1, 3},
		},
		{
			name:    "In-place round",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float64{1.1, 2.6, -1.1, -2.6},
			want:    []float64{1, 3, -1, -3},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.URoundStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUNormcdfF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic normcdf",
			numel: 3,
			inp:   []float32{0, 1, -1},
			out:   make([]float32, 3),
			want:  []float32{0.5, float32(0.5 * (1 + math.Erf(1/math.Sqrt(2)))), float32(0.5 * (1 + math.Erf(-1/math.Sqrt(2))))},
		},
		{
			name:  "In-place normcdf",
			numel: 3,
			inp:   nil,
			out:   []float32{0, 1, -1},
			want:  []float32{0.5, float32(0.5 * (1 + math.Erf(1/math.Sqrt(2)))), float32(0.5 * (1 + math.Erf(-1/math.Sqrt(2))))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UNormcdfF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUNormcdfStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{0, 1, -1},
			out:     make([]float32, 3),
			want:    []float32{0.5, float32(0.5 * (1 + math.Erf(1/math.Sqrt(2)))), float32(0.5 * (1 + math.Erf(-1/math.Sqrt(2))))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{0, 0, 1, 0},
			out:     make([]float32, 2),
			want:    []float32{0.5, float32(0.5 * (1 + math.Erf(1/math.Sqrt(2))))},
		},
		{
			name:    "In-place normcdf",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{0, 1, -1},
			want:    []float32{0.5, float32(0.5 * (1 + math.Erf(1/math.Sqrt(2)))), float32(0.5 * (1 + math.Erf(-1/math.Sqrt(2))))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UNormcdfStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUNormcdfF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic normcdf",
			numel: 3,
			inp:   []float64{0, 1, -1},
			out:   make([]float64, 3),
			want:  []float64{0.5, 0.5 * (1 + math.Erf(1/math.Sqrt(2))), 0.5 * (1 + math.Erf(-1/math.Sqrt(2)))},
		},
		{
			name:  "In-place normcdf",
			numel: 3,
			inp:   nil,
			out:   []float64{0, 1, -1},
			want:  []float64{0.5, 0.5 * (1 + math.Erf(1/math.Sqrt(2))), 0.5 * (1 + math.Erf(-1/math.Sqrt(2)))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UNormcdfF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUNormcdfStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{0, 1, -1},
			out:     make([]float64, 3),
			want:    []float64{0.5, 0.5 * (1 + math.Erf(1/math.Sqrt(2))), 0.5 * (1 + math.Erf(-1/math.Sqrt(2)))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{0, 0, 1, 0},
			out:     make([]float64, 2),
			want:    []float64{0.5, 0.5 * (1 + math.Erf(1/math.Sqrt(2)))},
		},
		{
			name:    "In-place normcdf",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{0, 1, -1},
			want:    []float64{0.5, 0.5 * (1 + math.Erf(1/math.Sqrt(2))), 0.5 * (1 + math.Erf(-1/math.Sqrt(2)))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UNormcdfStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUAbsF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic abs",
			numel: 4,
			inp:   []float32{1, -1, 0, -0},
			out:   make([]float32, 4),
			want:  []float32{1, 1, 0, 0},
		},
		{
			name:  "In-place abs",
			numel: 4,
			inp:   nil,
			out:   []float32{1, -1, 0, -0},
			want:  []float32{1, 1, 0, 0},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UAbsF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUAbsStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float32{1, -1, 0, -0},
			out:     make([]float32, 4),
			want:    []float32{1, 1, 0, 0},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{1, 0, -1, 0},
			out:     make([]float32, 2),
			want:    []float32{1, 1},
		},
		{
			name:    "In-place abs",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float32{1, -1, 0, -0},
			want:    []float32{1, 1, 0, 0},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UAbsStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUAbsF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic abs",
			numel: 4,
			inp:   []float64{1, -1, 0, -0},
			out:   make([]float64, 4),
			want:  []float64{1, 1, 0, 0},
		},
		{
			name:  "In-place abs",
			numel: 4,
			inp:   nil,
			out:   []float64{1, -1, 0, -0},
			want:  []float64{1, 1, 0, 0},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UAbsF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUAbsStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float64{1, -1, 0, -0},
			out:     make([]float64, 4),
			want:    []float64{1, 1, 0, 0},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{1, 0, -1, 0},
			out:     make([]float64, 2),
			want:    []float64{1, 1},
		},
		{
			name:    "In-place abs",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float64{1, -1, 0, -0},
			want:    []float64{1, 1, 0, 0},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UAbsStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSqrF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic sqr",
			numel: 3,
			inp:   []float32{1, 2, 3},
			out:   make([]float32, 3),
			want:  []float32{1, 4, 9},
		},
		{
			name:  "In-place sqr",
			numel: 3,
			inp:   nil,
			out:   []float32{1, 2, 3},
			want:  []float32{1, 4, 9},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USqrF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSqrStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{1, 2, 3},
			out:     make([]float32, 3),
			want:    []float32{1, 4, 9},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{1, 0, 2, 0},
			out:     make([]float32, 2),
			want:    []float32{1, 4},
		},
		{
			name:    "In-place sqr",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{1, 2, 3},
			want:    []float32{1, 4, 9},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USqrStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSqrF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic sqr",
			numel: 3,
			inp:   []float64{1, 2, 3},
			out:   make([]float64, 3),
			want:  []float64{1, 4, 9},
		},
		{
			name:  "In-place sqr",
			numel: 3,
			inp:   nil,
			out:   []float64{1, 2, 3},
			want:  []float64{1, 4, 9},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USqrF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSqrStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{1, 2, 3},
			out:     make([]float64, 3),
			want:    []float64{1, 4, 9},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{1, 0, 2, 0},
			out:     make([]float64, 2),
			want:    []float64{1, 4},
		},
		{
			name:    "In-place sqr",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{1, 2, 3},
			want:    []float64{1, 4, 9},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USqrStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSqrtF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic sqrt",
			numel: 3,
			inp:   []float32{1, 4, 9},
			out:   make([]float32, 3),
			want:  []float32{1, 2, 3},
		},
		{
			name:  "In-place sqrt",
			numel: 3,
			inp:   nil,
			out:   []float32{1, 4, 9},
			want:  []float32{1, 2, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USqrtF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSqrtStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{1, 4, 9},
			out:     make([]float32, 3),
			want:    []float32{1, 2, 3},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{1, 0, 4, 0},
			out:     make([]float32, 2),
			want:    []float32{1, 2},
		},
		{
			name:    "In-place sqrt",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{1, 4, 9},
			want:    []float32{1, 2, 3},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USqrtStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSqrtF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic sqrt",
			numel: 3,
			inp:   []float64{1, 4, 9},
			out:   make([]float64, 3),
			want:  []float64{1, 2, 3},
		},
		{
			name:  "In-place sqrt",
			numel: 3,
			inp:   nil,
			out:   []float64{1, 4, 9},
			want:  []float64{1, 2, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USqrtF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSqrtStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{1, 4, 9},
			out:     make([]float64, 3),
			want:    []float64{1, 2, 3},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{1, 0, 4, 0},
			out:     make([]float64, 2),
			want:    []float64{1, 2},
		},
		{
			name:    "In-place sqrt",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{1, 4, 9},
			want:    []float64{1, 2, 3},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USqrtStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUGeluF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic gelu",
			numel: 3,
			inp:   []float32{0, 1, -1},
			out:   make([]float32, 3),
			want:  []float32{0, float32(0.5 * 1 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(1+0.044715*1)))), float32(0.5 * (-1) * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(-1+0.044715*(-1)))))},
		},
		{
			name:  "In-place gelu",
			numel: 3,
			inp:   nil,
			out:   []float32{0, 1, -1},
			want:  []float32{0, float32(0.5 * 1 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(1+0.044715*1)))), float32(0.5 * (-1) * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(-1+0.044715*(-1)))))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UGeluF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUGeluStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{0, 1, -1},
			out:     make([]float32, 3),
			want:    []float32{0, float32(0.5 * 1 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(1+0.044715*1)))), float32(0.5 * (-1) * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(-1+0.044715*(-1)))))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{0, 0, 1, 0},
			out:     make([]float32, 2),
			want:    []float32{0, float32(0.5 * 1 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(1+0.044715*1))))},
		},
		{
			name:    "In-place gelu",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{0, 1, -1},
			want:    []float32{0, float32(0.5 * 1 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(1+0.044715*1)))), float32(0.5 * (-1) * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(-1+0.044715*(-1)))))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UGeluStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUGeluF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic gelu",
			numel: 3,
			inp:   []float64{0, 1, -1},
			out:   make([]float64, 3),
			want:  []float64{0, 0.5 * 1 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(1+0.044715*1))), 0.5 * (-1) * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(-1+0.044715*(-1))))},
		},
		{
			name:  "In-place gelu",
			numel: 3,
			inp:   nil,
			out:   []float64{0, 1, -1},
			want:  []float64{0, 0.5 * 1 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(1+0.044715*1))), 0.5 * (-1) * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(-1+0.044715*(-1))))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UGeluF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUGeluStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{0, 1, -1},
			out:     make([]float64, 3),
			want:    []float64{0, 0.5 * 1 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(1+0.044715*1))), 0.5 * (-1) * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(-1+0.044715*(-1))))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{0, 0, 1, 0},
			out:     make([]float64, 2),
			want:    []float64{0, 0.5 * 1 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(1+0.044715*1)))},
		},
		{
			name:    "In-place gelu",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{0, 1, -1},
			want:    []float64{0, 0.5 * 1 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(1+0.044715*1))), 0.5 * (-1) * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(-1+0.044715*(-1))))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UGeluStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUGeluErfF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic gelu_erf",
			numel: 3,
			inp:   []float32{0, 1, -1},
			out:   make([]float32, 3),
			want:  []float32{0, float32(1 * 0.5 * (1 + math.Erf(1/math.Sqrt(2)))), float32((-1) * 0.5 * (1 + math.Erf(-1/math.Sqrt(2))))},
		},
		{
			name:  "In-place gelu_erf",
			numel: 3,
			inp:   nil,
			out:   []float32{0, 1, -1},
			want:  []float32{0, float32(1 * 0.5 * (1 + math.Erf(1/math.Sqrt(2)))), float32((-1) * 0.5 * (1 + math.Erf(-1/math.Sqrt(2))))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UGeluErfF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUGeluErfStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{0, 1, -1},
			out:     make([]float32, 3),
			want:    []float32{0, float32(1 * 0.5 * (1 + math.Erf(1/math.Sqrt(2)))), float32((-1) * 0.5 * (1 + math.Erf(-1/math.Sqrt(2))))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{0, 0, 1, 0},
			out:     make([]float32, 2),
			want:    []float32{0, float32(1 * 0.5 * (1 + math.Erf(1/math.Sqrt(2))))},
		},
		{
			name:    "In-place gelu_erf",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{0, 1, -1},
			want:    []float32{0, float32(1 * 0.5 * (1 + math.Erf(1/math.Sqrt(2)))), float32((-1) * 0.5 * (1 + math.Erf(-1/math.Sqrt(2))))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UGeluErfStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUGeluErfF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic gelu_erf",
			numel: 3,
			inp:   []float64{0, 1, -1},
			out:   make([]float64, 3),
			want:  []float64{0, 1 * 0.5 * (1 + math.Erf(1/math.Sqrt(2))), (-1) * 0.5 * (1 + math.Erf(-1/math.Sqrt(2)))},
		},
		{
			name:  "In-place gelu_erf",
			numel: 3,
			inp:   nil,
			out:   []float64{0, 1, -1},
			want:  []float64{0, 1 * 0.5 * (1 + math.Erf(1/math.Sqrt(2))), (-1) * 0.5 * (1 + math.Erf(-1/math.Sqrt(2)))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UGeluErfF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUGeluErfStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{0, 1, -1},
			out:     make([]float64, 3),
			want:    []float64{0, 1 * 0.5 * (1 + math.Erf(1/math.Sqrt(2))), (-1) * 0.5 * (1 + math.Erf(-1/math.Sqrt(2)))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{0, 0, 1, 0},
			out:     make([]float64, 2),
			want:    []float64{0, 1 * 0.5 * (1 + math.Erf(1/math.Sqrt(2)))},
		},
		{
			name:    "In-place gelu_erf",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{0, 1, -1},
			want:    []float64{0, 1 * 0.5 * (1 + math.Erf(1/math.Sqrt(2))), (-1) * 0.5 * (1 + math.Erf(-1/math.Sqrt(2)))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UGeluErfStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUReluF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic relu",
			numel: 4,
			inp:   []float32{-2, -1, 0, 1},
			out:   make([]float32, 4),
			want:  []float32{0, 0, 0, 1},
		},
		{
			name:  "In-place relu",
			numel: 4,
			inp:   nil,
			out:   []float32{-2, -1, 0, 1},
			want:  []float32{0, 0, 0, 1},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UReluF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUReluStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float32{-2, -1, 0, 1},
			out:     make([]float32, 4),
			want:    []float32{0, 0, 0, 1},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{-1, 0, 1, 0},
			out:     make([]float32, 2),
			want:    []float32{0, 1},
		},
		{
			name:    "In-place relu",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float32{-2, -1, 0, 1},
			want:    []float32{0, 0, 0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UReluStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUReluF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic relu",
			numel: 4,
			inp:   []float64{-2, -1, 0, 1},
			out:   make([]float64, 4),
			want:  []float64{0, 0, 0, 1},
		},
		{
			name:  "In-place relu",
			numel: 4,
			inp:   nil,
			out:   []float64{-2, -1, 0, 1},
			want:  []float64{0, 0, 0, 1},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UReluF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUReluStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float64{-2, -1, 0, 1},
			out:     make([]float64, 4),
			want:    []float64{0, 0, 0, 1},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{-1, 0, 1, 0},
			out:     make([]float64, 2),
			want:    []float64{0, 1},
		},
		{
			name:    "In-place relu",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float64{-2, -1, 0, 1},
			want:    []float64{0, 0, 0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UReluStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUEluF32(t *testing.T) {
	tests := []struct {
		name  string
		alpha float32
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic elu",
			alpha: 1.0,
			numel: 3,
			inp:   []float32{-2, 0, 1},
			out:   make([]float32, 3),
			want:  []float32{1 * (float32(math.Exp(float64(-2))) - 1), 0, 1},
		},
		{
			name:  "In-place elu",
			alpha: 1.0,
			numel: 3,
			inp:   nil,
			out:   []float32{-2, 0, 1},
			want:  []float32{1 * (float32(math.Exp(float64(-2))) - 1), 0, 1},
		},
		{
			name:  "Empty",
			alpha: 1.0,
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UEluF32(tt.alpha, tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUEluStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		alpha   float32
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			alpha:   1.0,
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{-2, 0, 1},
			out:     make([]float32, 3),
			want:    []float32{1 * (float32(math.Exp(float64(-2))) - 1), 0, 1},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			alpha:   1.0,
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{-2, 0, 1, 0},
			out:     make([]float32, 2),
			want:    []float32{1 * (float32(math.Exp(float64(-2))) - 1), 1},
		},
		{
			name:    "In-place elu",
			alpha:   1.0,
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{-2, 0, 1},
			want:    []float32{1 * (float32(math.Exp(float64(-2))) - 1), 0, 1},
		},
		{
			name:    "Empty",
			alpha:   1.0,
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UEluStridedF32(tt.alpha, tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUEluF64(t *testing.T) {
	tests := []struct {
		name  string
		alpha float64
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic elu",
			alpha: 1.0,
			numel: 3,
			inp:   []float64{-2, 0, 1},
			out:   make([]float64, 3),
			want:  []float64{1 * (math.Exp(-2) - 1), 0, 1},
		},
		{
			name:  "In-place elu",
			alpha: 1.0,
			numel: 3,
			inp:   nil,
			out:   []float64{-2, 0, 1},
			want:  []float64{1 * (math.Exp(-2) - 1), 0, 1},
		},
		{
			name:  "Empty",
			alpha: 1.0,
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UEluF64(tt.alpha, tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUEluStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		alpha   float64
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			alpha:   1.0,
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{-2, 0, 1},
			out:     make([]float64, 3),
			want:    []float64{1 * (math.Exp(-2) - 1), 0, 1},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			alpha:   1.0,
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{-2, 0, 1, 0},
			out:     make([]float64, 2),
			want:    []float64{1 * (math.Exp(-2) - 1), 1},
		},
		{
			name:    "In-place elu",
			alpha:   1.0,
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{-2, 0, 1},
			want:    []float64{1 * (math.Exp(-2) - 1), 0, 1},
		},
		{
			name:    "Empty",
			alpha:   1.0,
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UEluStridedF64(tt.alpha, tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSiluF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic silu",
			numel: 3,
			inp:   []float32{0, 1, -1},
			out:   make([]float32, 3),
			want:  []float32{0, float32(1 / (1 + math.Exp(-1))), float32(-1 / (1 + math.Exp(1)))},
		},
		{
			name:  "In-place silu",
			numel: 3,
			inp:   nil,
			out:   []float32{0, 1, -1},
			want:  []float32{0, float32(1 / (1 + math.Exp(-1))), float32(-1 / (1 + math.Exp(1)))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USiluF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSiluStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{0, 1, -1},
			out:     make([]float32, 3),
			want:    []float32{0, float32(1 / (1 + math.Exp(-1))), float32(-1 / (1 + math.Exp(1)))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{0, 0, 1, 0},
			out:     make([]float32, 2),
			want:    []float32{0, float32(1 / (1 + math.Exp(-1)))},
		},
		{
			name:    "In-place silu",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{0, 1, -1},
			want:    []float32{0, float32(1 / (1 + math.Exp(-1))), float32(-1 / (1 + math.Exp(1)))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USiluStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSiluF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic silu",
			numel: 3,
			inp:   []float64{0, 1, -1},
			out:   make([]float64, 3),
			want:  []float64{0, 1 / (1 + math.Exp(-1)), -1 / (1 + math.Exp(1))},
		},
		{
			name:  "In-place silu",
			numel: 3,
			inp:   nil,
			out:   []float64{0, 1, -1},
			want:  []float64{0, 1 / (1 + math.Exp(-1)), -1 / (1 + math.Exp(1))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USiluF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSiluStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{0, 1, -1},
			out:     make([]float64, 3),
			want:    []float64{0, 1 / (1 + math.Exp(-1)), -1 / (1 + math.Exp(1))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{0, 0, 1, 0},
			out:     make([]float64, 2),
			want:    []float64{0, 1 / (1 + math.Exp(-1))},
		},
		{
			name:    "In-place silu",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{0, 1, -1},
			want:    []float64{0, 1 / (1 + math.Exp(-1)), -1 / (1 + math.Exp(1))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USiluStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUPowfF32(t *testing.T) {
	tests := []struct {
		name  string
		param float32
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic powf",
			param: 2.0,
			numel: 3,
			inp:   []float32{1, 2, 3},
			out:   make([]float32, 3),
			want:  []float32{1, 4, 9},
		},
		{
			name:  "In-place powf",
			param: 2.0,
			numel: 3,
			inp:   nil,
			out:   []float32{1, 2, 3},
			want:  []float32{1, 4, 9},
		},
		{
			name:  "Empty",
			param: 2.0,
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UPowfF32(tt.param, tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUPowfStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		param   float32
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			param:   2.0,
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{1, 2, 3},
			out:     make([]float32, 3),
			want:    []float32{1, 4, 9},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			param:   2.0,
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{1, 0, 2, 0},
			out:     make([]float32, 2),
			want:    []float32{1, 4},
		},
		{
			name:    "In-place powf",
			param:   2.0,
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{1, 2, 3},
			want:    []float32{1, 4, 9},
		},
		{
			name:    "Empty",
			param:   2.0,
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UPowfStridedF32(tt.param, tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUPowfF64(t *testing.T) {
	tests := []struct {
		name  string
		param float64
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic powf",
			param: 2.0,
			numel: 3,
			inp:   []float64{1, 2, 3},
			out:   make([]float64, 3),
			want:  []float64{1, 4, 9},
		},
		{
			name:  "In-place powf",
			param: 2.0,
			numel: 3,
			inp:   nil,
			out:   []float64{1, 2, 3},
			want:  []float64{1, 4, 9},
		},
		{
			name:  "Empty",
			param: 2.0,
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UPowfF64(tt.param, tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUPowfStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		param   float64
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			param:   2.0,
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{1, 2, 3},
			out:     make([]float64, 3),
			want:    []float64{1, 4, 9},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			param:   2.0,
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{1, 0, 2, 0},
			out:     make([]float64, 2),
			want:    []float64{1, 4},
		},
		{
			name:    "In-place powf",
			param:   2.0,
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{1, 2, 3},
			want:    []float64{1, 4, 9},
		},
		{
			name:    "Empty",
			param:   2.0,
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.UPowfStridedF64(tt.param, tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSignF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic sign",
			numel: 4,
			inp:   []float32{-2, -1, 0, 1},
			out:   make([]float32, 4),
			want:  []float32{-1, -1, 0, 1},
		},
		{
			name:  "In-place sign",
			numel: 4,
			inp:   nil,
			out:   []float32{-2, -1, 0, 1},
			want:  []float32{-1, -1, 0, 1},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USignF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSignStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float32{-2, -1, 0, 1},
			out:     make([]float32, 4),
			want:    []float32{-1, -1, 0, 1},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{-1, 0, 1, 0},
			out:     make([]float32, 2),
			want:    []float32{-1, 1},
		},
		{
			name:    "In-place sign",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float32{-2, -1, 0, 1},
			want:    []float32{-1, -1, 0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USignStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSignF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic sign",
			numel: 4,
			inp:   []float64{-2, -1, 0, 1},
			out:   make([]float64, 4),
			want:  []float64{-1, -1, 0, 1},
		},
		{
			name:  "In-place sign",
			numel: 4,
			inp:   nil,
			out:   []float64{-2, -1, 0, 1},
			want:  []float64{-1, -1, 0, 1},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USignF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSignStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     []float64{-2, -1, 0, 1},
			out:     make([]float64, 4),
			want:    []float64{-1, -1, 0, 1},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{-1, 0, 1, 0},
			out:     make([]float64, 2),
			want:    []float64{-1, 1},
		},
		{
			name:    "In-place sign",
			numel:   4,
			numDims: 1,
			dims:    []int{4},
			strides: []int{1},
			inp:     nil,
			out:     []float64{-2, -1, 0, 1},
			want:    []float64{-1, -1, 0, 1},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USignStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSigmoidF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float32
		out   []float32
		want  []float32
	}{
		{
			name:  "Basic sigmoid",
			numel: 3,
			inp:   []float32{0, 1, -1},
			out:   make([]float32, 3),
			want:  []float32{float32(1 / (1 + math.Exp(0))), float32(1 / (1 + math.Exp(-1))), float32(1 / (1 + math.Exp(1)))},
		},
		{
			name:  "In-place sigmoid",
			numel: 3,
			inp:   nil,
			out:   []float32{0, 1, -1},
			want:  []float32{float32(1 / (1 + math.Exp(0))), float32(1 / (1 + math.Exp(-1))), float32(1 / (1 + math.Exp(1)))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float32{},
			out:   make([]float32, 0),
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USigmoidF32(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSigmoidStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float32
		out     []float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float32{0, 1, -1},
			out:     make([]float32, 3),
			want:    []float32{float32(1 / (1 + math.Exp(0))), float32(1 / (1 + math.Exp(-1))), float32(1 / (1 + math.Exp(1)))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float32{0, 0, 1, 0},
			out:     make([]float32, 2),
			want:    []float32{float32(1 / (1 + math.Exp(0))), float32(1 / (1 + math.Exp(-1)))},
		},
		{
			name:    "In-place sigmoid",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float32{0, 1, -1},
			want:    []float32{float32(1 / (1 + math.Exp(0))), float32(1 / (1 + math.Exp(-1))), float32(1 / (1 + math.Exp(1)))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float32{},
			out:     make([]float32, 0),
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float32, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USigmoidStridedF32(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSigmoidF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		inp   []float64
		out   []float64
		want  []float64
	}{
		{
			name:  "Basic sigmoid",
			numel: 3,
			inp:   []float64{0, 1, -1},
			out:   make([]float64, 3),
			want:  []float64{1 / (1 + math.Exp(0)), 1 / (1 + math.Exp(-1)), 1 / (1 + math.Exp(1))},
		},
		{
			name:  "In-place sigmoid",
			numel: 3,
			inp:   nil,
			out:   []float64{0, 1, -1},
			want:  []float64{1 / (1 + math.Exp(0)), 1 / (1 + math.Exp(-1)), 1 / (1 + math.Exp(1))},
		},
		{
			name:  "Empty",
			numel: 0,
			inp:   []float64{},
			out:   make([]float64, 0),
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USigmoidF64(tt.numel, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

func TestUSigmoidStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		numDims int
		dims    []int
		strides []int
		inp     []float64
		out     []float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     []float64{0, 1, -1},
			out:     make([]float64, 3),
			want:    []float64{1 / (1 + math.Exp(0)), 1 / (1 + math.Exp(-1)), 1 / (1 + math.Exp(1))},
		},
		{
			name:    "Non-contiguous 2D (strided inp)",
			numel:   2,
			numDims: 2,
			dims:    []int{2, 1},
			strides: []int{2, 1},
			inp:     []float64{0, 0, 1, 0},
			out:     make([]float64, 2),
			want:    []float64{1 / (1 + math.Exp(0)), 1 / (1 + math.Exp(-1))},
		},
		{
			name:    "In-place sigmoid",
			numel:   3,
			numDims: 1,
			dims:    []int{3},
			strides: []int{1},
			inp:     nil,
			out:     []float64{0, 1, -1},
			want:    []float64{1 / (1 + math.Exp(0)), 1 / (1 + math.Exp(-1)), 1 / (1 + math.Exp(1))},
		},
		{
			name:    "Empty",
			numel:   0,
			numDims: 0,
			dims:    []int{},
			strides: []int{},
			inp:     []float64{},
			out:     make([]float64, 0),
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outCopy := make([]float64, len(tt.out))
			copy(outCopy, tt.out)
			kernels.USigmoidStridedF64(tt.numel, tt.numDims, tt.dims, tt.strides, tt.inp, outCopy)
			if !slices.EqualFunc(outCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", outCopy, tt.want)
			}
		})
	}
}

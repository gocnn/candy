package kernels_test

import (
	"math"
	"slices"
	"testing"

	"github.com/gocnn/spark/internal/cpu/kernels"
)

func TestAsortAscF32(t *testing.T) {
	tests := []struct {
		name        string
		ncols       int
		src         []float32
		dst         []float32
		want        []float32
		wantIndices []uint32
	}{
		{
			name:        "Basic ascending sort 1 row",
			ncols:       3,
			src:         []float32{3, 1, 2},
			dst:         make([]float32, 3),
			want:        []float32{1, 2, 0},
			wantIndices: []uint32{1, 2, 0},
		},
		{
			name:        "Ascending sort 2 rows",
			ncols:       2,
			src:         []float32{4, 2, 1, 3},
			dst:         make([]float32, 4),
			want:        []float32{1, 0, 0, 1},
			wantIndices: []uint32{1, 0, 0, 1},
		},
		{
			name:        "Empty",
			ncols:       0,
			src:         []float32{},
			dst:         make([]float32, 0),
			want:        []float32{},
			wantIndices: []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]float32, len(tt.dst))
			copy(dstCopy, tt.dst)
			gotIndices := kernels.AsortAscF32(tt.ncols, tt.src, dstCopy)
			if !slices.EqualFunc(dstCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
			if !slices.Equal(gotIndices, tt.wantIndices) {
				t.Errorf("indices: got %v, want %v", gotIndices, tt.wantIndices)
			}
		})
	}
}

func TestAsortAscStridedF32(t *testing.T) {
	tests := []struct {
		name        string
		ncols       int
		numDims     int
		dims        []int
		strides     []int
		src         []float32
		dst         []float32
		want        []float32
		wantIndices []uint32
	}{
		{
			name:        "Contiguous 1D",
			ncols:       3,
			numDims:     1,
			dims:        []int{3},
			strides:     []int{1},
			src:         []float32{3, 1, 2},
			dst:         make([]float32, 3),
			want:        []float32{1, 2, 0},
			wantIndices: []uint32{1, 2, 0},
		},
		{
			name:        "Non-contiguous 2D (strided src)",
			ncols:       2,
			numDims:     2,
			dims:        []int{2, 2},
			strides:     []int{1, 2},
			src:         []float32{4, 2, 1, 3},
			dst:         make([]float32, 4),
			want:        []float32{1, 0, 0, 1},
			wantIndices: []uint32{1, 0, 0, 1},
		},
		{
			name:        "Empty",
			ncols:       0,
			numDims:     0,
			dims:        []int{},
			strides:     []int{},
			src:         []float32{},
			dst:         make([]float32, 0),
			want:        []float32{},
			wantIndices: []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]float32, len(tt.dst))
			copy(dstCopy, tt.dst)
			gotIndices := kernels.AsortAscStridedF32(tt.ncols, tt.numDims, tt.dims, tt.strides, tt.src, dstCopy)
			if !slices.EqualFunc(dstCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
			if !slices.Equal(gotIndices, tt.wantIndices) {
				t.Errorf("indices: got %v, want %v", gotIndices, tt.wantIndices)
			}
		})
	}
}

func TestAsortAscF64(t *testing.T) {
	tests := []struct {
		name        string
		ncols       int
		src         []float64
		dst         []float64
		want        []float64
		wantIndices []uint32
	}{
		{
			name:        "Basic ascending sort 1 row",
			ncols:       3,
			src:         []float64{3, 1, 2},
			dst:         make([]float64, 3),
			want:        []float64{1, 2, 0},
			wantIndices: []uint32{1, 2, 0},
		},
		{
			name:        "Ascending sort 2 rows",
			ncols:       2,
			src:         []float64{4, 2, 1, 3},
			dst:         make([]float64, 4),
			want:        []float64{1, 0, 0, 1},
			wantIndices: []uint32{1, 0, 0, 1},
		},
		{
			name:        "Empty",
			ncols:       0,
			src:         []float64{},
			dst:         make([]float64, 0),
			want:        []float64{},
			wantIndices: []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]float64, len(tt.dst))
			copy(dstCopy, tt.dst)
			gotIndices := kernels.AsortAscF64(tt.ncols, tt.src, dstCopy)
			if !slices.EqualFunc(dstCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
			if !slices.Equal(gotIndices, tt.wantIndices) {
				t.Errorf("indices: got %v, want %v", gotIndices, tt.wantIndices)
			}
		})
	}
}

func TestAsortAscStridedF64(t *testing.T) {
	tests := []struct {
		name        string
		ncols       int
		numDims     int
		dims        []int
		strides     []int
		src         []float64
		dst         []float64
		want        []float64
		wantIndices []uint32
	}{
		{
			name:        "Contiguous 1D",
			ncols:       3,
			numDims:     1,
			dims:        []int{3},
			strides:     []int{1},
			src:         []float64{3, 1, 2},
			dst:         make([]float64, 3),
			want:        []float64{1, 2, 0},
			wantIndices: []uint32{1, 2, 0},
		},
		{
			name:        "Non-contiguous 2D (strided src)",
			ncols:       2,
			numDims:     2,
			dims:        []int{2, 2},
			strides:     []int{1, 2},
			src:         []float64{4, 2, 1, 3},
			dst:         make([]float64, 4),
			want:        []float64{1, 0, 0, 1},
			wantIndices: []uint32{1, 0, 0, 1},
		},
		{
			name:        "Empty",
			ncols:       0,
			numDims:     0,
			dims:        []int{},
			strides:     []int{},
			src:         []float64{},
			dst:         make([]float64, 0),
			want:        []float64{},
			wantIndices: []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]float64, len(tt.dst))
			copy(dstCopy, tt.dst)
			gotIndices := kernels.AsortAscStridedF64(tt.ncols, tt.numDims, tt.dims, tt.strides, tt.src, dstCopy)
			if !slices.EqualFunc(dstCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
			if !slices.Equal(gotIndices, tt.wantIndices) {
				t.Errorf("indices: got %v, want %v", gotIndices, tt.wantIndices)
			}
		})
	}
}

func TestAsortDescF32(t *testing.T) {
	tests := []struct {
		name        string
		ncols       int
		src         []float32
		dst         []float32
		want        []float32
		wantIndices []uint32
	}{
		{
			name:        "Basic descending sort 1 row",
			ncols:       3,
			src:         []float32{1, 3, 2},
			dst:         make([]float32, 3),
			want:        []float32{1, 2, 0},
			wantIndices: []uint32{1, 2, 0},
		},
		{
			name:        "Descending sort 2 rows",
			ncols:       2,
			src:         []float32{2, 4, 3, 1},
			dst:         make([]float32, 4),
			want:        []float32{1, 0, 0, 1},
			wantIndices: []uint32{1, 0, 0, 1},
		},
		{
			name:        "Empty",
			ncols:       0,
			src:         []float32{},
			dst:         make([]float32, 0),
			want:        []float32{},
			wantIndices: []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]float32, len(tt.dst))
			copy(dstCopy, tt.dst)
			gotIndices := kernels.AsortDescF32(tt.ncols, tt.src, dstCopy)
			if !slices.EqualFunc(dstCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
			if !slices.Equal(gotIndices, tt.wantIndices) {
				t.Errorf("indices: got %v, want %v", gotIndices, tt.wantIndices)
			}
		})
	}
}

func TestAsortDescStridedF32(t *testing.T) {
	tests := []struct {
		name        string
		ncols       int
		numDims     int
		dims        []int
		strides     []int
		src         []float32
		dst         []float32
		want        []float32
		wantIndices []uint32
	}{
		{
			name:        "Contiguous 1D",
			ncols:       3,
			numDims:     1,
			dims:        []int{3},
			strides:     []int{1},
			src:         []float32{1, 3, 2},
			dst:         make([]float32, 3),
			want:        []float32{1, 2, 0},
			wantIndices: []uint32{1, 2, 0},
		},
		{
			name:        "Non-contiguous 2D (strided src)",
			ncols:       2,
			numDims:     2,
			dims:        []int{2, 2},
			strides:     []int{1, 2},
			src:         []float32{4, 2, 1, 3},
			dst:         make([]float32, 4),
			want:        []float32{0, 1, 1, 0},
			wantIndices: []uint32{0, 1, 1, 0},
		},
		{
			name:        "Empty",
			ncols:       0,
			numDims:     0,
			dims:        []int{},
			strides:     []int{},
			src:         []float32{},
			dst:         make([]float32, 0),
			want:        []float32{},
			wantIndices: []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]float32, len(tt.dst))
			copy(dstCopy, tt.dst)
			gotIndices := kernels.AsortDescStridedF32(tt.ncols, tt.numDims, tt.dims, tt.strides, tt.src, dstCopy)
			if !slices.EqualFunc(dstCopy, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
			if !slices.Equal(gotIndices, tt.wantIndices) {
				t.Errorf("indices: got %v, want %v", gotIndices, tt.wantIndices)
			}
		})
	}
}

func TestAsortDescF64(t *testing.T) {
	tests := []struct {
		name        string
		ncols       int
		src         []float64
		dst         []float64
		want        []float64
		wantIndices []uint32
	}{
		{
			name:        "Basic descending sort 1 row",
			ncols:       3,
			src:         []float64{1, 3, 2},
			dst:         make([]float64, 3),
			want:        []float64{1, 2, 0},
			wantIndices: []uint32{1, 2, 0},
		},
		{
			name:        "Descending sort 2 rows",
			ncols:       2,
			src:         []float64{2, 4, 3, 1},
			dst:         make([]float64, 4),
			want:        []float64{1, 0, 0, 1},
			wantIndices: []uint32{1, 0, 0, 1},
		},
		{
			name:        "Empty",
			ncols:       0,
			src:         []float64{},
			dst:         make([]float64, 0),
			want:        []float64{},
			wantIndices: []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]float64, len(tt.dst))
			copy(dstCopy, tt.dst)
			gotIndices := kernels.AsortDescF64(tt.ncols, tt.src, dstCopy)
			if !slices.EqualFunc(dstCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
			if !slices.Equal(gotIndices, tt.wantIndices) {
				t.Errorf("indices: got %v, want %v", gotIndices, tt.wantIndices)
			}
		})
	}
}

func TestAsortDescStridedF64(t *testing.T) {
	tests := []struct {
		name        string
		ncols       int
		numDims     int
		dims        []int
		strides     []int
		src         []float64
		dst         []float64
		want        []float64
		wantIndices []uint32
	}{
		{
			name:        "Contiguous 1D",
			ncols:       3,
			numDims:     1,
			dims:        []int{3},
			strides:     []int{1},
			src:         []float64{1, 3, 2},
			dst:         make([]float64, 3),
			want:        []float64{1, 2, 0},
			wantIndices: []uint32{1, 2, 0},
		},
		{
			name:        "Non-contiguous 2D (strided src)",
			ncols:       2,
			numDims:     2,
			dims:        []int{2, 2},
			strides:     []int{1, 2},
			src:         []float64{4, 2, 1, 3},
			dst:         make([]float64, 4),
			want:        []float64{0, 1, 1, 0},
			wantIndices: []uint32{0, 1, 1, 0},
		},
		{
			name:        "Empty",
			ncols:       0,
			numDims:     0,
			dims:        []int{},
			strides:     []int{},
			src:         []float64{},
			dst:         make([]float64, 0),
			want:        []float64{},
			wantIndices: []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]float64, len(tt.dst))
			copy(dstCopy, tt.dst)
			gotIndices := kernels.AsortDescStridedF64(tt.ncols, tt.numDims, tt.dims, tt.strides, tt.src, dstCopy)
			if !slices.EqualFunc(dstCopy, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
			if !slices.Equal(gotIndices, tt.wantIndices) {
				t.Errorf("indices: got %v, want %v", gotIndices, tt.wantIndices)
			}
		})
	}
}

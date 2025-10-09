package kernels_test

import (
	"math"
	"testing"

	"slices"

	"github.com/gocnn/spark/internal/cpu/kernels"
)

func TestIndexSelectI64F32(t *testing.T) {
	tests := []struct {
		name                                        string
		numel                                       int
		ids                                         []int64
		inp, want                                   []float32
		leftSize, srcDimSize, idsDimSize, rightSize int
	}{
		{
			name:       "Basic select",
			numel:      3,
			ids:        []int64{1, 3, 0},
			inp:        []float32{10, 20, 30, 40, 50},
			want:       []float32{20, 40, 10},
			leftSize:   1,
			srcDimSize: 5,
			idsDimSize: 3,
			rightSize:  1,
		},
		{
			name:       "With max value",
			numel:      3,
			ids:        []int64{1, 9223372036854775807, 0},
			inp:        []float32{10, 20, 30, 40, 50},
			want:       []float32{20, 0, 10},
			leftSize:   1,
			srcDimSize: 5,
			idsDimSize: 3,
			rightSize:  1,
		},
		{
			name:       "Multi dim",
			numel:      12,
			ids:        []int64{0, 2, 1},
			inp:        []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			want:       []float32{0, 1, 4, 5, 2, 3, 8, 9, 12, 13, 10, 11},
			leftSize:   2,
			srcDimSize: 4,
			idsDimSize: 3,
			rightSize:  2,
		},
		{
			name:       "Empty",
			numel:      0,
			ids:        []int64{},
			inp:        []float32{},
			want:       []float32{},
			leftSize:   1,
			srcDimSize: 0,
			idsDimSize: 0,
			rightSize:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := make([]float32, tt.numel)
			kernels.IndexSelectI64F32(tt.numel, tt.ids, tt.inp, out, tt.leftSize, tt.srcDimSize, tt.idsDimSize, tt.rightSize)
			if !slices.EqualFunc(out, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", out, tt.want)
			}
		})
	}
}

func TestIndexSelectStridedI64F32(t *testing.T) {
	tests := []struct {
		name                                        string
		numel, ndims                                int
		dims, strides                               []int
		ids                                         []int64
		inp, want                                   []float32
		leftSize, srcDimSize, idsDimSize, rightSize int
	}{
		{
			name:       "Basic select",
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			strides:    []int{1},
			ids:        []int64{1, 3, 0},
			inp:        []float32{10, 20, 30, 40, 50},
			want:       []float32{20, 40, 10},
			leftSize:   1,
			srcDimSize: 5,
			idsDimSize: 3,
			rightSize:  1,
		},
		{
			name:       "With max value",
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			strides:    []int{1},
			ids:        []int64{1, 9223372036854775807, 0},
			inp:        []float32{10, 20, 30, 40, 50},
			want:       []float32{20, 0, 10},
			leftSize:   1,
			srcDimSize: 5,
			idsDimSize: 3,
			rightSize:  1,
		},
		{
			name:       "Multi dim",
			numel:      12,
			ndims:      3,
			dims:       []int{2, 4, 2},
			strides:    []int{8, 2, 1},
			ids:        []int64{0, 2, 1},
			inp:        []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			want:       []float32{0, 1, 4, 5, 2, 3, 8, 9, 12, 13, 10, 11},
			leftSize:   2,
			srcDimSize: 4,
			idsDimSize: 3,
			rightSize:  2,
		},
		{
			name:       "Non contiguous",
			numel:      2,
			ndims:      1,
			dims:       []int{3},
			strides:    []int{2},
			ids:        []int64{0, 2},
			inp:        []float32{10, 99, 20, 99, 30},
			want:       []float32{10, 30},
			leftSize:   1,
			srcDimSize: 3,
			idsDimSize: 2,
			rightSize:  1,
		},
		{
			name:       "Empty",
			numel:      0,
			ndims:      1,
			dims:       []int{0},
			strides:    []int{1},
			ids:        []int64{},
			inp:        []float32{},
			want:       []float32{},
			leftSize:   1,
			srcDimSize: 0,
			idsDimSize: 0,
			rightSize:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := make([]float32, tt.numel)
			kernels.IndexSelectStridedI64F32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.ids, tt.inp, out, tt.leftSize, tt.srcDimSize, tt.idsDimSize, tt.rightSize)
			if !slices.EqualFunc(out, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", out, tt.want)
			}
		})
	}
}

func TestGatherI64F32(t *testing.T) {
	tests := []struct {
		name                                        string
		numel                                       int
		ids                                         []int64
		inp, want                                   []float32
		leftSize, srcDimSize, idsDimSize, rightSize int
	}{
		{
			name:       "Basic gather",
			numel:      3,
			ids:        []int64{1, 2, 0},
			inp:        []float32{10, 20, 30, 40, 50},
			want:       []float32{20, 30, 10},
			leftSize:   1,
			srcDimSize: 5,
			idsDimSize: 3,
			rightSize:  1,
		},
		{
			name:       "With max value",
			numel:      3,
			ids:        []int64{1, 9223372036854775807, 0},
			inp:        []float32{10, 20, 30, 40, 50},
			want:       []float32{20, 0, 10},
			leftSize:   1,
			srcDimSize: 5,
			idsDimSize: 3,
			rightSize:  1,
		},
		{
			name:       "Multi dim",
			numel:      12,
			ids:        []int64{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
			inp:        []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			want:       []float32{0, 3, 4, 7, 0, 3, 12, 15, 8, 11, 12, 15},
			leftSize:   2,
			srcDimSize: 4,
			idsDimSize: 3,
			rightSize:  2,
		},
		{
			name:       "Empty",
			numel:      0,
			ids:        []int64{},
			inp:        []float32{},
			want:       []float32{},
			leftSize:   1,
			srcDimSize: 0,
			idsDimSize: 0,
			rightSize:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := make([]float32, tt.numel)
			kernels.GatherI64F32(tt.numel, tt.ids, tt.inp, out, tt.leftSize, tt.srcDimSize, tt.idsDimSize, tt.rightSize)
			if !slices.EqualFunc(out, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", out, tt.want)
			}
		})
	}
}

func TestIndexAddI64F32(t *testing.T) {
	tests := []struct {
		name                                        string
		ids                                         []int64
		inp, want                                   []float32
		leftSize, idsDimSize, dstDimSize, rightSize int
	}{
		{
			name:       "Basic add",
			ids:        []int64{1, 3, 0},
			inp:        []float32{2, 4, 1},
			want:       []float32{1, 2, 0, 4, 0},
			leftSize:   1,
			idsDimSize: 3,
			dstDimSize: 5,
			rightSize:  1,
		},
		{
			name:       "With max value skip",
			ids:        []int64{1, 9223372036854775807, 0},
			inp:        []float32{2, 4, 1},
			want:       []float32{1, 2, 0, 0, 0},
			leftSize:   1,
			idsDimSize: 3,
			dstDimSize: 5,
			rightSize:  1,
		},
		{
			name:       "Multi dim",
			ids:        []int64{0, 2, 1},
			inp:        []float32{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1},
			want:       []float32{0, 0.1, 0.4, 0.5, 0.2, 0.3, 0, 0, 0, 0, 0.6, 0.7, 1, 1.1, 0.8, 0.9, 0, 0, 0, 0},
			leftSize:   2,
			idsDimSize: 3,
			dstDimSize: 5,
			rightSize:  2,
		},
		{
			name:       "Empty",
			ids:        []int64{},
			inp:        []float32{},
			want:       []float32{},
			leftSize:   1,
			idsDimSize: 0,
			dstDimSize: 0,
			rightSize:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outLen := tt.leftSize * tt.dstDimSize * tt.rightSize
			out := make([]float32, outLen)
			kernels.IndexAddI64F32(tt.leftSize, tt.idsDimSize, tt.inp, out, tt.dstDimSize, tt.rightSize, tt.ids)
			if !slices.EqualFunc(out, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", out, tt.want)
			}
		})
	}
}

func TestScatterI64F32(t *testing.T) {
	tests := []struct {
		name                                        string
		ids                                         []int64
		inp, want                                   []float32
		leftSize, srcDimSize, dstDimSize, rightSize int
	}{
		{
			name:       "Basic scatter",
			ids:        []int64{1, 3, 0},
			inp:        []float32{2, 4, 1},
			want:       []float32{1, 2, 0, 4, 0},
			leftSize:   1,
			srcDimSize: 3,
			dstDimSize: 5,
			rightSize:  1,
		},
		{
			name:       "With max value skip",
			ids:        []int64{1, 9223372036854775807, 0},
			inp:        []float32{2, 4, 1},
			want:       []float32{1, 2, 0, 0, 0},
			leftSize:   1,
			srcDimSize: 3,
			dstDimSize: 5,
			rightSize:  1,
		},
		{
			name:       "With conflict overwrite",
			ids:        []int64{0, 2, 0},
			inp:        []float32{1, 3, 5},
			want:       []float32{5, 0, 3, 0, 0},
			leftSize:   1,
			srcDimSize: 3,
			dstDimSize: 5,
			rightSize:  1,
		},
		{
			name:       "Multi dim",
			ids:        []int64{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
			inp:        []float32{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1},
			want:       []float32{0.4, 0, 0, 0.5, 0.2, 0, 0, 0.3, 0.8, 0, 0, 0.9, 1, 0, 0, 1.1},
			leftSize:   2,
			srcDimSize: 3,
			dstDimSize: 4,
			rightSize:  2,
		},
		{
			name:       "Empty",
			ids:        []int64{},
			inp:        []float32{},
			want:       []float32{},
			leftSize:   1,
			srcDimSize: 0,
			dstDimSize: 0,
			rightSize:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outLen := tt.leftSize * tt.dstDimSize * tt.rightSize
			out := make([]float32, outLen)
			kernels.ScatterI64F32(tt.leftSize, tt.srcDimSize, tt.dstDimSize, tt.rightSize, tt.ids, tt.inp, out)
			if !slices.EqualFunc(out, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", out, tt.want)
			}
		})
	}
}

func TestScatterAddI64F32(t *testing.T) {
	tests := []struct {
		name                                        string
		ids                                         []int64
		inp, want                                   []float32
		leftSize, srcDimSize, dstDimSize, rightSize int
	}{
		{
			name:       "Basic scatter add",
			ids:        []int64{1, 3, 0},
			inp:        []float32{2, 4, 1},
			want:       []float32{1, 2, 0, 4, 0},
			leftSize:   1,
			srcDimSize: 3,
			dstDimSize: 5,
			rightSize:  1,
		},
		{
			name:       "With max value skip",
			ids:        []int64{1, 9223372036854775807, 0},
			inp:        []float32{2, 4, 1},
			want:       []float32{1, 2, 0, 0, 0},
			leftSize:   1,
			srcDimSize: 3,
			dstDimSize: 5,
			rightSize:  1,
		},
		{
			name:       "With conflict add",
			ids:        []int64{0, 2, 0},
			inp:        []float32{1, 3, 5},
			want:       []float32{6, 0, 3, 0, 0},
			leftSize:   1,
			srcDimSize: 3,
			dstDimSize: 5,
			rightSize:  1,
		},
		{
			name:       "Multi dim",
			ids:        []int64{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
			inp:        []float32{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1},
			want:       []float32{0.4, 0, 0, 0.6, 0.2, 0, 0, 0.3, 0.8, 0, 0, 0.9, 1.6, 0, 0, 1.8},
			leftSize:   2,
			srcDimSize: 3,
			dstDimSize: 4,
			rightSize:  2,
		},
		{
			name:       "Empty",
			ids:        []int64{},
			inp:        []float32{},
			want:       []float32{},
			leftSize:   1,
			srcDimSize: 0,
			dstDimSize: 0,
			rightSize:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outLen := tt.leftSize * tt.dstDimSize * tt.rightSize
			out := make([]float32, outLen)
			kernels.ScatterAddI64F32(tt.leftSize, tt.srcDimSize, tt.dstDimSize, tt.rightSize, tt.ids, tt.inp, out)
			if !slices.EqualFunc(out, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", out, tt.want)
			}
		})
	}
}

func TestIndexSelectI64I64(t *testing.T) {
	tests := []struct {
		name                                        string
		numel                                       int
		ids                                         []int64
		inp, want                                   []int64
		leftSize, srcDimSize, idsDimSize, rightSize int
	}{
		{
			name:       "Basic select",
			numel:      3,
			ids:        []int64{1, 3, 0},
			inp:        []int64{10, 20, 30, 40, 50},
			want:       []int64{20, 40, 10},
			leftSize:   1,
			srcDimSize: 5,
			idsDimSize: 3,
			rightSize:  1,
		},
		{
			name:       "With max value",
			numel:      3,
			ids:        []int64{1, 9223372036854775807, 0},
			inp:        []int64{10, 20, 30, 40, 50},
			want:       []int64{20, 0, 10},
			leftSize:   1,
			srcDimSize: 5,
			idsDimSize: 3,
			rightSize:  1,
		},
		{
			name:       "Multi dim",
			numel:      12,
			ids:        []int64{0, 2, 1},
			inp:        []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			want:       []int64{0, 1, 4, 5, 2, 3, 8, 9, 12, 13, 10, 11},
			leftSize:   2,
			srcDimSize: 4,
			idsDimSize: 3,
			rightSize:  2,
		},
		{
			name:       "Empty",
			numel:      0,
			ids:        []int64{},
			inp:        []int64{},
			want:       []int64{},
			leftSize:   1,
			srcDimSize: 0,
			idsDimSize: 0,
			rightSize:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := make([]int64, tt.numel)
			kernels.IndexSelectI64I64(tt.numel, tt.ids, tt.inp, out, tt.leftSize, tt.srcDimSize, tt.idsDimSize, tt.rightSize)
			if !slices.Equal(out, tt.want) {
				t.Errorf("got %v, want %v", out, tt.want)
			}
		})
	}
}

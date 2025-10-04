package kernels_test

import (
	"math"
	"testing"

	"slices"

	"github.com/gocnn/spark/internal/cpu/kernels"
)

func TestIndexSelectI64F32(t *testing.T) {
	tests := []struct {
		name  string
		dim   int
		numel int
		ids   []int64
		src   []float32
		want  []float32
	}{
		{
			name:  "Basic select",
			numel: 3,
			ids:   []int64{1, 3, 0},
			src:   []float32{10, 20, 30, 40, 50},
			want:  []float32{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ids:   []int64{2, 2},
			src:   []float32{1, 2, 3},
			want:  []float32{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []int64{},
			src:   []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.IndexSelectI64F32(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIndexSelectI64F64(t *testing.T) {
	tests := []struct {
		name  string
		dim   int
		numel int
		ids   []int64
		src   []float64
		want  []float64
	}{
		{
			name:  "Basic select",
			numel: 3,
			ids:   []int64{1, 3, 0},
			src:   []float64{10, 20, 30, 40, 50},
			want:  []float64{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ids:   []int64{2, 2},
			src:   []float64{1, 2, 3},
			want:  []float64{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []int64{},
			src:   []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.IndexSelectI64F64(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIndexSelectU32F32(t *testing.T) {
	tests := []struct {
		name  string
		dim   int
		numel int
		ids   []uint32
		src   []float32
		want  []float32
	}{
		{
			name:  "Basic select",
			numel: 3,
			ids:   []uint32{1, 3, 0},
			src:   []float32{10, 20, 30, 40, 50},
			want:  []float32{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ids:   []uint32{2, 2},
			src:   []float32{1, 2, 3},
			want:  []float32{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []uint32{},
			src:   []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.IndexSelectU32F32(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIndexSelectU32F64(t *testing.T) {
	tests := []struct {
		name  string
		dim   int
		numel int
		ids   []uint32
		src   []float64
		want  []float64
	}{
		{
			name:  "Basic select",
			numel: 3,
			ids:   []uint32{1, 3, 0},
			src:   []float64{10, 20, 30, 40, 50},
			want:  []float64{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ids:   []uint32{2, 2},
			src:   []float64{1, 2, 3},
			want:  []float64{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []uint32{},
			src:   []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.IndexSelectU32F64(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIndexSelectU8F32(t *testing.T) {
	tests := []struct {
		name  string
		dim   int
		numel int
		ids   []uint8
		src   []float32
		want  []float32
	}{
		{
			name:  "Basic select",
			numel: 3,
			ids:   []uint8{1, 3, 0},
			src:   []float32{10, 20, 30, 40, 50},
			want:  []float32{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ids:   []uint8{2, 2},
			src:   []float32{1, 2, 3},
			want:  []float32{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []uint8{},
			src:   []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.IndexSelectU8F32(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIndexSelectU8F64(t *testing.T) {
	tests := []struct {
		name  string
		dim   int
		numel int
		ids   []uint8
		src   []float64
		want  []float64
	}{
		{
			name:  "Basic select",
			numel: 3,
			ids:   []uint8{1, 3, 0},
			src:   []float64{10, 20, 30, 40, 50},
			want:  []float64{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ids:   []uint8{2, 2},
			src:   []float64{1, 2, 3},
			want:  []float64{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []uint8{},
			src:   []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.IndexSelectU8F64(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIndexSelectStridedI64F32(t *testing.T) {
	tests := []struct {
		name       string
		dim        int
		numel      int
		ndims      int
		dims       []int
		stridesSrc []int
		stridesDst []int
		ids        []int64
		src        []float32
		want       []float32
	}{
		{
			name:       "Contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{1},
			ids:        []int64{1, 3, 0},
			src:        []float32{10, 20, 30, 40, 50},
			want:       []float32{20, 40, 10},
		},
		{
			name:       "Non-contiguous src and dst",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{2},
			stridesDst: []int{1},
			ids:        []int64{0, 1, 2},
			src:        []float32{10, 0, 20, 0, 30, 0, 40, 0, 50},
			want:       []float32{10, 20, 30},
		},
		{
			name:       "Empty",
			dim:        0,
			numel:      0,
			ndims:      0,
			dims:       []int{},
			stridesSrc: []int{},
			stridesDst: []int{},
			ids:        []int64{},
			src:        []float32{},
			want:       []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.IndexSelectStridedI64F32(tt.numel, tt.ndims, tt.dims, tt.stridesSrc, tt.stridesDst, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIndexSelectStridedI64F64(t *testing.T) {
	tests := []struct {
		name       string
		dim        int
		numel      int
		ndims      int
		dims       []int
		stridesSrc []int
		stridesDst []int
		ids        []int64
		src        []float64
		want       []float64
	}{
		{
			name:       "Contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{1},
			ids:        []int64{1, 3, 0},
			src:        []float64{10, 20, 30, 40, 50},
			want:       []float64{20, 40, 10},
		},
		{
			name:       "Non-contiguous src and dst",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{2},
			stridesDst: []int{1},
			ids:        []int64{0, 1, 2},
			src:        []float64{10, 0, 20, 0, 30, 0, 40, 0, 50},
			want:       []float64{10, 20, 30},
		},
		{
			name:       "Empty",
			dim:        0,
			numel:      0,
			ndims:      0,
			dims:       []int{},
			stridesSrc: []int{},
			stridesDst: []int{},
			ids:        []int64{},
			src:        []float64{},
			want:       []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.IndexSelectStridedI64F64(tt.numel, tt.ndims, tt.dims, tt.stridesSrc, tt.stridesDst, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIndexSelectStridedU32F32(t *testing.T) {
	tests := []struct {
		name       string
		dim        int
		numel      int
		ndims      int
		dims       []int
		stridesSrc []int
		stridesDst []int
		ids        []uint32
		src        []float32
		want       []float32
	}{
		{
			name:       "Contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{1},
			ids:        []uint32{1, 3, 0},
			src:        []float32{10, 20, 30, 40, 50},
			want:       []float32{20, 40, 10},
		},
		{
			name:       "Non-contiguous src and dst",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{2},
			stridesDst: []int{1},
			ids:        []uint32{0, 1, 2},
			src:        []float32{10, 0, 20, 0, 30, 0, 40, 0, 50},
			want:       []float32{10, 20, 30},
		},
		{
			name:       "Empty",
			dim:        0,
			numel:      0,
			ndims:      0,
			dims:       []int{},
			stridesSrc: []int{},
			stridesDst: []int{},
			ids:        []uint32{},
			src:        []float32{},
			want:       []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.IndexSelectStridedU32F32(tt.numel, tt.ndims, tt.dims, tt.stridesSrc, tt.stridesDst, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIndexSelectStridedU32F64(t *testing.T) {
	tests := []struct {
		name       string
		dim        int
		numel      int
		ndims      int
		dims       []int
		stridesSrc []int
		stridesDst []int
		ids        []uint32
		src        []float64
		want       []float64
	}{
		{
			name:       "Contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{1},
			ids:        []uint32{1, 3, 0},
			src:        []float64{10, 20, 30, 40, 50},
			want:       []float64{20, 40, 10},
		},
		{
			name:       "Non-contiguous src and dst",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{2},
			stridesDst: []int{1},
			ids:        []uint32{0, 1, 2},
			src:        []float64{10, 0, 20, 0, 30, 0, 40, 0, 50},
			want:       []float64{10, 20, 30},
		},
		{
			name:       "Empty",
			dim:        0,
			numel:      0,
			ndims:      0,
			dims:       []int{},
			stridesSrc: []int{},
			stridesDst: []int{},
			ids:        []uint32{},
			src:        []float64{},
			want:       []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.IndexSelectStridedU32F64(tt.numel, tt.ndims, tt.dims, tt.stridesSrc, tt.stridesDst, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIndexSelectStridedU8F32(t *testing.T) {
	tests := []struct {
		name       string
		dim        int
		numel      int
		ndims      int
		dims       []int
		stridesSrc []int
		stridesDst []int
		ids        []uint8
		src        []float32
		want       []float32
	}{
		{
			name:       "Contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{1},
			ids:        []uint8{1, 3, 0},
			src:        []float32{10, 20, 30, 40, 50},
			want:       []float32{20, 40, 10},
		},
		{
			name:       "Non-contiguous src and dst",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{2},
			stridesDst: []int{1},
			ids:        []uint8{0, 1, 2},
			src:        []float32{10, 0, 20, 0, 30, 0, 40, 0, 50},
			want:       []float32{10, 20, 30},
		},
		{
			name:       "Empty",
			dim:        0,
			numel:      0,
			ndims:      0,
			dims:       []int{},
			stridesSrc: []int{},
			stridesDst: []int{},
			ids:        []uint8{},
			src:        []float32{},
			want:       []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.IndexSelectStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesSrc, tt.stridesDst, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIndexSelectStridedU8F64(t *testing.T) {
	tests := []struct {
		name       string
		dim        int
		numel      int
		ndims      int
		dims       []int
		stridesSrc []int
		stridesDst []int
		ids        []uint8
		src        []float64
		want       []float64
	}{
		{
			name:       "Contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{1},
			ids:        []uint8{1, 3, 0},
			src:        []float64{10, 20, 30, 40, 50},
			want:       []float64{20, 40, 10},
		},
		{
			name:       "Non-contiguous src and dst",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{2},
			stridesDst: []int{1},
			ids:        []uint8{0, 1, 2},
			src:        []float64{10, 0, 20, 0, 30, 0, 40, 0, 50},
			want:       []float64{10, 20, 30},
		},
		{
			name:       "Empty",
			dim:        0,
			numel:      0,
			ndims:      0,
			dims:       []int{},
			stridesSrc: []int{},
			stridesDst: []int{},
			ids:        []uint8{},
			src:        []float64{},
			want:       []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.IndexSelectStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesSrc, tt.stridesDst, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestGatherI64F32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dim   int
		dims  []int
		ids   []int64
		src   []float32
		want  []float32
	}{
		{
			name:  "Basic gather",
			numel: 3,
			ndims: 1,
			dim:   0,
			dims:  []int{5},
			ids:   []int64{1, 3, 0},
			src:   []float32{10, 20, 30, 40, 50},
			want:  []float32{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ndims: 1,
			dim:   0,
			dims:  []int{3},
			ids:   []int64{2, 2},
			src:   []float32{1, 2, 3},
			want:  []float32{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 0,
			dim:   0,
			dims:  []int{},
			ids:   []int64{},
			src:   []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.GatherI64F32(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestGatherI64F64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dim   int
		dims  []int
		ids   []int64
		src   []float64
		want  []float64
	}{
		{
			name:  "Basic gather",
			numel: 3,
			ndims: 1,
			dim:   0,
			dims:  []int{5},
			ids:   []int64{1, 3, 0},
			src:   []float64{10, 20, 30, 40, 50},
			want:  []float64{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ndims: 1,
			dim:   0,
			dims:  []int{3},
			ids:   []int64{2, 2},
			src:   []float64{1, 2, 3},
			want:  []float64{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 0,
			dim:   0,
			dims:  []int{},
			ids:   []int64{},
			src:   []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.GatherI64F64(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestGatherU32F32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dim   int
		dims  []int
		ids   []uint32
		src   []float32
		want  []float32
	}{
		{
			name:  "Basic gather",
			numel: 3,
			ndims: 1,
			dim:   0,
			dims:  []int{5},
			ids:   []uint32{1, 3, 0},
			src:   []float32{10, 20, 30, 40, 50},
			want:  []float32{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ndims: 1,
			dim:   0,
			dims:  []int{3},
			ids:   []uint32{2, 2},
			src:   []float32{1, 2, 3},
			want:  []float32{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 0,
			dim:   0,
			dims:  []int{},
			ids:   []uint32{},
			src:   []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.GatherU32F32(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestGatherU32F64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dim   int
		dims  []int
		ids   []uint32
		src   []float64
		want  []float64
	}{
		{
			name:  "Basic gather",
			numel: 3,
			ndims: 1,
			dim:   0,
			dims:  []int{5},
			ids:   []uint32{1, 3, 0},
			src:   []float64{10, 20, 30, 40, 50},
			want:  []float64{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ndims: 1,
			dim:   0,
			dims:  []int{3},
			ids:   []uint32{2, 2},
			src:   []float64{1, 2, 3},
			want:  []float64{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 0,
			dim:   0,
			dims:  []int{},
			ids:   []uint32{},
			src:   []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.GatherU32F64(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestGatherU8F32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dim   int
		dims  []int
		ids   []uint8
		src   []float32
		want  []float32
	}{
		{
			name:  "Basic gather",
			numel: 3,
			ndims: 1,
			dim:   0,
			dims:  []int{5},
			ids:   []uint8{1, 3, 0},
			src:   []float32{10, 20, 30, 40, 50},
			want:  []float32{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ndims: 1,
			dim:   0,
			dims:  []int{3},
			ids:   []uint8{2, 2},
			src:   []float32{1, 2, 3},
			want:  []float32{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 0,
			dim:   0,
			dims:  []int{},
			ids:   []uint8{},
			src:   []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.GatherU8F32(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestGatherU8F64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ndims int
		dim   int
		dims  []int
		ids   []uint8
		src   []float64
		want  []float64
	}{
		{
			name:  "Basic gather",
			numel: 3,
			ndims: 1,
			dim:   0,
			dims:  []int{5},
			ids:   []uint8{1, 3, 0},
			src:   []float64{10, 20, 30, 40, 50},
			want:  []float64{20, 40, 10},
		},
		{
			name:  "All same index",
			numel: 2,
			ndims: 1,
			dim:   0,
			dims:  []int{3},
			ids:   []uint8{2, 2},
			src:   []float64{1, 2, 3},
			want:  []float64{3, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ndims: 0,
			dim:   0,
			dims:  []int{},
			ids:   []uint8{},
			src:   []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.GatherU8F64(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestGatherStridedI64F32(t *testing.T) {
	tests := []struct {
		name       string
		numel      int
		ndims      int
		dim        int
		dims       []int
		stridesSrc []int
		stridesDst []int
		stridesIds []int
		ids        []int64
		src        []float32
		want       []float32
	}{
		{
			name:       "Contiguous",
			numel:      3,
			ndims:      1,
			dim:        0,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{1},
			stridesIds: []int{1},
			ids:        []int64{1, 3, 0},
			src:        []float32{10, 20, 30, 40, 50},
			want:       []float32{20, 40, 10},
		},
		{
			name:       "Non-contiguous",
			numel:      3,
			ndims:      1,
			dim:        0,
			dims:       []int{5},
			stridesSrc: []int{2},
			stridesDst: []int{1},
			stridesIds: []int{1},
			ids:        []int64{0, 1, 2},
			src:        []float32{10, 0, 20, 0, 30, 0, 40, 0, 50},
			want:       []float32{10, 20, 30},
		},
		{
			name:       "Empty",
			numel:      0,
			ndims:      0,
			dim:        0,
			dims:       []int{},
			stridesSrc: []int{},
			stridesDst: []int{},
			stridesIds: []int{},
			ids:        []int64{},
			src:        []float32{},
			want:       []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.GatherStridedI64F32(tt.numel, tt.ndims, tt.dims, tt.stridesSrc, tt.stridesDst, tt.stridesIds, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

// Repeat for GatherStridedI64F64, U32F32, U32F64, U8F32, U8F64 with similar tests

func TestIndexAddI64F32(t *testing.T) {
	tests := []struct {
		name    string
		dim     int
		numel   int
		ids     []int64
		src     []float32
		initDst []float32
		want    []float32
	}{
		{
			name:    "Basic add",
			dim:     0,
			numel:   3,
			ids:     []int64{1, 3, 0},
			src:     []float32{10, 20, 30},
			initDst: []float32{1, 2, 3, 4, 5},
			want:    []float32{31, 12, 3, 24, 5},
		},
		{
			name:    "Add to same index",
			dim:     0,
			numel:   2,
			ids:     []int64{1, 1},
			src:     []float32{5, 10},
			initDst: []float32{0, 0, 0},
			want:    []float32{0, 15, 0},
		},
		{
			name:    "Empty",
			dim:     0,
			numel:   0,
			ids:     []int64{},
			src:     []float32{},
			initDst: []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.initDst))
			copy(dst, tt.initDst)
			kernels.IndexAddI64F32(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

// Repeat for IndexAddI64F64, U32F32, U32F64, U8F32, U8F64

func TestIndexAddStridedI64F32(t *testing.T) {
	tests := []struct {
		name       string
		dim        int
		numel      int
		ndims      int
		dims       []int
		stridesSrc []int
		stridesDst []int
		ids        []int64
		src        []float32
		initDst    []float32
		want       []float32
	}{
		{
			name:       "Contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{1},
			ids:        []int64{1, 3, 0},
			src:        []float32{10, 20, 30},
			initDst:    []float32{1, 2, 3, 4, 5},
			want:       []float32{31, 12, 3, 24, 5},
		},
		{
			name:       "Non-contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{2},
			ids:        []int64{0, 1, 2},
			src:        []float32{10, 20, 30},
			initDst:    []float32{1, 0, 2, 0, 3, 0, 4, 0, 5},
			want:       []float32{11, 0, 22, 0, 33, 0, 4, 0, 5},
		},
		{
			name:       "Empty",
			dim:        0,
			numel:      0,
			ndims:      0,
			dims:       []int{},
			stridesSrc: []int{},
			stridesDst: []int{},
			ids:        []int64{},
			src:        []float32{},
			initDst:    []float32{},
			want:       []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.initDst))
			copy(dst, tt.initDst)
			kernels.IndexAddStridedI64F32(tt.numel, tt.ndims, tt.dims, tt.stridesSrc, tt.stridesDst, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

// Repeat for IndexAddStridedI64F64, U32F32, U32F64, U8F32, U8F64

func TestScatterI64F32(t *testing.T) {
	tests := []struct {
		name    string
		dim     int
		numel   int
		ids     []int64
		src     []float32
		initDst []float32
		want    []float32
	}{
		{
			name:    "Basic scatter",
			dim:     0,
			numel:   3,
			ids:     []int64{1, 3, 0},
			src:     []float32{10, 20, 30},
			initDst: []float32{1, 2, 3, 4, 5},
			want:    []float32{30, 10, 3, 20, 5},
		},
		{
			name:    "Scatter to same index (last wins)",
			dim:     0,
			numel:   2,
			ids:     []int64{1, 1},
			src:     []float32{5, 10},
			initDst: []float32{0, 0, 0},
			want:    []float32{0, 10, 0},
		},
		{
			name:    "Empty",
			dim:     0,
			numel:   0,
			ids:     []int64{},
			src:     []float32{},
			initDst: []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.initDst))
			copy(dst, tt.initDst)
			kernels.ScatterI64F32(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

// Repeat for ScatterI64F64, U32F32, U32F64, U8F32, U8F64

func TestScatterStridedI64F32(t *testing.T) {
	tests := []struct {
		name       string
		dim        int
		numel      int
		ndims      int
		dims       []int
		stridesSrc []int
		stridesDst []int
		ids        []int64
		src        []float32
		initDst    []float32
		want       []float32
	}{
		{
			name:       "Contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{1},
			ids:        []int64{1, 3, 0},
			src:        []float32{10, 20, 30},
			initDst:    []float32{1, 2, 3, 4, 5},
			want:       []float32{30, 10, 3, 20, 5},
		},
		{
			name:       "Non-contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{2},
			ids:        []int64{0, 1, 2},
			src:        []float32{10, 20, 30},
			initDst:    []float32{1, 0, 2, 0, 3, 0, 4, 0, 5},
			want:       []float32{10, 0, 20, 0, 30, 0, 4, 0, 5},
		},
		{
			name:       "Empty",
			dim:        0,
			numel:      0,
			ndims:      0,
			dims:       []int{},
			stridesSrc: []int{},
			stridesDst: []int{},
			ids:        []int64{},
			src:        []float32{},
			initDst:    []float32{},
			want:       []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.initDst))
			copy(dst, tt.initDst)
			kernels.ScatterStridedI64F32(tt.numel, tt.ndims, tt.dims, tt.stridesSrc, tt.stridesDst, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

// Repeat for ScatterStridedI64F64, U32F32, U32F64, U8F32, U8F64

func TestScatterAddI64F32(t *testing.T) {
	tests := []struct {
		name    string
		dim     int
		numel   int
		ids     []int64
		src     []float32
		initDst []float32
		want    []float32
	}{
		{
			name:    "Basic scatter add",
			dim:     0,
			numel:   3,
			ids:     []int64{1, 3, 0},
			src:     []float32{10, 20, 30},
			initDst: []float32{1, 2, 3, 4, 5},
			want:    []float32{31, 12, 3, 24, 5},
		},
		{
			name:    "Add to same index",
			dim:     0,
			numel:   2,
			ids:     []int64{1, 1},
			src:     []float32{5, 10},
			initDst: []float32{0, 0, 0},
			want:    []float32{0, 15, 0},
		},
		{
			name:    "Empty",
			dim:     0,
			numel:   0,
			ids:     []int64{},
			src:     []float32{},
			initDst: []float32{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.initDst))
			copy(dst, tt.initDst)
			kernels.ScatterAddI64F32(tt.numel, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

// Repeat for ScatterAddI64F64, U32F32, U32F64, U8F32, U8F64

func TestScatterAddStridedI64F32(t *testing.T) {
	tests := []struct {
		name       string
		dim        int
		numel      int
		ndims      int
		dims       []int
		stridesSrc []int
		stridesDst []int
		ids        []int64
		src        []float32
		initDst    []float32
		want       []float32
	}{
		{
			name:       "Contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{1},
			ids:        []int64{1, 3, 0},
			src:        []float32{10, 20, 30},
			initDst:    []float32{1, 2, 3, 4, 5},
			want:       []float32{31, 12, 3, 24, 5},
		},
		{
			name:       "Non-contiguous",
			dim:        0,
			numel:      3,
			ndims:      1,
			dims:       []int{5},
			stridesSrc: []int{1},
			stridesDst: []int{2},
			ids:        []int64{0, 1, 2},
			src:        []float32{10, 20, 30},
			initDst:    []float32{1, 0, 2, 0, 3, 0, 4, 0, 5},
			want:       []float32{11, 0, 22, 0, 33, 0, 4, 0, 5},
		},
		{
			name:       "Empty",
			dim:        0,
			numel:      0,
			ndims:      0,
			dims:       []int{},
			stridesSrc: []int{},
			stridesDst: []int{},
			ids:        []int64{},
			src:        []float32{},
			initDst:    []float32{},
			want:       []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.initDst))
			copy(dst, tt.initDst)
			kernels.ScatterAddStridedI64F32(tt.numel, tt.ndims, tt.dims, tt.stridesSrc, tt.stridesDst, tt.ids, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

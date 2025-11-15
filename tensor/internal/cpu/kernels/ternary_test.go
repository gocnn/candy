package kernels_test

import (
	"math"
	"testing"

	"slices"

	"github.com/gocnn/candy/tensor/internal/cpu/kernels"
)

func TestWhereI64F32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ids   []int64
		t     []float32
		f     []float32
		want  []float32
	}{
		{
			name:  "Basic where",
			numel: 5,
			ids:   []int64{0, 1, 0, 1, 0},
			t:     []float32{10, 20, 30, 40, 50},
			f:     []float32{1, 2, 3, 4, 5},
			want:  []float32{1, 20, 3, 40, 5},
		},
		{
			name:  "All true",
			numel: 3,
			ids:   []int64{1, 1, 1},
			t:     []float32{10, 20, 30},
			f:     []float32{1, 2, 3},
			want:  []float32{10, 20, 30},
		},
		{
			name:  "All false",
			numel: 3,
			ids:   []int64{0, 0, 0},
			t:     []float32{10, 20, 30},
			f:     []float32{1, 2, 3},
			want:  []float32{1, 2, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []int64{},
			t:     []float32{},
			f:     []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.WhereI64F32(tt.numel, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestWhereI64F64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ids   []int64
		t     []float64
		f     []float64
		want  []float64
	}{
		{
			name:  "Basic where",
			numel: 5,
			ids:   []int64{0, 1, 0, 1, 0},
			t:     []float64{10, 20, 30, 40, 50},
			f:     []float64{1, 2, 3, 4, 5},
			want:  []float64{1, 20, 3, 40, 5},
		},
		{
			name:  "All true",
			numel: 3,
			ids:   []int64{1, 1, 1},
			t:     []float64{10, 20, 30},
			f:     []float64{1, 2, 3},
			want:  []float64{10, 20, 30},
		},
		{
			name:  "All false",
			numel: 3,
			ids:   []int64{0, 0, 0},
			t:     []float64{10, 20, 30},
			f:     []float64{1, 2, 3},
			want:  []float64{1, 2, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []int64{},
			t:     []float64{},
			f:     []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.WhereI64F64(tt.numel, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestWhereU32F32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ids   []uint32
		t     []float32
		f     []float32
		want  []float32
	}{
		{
			name:  "Basic where",
			numel: 5,
			ids:   []uint32{0, 1, 0, 1, 0},
			t:     []float32{10, 20, 30, 40, 50},
			f:     []float32{1, 2, 3, 4, 5},
			want:  []float32{1, 20, 3, 40, 5},
		},
		{
			name:  "All true",
			numel: 3,
			ids:   []uint32{1, 1, 1},
			t:     []float32{10, 20, 30},
			f:     []float32{1, 2, 3},
			want:  []float32{10, 20, 30},
		},
		{
			name:  "All false",
			numel: 3,
			ids:   []uint32{0, 0, 0},
			t:     []float32{10, 20, 30},
			f:     []float32{1, 2, 3},
			want:  []float32{1, 2, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []uint32{},
			t:     []float32{},
			f:     []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.WhereU32F32(tt.numel, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestWhereU32F64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ids   []uint32
		t     []float64
		f     []float64
		want  []float64
	}{
		{
			name:  "Basic where",
			numel: 5,
			ids:   []uint32{0, 1, 0, 1, 0},
			t:     []float64{10, 20, 30, 40, 50},
			f:     []float64{1, 2, 3, 4, 5},
			want:  []float64{1, 20, 3, 40, 5},
		},
		{
			name:  "All true",
			numel: 3,
			ids:   []uint32{1, 1, 1},
			t:     []float64{10, 20, 30},
			f:     []float64{1, 2, 3},
			want:  []float64{10, 20, 30},
		},
		{
			name:  "All false",
			numel: 3,
			ids:   []uint32{0, 0, 0},
			t:     []float64{10, 20, 30},
			f:     []float64{1, 2, 3},
			want:  []float64{1, 2, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []uint32{},
			t:     []float64{},
			f:     []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.WhereU32F64(tt.numel, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestWhereU8F32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ids   []uint8
		t     []float32
		f     []float32
		want  []float32
	}{
		{
			name:  "Basic where",
			numel: 5,
			ids:   []uint8{0, 1, 0, 1, 0},
			t:     []float32{10, 20, 30, 40, 50},
			f:     []float32{1, 2, 3, 4, 5},
			want:  []float32{1, 20, 3, 40, 5},
		},
		{
			name:  "All true",
			numel: 3,
			ids:   []uint8{1, 1, 1},
			t:     []float32{10, 20, 30},
			f:     []float32{1, 2, 3},
			want:  []float32{10, 20, 30},
		},
		{
			name:  "All false",
			numel: 3,
			ids:   []uint8{0, 0, 0},
			t:     []float32{10, 20, 30},
			f:     []float32{1, 2, 3},
			want:  []float32{1, 2, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []uint8{},
			t:     []float32{},
			f:     []float32{},
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.WhereU8F32(tt.numel, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestWhereU8F64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		ids   []uint8
		t     []float64
		f     []float64
		want  []float64
	}{
		{
			name:  "Basic where",
			numel: 5,
			ids:   []uint8{0, 1, 0, 1, 0},
			t:     []float64{10, 20, 30, 40, 50},
			f:     []float64{1, 2, 3, 4, 5},
			want:  []float64{1, 20, 3, 40, 5},
		},
		{
			name:  "All true",
			numel: 3,
			ids:   []uint8{1, 1, 1},
			t:     []float64{10, 20, 30},
			f:     []float64{1, 2, 3},
			want:  []float64{10, 20, 30},
		},
		{
			name:  "All false",
			numel: 3,
			ids:   []uint8{0, 0, 0},
			t:     []float64{10, 20, 30},
			f:     []float64{1, 2, 3},
			want:  []float64{1, 2, 3},
		},
		{
			name:  "Empty",
			numel: 0,
			ids:   []uint8{},
			t:     []float64{},
			f:     []float64{},
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.WhereU8F64(tt.numel, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestWhereStridedI64F32(t *testing.T) {
	tests := []struct {
		name     string
		numel    int
		ndims    int
		dims     []int
		strides  []int
		stridesT []int
		stridesF []int
		ids      []int64
		t        []float32
		f        []float32
		want     []float32
	}{
		{
			name:     "Contiguous",
			numel:    5,
			ndims:    1,
			dims:     []int{5},
			strides:  []int{1},
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []int64{0, 1, 0, 1, 0},
			t:        []float32{10, 20, 30, 40, 50},
			f:        []float32{1, 2, 3, 4, 5},
			want:     []float32{1, 20, 3, 40, 5},
		},
		{
			name:     "Non-contiguous strides",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{2}, // ids stored with stride 2
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []int64{0, 0, 1, 0, 1, 0}, // needs 6 elements for stride=2
			t:        []float32{10, 20, 30},
			f:        []float32{1, 2, 3},
			want:     []float32{1, 20, 30},
		},
		{
			name:     "Non-contiguous t and f",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{1},
			stridesT: []int{2}, // t stored with stride 2
			stridesF: []int{2}, // f stored with stride 2
			ids:      []int64{0, 1, 0},
			t:        []float32{10, 0, 20, 0, 30},
			f:        []float32{1, 0, 2, 0, 3},
			want:     []float32{1, 20, 3},
		},
		{
			name:     "Empty",
			numel:    0,
			ndims:    0,
			dims:     []int{},
			strides:  []int{},
			stridesT: []int{},
			stridesF: []int{},
			ids:      []int64{},
			t:        []float32{},
			f:        []float32{},
			want:     []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.WhereStridedI64F32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.stridesT, tt.stridesF, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestWhereStridedI64F64(t *testing.T) {
	tests := []struct {
		name     string
		numel    int
		ndims    int
		dims     []int
		strides  []int
		stridesT []int
		stridesF []int
		ids      []int64
		t        []float64
		f        []float64
		want     []float64
	}{
		{
			name:     "Contiguous",
			numel:    5,
			ndims:    1,
			dims:     []int{5},
			strides:  []int{1},
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []int64{0, 1, 0, 1, 0},
			t:        []float64{10, 20, 30, 40, 50},
			f:        []float64{1, 2, 3, 4, 5},
			want:     []float64{1, 20, 3, 40, 5},
		},
		{
			name:     "Non-contiguous strides",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{2},
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []int64{0, 0, 1, 0, 1, 0},
			t:        []float64{10, 20, 30},
			f:        []float64{1, 2, 3},
			want:     []float64{1, 20, 30},
		},
		{
			name:     "Non-contiguous t and f",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{1},
			stridesT: []int{2},
			stridesF: []int{2},
			ids:      []int64{0, 1, 0},
			t:        []float64{10, 0, 20, 0, 30},
			f:        []float64{1, 0, 2, 0, 3},
			want:     []float64{1, 20, 3},
		},
		{
			name:     "Empty",
			numel:    0,
			ndims:    0,
			dims:     []int{},
			strides:  []int{},
			stridesT: []int{},
			stridesF: []int{},
			ids:      []int64{},
			t:        []float64{},
			f:        []float64{},
			want:     []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.WhereStridedI64F64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.stridesT, tt.stridesF, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestWhereStridedU32F32(t *testing.T) {
	tests := []struct {
		name     string
		numel    int
		ndims    int
		dims     []int
		strides  []int
		stridesT []int
		stridesF []int
		ids      []uint32
		t        []float32
		f        []float32
		want     []float32
	}{
		{
			name:     "Contiguous",
			numel:    5,
			ndims:    1,
			dims:     []int{5},
			strides:  []int{1},
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []uint32{0, 1, 0, 1, 0},
			t:        []float32{10, 20, 30, 40, 50},
			f:        []float32{1, 2, 3, 4, 5},
			want:     []float32{1, 20, 3, 40, 5},
		},
		{
			name:     "Non-contiguous strides",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{2},
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []uint32{0, 0, 1, 0, 1, 0},
			t:        []float32{10, 20, 30},
			f:        []float32{1, 2, 3},
			want:     []float32{1, 20, 30},
		},
		{
			name:     "Non-contiguous t and f",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{1},
			stridesT: []int{2},
			stridesF: []int{2},
			ids:      []uint32{0, 1, 0},
			t:        []float32{10, 0, 20, 0, 30},
			f:        []float32{1, 0, 2, 0, 3},
			want:     []float32{1, 20, 3},
		},
		{
			name:     "Empty",
			numel:    0,
			ndims:    0,
			dims:     []int{},
			strides:  []int{},
			stridesT: []int{},
			stridesF: []int{},
			ids:      []uint32{},
			t:        []float32{},
			f:        []float32{},
			want:     []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.WhereStridedU32F32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.stridesT, tt.stridesF, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestWhereStridedU32F64(t *testing.T) {
	tests := []struct {
		name     string
		numel    int
		ndims    int
		dims     []int
		strides  []int
		stridesT []int
		stridesF []int
		ids      []uint32
		t        []float64
		f        []float64
		want     []float64
	}{
		{
			name:     "Contiguous",
			numel:    5,
			ndims:    1,
			dims:     []int{5},
			strides:  []int{1},
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []uint32{0, 1, 0, 1, 0},
			t:        []float64{10, 20, 30, 40, 50},
			f:        []float64{1, 2, 3, 4, 5},
			want:     []float64{1, 20, 3, 40, 5},
		},
		{
			name:     "Non-contiguous strides",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{2},
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []uint32{0, 0, 1, 0, 1, 0},
			t:        []float64{10, 20, 30},
			f:        []float64{1, 2, 3},
			want:     []float64{1, 20, 30},
		},
		{
			name:     "Non-contiguous t and f",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{1},
			stridesT: []int{2},
			stridesF: []int{2},
			ids:      []uint32{0, 1, 0},
			t:        []float64{10, 0, 20, 0, 30},
			f:        []float64{1, 0, 2, 0, 3},
			want:     []float64{1, 20, 3},
		},
		{
			name:     "Empty",
			numel:    0,
			ndims:    0,
			dims:     []int{},
			strides:  []int{},
			stridesT: []int{},
			stridesF: []int{},
			ids:      []uint32{},
			t:        []float64{},
			f:        []float64{},
			want:     []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.WhereStridedU32F64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.stridesT, tt.stridesF, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestWhereStridedU8F32(t *testing.T) {
	tests := []struct {
		name     string
		numel    int
		ndims    int
		dims     []int
		strides  []int
		stridesT []int
		stridesF []int
		ids      []uint8
		t        []float32
		f        []float32
		want     []float32
	}{
		{
			name:     "Contiguous",
			numel:    5,
			ndims:    1,
			dims:     []int{5},
			strides:  []int{1},
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []uint8{0, 1, 0, 1, 0},
			t:        []float32{10, 20, 30, 40, 50},
			f:        []float32{1, 2, 3, 4, 5},
			want:     []float32{1, 20, 3, 40, 5},
		},
		{
			name:     "Non-contiguous strides",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{2},
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []uint8{0, 0, 1, 0, 1, 0},
			t:        []float32{10, 20, 30},
			f:        []float32{1, 2, 3},
			want:     []float32{1, 20, 30},
		},
		{
			name:     "Non-contiguous t and f",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{1},
			stridesT: []int{2},
			stridesF: []int{2},
			ids:      []uint8{0, 1, 0},
			t:        []float32{10, 0, 20, 0, 30},
			f:        []float32{1, 0, 2, 0, 3},
			want:     []float32{1, 20, 3},
		},
		{
			name:     "Empty",
			numel:    0,
			ndims:    0,
			dims:     []int{},
			strides:  []int{},
			stridesT: []int{},
			stridesF: []int{},
			ids:      []uint8{},
			t:        []float32{},
			f:        []float32{},
			want:     []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.WhereStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.stridesT, tt.stridesF, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestWhereStridedU8F64(t *testing.T) {
	tests := []struct {
		name     string
		numel    int
		ndims    int
		dims     []int
		strides  []int
		stridesT []int
		stridesF []int
		ids      []uint8
		t        []float64
		f        []float64
		want     []float64
	}{
		{
			name:     "Contiguous",
			numel:    5,
			ndims:    1,
			dims:     []int{5},
			strides:  []int{1},
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []uint8{0, 1, 0, 1, 0},
			t:        []float64{10, 20, 30, 40, 50},
			f:        []float64{1, 2, 3, 4, 5},
			want:     []float64{1, 20, 3, 40, 5},
		},
		{
			name:     "Non-contiguous strides",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{2},
			stridesT: []int{1},
			stridesF: []int{1},
			ids:      []uint8{0, 0, 1, 0, 1, 0},
			t:        []float64{10, 20, 30},
			f:        []float64{1, 2, 3},
			want:     []float64{1, 20, 30},
		},
		{
			name:     "Non-contiguous t and f",
			numel:    3,
			ndims:    1,
			dims:     []int{3},
			strides:  []int{1},
			stridesT: []int{2},
			stridesF: []int{2},
			ids:      []uint8{0, 1, 0},
			t:        []float64{10, 0, 20, 0, 30},
			f:        []float64{1, 0, 2, 0, 3},
			want:     []float64{1, 20, 3},
		},
		{
			name:     "Empty",
			numel:    0,
			ndims:    0,
			dims:     []int{},
			strides:  []int{},
			stridesT: []int{},
			stridesF: []int{},
			ids:      []uint8{},
			t:        []float64{},
			f:        []float64{},
			want:     []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.WhereStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.stridesT, tt.stridesF, tt.ids, tt.t, tt.f, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

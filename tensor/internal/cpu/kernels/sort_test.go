package kernels_test

import (
	"slices"
	"testing"

	"github.com/gocnn/candy/tensor/internal/cpu/kernels"
)

func TestAsortAscI64F32(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float32
		dst   []int64
		want  []int64
	}{
		{
			name:  "Basic ascending sort 1 row",
			ncols: 3,
			src:   []float32{3, 1, 2},
			dst:   make([]int64, 3),
			want:  []int64{1, 2, 0},
		},
		{
			name:  "Ascending sort 2 rows",
			ncols: 2,
			src:   []float32{4, 2, 1, 3},
			dst:   make([]int64, 4),
			want:  []int64{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]int64, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscI64F32(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortAscStridedI64F32(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float32
		dst        []int64
		want       []int64
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float32{3, 1, 2},
			dst:        make([]int64, 3),
			want:       []int64{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float32{4, 2, 1, 3},
			dst:        make([]int64, 4),
			want:       []int64{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]int64, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscStridedI64F32(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortAscI64F64(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float64
		dst   []int64
		want  []int64
	}{
		{
			name:  "Basic ascending sort 1 row",
			ncols: 3,
			src:   []float64{3, 1, 2},
			dst:   make([]int64, 3),
			want:  []int64{1, 2, 0},
		},
		{
			name:  "Ascending sort 2 rows",
			ncols: 2,
			src:   []float64{4, 2, 1, 3},
			dst:   make([]int64, 4),
			want:  []int64{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]int64, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscI64F64(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortAscStridedI64F64(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float64
		dst        []int64
		want       []int64
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float64{3, 1, 2},
			dst:        make([]int64, 3),
			want:       []int64{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float64{4, 2, 1, 3},
			dst:        make([]int64, 4),
			want:       []int64{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]int64, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscStridedI64F64(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortAscU32F32(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float32
		dst   []uint32
		want  []uint32
	}{
		{
			name:  "Basic ascending sort 1 row",
			ncols: 3,
			src:   []float32{3, 1, 2},
			dst:   make([]uint32, 3),
			want:  []uint32{1, 2, 0},
		},
		{
			name:  "Ascending sort 2 rows",
			ncols: 2,
			src:   []float32{4, 2, 1, 3},
			dst:   make([]uint32, 4),
			want:  []uint32{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint32, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscU32F32(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortAscStridedU32F32(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float32
		dst        []uint32
		want       []uint32
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float32{3, 1, 2},
			dst:        make([]uint32, 3),
			want:       []uint32{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float32{4, 2, 1, 3},
			dst:        make([]uint32, 4),
			want:       []uint32{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint32, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscStridedU32F32(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortAscU32F64(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float64
		dst   []uint32
		want  []uint32
	}{
		{
			name:  "Basic ascending sort 1 row",
			ncols: 3,
			src:   []float64{3, 1, 2},
			dst:   make([]uint32, 3),
			want:  []uint32{1, 2, 0},
		},
		{
			name:  "Ascending sort 2 rows",
			ncols: 2,
			src:   []float64{4, 2, 1, 3},
			dst:   make([]uint32, 4),
			want:  []uint32{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint32, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscU32F64(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortAscStridedU32F64(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float64
		dst        []uint32
		want       []uint32
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float64{3, 1, 2},
			dst:        make([]uint32, 3),
			want:       []uint32{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float64{4, 2, 1, 3},
			dst:        make([]uint32, 4),
			want:       []uint32{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint32, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscStridedU32F64(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortAscU8F32(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float32
		dst   []uint8
		want  []uint8
	}{
		{
			name:  "Basic ascending sort 1 row",
			ncols: 3,
			src:   []float32{3, 1, 2},
			dst:   make([]uint8, 3),
			want:  []uint8{1, 2, 0},
		},
		{
			name:  "Ascending sort 2 rows",
			ncols: 2,
			src:   []float32{4, 2, 1, 3},
			dst:   make([]uint8, 4),
			want:  []uint8{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint8, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscU8F32(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortAscStridedU8F32(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float32
		dst        []uint8
		want       []uint8
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float32{3, 1, 2},
			dst:        make([]uint8, 3),
			want:       []uint8{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float32{4, 2, 1, 3},
			dst:        make([]uint8, 4),
			want:       []uint8{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint8, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscStridedU8F32(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortAscU8F64(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float64
		dst   []uint8
		want  []uint8
	}{
		{
			name:  "Basic ascending sort 1 row",
			ncols: 3,
			src:   []float64{3, 1, 2},
			dst:   make([]uint8, 3),
			want:  []uint8{1, 2, 0},
		},
		{
			name:  "Ascending sort 2 rows",
			ncols: 2,
			src:   []float64{4, 2, 1, 3},
			dst:   make([]uint8, 4),
			want:  []uint8{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint8, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscU8F64(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortAscStridedU8F64(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float64
		dst        []uint8
		want       []uint8
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float64{3, 1, 2},
			dst:        make([]uint8, 3),
			want:       []uint8{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float64{4, 2, 1, 3},
			dst:        make([]uint8, 4),
			want:       []uint8{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint8, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortAscStridedU8F64(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescI64F32(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float32
		dst   []int64
		want  []int64
	}{
		{
			name:  "Basic descending sort 1 row",
			ncols: 3,
			src:   []float32{1, 3, 2},
			dst:   make([]int64, 3),
			want:  []int64{1, 2, 0},
		},
		{
			name:  "Descending sort 2 rows",
			ncols: 2,
			src:   []float32{2, 4, 3, 1},
			dst:   make([]int64, 4),
			want:  []int64{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]int64, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescI64F32(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescStridedI64F32(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float32
		dst        []int64
		want       []int64
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float32{1, 3, 2},
			dst:        make([]int64, 3),
			want:       []int64{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float32{4, 2, 1, 3},
			dst:        make([]int64, 4),
			want:       []int64{0, 1, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]int64, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescStridedI64F32(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescI64F64(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float64
		dst   []int64
		want  []int64
	}{
		{
			name:  "Basic descending sort 1 row",
			ncols: 3,
			src:   []float64{1, 3, 2},
			dst:   make([]int64, 3),
			want:  []int64{1, 2, 0},
		},
		{
			name:  "Descending sort 2 rows",
			ncols: 2,
			src:   []float64{2, 4, 3, 1},
			dst:   make([]int64, 4),
			want:  []int64{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]int64, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescI64F64(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescStridedI64F64(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float64
		dst        []int64
		want       []int64
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float64{1, 3, 2},
			dst:        make([]int64, 3),
			want:       []int64{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float64{4, 2, 1, 3},
			dst:        make([]int64, 4),
			want:       []int64{0, 1, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]int64, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescStridedI64F64(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescU32F32(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float32
		dst   []uint32
		want  []uint32
	}{
		{
			name:  "Basic descending sort 1 row",
			ncols: 3,
			src:   []float32{1, 3, 2},
			dst:   make([]uint32, 3),
			want:  []uint32{1, 2, 0},
		},
		{
			name:  "Descending sort 2 rows",
			ncols: 2,
			src:   []float32{2, 4, 3, 1},
			dst:   make([]uint32, 4),
			want:  []uint32{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint32, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescU32F32(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescStridedU32F32(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float32
		dst        []uint32
		want       []uint32
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float32{1, 3, 2},
			dst:        make([]uint32, 3),
			want:       []uint32{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float32{4, 2, 1, 3},
			dst:        make([]uint32, 4),
			want:       []uint32{0, 1, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint32, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescStridedU32F32(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescU32F64(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float64
		dst   []uint32
		want  []uint32
	}{
		{
			name:  "Basic descending sort 1 row",
			ncols: 3,
			src:   []float64{1, 3, 2},
			dst:   make([]uint32, 3),
			want:  []uint32{1, 2, 0},
		},
		{
			name:  "Descending sort 2 rows",
			ncols: 2,
			src:   []float64{2, 4, 3, 1},
			dst:   make([]uint32, 4),
			want:  []uint32{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint32, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescU32F64(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescStridedU32F64(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float64
		dst        []uint32
		want       []uint32
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float64{1, 3, 2},
			dst:        make([]uint32, 3),
			want:       []uint32{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float64{4, 2, 1, 3},
			dst:        make([]uint32, 4),
			want:       []uint32{0, 1, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint32, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescStridedU32F64(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescU8F32(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float32
		dst   []uint8
		want  []uint8
	}{
		{
			name:  "Basic descending sort 1 row",
			ncols: 3,
			src:   []float32{1, 3, 2},
			dst:   make([]uint8, 3),
			want:  []uint8{1, 2, 0},
		},
		{
			name:  "Descending sort 2 rows",
			ncols: 2,
			src:   []float32{2, 4, 3, 1},
			dst:   make([]uint8, 4),
			want:  []uint8{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint8, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescU8F32(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescStridedU8F32(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float32
		dst        []uint8
		want       []uint8
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float32{1, 3, 2},
			dst:        make([]uint8, 3),
			want:       []uint8{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float32{4, 2, 1, 3},
			dst:        make([]uint8, 4),
			want:       []uint8{0, 1, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint8, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescStridedU8F32(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescU8F64(t *testing.T) {
	tests := []struct {
		name  string
		ncols int
		src   []float64
		dst   []uint8
		want  []uint8
	}{
		{
			name:  "Basic descending sort 1 row",
			ncols: 3,
			src:   []float64{1, 3, 2},
			dst:   make([]uint8, 3),
			want:  []uint8{1, 2, 0},
		},
		{
			name:  "Descending sort 2 rows",
			ncols: 2,
			src:   []float64{2, 4, 3, 1},
			dst:   make([]uint8, 4),
			want:  []uint8{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint8, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescU8F64(tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

func TestAsortDescStridedU8F64(t *testing.T) {
	tests := []struct {
		name       string
		ndims      int
		dims       []int
		strides    []int
		stridesDst []int
		ncols      int
		src        []float64
		dst        []uint8
		want       []uint8
	}{
		{
			name:       "Contiguous 1D",
			ndims:      1,
			dims:       []int{3},
			strides:    []int{1},
			stridesDst: []int{1},
			ncols:      3,
			src:        []float64{1, 3, 2},
			dst:        make([]uint8, 3),
			want:       []uint8{1, 2, 0},
		},
		{
			name:       "Non-contiguous 2D (strided src)",
			ndims:      2,
			dims:       []int{2, 2},
			strides:    []int{1, 2},
			stridesDst: []int{2, 1},
			ncols:      2,
			src:        []float64{4, 2, 1, 3},
			dst:        make([]uint8, 4),
			want:       []uint8{0, 1, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstCopy := make([]uint8, len(tt.dst))
			copy(dstCopy, tt.dst)
			kernels.AsortDescStridedU8F64(tt.ndims, tt.dims, tt.strides, tt.stridesDst, tt.ncols, tt.src, dstCopy)
			if !slices.Equal(dstCopy, tt.want) {
				t.Errorf("dst: got %v, want %v", dstCopy, tt.want)
			}
		})
	}
}

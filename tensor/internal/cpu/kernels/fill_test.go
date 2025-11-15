package kernels_test

import (
	"slices"
	"testing"

	"github.com/gocnn/candy/tensor/internal/cpu/kernels"
)

func TestFillF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		val   float32
		want  []float32
	}{
		{
			name:  "Basic fill",
			numel: 5,
			val:   3.14,
			want:  []float32{3.14, 3.14, 3.14, 3.14, 3.14},
		},
		{
			name:  "Zero fill",
			numel: 3,
			val:   0,
			want:  []float32{0, 0, 0},
		},
		{
			name:  "Negative fill",
			numel: 4,
			val:   -1.5,
			want:  []float32{-1.5, -1.5, -1.5, -1.5},
		},
		{
			name:  "Empty",
			numel: 0,
			val:   1,
			want:  []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.numel)
			kernels.FillF32(tt.numel, tt.val, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFillF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		val   float64
		want  []float64
	}{
		{
			name:  "Basic fill",
			numel: 5,
			val:   3.14,
			want:  []float64{3.14, 3.14, 3.14, 3.14, 3.14},
		},
		{
			name:  "Zero fill",
			numel: 3,
			val:   0,
			want:  []float64{0, 0, 0},
		},
		{
			name:  "Negative fill",
			numel: 4,
			val:   -1.5,
			want:  []float64{-1.5, -1.5, -1.5, -1.5},
		},
		{
			name:  "Empty",
			numel: 0,
			val:   1,
			want:  []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.numel)
			kernels.FillF64(tt.numel, tt.val, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFillStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		val     float32
		want    []float32
	}{
		{
			name:    "Contiguous 1D",
			numel:   5,
			ndims:   1,
			dims:    []int{5},
			strides: []int{1},
			val:     2.5,
			want:    []float32{2.5, 2.5, 2.5, 2.5, 2.5},
		},
		{
			name:    "Non-contiguous 2D (transposed view)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			val:     1.0,
			want:    []float32{1.0, 1.0, 1.0, 1.0},
		},
		{
			name:    "Broadcast-like (zero stride)",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{0, 1},
			val:     0.5,
			want:    []float32{0.5, 0.5},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   0,
			dims:    []int{},
			strides: []int{},
			val:     1,
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstSize := 0
			if tt.numel > 0 {
				maxIdx := 0
				for d := 0; d < tt.ndims; d++ {
					maxIdx += (tt.dims[d] - 1) * tt.strides[d]
				}
				dstSize = maxIdx + 1
			}
			dst := make([]float32, dstSize)
			kernels.FillStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.val, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestFillStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		val     float64
		want    []float64
	}{
		{
			name:    "Contiguous 1D",
			numel:   5,
			ndims:   1,
			dims:    []int{5},
			strides: []int{1},
			val:     2.5,
			want:    []float64{2.5, 2.5, 2.5, 2.5, 2.5},
		},
		{
			name:    "Non-contiguous 2D (transposed view)",
			numel:   4,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			val:     1.0,
			want:    []float64{1.0, 1.0, 1.0, 1.0},
		},
		{
			name:    "Broadcast-like (zero stride)",
			numel:   6,
			ndims:   2,
			dims:    []int{3, 2},
			strides: []int{0, 1},
			val:     0.5,
			want:    []float64{0.5, 0.5},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   0,
			dims:    []int{},
			strides: []int{},
			val:     1,
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstSize := 0
			if tt.numel > 0 {
				maxIdx := 0
				for d := 0; d < tt.ndims; d++ {
					maxIdx += (tt.dims[d] - 1) * tt.strides[d]
				}
				dstSize = maxIdx + 1
			}
			dst := make([]float64, dstSize)
			kernels.FillStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.val, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestCopy2dStridedF32(t *testing.T) {
	tests := []struct {
		name      string
		rows      int
		cols      int
		lda       int // src leading dimension
		ldc       int // dst leading dimension
		srcOffset int
		dstOffset int
		src       []float32
		want      []float32
	}{
		{
			name:      "Src with padding (lda > cols)",
			rows:      2,
			cols:      2,
			lda:       3,
			ldc:       2,
			srcOffset: 0,
			dstOffset: 0,
			src:       []float32{1, 2, 0, 3, 4, 0},
			want:      []float32{1, 2, 3, 4},
		},
		{
			name:      "Dst with padding (ldc > cols)",
			rows:      2,
			cols:      2,
			lda:       2,
			ldc:       3,
			srcOffset: 0,
			dstOffset: 0,
			src:       []float32{1, 2, 3, 4},
			want:      []float32{1, 2, 0, 3, 4, 0},
		},
		{
			name:      "Strided with offsets",
			rows:      2,
			cols:      2,
			lda:       3,
			ldc:       3,
			srcOffset: 1,
			dstOffset: 1,
			src:       []float32{0, 1, 2, 0, 3, 4, 0},
			want:      []float32{0, 1, 2, 0, 3, 4, 0},
		},
		{
			name:      "Empty",
			rows:      0,
			cols:      0,
			lda:       0,
			ldc:       0,
			srcOffset: 0,
			dstOffset: 0,
			src:       []float32{},
			want:      []float32{},
		},
		{
			name:      "Single row with padding",
			rows:      1,
			cols:      3,
			lda:       4,
			ldc:       3,
			srcOffset: 1,
			dstOffset: 0,
			src:       []float32{0, 1, 2, 3},
			want:      []float32{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstSize := tt.dstOffset + tt.rows*tt.ldc
			dst := make([]float32, dstSize)
			kernels.Copy2dF32(tt.rows, tt.cols, tt.lda, tt.ldc, tt.srcOffset, tt.dstOffset, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestCopy2dStridedF64(t *testing.T) {
	tests := []struct {
		name      string
		rows      int
		cols      int
		lda       int // src leading dimension
		ldc       int // dst leading dimension
		srcOffset int
		dstOffset int
		src       []float64
		want      []float64
	}{
		{
			name:      "Src with padding (lda > cols)",
			rows:      2,
			cols:      2,
			lda:       3,
			ldc:       2,
			srcOffset: 0,
			dstOffset: 0,
			src:       []float64{1, 2, 0, 3, 4, 0},
			want:      []float64{1, 2, 3, 4},
		},
		{
			name:      "Dst with padding (ldc > cols)",
			rows:      2,
			cols:      2,
			lda:       2,
			ldc:       3,
			srcOffset: 0,
			dstOffset: 0,
			src:       []float64{1, 2, 3, 4},
			want:      []float64{1, 2, 0, 3, 4, 0},
		},
		{
			name:      "Strided with offsets",
			rows:      2,
			cols:      2,
			lda:       3,
			ldc:       3,
			srcOffset: 1,
			dstOffset: 1,
			src:       []float64{0, 1, 2, 0, 3, 4, 0},
			want:      []float64{0, 1, 2, 0, 3, 4, 0},
		},
		{
			name:      "Empty",
			rows:      0,
			cols:      0,
			lda:       0,
			ldc:       0,
			srcOffset: 0,
			dstOffset: 0,
			src:       []float64{},
			want:      []float64{},
		},
		{
			name:      "Single row with padding",
			rows:      1,
			cols:      3,
			lda:       4,
			ldc:       3,
			srcOffset: 1,
			dstOffset: 0,
			src:       []float64{0, 1, 2, 3},
			want:      []float64{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstSize := tt.dstOffset + tt.rows*tt.ldc
			dst := make([]float64, dstSize)
			kernels.Copy2dF64(tt.rows, tt.cols, tt.lda, tt.ldc, tt.srcOffset, tt.dstOffset, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestCopy2dStridedU8(t *testing.T) {
	tests := []struct {
		name      string
		rows      int
		cols      int
		lda       int // src leading dimension
		ldc       int // dst leading dimension
		srcOffset int
		dstOffset int
		src       []uint8
		want      []uint8
	}{
		{
			name:      "Src with padding (lda > cols)",
			rows:      2,
			cols:      2,
			lda:       3,
			ldc:       2,
			srcOffset: 0,
			dstOffset: 0,
			src:       []uint8{1, 2, 0, 3, 4, 0},
			want:      []uint8{1, 2, 3, 4},
		},
		{
			name:      "Dst with padding (ldc > cols)",
			rows:      2,
			cols:      2,
			lda:       2,
			ldc:       3,
			srcOffset: 0,
			dstOffset: 0,
			src:       []uint8{1, 2, 3, 4},
			want:      []uint8{1, 2, 0, 3, 4, 0},
		},
		{
			name:      "Strided with offsets",
			rows:      2,
			cols:      2,
			lda:       3,
			ldc:       3,
			srcOffset: 1,
			dstOffset: 1,
			src:       []uint8{0, 1, 2, 0, 3, 4, 0},
			want:      []uint8{0, 1, 2, 0, 3, 4, 0},
		},
		{
			name:      "Empty",
			rows:      0,
			cols:      0,
			lda:       0,
			ldc:       0,
			srcOffset: 0,
			dstOffset: 0,
			src:       []uint8{},
			want:      []uint8{},
		},
		{
			name:      "Single row with padding",
			rows:      1,
			cols:      3,
			lda:       4,
			ldc:       3,
			srcOffset: 1,
			dstOffset: 0,
			src:       []uint8{0, 1, 2, 3},
			want:      []uint8{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstSize := tt.dstOffset + tt.rows*tt.ldc
			dst := make([]uint8, dstSize)
			kernels.Copy2dU8(tt.rows, tt.cols, tt.lda, tt.ldc, tt.srcOffset, tt.dstOffset, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestCopy2dU32(t *testing.T) {
	tests := []struct {
		name      string
		rows      int
		cols      int
		lda       int // src leading dimension
		ldc       int // dst leading dimension
		srcOffset int
		dstOffset int
		src       []uint32
		want      []uint32
	}{
		{
			name:      "Contiguous copy",
			rows:      2,
			cols:      3,
			lda:       3,
			ldc:       3,
			srcOffset: 0,
			dstOffset: 0,
			src:       []uint32{1, 2, 3, 4, 5, 6},
			want:      []uint32{1, 2, 3, 4, 5, 6},
		},
		{
			name:      "Contiguous with offsets",
			rows:      2,
			cols:      2,
			lda:       2,
			ldc:       2,
			srcOffset: 2,
			dstOffset: 1,
			src:       []uint32{0, 0, 1, 2, 3, 4},
			want:      []uint32{0, 1, 2, 3, 4},
		},
		{
			name:      "Empty",
			rows:      0,
			cols:      0,
			lda:       0,
			ldc:       0,
			srcOffset: 0,
			dstOffset: 0,
			src:       []uint32{},
			want:      []uint32{},
		},
		{
			name:      "Single row",
			rows:      1,
			cols:      4,
			lda:       4,
			ldc:       4,
			srcOffset: 0,
			dstOffset: 0,
			src:       []uint32{1, 2, 3, 4},
			want:      []uint32{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstSize := tt.dstOffset + tt.rows*tt.ldc
			dst := make([]uint32, dstSize)
			kernels.Copy2dU32(tt.rows, tt.cols, tt.lda, tt.ldc, tt.srcOffset, tt.dstOffset, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestCopy2dI64(t *testing.T) {
	tests := []struct {
		name      string
		rows      int
		cols      int
		lda       int // src leading dimension
		ldc       int // dst leading dimension
		srcOffset int
		dstOffset int
		src       []int64
		want      []int64
	}{
		{
			name:      "Contiguous copy",
			rows:      2,
			cols:      3,
			lda:       3,
			ldc:       3,
			srcOffset: 0,
			dstOffset: 0,
			src:       []int64{1, 2, 3, 4, 5, 6},
			want:      []int64{1, 2, 3, 4, 5, 6},
		},
		{
			name:      "Contiguous with offsets",
			rows:      2,
			cols:      2,
			lda:       2,
			ldc:       2,
			srcOffset: 2,
			dstOffset: 1,
			src:       []int64{0, 0, 1, 2, 3, 4},
			want:      []int64{0, 1, 2, 3, 4},
		},
		{
			name:      "Empty",
			rows:      0,
			cols:      0,
			lda:       0,
			ldc:       0,
			srcOffset: 0,
			dstOffset: 0,
			src:       []int64{},
			want:      []int64{},
		},
		{
			name:      "Single row",
			rows:      1,
			cols:      4,
			lda:       4,
			ldc:       4,
			srcOffset: 0,
			dstOffset: 0,
			src:       []int64{1, 2, 3, 4},
			want:      []int64{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstSize := tt.dstOffset + tt.rows*tt.ldc
			dst := make([]int64, dstSize)
			kernels.Copy2dI64(tt.rows, tt.cols, tt.lda, tt.ldc, tt.srcOffset, tt.dstOffset, tt.src, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestConstSetF32(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		val   float32
		ids   []int
		want  []float32
	}{
		{
			name:  "Basic set",
			numel: 3,
			val:   5.0,
			ids:   []int{0, 2, 4},
			want:  []float32{5.0, 0, 5.0, 0, 5.0}, // dst size inferred as max id +1 =5
		},
		{
			name:  "Set all",
			numel: 4,
			val:   -1,
			ids:   []int{0, 1, 2, 3},
			want:  []float32{-1, -1, -1, -1},
		},
		{
			name:  "Empty",
			numel: 0,
			val:   1,
			ids:   []int{},
			want:  []float32{},
		},
		{
			name:  "Sparse set",
			numel: 2,
			val:   10,
			ids:   []int{1, 3},
			want:  []float32{0, 10, 0, 10},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstSize := 0
			for _, id := range tt.ids {
				if id+1 > dstSize {
					dstSize = id + 1
				}
			}
			dst := make([]float32, dstSize)
			kernels.ConstSetF32(tt.numel, tt.val, tt.ids, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestConstSetF64(t *testing.T) {
	tests := []struct {
		name  string
		numel int
		val   float64
		ids   []int
		want  []float64
	}{
		{
			name:  "Basic set",
			numel: 3,
			val:   5.0,
			ids:   []int{0, 2, 4},
			want:  []float64{5.0, 0, 5.0, 0, 5.0},
		},
		{
			name:  "Set all",
			numel: 4,
			val:   -1,
			ids:   []int{0, 1, 2, 3},
			want:  []float64{-1, -1, -1, -1},
		},
		{
			name:  "Empty",
			numel: 0,
			val:   1,
			ids:   []int{},
			want:  []float64{},
		},
		{
			name:  "Sparse set",
			numel: 2,
			val:   10,
			ids:   []int{1, 3},
			want:  []float64{0, 10, 0, 10},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstSize := 0
			for _, id := range tt.ids {
				if id+1 > dstSize {
					dstSize = id + 1
				}
			}
			dst := make([]float64, dstSize)
			kernels.ConstSetF64(tt.numel, tt.val, tt.ids, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestConstSetStridedF32(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		val     float32
		ids     []int
		want    []float32
	}{
		{
			name:    "Contiguous",
			numel:   3,
			ndims:   1,
			dims:    []int{5},
			strides: []int{1},
			val:     5.0,
			ids:     []int{0, 2, 4},
			want:    []float32{5.0, 0, 5.0, 0, 5.0},
		},
		{
			name:    "Non-contiguous (strided ids)",
			numel:   2,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			val:     10,
			ids:     []int{0, 2},
			want:    []float32{10, 10, 0},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   0,
			dims:    []int{},
			strides: []int{},
			val:     1,
			ids:     []int{},
			want:    []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstSize := 0
			for _, id := range tt.ids {
				if id+1 > dstSize {
					dstSize = id + 1
				}
			}
			dst := make([]float32, dstSize)
			kernels.ConstSetStridedF32(tt.numel, tt.ndims, tt.dims, tt.strides, tt.val, tt.ids, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestConstSetStridedF64(t *testing.T) {
	tests := []struct {
		name    string
		numel   int
		ndims   int
		dims    []int
		strides []int
		val     float64
		ids     []int
		want    []float64
	}{
		{
			name:    "Contiguous",
			numel:   3,
			ndims:   1,
			dims:    []int{5},
			strides: []int{1},
			val:     5.0,
			ids:     []int{0, 2, 4},
			want:    []float64{5.0, 0, 5.0, 0, 5.0},
		},
		{
			name:    "Non-contiguous (strided ids)",
			numel:   2,
			ndims:   2,
			dims:    []int{2, 2},
			strides: []int{1, 2},
			val:     10,
			ids:     []int{0, 2},
			want:    []float64{10, 10, 0},
		},
		{
			name:    "Empty",
			numel:   0,
			ndims:   0,
			dims:    []int{},
			strides: []int{},
			val:     1,
			ids:     []int{},
			want:    []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dstSize := 0
			for _, id := range tt.ids {
				if id+1 > dstSize {
					dstSize = id + 1
				}
			}
			dst := make([]float64, dstSize)
			kernels.ConstSetStridedF64(tt.numel, tt.ndims, tt.dims, tt.strides, tt.val, tt.ids, dst)
			if !slices.Equal(dst, tt.want) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

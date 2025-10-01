package kernels_test

import (
	"math"
	"slices"
	"testing"

	"github.com/gocnn/spark/internal/cpu/kernels"
)

func TestNaiveConv1dF32(t *testing.T) {
	tests := []struct {
		name                         string
		bSize, cIn, lIn, cOut, kSize int
		stride, padding, dilation    int
		src, kernel                  []float32
		want                         []float32
	}{
		{
			name:     "Basic 1D convolution",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			cOut:     1,
			kSize:    3,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5},
			kernel:   []float32{1, 0, -1},
			want:     []float32{-2, -2, -2},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			lIn:      0,
			cOut:     1,
			kSize:    1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{},
			kernel:   []float32{0},
			want:     []float32{},
		},
		{
			name:     "With padding",
			bSize:    2,
			cIn:      1,
			lIn:      3,
			cOut:     1,
			kSize:    2,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6},
			kernel:   []float32{1, -1},
			want:     []float32{-1, -1, -1, 3, -4, -1, -1, 6},
		},
		{
			name:     "Multiple channels",
			bSize:    1,
			cIn:      2,
			lIn:      4,
			cOut:     1,
			kSize:    3,
			stride:   2,
			padding:  0,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8},
			kernel:   []float32{1, 0, -1, 1, 0, -1},
			want:     []float32{-4},
		},
		{
			name:     "With dilation",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			cOut:     1,
			kSize:    3,
			stride:   1,
			padding:  1,
			dilation: 2,
			src:      []float32{1, 2, 3, 4, 5},
			kernel:   []float32{1, 0, -1},
			want:     []float32{-4, -4, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lOut := max(0, (tt.lIn+2*tt.padding-tt.dilation*(tt.kSize-1)-1)/tt.stride+1)
			dst := make([]float32, tt.bSize*tt.cOut*lOut)
			kernels.NaiveConv1dF32(tt.bSize, tt.cIn, tt.lIn, tt.cOut, tt.kSize, tt.stride, tt.padding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestNaiveConv1dF64(t *testing.T) {
	tests := []struct {
		name                         string
		bSize, cIn, lIn, cOut, kSize int
		stride, padding, dilation    int
		src, kernel                  []float64
		want                         []float64
	}{
		{
			name:     "Basic 1D convolution",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			cOut:     1,
			kSize:    3,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5},
			kernel:   []float64{1, 0, -1},
			want:     []float64{-2, -2, -2},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			lIn:      0,
			cOut:     1,
			kSize:    1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{},
			kernel:   []float64{0},
			want:     []float64{},
		},
		{
			name:     "With padding",
			bSize:    2,
			cIn:      1,
			lIn:      3,
			cOut:     1,
			kSize:    2,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6},
			kernel:   []float64{1, -1},
			want:     []float64{-1, -1, -1, 3, -4, -1, -1, 6},
		},
		{
			name:     "Multiple channels",
			bSize:    1,
			cIn:      2,
			lIn:      4,
			cOut:     1,
			kSize:    3,
			stride:   2,
			padding:  0,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8},
			kernel:   []float64{1, 0, -1, 1, 0, -1},
			want:     []float64{-4},
		},
		{
			name:     "With dilation",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			cOut:     1,
			kSize:    3,
			stride:   1,
			padding:  1,
			dilation: 2,
			src:      []float64{1, 2, 3, 4, 5},
			kernel:   []float64{1, 0, -1},
			want:     []float64{-4, -4, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lOut := max(0, (tt.lIn+2*tt.padding-tt.dilation*(tt.kSize-1)-1)/tt.stride+1)
			dst := make([]float64, tt.bSize*tt.cOut*lOut)
			kernels.NaiveConv1dF64(tt.bSize, tt.cIn, tt.lIn, tt.cOut, tt.kSize, tt.stride, tt.padding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestNaiveConv1dStridedF32(t *testing.T) {
	tests := []struct {
		name                                  string
		bSize, cIn, lIn, cOut, kSize          int
		stride, padding, dilation             int
		src, kernel                           []float32
		srcStrides, kernelStrides, dstStrides []int
		want                                  []float32
	}{
		{
			name:          "Basic strided convolution - contiguous",
			bSize:         1,
			cIn:           1,
			lIn:           5,
			cOut:          1,
			kSize:         3,
			stride:        1,
			padding:       0,
			dilation:      1,
			src:           []float32{1, 2, 3, 4, 5},
			kernel:        []float32{1, 0, -1},
			srcStrides:    []int{5, 5, 1},        // contiguous: [b_stride = cIn*lIn, c_stride = lIn, l_stride = 1]
			kernelStrides: []int{3, 3, 1},        // contiguous: [co_stride = cIn*kSize, ci_stride = kSize, k_stride = 1]
			dstStrides:    []int{3, 3, 1},        // contiguous: [b_stride = cOut*lOut=3, co_stride = lOut=3, l_stride = 1]
			want:          []float32{-2, -2, -2}, // Verified with PyTorch: torch.nn.Conv1d(1,1,3)(torch.tensor([[[1,2,3,4,5]]]).float()) -> [[[-2,-2,-2]]]
		},
		{
			name:          "Transposed input tensor (non-contiguous src)",
			bSize:         1,
			cIn:           2,
			lIn:           3,
			cOut:          1,
			kSize:         2,
			stride:        1,
			padding:       0,
			dilation:      1,
			src:           []float32{1, 4, 2, 5, 3, 6}, // Underlying storage: accesses as ch0:[1,2,3], ch1:[4,5,6] via strides
			kernel:        []float32{1, -1, 0.5, -0.5}, // contiguous: ch0:[1,-1], ch1:[0.5,-0.5]
			srcStrides:    []int{6, 1, 2},              // Non-contiguous: b=6 (unused), ci=1 (transposed channels), li=2 (every other element per channel)
			kernelStrides: []int{4, 2, 1},              // contiguous
			dstStrides:    []int{2, 2, 1},              // contiguous, lOut=2
			want:          []float32{-1.5, -1.5},       // Verified with PyTorch: using torch.as_strided for src strides, Conv1d(2,1,2)(input) -> [[[-1.5, -1.5]]]
		},
		{
			name:          "Strided kernel (non-contiguous kernel)",
			bSize:         1,
			cIn:           1,
			lIn:           4,
			cOut:          2,
			kSize:         2,
			stride:        1,
			padding:       0,
			dilation:      1,
			src:           []float32{1, 2, 3, 4},          // contiguous
			kernel:        []float32{1, 2, -1, 0},         // Underlying storage: co=0: k=[1,-1] (idx 0,2); co=1: k=[2,0] (idx 1,3)
			srcStrides:    []int{4, 4, 1},                 // contiguous
			kernelStrides: []int{1, 4, 2},                 // Non-contiguous: co_stride=1 (interleaved cos), ci=4 (unused since cIn=1), k_stride=2 (skip every other)
			dstStrides:    []int{6, 3, 1},                 // contiguous, lOut=3, total 6
			want:          []float32{-1, -1, -1, 2, 4, 6}, // Verified with PyTorch: as_strided for kernel, Conv1d(1,2,2)(input) -> co0:[-1,-1,-1], co1:[2,4,6]
		},
		{
			name:          "With padding and strides",
			bSize:         1,
			cIn:           1,
			lIn:           3,
			cOut:          1,
			kSize:         2,
			stride:        1,
			padding:       1,
			dilation:      1,
			src:           []float32{1, 2, 3},
			kernel:        []float32{1, -1},
			srcStrides:    []int{3, 3, 1},           // contiguous
			kernelStrides: []int{2, 2, 1},           // contiguous
			dstStrides:    []int{4, 4, 1},           // contiguous, lOut=3
			want:          []float32{-1, -1, -1, 3}, // Verified with PyTorch: Conv1d(1,1,2,padding=1)([[1,2,3]]) -> [[-1,-1,-1]] (lo0: only src[0]*(-1); lo1:1*1+2*(-1); lo2:2*1+3*(-1))
		},
		{
			name:          "Broadcast-like source (zero batch stride)",
			bSize:         2,
			cIn:           1,
			lIn:           3,
			cOut:          1,
			kSize:         2,
			stride:        1,
			padding:       0,
			dilation:      1,
			src:           []float32{1, 2, 3}, // Shared across batches via stride[0]=0
			kernel:        []float32{1, -1},
			srcStrides:    []int{0, 3, 1},            // batch_stride=0 (broadcast), others contiguous
			kernelStrides: []int{2, 2, 1},            // contiguous
			dstStrides:    []int{2, 2, 1},            // contiguous, lOut=2, total 4 per batch but shared computation
			want:          []float32{-1, -1, -1, -1}, // Verified with PyTorch: as_strided with batch_stride=0, Conv1d -> both batches [-1,-1]
		},
		{
			name:          "Empty input",
			bSize:         1,
			cIn:           1,
			lIn:           0,
			cOut:          1,
			kSize:         1,
			stride:        1,
			padding:       0,
			dilation:      1,
			src:           []float32{},
			kernel:        []float32{1},
			srcStrides:    []int{0, 0, 1},
			kernelStrides: []int{1, 1, 1},
			dstStrides:    []int{0, 0, 1},
			want:          []float32{}, // lOut=0
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lOut := max(0, (tt.lIn+2*tt.padding-tt.dilation*(tt.kSize-1)-1)/tt.stride+1)
			dst := make([]float32, tt.bSize*tt.cOut*lOut) // Note: actual storage size based on contiguous, but function handles strides
			kernels.NaiveConv1dStridedF32(tt.bSize, tt.cIn, tt.lIn, tt.cOut, tt.kSize, tt.stride, tt.padding, tt.dilation, tt.src, tt.kernel, dst, tt.srcStrides, tt.kernelStrides, tt.dstStrides)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestNaiveConv1dStridedF64(t *testing.T) {
	tests := []struct {
		name                                  string
		bSize, cIn, lIn, cOut, kSize          int
		stride, padding, dilation             int
		src, kernel                           []float64
		srcStrides, kernelStrides, dstStrides []int
		want                                  []float64
	}{
		{
			name:          "Basic strided convolution - contiguous",
			bSize:         1,
			cIn:           1,
			lIn:           5,
			cOut:          1,
			kSize:         3,
			stride:        1,
			padding:       0,
			dilation:      1,
			src:           []float64{1, 2, 3, 4, 5},
			kernel:        []float64{1, 0, -1},
			srcStrides:    []int{5, 5, 1},        // contiguous: [b_stride = cIn*lIn, c_stride = lIn, l_stride = 1]
			kernelStrides: []int{3, 3, 1},        // contiguous: [co_stride = cIn*kSize, ci_stride = kSize, k_stride = 1]
			dstStrides:    []int{3, 3, 1},        // contiguous: [b_stride = cOut*lOut=3, co_stride = lOut=3, l_stride = 1]
			want:          []float64{-2, -2, -2}, // Verified with PyTorch: torch.nn.Conv1d(1,1,3)(torch.tensor([[[1,2,3,4,5]]]).float()) -> [[[-2,-2,-2]]]
		},
		{
			name:          "Transposed input tensor (non-contiguous src)",
			bSize:         1,
			cIn:           2,
			lIn:           3,
			cOut:          1,
			kSize:         2,
			stride:        1,
			padding:       0,
			dilation:      1,
			src:           []float64{1, 4, 2, 5, 3, 6}, // Underlying storage: accesses as ch0:[1,2,3], ch1:[4,5,6] via strides
			kernel:        []float64{1, -1, 0.5, -0.5}, // contiguous: ch0:[1,-1], ch1:[0.5,-0.5]
			srcStrides:    []int{6, 1, 2},              // Non-contiguous: b=6 (unused), ci=1 (transposed channels), li=2 (every other element per channel)
			kernelStrides: []int{4, 2, 1},              // contiguous
			dstStrides:    []int{2, 2, 1},              // contiguous, lOut=2
			want:          []float64{-1.5, -1.5},       // Verified with PyTorch: using torch.as_strided for src strides, Conv1d(2,1,2)(input) -> [[[-1.5, -1.5]]]
		},
		{
			name:          "Strided kernel (non-contiguous kernel)",
			bSize:         1,
			cIn:           1,
			lIn:           4,
			cOut:          2,
			kSize:         2,
			stride:        1,
			padding:       0,
			dilation:      1,
			src:           []float64{1, 2, 3, 4},          // contiguous
			kernel:        []float64{1, 2, -1, 0},         // Underlying storage: co=0: k=[1,-1] (idx 0,2); co=1: k=[2,0] (idx 1,3)
			srcStrides:    []int{4, 4, 1},                 // contiguous
			kernelStrides: []int{1, 4, 2},                 // Non-contiguous: co_stride=1 (interleaved cos), ci=4 (unused since cIn=1), k_stride=2 (skip every other)
			dstStrides:    []int{6, 3, 1},                 // contiguous, lOut=3, total 6
			want:          []float64{-1, -1, -1, 2, 4, 6}, // Verified with PyTorch: as_strided for kernel, Conv1d(1,2,2)(input) -> co0:[-1,-1,-1], co1:[2,4,6]
		},
		{
			name:          "With padding and strides",
			bSize:         1,
			cIn:           1,
			lIn:           3,
			cOut:          1,
			kSize:         2,
			stride:        1,
			padding:       1,
			dilation:      1,
			src:           []float64{1, 2, 3},
			kernel:        []float64{1, -1},
			srcStrides:    []int{3, 3, 1},           // contiguous
			kernelStrides: []int{2, 2, 1},           // contiguous
			dstStrides:    []int{4, 4, 1},           // contiguous, lOut=3
			want:          []float64{-1, -1, -1, 3}, // Verified with PyTorch: Conv1d(1,1,2,padding=1)([[1,2,3]]) -> [[-1,-1,-1]] (lo0: only src[0]*(-1); lo1:1*1+2*(-1); lo2:2*1+3*(-1))
		},
		{
			name:          "Broadcast-like source (zero batch stride)",
			bSize:         2,
			cIn:           1,
			lIn:           3,
			cOut:          1,
			kSize:         2,
			stride:        1,
			padding:       0,
			dilation:      1,
			src:           []float64{1, 2, 3}, // Shared across batches via stride[0]=0
			kernel:        []float64{1, -1},
			srcStrides:    []int{0, 3, 1},            // batch_stride=0 (broadcast), others contiguous
			kernelStrides: []int{2, 2, 1},            // contiguous
			dstStrides:    []int{2, 2, 1},            // contiguous, lOut=2, total 4 per batch but shared computation
			want:          []float64{-1, -1, -1, -1}, // Verified with PyTorch: as_strided with batch_stride=0, Conv1d -> both batches [-1,-1]
		},
		{
			name:          "Empty input",
			bSize:         1,
			cIn:           1,
			lIn:           0,
			cOut:          1,
			kSize:         1,
			stride:        1,
			padding:       0,
			dilation:      1,
			src:           []float64{},
			kernel:        []float64{1},
			srcStrides:    []int{0, 0, 1},
			kernelStrides: []int{1, 1, 1},
			dstStrides:    []int{0, 0, 1},
			want:          []float64{}, // lOut=0
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lOut := max(0, (tt.lIn+2*tt.padding-tt.dilation*(tt.kSize-1)-1)/tt.stride+1)
			dst := make([]float64, tt.bSize*tt.cOut*lOut) // Note: actual storage size based on contiguous, but function handles strides
			kernels.NaiveConv1dStridedF64(tt.bSize, tt.cIn, tt.lIn, tt.cOut, tt.kSize, tt.stride, tt.padding, tt.dilation, tt.src, tt.kernel, dst, tt.srcStrides, tt.kernelStrides, tt.dstStrides)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIm2colConv1dF32(t *testing.T) {
	tests := []struct {
		name                         string
		bSize, cIn, lIn, cOut, kSize int
		stride, padding, dilation    int
		src, kernel                  []float32
		want                         []float32
	}{
		{
			name:     "Basic 1D convolution",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			cOut:     1,
			kSize:    3,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5},
			kernel:   []float32{1, 0, -1},
			want:     []float32{-2, -2, -2},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			lIn:      0,
			cOut:     1,
			kSize:    1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{},
			kernel:   []float32{0},
			want:     []float32{},
		},
		{
			name:     "With padding",
			bSize:    2,
			cIn:      1,
			lIn:      3,
			cOut:     1,
			kSize:    2,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6},
			kernel:   []float32{1, -1},
			want:     []float32{-1, -1, -1, 3, -4, -1, -1, 6},
		},
		{
			name:     "Multiple channels",
			bSize:    1,
			cIn:      2,
			lIn:      4,
			cOut:     1,
			kSize:    3,
			stride:   2,
			padding:  0,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8},
			kernel:   []float32{1, 0, -1, 1, 0, -1},
			want:     []float32{-4},
		},
		{
			name:     "With dilation",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			cOut:     1,
			kSize:    3,
			stride:   1,
			padding:  1,
			dilation: 2,
			src:      []float32{1, 2, 3, 4, 5},
			kernel:   []float32{1, 0, -1},
			want:     []float32{-4, -4, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lOut := max(0, (tt.lIn+2*tt.padding-tt.dilation*(tt.kSize-1)-1)/tt.stride+1)
			dst := make([]float32, tt.bSize*tt.cOut*lOut)
			kernels.Im2colConv1dF32(tt.bSize, tt.cIn, tt.lIn, tt.cOut, tt.kSize, tt.stride, tt.padding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIm2colConv1dF64(t *testing.T) {
	tests := []struct {
		name                         string
		bSize, cIn, lIn, cOut, kSize int
		stride, padding, dilation    int
		src, kernel                  []float64
		want                         []float64
	}{
		{
			name:     "Basic 1D convolution",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			cOut:     1,
			kSize:    3,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5},
			kernel:   []float64{1, 0, -1},
			want:     []float64{-2, -2, -2},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			lIn:      0,
			cOut:     1,
			kSize:    1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{},
			kernel:   []float64{0},
			want:     []float64{},
		},
		{
			name:     "With padding",
			bSize:    2,
			cIn:      1,
			lIn:      3,
			cOut:     1,
			kSize:    2,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6},
			kernel:   []float64{1, -1},
			want:     []float64{-1, -1, -1, 3, -4, -1, -1, 6},
		},
		{
			name:     "Multiple channels",
			bSize:    1,
			cIn:      2,
			lIn:      4,
			cOut:     1,
			kSize:    3,
			stride:   2,
			padding:  0,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8},
			kernel:   []float64{1, 0, -1, 1, 0, -1},
			want:     []float64{-4},
		},
		{
			name:     "With dilation",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			cOut:     1,
			kSize:    3,
			stride:   1,
			padding:  1,
			dilation: 2,
			src:      []float64{1, 2, 3, 4, 5},
			kernel:   []float64{1, 0, -1},
			want:     []float64{-4, -4, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lOut := max(0, (tt.lIn+2*tt.padding-tt.dilation*(tt.kSize-1)-1)/tt.stride+1)
			dst := make([]float64, tt.bSize*tt.cOut*lOut)
			kernels.Im2colConv1dF64(tt.bSize, tt.cIn, tt.lIn, tt.cOut, tt.kSize, tt.stride, tt.padding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestNaiveConv2dF32(t *testing.T) {
	tests := []struct {
		name                               string
		bSize, cIn, hIn, wIn, cOut, hK, wK int
		stride, padding, dilation          int
		src, kernel                        []float32
		want                               []float32
	}{
		{
			name:     "Basic 2D convolution",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			kernel:   []float32{1, 0, 0, -1},
			want:     []float32{-4, -4, -4, -4},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			hIn:      0,
			wIn:      0,
			cOut:     1,
			hK:       1,
			wK:       1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{},
			kernel:   []float32{0},
			want:     []float32{},
		},
		{
			name:     "With padding",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       3,
			wK:       3,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			kernel:   []float32{1, 1, 1, 1, 1, 1, 1, 1, 1},
			want:     []float32{12, 21, 16, 27, 45, 33, 24, 39, 28},
		},
		{
			name:     "Multiple channels",
			bSize:    1,
			cIn:      2,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
			kernel:   []float32{1, 1, 1, 1, 1, 1, 1, 1},
			want:     []float32{60, 68, 84, 92},
		},
		{
			name:     "Larger input",
			bSize:    1,
			cIn:      1,
			hIn:      4,
			wIn:      4,
			cOut:     1,
			hK:       3,
			wK:       3,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			kernel:   []float32{1, 0, -1, 1, 0, -1, 1, 0, -1},
			want:     []float32{-8, -4, -4, 10, -18, -6, -6, 21, -30, -6, -6, 33, -24, -4, -4, 26},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hOut := max(0, (tt.hIn+2*tt.padding-tt.dilation*(tt.hK-1)-1)/tt.stride+1)
			wOut := max(0, (tt.wIn+2*tt.padding-tt.dilation*(tt.wK-1)-1)/tt.stride+1)
			dst := make([]float32, tt.bSize*tt.cOut*hOut*wOut)
			kernels.NaiveConv2dF32(tt.bSize, tt.cIn, tt.hIn, tt.wIn, tt.cOut, tt.hK, tt.wK, tt.stride, tt.padding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestNaiveConv2dF64(t *testing.T) {
	tests := []struct {
		name                               string
		bSize, cIn, hIn, wIn, cOut, hK, wK int
		stride, padding, dilation          int
		src, kernel                        []float64
		want                               []float64
	}{
		{
			name:     "Basic 2D convolution",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			kernel:   []float64{1, 0, 0, -1},
			want:     []float64{-4, -4, -4, -4},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			hIn:      0,
			wIn:      0,
			cOut:     1,
			hK:       1,
			wK:       1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{},
			kernel:   []float64{0},
			want:     []float64{},
		},
		{
			name:     "With padding",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       3,
			wK:       3,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			kernel:   []float64{1, 1, 1, 1, 1, 1, 1, 1, 1},
			want:     []float64{12, 21, 16, 27, 45, 33, 24, 39, 28},
		},
		{
			name:     "Multiple channels",
			bSize:    1,
			cIn:      2,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
			kernel:   []float64{1, 1, 1, 1, 1, 1, 1, 1},
			want:     []float64{60, 68, 84, 92},
		},
		{
			name:     "Larger input",
			bSize:    1,
			cIn:      1,
			hIn:      4,
			wIn:      4,
			cOut:     1,
			hK:       3,
			wK:       3,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			kernel:   []float64{1, 0, -1, 1, 0, -1, 1, 0, -1},
			want:     []float64{-8, -4, -4, 10, -18, -6, -6, 21, -30, -6, -6, 33, -24, -4, -4, 26},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hOut := max(0, (tt.hIn+2*tt.padding-tt.dilation*(tt.hK-1)-1)/tt.stride+1)
			wOut := max(0, (tt.wIn+2*tt.padding-tt.dilation*(tt.wK-1)-1)/tt.stride+1)
			dst := make([]float64, tt.bSize*tt.cOut*hOut*wOut)
			kernels.NaiveConv2dF64(tt.bSize, tt.cIn, tt.hIn, tt.wIn, tt.cOut, tt.hK, tt.wK, tt.stride, tt.padding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIm2colConv2dF32(t *testing.T) {
	tests := []struct {
		name                               string
		bSize, cIn, hIn, wIn, cOut, hK, wK int
		stride, padding, dilation          int
		src, kernel                        []float32
		want                               []float32
	}{
		{
			name:     "Basic 2D convolution",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			kernel:   []float32{1, 0, 0, -1},
			want:     []float32{-4, -4, -4, -4},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			hIn:      0,
			wIn:      0,
			cOut:     1,
			hK:       1,
			wK:       1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{},
			kernel:   []float32{0},
			want:     []float32{},
		},
		{
			name:     "With padding",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       3,
			wK:       3,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			kernel:   []float32{1, 1, 1, 1, 1, 1, 1, 1, 1},
			want:     []float32{12, 21, 16, 27, 45, 33, 24, 39, 28},
		},
		{
			name:     "Multiple channels",
			bSize:    1,
			cIn:      2,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
			kernel:   []float32{1, 1, 1, 1, 1, 1, 1, 1},
			want:     []float32{60, 68, 84, 92},
		},
		{
			name:     "Larger input",
			bSize:    1,
			cIn:      1,
			hIn:      4,
			wIn:      4,
			cOut:     1,
			hK:       3,
			wK:       3,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			kernel:   []float32{1, 0, -1, 1, 0, -1, 1, 0, -1},
			want:     []float32{-8, -4, -4, 10, -18, -6, -6, 21, -30, -6, -6, 33, -24, -4, -4, 26},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hOut := max(0, (tt.hIn+2*tt.padding-tt.dilation*(tt.hK-1)-1)/tt.stride+1)
			wOut := max(0, (tt.wIn+2*tt.padding-tt.dilation*(tt.wK-1)-1)/tt.stride+1)
			dst := make([]float32, tt.bSize*tt.cOut*hOut*wOut)
			kernels.Im2colConv2dF32(tt.bSize, tt.cIn, tt.hIn, tt.wIn, tt.cOut, tt.hK, tt.wK, tt.stride, tt.padding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIm2colConv2dF64(t *testing.T) {
	tests := []struct {
		name                               string
		bSize, cIn, hIn, wIn, cOut, hK, wK int
		stride, padding, dilation          int
		src, kernel                        []float64
		want                               []float64
	}{
		{
			name:     "Basic 2D convolution",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			kernel:   []float64{1, 0, 0, -1},
			want:     []float64{-4, -4, -4, -4},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			hIn:      0,
			wIn:      0,
			cOut:     1,
			hK:       1,
			wK:       1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{},
			kernel:   []float64{0},
			want:     []float64{},
		},
		{
			name:     "With padding",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       3,
			wK:       3,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			kernel:   []float64{1, 1, 1, 1, 1, 1, 1, 1, 1},
			want:     []float64{12, 21, 16, 27, 45, 33, 24, 39, 28},
		},
		{
			name:     "Multiple channels",
			bSize:    1,
			cIn:      2,
			hIn:      3,
			wIn:      3,
			cOut:     1,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
			kernel:   []float64{1, 1, 1, 1, 1, 1, 1, 1},
			want:     []float64{60, 68, 84, 92},
		},
		{
			name:     "Larger input",
			bSize:    1,
			cIn:      1,
			hIn:      4,
			wIn:      4,
			cOut:     1,
			hK:       3,
			wK:       3,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			kernel:   []float64{1, 0, -1, 1, 0, -1, 1, 0, -1},
			want:     []float64{-8, -4, -4, 10, -18, -6, -6, 21, -30, -6, -6, 33, -24, -4, -4, 26},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hOut := max(0, (tt.hIn+2*tt.padding-tt.dilation*(tt.hK-1)-1)/tt.stride+1)
			wOut := max(0, (tt.wIn+2*tt.padding-tt.dilation*(tt.wK-1)-1)/tt.stride+1)
			dst := make([]float64, tt.bSize*tt.cOut*hOut*wOut)
			kernels.Im2colConv2dF64(tt.bSize, tt.cIn, tt.hIn, tt.wIn, tt.cOut, tt.hK, tt.wK, tt.stride, tt.padding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestNaiveConvTranspose1dF32(t *testing.T) {
	tests := []struct {
		name                                  string
		bSize, cIn, lIn, cOut, kSize          int
		stride, padding, outPadding, dilation int
		src, kernel                           []float32
		want                                  []float32
	}{
		{
			name:       "Basic transpose convolution",
			bSize:      1,
			cIn:        1,
			lIn:        3,
			cOut:       1,
			kSize:      3,
			stride:     1,
			padding:    0,
			outPadding: 0,
			dilation:   1,
			src:        []float32{-2, -2, -2},
			kernel:     []float32{1, 0, -1},
			want:       []float32{-2, -2, 0, 2, 2},
		},
		{
			name:       "Empty input",
			bSize:      1,
			cIn:        1,
			lIn:        0,
			cOut:       1,
			kSize:      1,
			stride:     1,
			padding:    0,
			outPadding: 0,
			dilation:   1,
			src:        []float32{},
			kernel:     []float32{0},
			want:       []float32{},
		},
		{
			name:       "With stride",
			bSize:      1,
			cIn:        1,
			lIn:        2,
			cOut:       1,
			kSize:      2,
			stride:     2,
			padding:    0,
			outPadding: 0,
			dilation:   1,
			src:        []float32{1, 2},
			kernel:     []float32{3, 4},
			want:       []float32{3, 4, 6, 8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lOut := max(0, (tt.lIn-1)*tt.stride+tt.dilation*(tt.kSize-1)+tt.outPadding-2*tt.padding+1)
			dst := make([]float32, tt.bSize*tt.cOut*lOut)
			kernels.NaiveConvTranspose1dF32(tt.bSize, tt.cIn, tt.lIn, tt.cOut, tt.kSize, tt.stride, tt.padding, tt.outPadding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestNaiveConvTranspose1dF64(t *testing.T) {
	tests := []struct {
		name                                  string
		bSize, cIn, lIn, cOut, kSize          int
		stride, padding, outPadding, dilation int
		src, kernel                           []float64
		want                                  []float64
	}{
		{
			name:       "Basic transpose convolution",
			bSize:      1,
			cIn:        1,
			lIn:        3,
			cOut:       1,
			kSize:      3,
			stride:     1,
			padding:    0,
			outPadding: 0,
			dilation:   1,
			src:        []float64{-2, -2, -2},
			kernel:     []float64{1, 0, -1},
			want:       []float64{-2, -2, 0, 2, 2},
		},
		{
			name:       "Empty input",
			bSize:      1,
			cIn:        1,
			lIn:        0,
			cOut:       1,
			kSize:      1,
			stride:     1,
			padding:    0,
			outPadding: 0,
			dilation:   1,
			src:        []float64{},
			kernel:     []float64{0},
			want:       []float64{},
		},
		{
			name:       "With stride",
			bSize:      1,
			cIn:        1,
			lIn:        2,
			cOut:       1,
			kSize:      2,
			stride:     2,
			padding:    0,
			outPadding: 0,
			dilation:   1,
			src:        []float64{1, 2},
			kernel:     []float64{3, 4},
			want:       []float64{3, 4, 6, 8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lOut := max(0, (tt.lIn-1)*tt.stride+tt.dilation*(tt.kSize-1)+tt.outPadding-2*tt.padding+1)
			dst := make([]float64, tt.bSize*tt.cOut*lOut)
			kernels.NaiveConvTranspose1dF64(tt.bSize, tt.cIn, tt.lIn, tt.cOut, tt.kSize, tt.stride, tt.padding, tt.outPadding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestNaiveConvTranspose2dF32(t *testing.T) {
	tests := []struct {
		name                                  string
		bSize, cIn, hIn, wIn, cOut, hK, wK    int
		stride, padding, outPadding, dilation int
		src, kernel                           []float32
		want                                  []float32
	}{
		{
			name:       "Basic transpose 2d",
			bSize:      1,
			cIn:        1,
			hIn:        2,
			wIn:        2,
			cOut:       1,
			hK:         2,
			wK:         2,
			stride:     1,
			padding:    0,
			outPadding: 0,
			dilation:   1,
			src:        []float32{-4, -4, -4, -4},
			kernel:     []float32{1, 0, 0, -1},
			want:       []float32{-4, -4, 0, -4, 0, 4, 0, 4, 4},
		},
		{
			name:       "Empty input",
			bSize:      1,
			cIn:        1,
			hIn:        0,
			wIn:        0,
			cOut:       1,
			hK:         1,
			wK:         1,
			stride:     1,
			padding:    0,
			outPadding: 0,
			dilation:   1,
			src:        []float32{},
			kernel:     []float32{0},
			want:       []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hOut := max(0, (tt.hIn-1)*tt.stride+tt.dilation*(tt.hK-1)+tt.outPadding-2*tt.padding+1)
			wOut := max(0, (tt.wIn-1)*tt.stride+tt.dilation*(tt.wK-1)+tt.outPadding-2*tt.padding+1)
			dst := make([]float32, tt.bSize*tt.cOut*hOut*wOut)
			kernels.NaiveConvTranspose2dF32(tt.bSize, tt.cIn, tt.hIn, tt.wIn, tt.cOut, tt.hK, tt.wK, tt.stride, tt.padding, tt.outPadding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestNaiveConvTranspose2dF64(t *testing.T) {
	tests := []struct {
		name                                  string
		bSize, cIn, hIn, wIn, cOut, hK, wK    int
		stride, padding, outPadding, dilation int
		src, kernel                           []float64
		want                                  []float64
	}{
		{
			name:       "Basic transpose 2d",
			bSize:      1,
			cIn:        1,
			hIn:        2,
			wIn:        2,
			cOut:       1,
			hK:         2,
			wK:         2,
			stride:     1,
			padding:    0,
			outPadding: 0,
			dilation:   1,
			src:        []float64{-4, -4, -4, -4},
			kernel:     []float64{1, 0, 0, -1},
			want:       []float64{-4, -4, 0, -4, 0, 4, 0, 4, 4},
		},
		{
			name:       "Empty input",
			bSize:      1,
			cIn:        1,
			hIn:        0,
			wIn:        0,
			cOut:       1,
			hK:         1,
			wK:         1,
			stride:     1,
			padding:    0,
			outPadding: 0,
			dilation:   1,
			src:        []float64{},
			kernel:     []float64{0},
			want:       []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hOut := max(0, (tt.hIn-1)*tt.stride+tt.dilation*(tt.hK-1)+tt.outPadding-2*tt.padding+1)
			wOut := max(0, (tt.wIn-1)*tt.stride+tt.dilation*(tt.wK-1)+tt.outPadding-2*tt.padding+1)
			dst := make([]float64, tt.bSize*tt.cOut*hOut*wOut)
			kernels.NaiveConvTranspose2dF64(tt.bSize, tt.cIn, tt.hIn, tt.wIn, tt.cOut, tt.hK, tt.wK, tt.stride, tt.padding, tt.outPadding, tt.dilation, tt.src, tt.kernel, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestAvgPool2dF32(t *testing.T) {
	tests := []struct {
		name                       string
		bSize, c, hIn, wIn, hK, wK int
		hStride, wStride           int
		src                        []float32
		want                       []float32
	}{
		{
			name:    "Basic avg pool",
			bSize:   1,
			c:       1,
			hIn:     3,
			wIn:     3,
			hK:      2,
			wK:      2,
			hStride: 1,
			wStride: 1,
			src:     []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want:    []float32{3, 4, 6, 7},
		},
		{
			name:    "Empty input",
			bSize:   1,
			c:       1,
			hIn:     0,
			wIn:     0,
			hK:      1,
			wK:      1,
			hStride: 1,
			wStride: 1,
			src:     []float32{},
			want:    []float32{},
		},
		{
			name:    "Larger input with stride",
			bSize:   1,
			c:       1,
			hIn:     4,
			wIn:     4,
			hK:      2,
			wK:      2,
			hStride: 2,
			wStride: 2,
			src:     []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			want:    []float32{3.5, 5.5, 11.5, 13.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hOut := max(0, (tt.hIn-tt.hK)/tt.hStride+1)
			wOut := max(0, (tt.wIn-tt.wK)/tt.wStride+1)
			dst := make([]float32, tt.bSize*tt.c*hOut*wOut)
			kernels.AvgPool2dF32(tt.bSize, tt.c, tt.hIn, tt.wIn, tt.hK, tt.wK, tt.hStride, tt.wStride, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestAvgPool2dF64(t *testing.T) {
	tests := []struct {
		name                       string
		bSize, c, hIn, wIn, hK, wK int
		hStride, wStride           int
		src                        []float64
		want                       []float64
	}{
		{
			name:    "Basic avg pool",
			bSize:   1,
			c:       1,
			hIn:     3,
			wIn:     3,
			hK:      2,
			wK:      2,
			hStride: 1,
			wStride: 1,
			src:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want:    []float64{3, 4, 6, 7},
		},
		{
			name:    "Empty input",
			bSize:   1,
			c:       1,
			hIn:     0,
			wIn:     0,
			hK:      1,
			wK:      1,
			hStride: 1,
			wStride: 1,
			src:     []float64{},
			want:    []float64{},
		},
		{
			name:    "Larger input with stride",
			bSize:   1,
			c:       1,
			hIn:     4,
			wIn:     4,
			hK:      2,
			wK:      2,
			hStride: 2,
			wStride: 2,
			src:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			want:    []float64{3.5, 5.5, 11.5, 13.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hOut := max(0, (tt.hIn-tt.hK)/tt.hStride+1)
			wOut := max(0, (tt.wIn-tt.wK)/tt.wStride+1)
			dst := make([]float64, tt.bSize*tt.c*hOut*wOut)
			kernels.AvgPool2dF64(tt.bSize, tt.c, tt.hIn, tt.wIn, tt.hK, tt.wK, tt.hStride, tt.wStride, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestMaxPool2dF32(t *testing.T) {
	tests := []struct {
		name                       string
		bSize, c, hIn, wIn, hK, wK int
		hStride, wStride           int
		src                        []float32
		want                       []float32
	}{
		{
			name:    "Basic max pool",
			bSize:   1,
			c:       1,
			hIn:     3,
			wIn:     3,
			hK:      2,
			wK:      2,
			hStride: 1,
			wStride: 1,
			src:     []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want:    []float32{5, 6, 8, 9},
		},
		{
			name:    "Empty input",
			bSize:   1,
			c:       1,
			hIn:     0,
			wIn:     0,
			hK:      1,
			wK:      1,
			hStride: 1,
			wStride: 1,
			src:     []float32{},
			want:    []float32{},
		},
		{
			name:    "Larger input with stride",
			bSize:   1,
			c:       1,
			hIn:     4,
			wIn:     4,
			hK:      2,
			wK:      2,
			hStride: 2,
			wStride: 2,
			src:     []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			want:    []float32{6, 8, 14, 16},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hOut := max(0, (tt.hIn-tt.hK)/tt.hStride+1)
			wOut := max(0, (tt.wIn-tt.wK)/tt.wStride+1)
			dst := make([]float32, tt.bSize*tt.c*hOut*wOut)
			kernels.MaxPool2dF32(tt.bSize, tt.c, tt.hIn, tt.wIn, tt.hK, tt.wK, tt.hStride, tt.wStride, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestMaxPool2dF64(t *testing.T) {
	tests := []struct {
		name                       string
		bSize, c, hIn, wIn, hK, wK int
		hStride, wStride           int
		src                        []float64
		want                       []float64
	}{
		{
			name:    "Basic max pool",
			bSize:   1,
			c:       1,
			hIn:     3,
			wIn:     3,
			hK:      2,
			wK:      2,
			hStride: 1,
			wStride: 1,
			src:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want:    []float64{5, 6, 8, 9},
		},
		{
			name:    "Empty input",
			bSize:   1,
			c:       1,
			hIn:     0,
			wIn:     0,
			hK:      1,
			wK:      1,
			hStride: 1,
			wStride: 1,
			src:     []float64{},
			want:    []float64{},
		},
		{
			name:    "Larger input with stride",
			bSize:   1,
			c:       1,
			hIn:     4,
			wIn:     4,
			hK:      2,
			wK:      2,
			hStride: 2,
			wStride: 2,
			src:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			want:    []float64{6, 8, 14, 16},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hOut := max(0, (tt.hIn-tt.hK)/tt.hStride+1)
			wOut := max(0, (tt.wIn-tt.wK)/tt.wStride+1)
			dst := make([]float64, tt.bSize*tt.c*hOut*wOut)
			kernels.MaxPool2dF64(tt.bSize, tt.c, tt.hIn, tt.wIn, tt.hK, tt.wK, tt.hStride, tt.wStride, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestUpsampleNearest2dF32(t *testing.T) {
	tests := []struct {
		name                           string
		bSize, c, hIn, wIn, hOut, wOut int
		hScale, wScale                 float64
		src                            []float32
		want                           []float32
	}{
		{
			name:   "Upsample 3x3 to 6x6",
			bSize:  1,
			c:      1,
			hIn:    3,
			wIn:    3,
			hOut:   6,
			wOut:   6,
			hScale: 2,
			wScale: 2,
			src:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want:   []float32{1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 7, 7, 8, 8, 9, 9},
		},
		{
			name:   "Empty input",
			bSize:  1,
			c:      1,
			hIn:    0,
			wIn:    0,
			hOut:   0,
			wOut:   0,
			hScale: 1,
			wScale: 1,
			src:    []float32{},
			want:   []float32{},
		},
		{
			name:   "Upsample 2x2 to 3x3",
			bSize:  1,
			c:      1,
			hIn:    2,
			wIn:    2,
			hOut:   3,
			wOut:   3,
			hScale: 1.5,
			wScale: 1.5,
			src:    []float32{1, 2, 3, 4},
			want:   []float32{1, 2, 2, 3, 4, 4, 3, 4, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.bSize*tt.c*tt.hOut*tt.wOut)
			kernels.UpsampleNearest2dF32(tt.bSize, tt.c, tt.hIn, tt.wIn, tt.hOut, tt.wOut, tt.hScale, tt.wScale, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestUpsampleNearest2dF64(t *testing.T) {
	tests := []struct {
		name                           string
		bSize, c, hIn, wIn, hOut, wOut int
		hScale, wScale                 float64
		src                            []float64
		want                           []float64
	}{
		{
			name:   "Upsample 3x3 to 6x6",
			bSize:  1,
			c:      1,
			hIn:    3,
			wIn:    3,
			hOut:   6,
			wOut:   6,
			hScale: 2,
			wScale: 2,
			src:    []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want:   []float64{1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 7, 7, 8, 8, 9, 9},
		},
		{
			name:   "Empty input",
			bSize:  1,
			c:      1,
			hIn:    0,
			wIn:    0,
			hOut:   0,
			wOut:   0,
			hScale: 1,
			wScale: 1,
			src:    []float64{},
			want:   []float64{},
		},
		{
			name:   "Upsample 2x2 to 3x3",
			bSize:  1,
			c:      1,
			hIn:    2,
			wIn:    2,
			hOut:   3,
			wOut:   3,
			hScale: 1.5,
			wScale: 1.5,
			src:    []float64{1, 2, 3, 4},
			want:   []float64{1, 2, 2, 3, 4, 4, 3, 4, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, tt.bSize*tt.c*tt.hOut*tt.wOut)
			kernels.UpsampleNearest2dF64(tt.bSize, tt.c, tt.hIn, tt.wIn, tt.hOut, tt.wOut, tt.hScale, tt.wScale, tt.src, dst)
			if !slices.EqualFunc(dst, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestIm2colF32(t *testing.T) {
	tests := []struct {
		name                                     string
		bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int
		stride, padding, dilation                int
		src                                      []float32
		want                                     []float32
	}{
		{
			name:     "Basic im2col2d",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			hOut:     2,
			wOut:     2,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want:     []float32{1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			hIn:      0,
			wIn:      0,
			hOut:     0,
			wOut:     0,
			hK:       1,
			wK:       1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{},
			want:     []float32{},
		},
		{
			name:     "With padding",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			hOut:     4,
			wOut:     4,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want:     []float32{0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3, 0, 0, 1, 0, 4, 1, 2, 4, 5, 2, 3, 5, 6, 3, 0, 6, 0, 0, 4, 0, 7, 4, 5, 7, 8, 5, 6, 8, 9, 6, 0, 9, 0, 0, 7, 0, 0, 7, 8, 0, 0, 8, 9, 0, 0, 9, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			colSize := tt.bSize * tt.hOut * tt.wOut * tt.cIn * tt.hK * tt.wK
			col := make([]float32, colSize)
			kernels.Im2colF32(tt.bSize, tt.cIn, tt.hIn, tt.wIn, tt.hOut, tt.wOut, tt.hK, tt.wK, tt.stride, tt.padding, tt.dilation, tt.src, col)
			if !slices.EqualFunc(col, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", col, tt.want)
			}
		})
	}
}

func TestIm2colF64(t *testing.T) {
	tests := []struct {
		name                                     string
		bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int
		stride, padding, dilation                int
		src                                      []float64
		want                                     []float64
	}{
		{
			name:     "Basic im2col2d",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			hOut:     2,
			wOut:     2,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want:     []float64{1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			hIn:      0,
			wIn:      0,
			hOut:     0,
			wOut:     0,
			hK:       1,
			wK:       1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{},
			want:     []float64{},
		},
		{
			name:     "With padding",
			bSize:    1,
			cIn:      1,
			hIn:      3,
			wIn:      3,
			hOut:     4,
			wOut:     4,
			hK:       2,
			wK:       2,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			want:     []float64{0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3, 0, 0, 1, 0, 4, 1, 2, 4, 5, 2, 3, 5, 6, 3, 0, 6, 0, 0, 4, 0, 7, 4, 5, 7, 8, 5, 6, 8, 9, 6, 0, 9, 0, 0, 7, 0, 0, 7, 8, 0, 0, 8, 9, 0, 0, 9, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			colSize := tt.bSize * tt.hOut * tt.wOut * tt.cIn * tt.hK * tt.wK
			col := make([]float64, colSize)
			kernels.Im2colF64(tt.bSize, tt.cIn, tt.hIn, tt.wIn, tt.hOut, tt.wOut, tt.hK, tt.wK, tt.stride, tt.padding, tt.dilation, tt.src, col)
			if !slices.EqualFunc(col, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", col, tt.want)
			}
		})
	}
}

func TestIm2col1dF32(t *testing.T) {
	tests := []struct {
		name                         string
		bSize, cIn, lIn, lOut, kSize int
		stride, padding, dilation    int
		src                          []float32
		want                         []float32
	}{
		{
			name:     "Basic im2col1d",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			lOut:     3,
			kSize:    3,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{1, 2, 3, 4, 5},
			want:     []float32{1, 2, 3, 2, 3, 4, 3, 4, 5},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			lIn:      0,
			lOut:     0,
			kSize:    1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float32{},
			want:     []float32{},
		},
		{
			name:     "With padding",
			bSize:    1,
			cIn:      1,
			lIn:      3,
			lOut:     3,
			kSize:    2,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float32{1, 2, 3},
			want:     []float32{0, 1, 1, 2, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			colSize := tt.bSize * tt.lOut * tt.cIn * tt.kSize
			col := make([]float32, colSize)
			kernels.Im2col1dF32(tt.bSize, tt.cIn, tt.lIn, tt.lOut, tt.kSize, tt.stride, tt.padding, tt.dilation, tt.src, col)
			if !slices.EqualFunc(col, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", col, tt.want)
			}
		})
	}
}

func TestIm2col1dF64(t *testing.T) {
	tests := []struct {
		name                         string
		bSize, cIn, lIn, lOut, kSize int
		stride, padding, dilation    int
		src                          []float64
		want                         []float64
	}{
		{
			name:     "Basic im2col1d",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			lOut:     3,
			kSize:    3,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{1, 2, 3, 4, 5},
			want:     []float64{1, 2, 3, 2, 3, 4, 3, 4, 5},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			lIn:      0,
			lOut:     0,
			kSize:    1,
			stride:   1,
			padding:  0,
			dilation: 1,
			src:      []float64{},
			want:     []float64{},
		},
		{
			name:     "With padding",
			bSize:    1,
			cIn:      1,
			lIn:      3,
			lOut:     3,
			kSize:    2,
			stride:   1,
			padding:  1,
			dilation: 1,
			src:      []float64{1, 2, 3},
			want:     []float64{0, 1, 1, 2, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			colSize := tt.bSize * tt.lOut * tt.cIn * tt.kSize
			col := make([]float64, colSize)
			kernels.Im2col1dF64(tt.bSize, tt.cIn, tt.lIn, tt.lOut, tt.kSize, tt.stride, tt.padding, tt.dilation, tt.src, col)
			if !slices.EqualFunc(col, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", col, tt.want)
			}
		})
	}
}

func TestCol2im1dF32(t *testing.T) {
	tests := []struct {
		name                         string
		bSize, cIn, lIn, lOut, kSize int
		stride, padding, dilation    int
		col                          []float32
		want                         []float32
	}{
		{
			name:     "Basic col2im1d",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			lOut:     3,
			kSize:    3,
			stride:   1,
			padding:  0,
			dilation: 1,
			col:      []float32{1, 2, 3, 2, 3, 4, 3, 4, 5},
			want:     []float32{1, 4, 9, 8, 5},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			lIn:      0,
			lOut:     0,
			kSize:    1,
			stride:   1,
			padding:  0,
			dilation: 1,
			col:      []float32{},
			want:     []float32{},
		},
		{
			name:     "With padding",
			bSize:    1,
			cIn:      1,
			lIn:      3,
			lOut:     3,
			kSize:    2,
			stride:   1,
			padding:  1,
			dilation: 1,
			col:      []float32{0, 1, 1, 2, 2, 3},
			want:     []float32{2, 4, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			im := make([]float32, tt.bSize*tt.cIn*tt.lIn)
			kernels.Col2im1dF32(tt.bSize, tt.cIn, tt.lIn, tt.lOut, tt.kSize, tt.stride, tt.padding, tt.dilation, tt.col, im)
			if !slices.EqualFunc(im, tt.want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
				t.Errorf("got %v, want %v", im, tt.want)
			}
		})
	}
}

func TestCol2im1dF64(t *testing.T) {
	tests := []struct {
		name                         string
		bSize, cIn, lIn, lOut, kSize int
		stride, padding, dilation    int
		col                          []float64
		want                         []float64
	}{
		{
			name:     "Basic col2im1d",
			bSize:    1,
			cIn:      1,
			lIn:      5,
			lOut:     3,
			kSize:    3,
			stride:   1,
			padding:  0,
			dilation: 1,
			col:      []float64{1, 2, 3, 2, 3, 4, 3, 4, 5},
			want:     []float64{1, 4, 9, 8, 5},
		},
		{
			name:     "Empty input",
			bSize:    1,
			cIn:      1,
			lIn:      0,
			lOut:     0,
			kSize:    1,
			stride:   1,
			padding:  0,
			dilation: 1,
			col:      []float64{},
			want:     []float64{},
		},
		{
			name:     "With padding",
			bSize:    1,
			cIn:      1,
			lIn:      3,
			lOut:     3,
			kSize:    2,
			stride:   1,
			padding:  1,
			dilation: 1,
			col:      []float64{0, 1, 1, 2, 2, 3},
			want:     []float64{2, 4, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			im := make([]float64, tt.bSize*tt.cIn*tt.lIn)
			kernels.Col2im1dF64(tt.bSize, tt.cIn, tt.lIn, tt.lOut, tt.kSize, tt.stride, tt.padding, tt.dilation, tt.col, im)
			if !slices.EqualFunc(im, tt.want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
				t.Errorf("got %v, want %v", im, tt.want)
			}
		})
	}
}

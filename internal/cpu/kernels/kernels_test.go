package kernels_test

import (
	"testing"

	"github.com/gocnn/spark/internal/cpu/kernels"
)

func TestIsContiguous(t *testing.T) {
	tests := []struct {
		numDims int
		dims    []int
		strides []int
		want    bool
	}{
		{0, nil, nil, true},
		{1, []int{3}, []int{1}, true},
		{1, []int{3}, []int{2}, false},
		{2, []int{2, 3}, []int{3, 1}, true},
		{2, []int{2, 3}, []int{1, 2}, false},
		{2, []int{3, 2}, []int{2, 1}, true},
		{2, []int{3, 2}, []int{1, 3}, false},
		{3, []int{2, 2, 2}, []int{4, 2, 1}, true},
		{3, []int{2, 2, 2}, []int{4, 1, 2}, false},
		{3, []int{2, 2, 2}, []int{1, 2, 4}, false},
	}

	for _, tt := range tests {
		got := kernels.IsContiguous(tt.numDims, tt.dims, tt.strides)
		if got != tt.want {
			t.Errorf("got %v, want %v", got, tt.want)
		}
	}
}

func TestGetStridedIndex(t *testing.T) {
	tests := []struct {
		linearIdx int
		numDims   int
		dims      []int
		strides   []int
		want      int
	}{
		// 0D
		{0, 0, nil, nil, 0},
		// 1D contiguous
		{0, 1, []int{3}, []int{1}, 0},
		{1, 1, []int{3}, []int{1}, 1},
		{2, 1, []int{3}, []int{1}, 2},
		// 1D strided
		{0, 1, []int{3}, []int{2}, 0},
		{1, 1, []int{3}, []int{2}, 2},
		{2, 1, []int{3}, []int{2}, 4},
		// 2D contiguous (row-major)
		{0, 2, []int{2, 3}, []int{3, 1}, 0}, // (0,0)
		{1, 2, []int{2, 3}, []int{3, 1}, 1}, // (0,1)
		{2, 2, []int{2, 3}, []int{3, 1}, 2}, // (0,2)
		{3, 2, []int{2, 3}, []int{3, 1}, 3}, // (1,0)
		{4, 2, []int{2, 3}, []int{3, 1}, 4}, // (1,1)
		{5, 2, []int{2, 3}, []int{3, 1}, 5}, // (1,2)
		// 2D transposed (column-major)
		{0, 2, []int{2, 3}, []int{1, 2}, 0}, // (0,0)
		{1, 2, []int{2, 3}, []int{1, 2}, 2}, // (0,1)
		{2, 2, []int{2, 3}, []int{1, 2}, 4}, // (0,2)
		{3, 2, []int{2, 3}, []int{1, 2}, 1}, // (1,0)
		{4, 2, []int{2, 3}, []int{1, 2}, 3}, // (1,1)
		{5, 2, []int{2, 3}, []int{1, 2}, 5}, // (1,2)
		// 3D contiguous
		{0, 3, []int{2, 2, 2}, []int{4, 2, 1}, 0}, // (0,0,0)
		{1, 3, []int{2, 2, 2}, []int{4, 2, 1}, 1}, // (0,0,1)
		{2, 3, []int{2, 2, 2}, []int{4, 2, 1}, 2}, // (0,1,0)
		{3, 3, []int{2, 2, 2}, []int{4, 2, 1}, 3}, // (0,1,1)
		{4, 3, []int{2, 2, 2}, []int{4, 2, 1}, 4}, // (1,0,0)
		{5, 3, []int{2, 2, 2}, []int{4, 2, 1}, 5}, // (1,0,1)
		{6, 3, []int{2, 2, 2}, []int{4, 2, 1}, 6}, // (1,1,0)
		{7, 3, []int{2, 2, 2}, []int{4, 2, 1}, 7}, // (1,1,1)
		// 3D strided (transposed dims 1 and 2)
		{0, 3, []int{2, 2, 2}, []int{4, 1, 2}, 0}, // (0,0,0)
		{1, 3, []int{2, 2, 2}, []int{4, 1, 2}, 2}, // (0,0,1)
		{2, 3, []int{2, 2, 2}, []int{4, 1, 2}, 1}, // (0,1,0)
		{3, 3, []int{2, 2, 2}, []int{4, 1, 2}, 3}, // (0,1,1)
		{4, 3, []int{2, 2, 2}, []int{4, 1, 2}, 4}, // (1,0,0)
		{5, 3, []int{2, 2, 2}, []int{4, 1, 2}, 6}, // (1,0,1)
		{6, 3, []int{2, 2, 2}, []int{4, 1, 2}, 5}, // (1,1,0)
		{7, 3, []int{2, 2, 2}, []int{4, 1, 2}, 7}, // (1,1,1)
	}

	for _, tt := range tests {
		got := kernels.GetStridedIndex(tt.linearIdx, tt.numDims, tt.dims, tt.strides)
		if got != tt.want {
			t.Errorf("got %v, want %v", got, tt.want)
		}
	}
}

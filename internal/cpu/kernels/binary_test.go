package kernels_test

import (
	"slices"
	"testing"

	"github.com/gocnn/candy/internal/cpu/kernels"
)

// Arithmetic tests

func TestBAddF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{5, 7, 9}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{15}},
		{4, []float32{-1, -2, 3, 4}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{0, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.BAddF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBAddF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{5, 7, 9}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{15}},
		{4, []float64{-1, -2, 3, 4}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{0, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.BAddF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBAddU8(t *testing.T) {
	tests := []struct {
		name      string
		numel     int
		x1, x2, y []uint8
		want      []uint8
	}{
		{
			name:  "Basic addition",
			numel: 3,
			x1:    []uint8{1, 2, 3},
			x2:    []uint8{4, 5, 6},
			y:     make([]uint8, 3),
			want:  []uint8{5, 7, 9},
		},
		{
			name:  "Empty array",
			numel: 0,
			x1:    []uint8{},
			x2:    []uint8{},
			y:     []uint8{},
			want:  []uint8{},
		},
		{
			name:  "Single element",
			numel: 1,
			x1:    []uint8{100},
			x2:    []uint8{50},
			y:     make([]uint8, 1),
			want:  []uint8{150},
		},
		{
			name:  "Overflow test",
			numel: 2,
			x1:    []uint8{200, 255},
			x2:    []uint8{100, 1},
			y:     make([]uint8, 2),
			want:  []uint8{44, 0}, // 200+100=300%256=44, 255+1=256%256=0
		},
		{
			name:  "Random values",
			numel: 4,
			x1:    []uint8{42, 127, 0, 200},
			x2:    []uint8{10, 50, 255, 80},
			y:     make([]uint8, 4),
			want:  []uint8{52, 177, 255, 24}, // Verified with PyTorch: torch.tensor([42,127,0,200], dtype=torch.uint8) + torch.tensor([10,50,255,80], dtype=torch.uint8)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kernels.BAddU8(tt.numel, tt.x1, tt.x2, tt.y)
			if !slices.Equal(tt.y, tt.want) {
				t.Errorf("got %v, want %v", tt.y, tt.want)
			}
		})
	}
}

func TestBAddU32(t *testing.T) {
	tests := []struct {
		name      string
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{
			name:  "Basic addition",
			numel: 3,
			x1:    []uint32{1000, 2000, 3000},
			x2:    []uint32{4000, 5000, 6000},
			y:     make([]uint32, 3),
			want:  []uint32{5000, 7000, 9000},
		},
		{
			name:  "Empty array",
			numel: 0,
			x1:    []uint32{},
			x2:    []uint32{},
			y:     []uint32{},
			want:  []uint32{},
		},
		{
			name:  "Single element",
			numel: 1,
			x1:    []uint32{100000},
			x2:    []uint32{50000},
			y:     make([]uint32, 1),
			want:  []uint32{150000},
		},
		{
			name:  "Overflow test",
			numel: 2,
			x1:    []uint32{4294967295, 4000000000},
			x2:    []uint32{1, 4000000000},
			y:     make([]uint32, 2),
			want:  []uint32{0, 3705032704}, // 4294967295+1=2^32=0, 4000000000+4000000000=8000000000%2^32=3705032704
		},
		{
			name:  "Random values",
			numel: 4,
			x1:    []uint32{123456, 789012, 345678, 901234},
			x2:    []uint32{654321, 210987, 876543, 987654},
			y:     make([]uint32, 4),
			want:  []uint32{777777, 999999, 1222221, 1888888}, // Verified with PyTorch: torch.tensor([123456,789012,345678,901234], dtype=torch.uint32) + torch.tensor([654321,210987,876543,987654], dtype=torch.uint32)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kernels.BAddU32(tt.numel, tt.x1, tt.x2, tt.y)
			if !slices.Equal(tt.y, tt.want) {
				t.Errorf("got %v, want %v", tt.y, tt.want)
			}
		})
	}
}

func TestBAddI64(t *testing.T) {
	tests := []struct {
		name      string
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{
			name:  "Basic addition",
			numel: 3,
			x1:    []int64{1, 2, 3},
			x2:    []int64{4, 5, 6},
			y:     make([]int64, 3),
			want:  []int64{5, 7, 9},
		},
		{
			name:  "Empty array",
			numel: 0,
			x1:    []int64{},
			x2:    []int64{},
			y:     []int64{},
			want:  []int64{},
		},
		{
			name:  "Single element",
			numel: 1,
			x1:    []int64{1000000000000},
			x2:    []int64{500000000000},
			y:     make([]int64, 1),
			want:  []int64{1500000000000},
		},
		{
			name:  "Overflow test",
			numel: 4,
			x1:    []int64{9223372036854775807, -9223372036854775808, 4611686018427387904, -4611686018427387905},
			x2:    []int64{1, -1, 4611686018427387904, -4611686018427387904},
			y:     make([]int64, 4),
			want:  []int64{-9223372036854775808, 9223372036854775807, -9223372036854775808, 9223372036854775807}, // max +1 overflows to min, min -1 overflows to max, 2^62 + 2^62 overflows to min, -(2^62+1) + (-2^62) overflows to max; Verified with PyTorch
		},
		{
			name:  "Random values",
			numel: 4,
			x1:    []int64{123456789012345, -67890123456789, 34567890123456, -90123456789012},
			x2:    []int64{65432109876543, 21098765432109, -87654321098765, 98765432109876},
			y:     make([]int64, 4),
			want:  []int64{188888898888888, -46791358024680, -53086430975309, 8641975320864}, // Verified with PyTorch: torch.tensor([123456789012345, -67890123456789, 34567890123456, -90123456789012], dtype=torch.int64) + torch.tensor([65432109876543, 21098765432109, -87654321098765, 98765432109876], dtype=torch.int64)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kernels.BAddI64(tt.numel, tt.x1, tt.x2, tt.y)
			if !slices.Equal(tt.y, tt.want) {
				t.Errorf("got %v, want %v", tt.y, tt.want)
			}
		})
	}
}

func TestBAddStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{5, 7, 9}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{4, 0, 5, 0, 6}, make([]float32, 3), []float32{5, 7, 9}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30, 40, 50, 60}, make([]float32, 6), []float32{11, 22, 33, 44, 55, 66}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 40, 20, 50, 30, 60}, make([]float32, 6), []float32{11, 22, 33, 44, 55, 66}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{11, 22, 33, 44, 55, 66, 77, 88}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{11, 22, 33, 44, 55, 66, 77, 88}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BAddStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBAddStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{5, 7, 9}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{4, 0, 5, 0, 6}, make([]float64, 3), []float64{5, 7, 9}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 20, 30, 40, 50, 60}, make([]float64, 6), []float64{11, 22, 33, 44, 55, 66}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 40, 20, 50, 30, 60}, make([]float64, 6), []float64{11, 22, 33, 44, 55, 66}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{11, 22, 33, 44, 55, 66, 77, 88}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{11, 22, 33, 44, 55, 66, 77, 88}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BAddStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBAddStridedU8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{1, 2, 3}, []uint8{4, 5, 6}, make([]uint8, 3), []uint8{5, 7, 9}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{1, 2, 3}, []uint8{4, 0, 5, 0, 6}, make([]uint8, 3), []uint8{5, 7, 9}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{1, 2, 3, 4, 5, 6}, []uint8{10, 20, 30, 40, 50, 60}, make([]uint8, 6), []uint8{11, 22, 33, 44, 55, 66}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{1, 2, 3, 4, 5, 6}, []uint8{10, 40, 20, 50, 30, 60}, make([]uint8, 6), []uint8{11, 22, 33, 44, 55, 66}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{1, 2, 3, 4, 5, 6, 7, 8}, []uint8{10, 20, 30, 40, 50, 60, 70, 80}, make([]uint8, 8), []uint8{11, 22, 33, 44, 55, 66, 77, 88}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{1, 3, 2, 4, 5, 7, 6, 8}, []uint8{10, 20, 30, 40, 50, 60, 70, 80}, make([]uint8, 8), []uint8{11, 22, 33, 44, 55, 66, 77, 88}},
		// 1D with overflow
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{200, 201, 202}, []uint8{100, 101, 102}, make([]uint8, 3), []uint8{44, 46, 48}},
		// 1D strided with overflow
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{200, 201, 202}, []uint8{100, 0, 101, 0, 102}, make([]uint8, 3), []uint8{44, 46, 48}},
		// Random 2D contiguous with overflow
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{142, 67, 76, 14, 26, 135}, []uint8{200, 120, 170, 237, 182, 206}, make([]uint8, 6), []uint8{86, 187, 246, 251, 208, 85}},
		// Random 2D x2 strided with overflow
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{142, 67, 76, 14, 26, 135}, []uint8{198, 166, 132, 116, 130, 213}, make([]uint8, 6), []uint8{84, 199, 206, 180, 142, 92}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BAddStridedU8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBAddStridedU32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint32
		want         []uint32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{1, 2, 3}, []uint32{4, 5, 6}, make([]uint32, 3), []uint32{5, 7, 9}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{1, 2, 3}, []uint32{4, 0, 5, 0, 6}, make([]uint32, 3), []uint32{5, 7, 9}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{1, 2, 3, 4, 5, 6}, []uint32{10, 20, 30, 40, 50, 60}, make([]uint32, 6), []uint32{11, 22, 33, 44, 55, 66}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{1, 2, 3, 4, 5, 6}, []uint32{10, 40, 20, 50, 30, 60}, make([]uint32, 6), []uint32{11, 22, 33, 44, 55, 66}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1, 2, 3, 4, 5, 6, 7, 8}, []uint32{10, 20, 30, 40, 50, 60, 70, 80}, make([]uint32, 8), []uint32{11, 22, 33, 44, 55, 66, 77, 88}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1, 3, 2, 4, 5, 7, 6, 8}, []uint32{10, 20, 30, 40, 50, 60, 70, 80}, make([]uint32, 8), []uint32{11, 22, 33, 44, 55, 66, 77, 88}},
		// 1D with overflow
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{4294967295, 4294967294, 4294967293}, []uint32{1, 2, 3}, make([]uint32, 3), []uint32{0, 0, 0}},
		// 1D strided with overflow
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{4294967295, 4294967294, 4294967293}, []uint32{1, 0, 2, 0, 3}, make([]uint32, 3), []uint32{0, 0, 0}},
		// Random 2D contiguous with overflow
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{1972458954, 1433267572, 613608295, 2795544706, 242285876, 3100961111}, []uint32{4031053213, 3344769, 4261516219, 2652062880, 2627030329, 30349564}, make([]uint32, 6), []uint32{1708544871, 1436612341, 580157218, 1152640290, 2869316205, 3131310675}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BAddStridedU32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBAddStridedI64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []int64
		want         []int64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{1, 2, 3}, []int64{4, 5, 6}, make([]int64, 3), []int64{5, 7, 9}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{1, 2, 3}, []int64{4, 0, 5, 0, 6}, make([]int64, 3), []int64{5, 7, 9}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{1, 2, 3, 4, 5, 6}, []int64{10, 20, 30, 40, 50, 60}, make([]int64, 6), []int64{11, 22, 33, 44, 55, 66}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{1, 2, 3, 4, 5, 6}, []int64{10, 40, 20, 50, 30, 60}, make([]int64, 6), []int64{11, 22, 33, 44, 55, 66}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{1, 2, 3, 4, 5, 6, 7, 8}, []int64{10, 20, 30, 40, 50, 60, 70, 80}, make([]int64, 8), []int64{11, 22, 33, 44, 55, 66, 77, 88}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{1, 3, 2, 4, 5, 7, 6, 8}, []int64{10, 20, 30, 40, 50, 60, 70, 80}, make([]int64, 8), []int64{11, 22, 33, 44, 55, 66, 77, 88}},
		// Random 2D contiguous with large values
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{-21171880, -111354918, -867845122, 743373311, -652577605, 43620622}, []int64{42767805, -559061571, -908206741, -118348045, -667603976, 658023396}, make([]int64, 6), []int64{21595925, -670416489, -1776051863, 625025266, -1320181581, 701644018}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BAddStridedI64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBSubF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{-3, -3, -3}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{-5}},
		{4, []float32{-1, -2, 3, 4}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{-2, -4, 6, 8}},
	}

	for _, tt := range tests {
		kernels.BSubF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBSubF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{-3, -3, -3}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{-5}},
		{4, []float64{-1, -2, 3, 4}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{-2, -4, 6, 8}},
	}

	for _, tt := range tests {
		kernels.BSubF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBSubU8(t *testing.T) {
	tests := []struct {
		name      string
		numel     int
		x1, x2, y []uint8
		want      []uint8
	}{
		{
			name:  "Basic subtraction",
			numel: 3,
			x1:    []uint8{10, 20, 30},
			x2:    []uint8{5, 10, 15},
			y:     make([]uint8, 3),
			want:  []uint8{5, 10, 15},
		},
		{
			name:  "Empty",
			numel: 0,
			x1:    nil,
			x2:    nil,
			y:     nil,
			want:  nil,
		},
		{
			name:  "Single element",
			numel: 1,
			x1:    []uint8{100},
			x2:    []uint8{50},
			y:     make([]uint8, 1),
			want:  []uint8{50},
		},
		{
			name:  "Underflow",
			numel: 3,
			x1:    []uint8{1, 2, 3},
			x2:    []uint8{4, 5, 6},
			y:     make([]uint8, 3),
			want:  []uint8{253, 253, 253},
		},
		{
			name:  "Single underflow",
			numel: 1,
			x1:    []uint8{0},
			x2:    []uint8{1},
			y:     make([]uint8, 1),
			want:  []uint8{255},
		},
		{
			name:  "Random",
			numel: 4,
			x1:    []uint8{102, 220, 225, 95},
			x2:    []uint8{179, 61, 234, 203},
			y:     make([]uint8, 4),
			want:  []uint8{179, 159, 247, 148},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kernels.BSubU8(tt.numel, tt.x1, tt.x2, tt.y)
			if !slices.Equal(tt.y, tt.want) {
				t.Errorf("got %v, want %v", tt.y, tt.want)
			}
		})
	}
}

func TestBSubU32(t *testing.T) {
	tests := []struct {
		name      string
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{
			name:  "Basic subtraction",
			numel: 3,
			x1:    []uint32{1000, 2000, 3000},
			x2:    []uint32{500, 1000, 1500},
			y:     make([]uint32, 3),
			want:  []uint32{500, 1000, 1500},
		},
		{
			name:  "Empty",
			numel: 0,
			x1:    nil,
			x2:    nil,
			y:     nil,
			want:  nil,
		},
		{
			name:  "Single element",
			numel: 1,
			x1:    []uint32{100000},
			x2:    []uint32{50000},
			y:     make([]uint32, 1),
			want:  []uint32{50000},
		},
		{
			name:  "Underflow",
			numel: 3,
			x1:    []uint32{1, 2, 3},
			x2:    []uint32{4, 5, 6},
			y:     make([]uint32, 3),
			want:  []uint32{4294967293, 4294967293, 4294967293},
		},
		{
			name:  "Single underflow",
			numel: 1,
			x1:    []uint32{0},
			x2:    []uint32{1},
			y:     make([]uint32, 1),
			want:  []uint32{4294967295},
		},
		{
			name:  "Random",
			numel: 4,
			x1:    []uint32{4083286876, 787846414, 3143890026, 3348747335},
			x2:    []uint32{2571218620, 2563451924, 670094950, 1914837113},
			y:     make([]uint32, 4),
			want:  []uint32{1512068256, 2519361786, 2473795076, 1433910222},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kernels.BSubU32(tt.numel, tt.x1, tt.x2, tt.y)
			if !slices.Equal(tt.y, tt.want) {
				t.Errorf("got %v, want %v", tt.y, tt.want)
			}
		})
	}
}

func TestBSubI64(t *testing.T) {
	tests := []struct {
		name      string
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{
			name:  "Basic subtraction",
			numel: 3,
			x1:    []int64{1, 2, 3},
			x2:    []int64{4, 5, 6},
			y:     make([]int64, 3),
			want:  []int64{-3, -3, -3},
		},
		{
			name:  "Empty",
			numel: 0,
			x1:    nil,
			x2:    nil,
			y:     nil,
			want:  nil,
		},
		{
			name:  "Single element",
			numel: 1,
			x1:    []int64{1000000},
			x2:    []int64{500000},
			y:     make([]int64, 1),
			want:  []int64{500000},
		},
		{
			name:  "Negative and positive",
			numel: 4,
			x1:    []int64{-100, 200, -300, 400},
			x2:    []int64{50, -150, 250, -350},
			y:     make([]int64, 4),
			want:  []int64{-150, 350, -550, 750},
		},
		{
			name:  "Random large values",
			numel: 4,
			x1:    []int64{-6345780979313412905, -8151918526507952693, 6754757701360545141, 1865242737500154728},
			x2:    []int64{3838261603483033731, -8843655056009921227, 8668306688712173912, 6132484236315524510},
			y:     make([]int64, 4),
			want:  []int64{8262701490913104980, 691736529501968534, -1913548987351628771, -4267241498815369782},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kernels.BSubI64(tt.numel, tt.x1, tt.x2, tt.y)
			if !slices.Equal(tt.y, tt.want) {
				t.Errorf("got %v, want %v", tt.y, tt.want)
			}
		})
	}
}

func TestBSubStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{-3, -3, -3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{4, 0, 5, 0, 6}, make([]float32, 3), []float32{-3, -3, -3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30, 40, 50, 60}, make([]float32, 6), []float32{-9, -18, -27, -36, -45, -54}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 40, 20, 50, 30, 60}, make([]float32, 6), []float32{-9, -18, -27, -36, -45, -54}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{-9, -18, -27, -36, -45, -54, -63, -72}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{-9, -18, -27, -36, -45, -54, -63, -72}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BSubStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBSubStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{-3, -3, -3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{4, 0, 5, 0, 6}, make([]float64, 3), []float64{-3, -3, -3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 20, 30, 40, 50, 60}, make([]float64, 6), []float64{-9, -18, -27, -36, -45, -54}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 40, 20, 50, 30, 60}, make([]float64, 6), []float64{-9, -18, -27, -36, -45, -54}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{-9, -18, -27, -36, -45, -54, -63, -72}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{-9, -18, -27, -36, -45, -54, -63, -72}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BSubStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBSubStridedU8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint8
		want         []uint8
	}{
		{
			numel:     3,
			ndims:     1,
			dims:      []int{3},
			stridesX1: []int{1},
			stridesX2: []int{1},
			stridesY:  []int{1},
			x1:        []uint8{1, 2, 3},
			x2:        []uint8{4, 5, 6},
			y:         make([]uint8, 3),
			want:      []uint8{253, 253, 253},
		},
		{
			numel:     3,
			ndims:     1,
			dims:      []int{3},
			stridesX1: []int{1},
			stridesX2: []int{2},
			stridesY:  []int{1},
			x1:        []uint8{1, 2, 3},
			x2:        []uint8{4, 0, 5, 0, 6},
			y:         make([]uint8, 3),
			want:      []uint8{253, 253, 253},
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{3, 1},
			stridesY:  []int{3, 1},
			x1:        []uint8{1, 2, 3, 4, 5, 6},
			x2:        []uint8{10, 20, 30, 40, 50, 60},
			y:         make([]uint8, 6),
			want:      []uint8{247, 238, 229, 220, 211, 202},
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{1, 2},
			stridesY:  []int{3, 1},
			x1:        []uint8{1, 2, 3, 4, 5, 6},
			x2:        []uint8{10, 40, 20, 50, 30, 60},
			y:         make([]uint8, 6),
			want:      []uint8{247, 238, 229, 220, 211, 202},
		},
		{
			numel:     8,
			ndims:     3,
			dims:      []int{2, 2, 2},
			stridesX1: []int{4, 2, 1},
			stridesX2: []int{4, 2, 1},
			stridesY:  []int{4, 2, 1},
			x1:        []uint8{1, 2, 3, 4, 5, 6, 7, 8},
			x2:        []uint8{10, 20, 30, 40, 50, 60, 70, 80},
			y:         make([]uint8, 8),
			want:      []uint8{247, 238, 229, 220, 211, 202, 193, 184},
		},
		{
			numel:     8,
			ndims:     3,
			dims:      []int{2, 2, 2},
			stridesX1: []int{4, 1, 2},
			stridesX2: []int{4, 2, 1},
			stridesY:  []int{4, 2, 1},
			x1:        []uint8{1, 3, 2, 4, 5, 7, 6, 8},
			x2:        []uint8{10, 20, 30, 40, 50, 60, 70, 80},
			y:         make([]uint8, 8),
			want:      []uint8{247, 238, 229, 220, 211, 202, 193, 184},
		},
		{
			numel:     0,
			ndims:     0,
			dims:      nil,
			stridesX1: nil,
			stridesX2: nil,
			stridesY:  nil,
			x1:        nil,
			x2:        nil,
			y:         nil,
			want:      nil,
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{3, 1},
			stridesY:  []int{3, 1},
			x1:        []uint8{102, 179, 92, 14, 106, 71},
			x2:        []uint8{188, 20, 102, 121, 210, 214},
			y:         make([]uint8, 6),
			want:      []uint8{170, 159, 246, 149, 152, 113},
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{1, 2},
			stridesY:  []int{3, 1},
			x1:        []uint8{102, 179, 92, 14, 106, 71},
			x2:        []uint8{188, 121, 20, 210, 102, 214},
			y:         make([]uint8, 6),
			want:      []uint8{170, 159, 246, 149, 152, 113},
		},
	}

	for _, tt := range tests {
		kernels.BSubStridedU8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBSubStridedU32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint32
		want         []uint32
	}{
		{
			numel:     3,
			ndims:     1,
			dims:      []int{3},
			stridesX1: []int{1},
			stridesX2: []int{1},
			stridesY:  []int{1},
			x1:        []uint32{1, 2, 3},
			x2:        []uint32{4, 5, 6},
			y:         make([]uint32, 3),
			want:      []uint32{4294967293, 4294967293, 4294967293},
		},
		{
			numel:     3,
			ndims:     1,
			dims:      []int{3},
			stridesX1: []int{1},
			stridesX2: []int{2},
			stridesY:  []int{1},
			x1:        []uint32{1, 2, 3},
			x2:        []uint32{4, 0, 5, 0, 6},
			y:         make([]uint32, 3),
			want:      []uint32{4294967293, 4294967293, 4294967293},
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{3, 1},
			stridesY:  []int{3, 1},
			x1:        []uint32{1, 2, 3, 4, 5, 6},
			x2:        []uint32{10, 20, 30, 40, 50, 60},
			y:         make([]uint32, 6),
			want:      []uint32{4294967287, 4294967278, 4294967269, 4294967260, 4294967251, 4294967242},
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{1, 2},
			stridesY:  []int{3, 1},
			x1:        []uint32{1, 2, 3, 4, 5, 6},
			x2:        []uint32{10, 40, 20, 50, 30, 60},
			y:         make([]uint32, 6),
			want:      []uint32{4294967287, 4294967278, 4294967269, 4294967260, 4294967251, 4294967242},
		},
		{
			numel:     8,
			ndims:     3,
			dims:      []int{2, 2, 2},
			stridesX1: []int{4, 2, 1},
			stridesX2: []int{4, 2, 1},
			stridesY:  []int{4, 2, 1},
			x1:        []uint32{1, 2, 3, 4, 5, 6, 7, 8},
			x2:        []uint32{10, 20, 30, 40, 50, 60, 70, 80},
			y:         make([]uint32, 8),
			want:      []uint32{4294967287, 4294967278, 4294967269, 4294967260, 4294967251, 4294967242, 4294967233, 4294967224},
		},
		{
			numel:     8,
			ndims:     3,
			dims:      []int{2, 2, 2},
			stridesX1: []int{4, 1, 2},
			stridesX2: []int{4, 2, 1},
			stridesY:  []int{4, 2, 1},
			x1:        []uint32{1, 3, 2, 4, 5, 7, 6, 8},
			x2:        []uint32{10, 20, 30, 40, 50, 60, 70, 80},
			y:         make([]uint32, 8),
			want:      []uint32{4294967287, 4294967278, 4294967269, 4294967260, 4294967251, 4294967242, 4294967233, 4294967224},
		},
		{
			numel:     0,
			ndims:     0,
			dims:      nil,
			stridesX1: nil,
			stridesX2: nil,
			stridesY:  nil,
			x1:        nil,
			x2:        nil,
			y:         nil,
			want:      nil,
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{3, 1},
			stridesY:  []int{3, 1},
			x1:        []uint32{1972458954, 1433267572, 613608295, 2795544706, 242285876, 3100961111},
			x2:        []uint32{4031053213, 3344769, 4261516219, 2652062880, 2627030329, 30349564},
			y:         make([]uint32, 6),
			want:      []uint32{2236373037, 1429922803, 647059372, 143481826, 1910222843, 3070611547},
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{1, 2},
			stridesY:  []int{3, 1},
			x1:        []uint32{1972458954, 1433267572, 613608295, 2795544706, 242285876, 3100961111},
			x2:        []uint32{4031053213, 2652062880, 3344769, 2627030329, 4261516219, 30349564},
			y:         make([]uint32, 6),
			want:      []uint32{2236373037, 1429922803, 647059372, 143481826, 1910222843, 3070611547},
		},
	}

	for _, tt := range tests {
		kernels.BSubStridedU32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBSubStridedI64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []int64
		want         []int64
	}{
		{
			numel:     3,
			ndims:     1,
			dims:      []int{3},
			stridesX1: []int{1},
			stridesX2: []int{1},
			stridesY:  []int{1},
			x1:        []int64{1, 2, 3},
			x2:        []int64{4, 5, 6},
			y:         make([]int64, 3),
			want:      []int64{-3, -3, -3},
		},
		{
			numel:     3,
			ndims:     1,
			dims:      []int{3},
			stridesX1: []int{1},
			stridesX2: []int{2},
			stridesY:  []int{1},
			x1:        []int64{1, 2, 3},
			x2:        []int64{4, 0, 5, 0, 6},
			y:         make([]int64, 3),
			want:      []int64{-3, -3, -3},
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{3, 1},
			stridesY:  []int{3, 1},
			x1:        []int64{1, 2, 3, 4, 5, 6},
			x2:        []int64{10, 20, 30, 40, 50, 60},
			y:         make([]int64, 6),
			want:      []int64{-9, -18, -27, -36, -45, -54},
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{1, 2},
			stridesY:  []int{3, 1},
			x1:        []int64{1, 2, 3, 4, 5, 6},
			x2:        []int64{10, 40, 20, 50, 30, 60},
			y:         make([]int64, 6),
			want:      []int64{-9, -18, -27, -36, -45, -54},
		},
		{
			numel:     8,
			ndims:     3,
			dims:      []int{2, 2, 2},
			stridesX1: []int{4, 2, 1},
			stridesX2: []int{4, 2, 1},
			stridesY:  []int{4, 2, 1},
			x1:        []int64{1, 2, 3, 4, 5, 6, 7, 8},
			x2:        []int64{10, 20, 30, 40, 50, 60, 70, 80},
			y:         make([]int64, 8),
			want:      []int64{-9, -18, -27, -36, -45, -54, -63, -72},
		},
		{
			numel:     8,
			ndims:     3,
			dims:      []int{2, 2, 2},
			stridesX1: []int{4, 1, 2},
			stridesX2: []int{4, 2, 1},
			stridesY:  []int{4, 2, 1},
			x1:        []int64{1, 3, 2, 4, 5, 7, 6, 8},
			x2:        []int64{10, 20, 30, 40, 50, 60, 70, 80},
			y:         make([]int64, 8),
			want:      []int64{-9, -18, -27, -36, -45, -54, -63, -72},
		},
		{
			numel:     0,
			ndims:     0,
			dims:      nil,
			stridesX1: nil,
			stridesX2: nil,
			stridesY:  nil,
			x1:        nil,
			x2:        nil,
			y:         nil,
			want:      nil,
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{3, 1},
			stridesY:  []int{3, 1},
			x1:        []int64{-974689448, -967335462, 643647998, -873314305, 961022651, -73996530},
			x2:        []int64{-684590147, -565277763, 568919915, 1001719027, 930989560, 472190436},
			y:         make([]int64, 6),
			want:      []int64{-290099301, -402057699, 74728083, -1875033332, 30033091, -546186966},
		},
		{
			numel:     6,
			ndims:     2,
			dims:      []int{2, 3},
			stridesX1: []int{3, 1},
			stridesX2: []int{1, 2},
			stridesY:  []int{3, 1},
			x1:        []int64{-974689448, -967335462, 643647998, -873314305, 961022651, -73996530},
			x2:        []int64{-684590147, 1001719027, -565277763, 930989560, 568919915, 472190436},
			y:         make([]int64, 6),
			want:      []int64{-290099301, -402057699, 74728083, -1875033332, 30033091, -546186966},
		},
	}

	for _, tt := range tests {
		kernels.BSubStridedI64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{4, 10, 18}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{50}},
		{4, []float32{-1, -2, 3, 4}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{-1, -4, -9, -16}},
	}

	for _, tt := range tests {
		kernels.BMulF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{4, 10, 18}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{50}},
		{4, []float64{-1, -2, 3, 4}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{-1, -4, -9, -16}},
	}

	for _, tt := range tests {
		kernels.BMulF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulU8(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint8
		want      []uint8
	}{
		{3, []uint8{2, 3, 4}, []uint8{5, 6, 7}, make([]uint8, 3), []uint8{10, 18, 28}},
		{0, nil, nil, nil, nil},
		{1, []uint8{10}, []uint8{20}, make([]uint8, 1), []uint8{200}},
		{2, []uint8{200, 128}, []uint8{2, 3}, make([]uint8, 2), []uint8{144, 128}},
		{4, []uint8{102, 179, 92, 14}, []uint8{106, 71, 188, 20}, make([]uint8, 4), []uint8{60, 165, 144, 24}},
	}

	for _, tt := range tests {
		kernels.BMulU8(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulU32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{3, []uint32{2, 3, 4}, []uint32{5, 6, 7}, make([]uint32, 3), []uint32{10, 18, 28}},
		{0, nil, nil, nil, nil},
		{1, []uint32{1000}, []uint32{2000}, make([]uint32, 1), []uint32{2000000}},
		{2, []uint32{4294967295, 2147483648}, []uint32{2, 3}, make([]uint32, 2), []uint32{4294967294, 2147483648}},
		{4, []uint32{3421126067, 787846414, 3348747335, 2563451924}, []uint32{1914837113, 429389014, 1972458954, 1433267572}, make([]uint32, 4), []uint32{1061548443, 2627596724, 1108611846, 3160571152}},
	}

	for _, tt := range tests {
		kernels.BMulU32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulI64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{3, []int64{2, 3, 4}, []int64{5, 6, 7}, make([]int64, 3), []int64{10, 18, 28}},
		{0, nil, nil, nil, nil},
		{1, []int64{1000}, []int64{2000}, make([]int64, 1), []int64{2000000}},
		{4, []int64{1, -2, 3, -4}, []int64{5, 6, -7, 8}, make([]int64, 4), []int64{5, -12, -21, -32}},
		{4, []int64{-2314326399425823309, 8314211556539077902, 4279532810384561223, 1819927849474927636}, []int64{-6345336139475183495, -6345780979313412906, -8151918526507952694, 6754757701360545140}, make([]int64, 4), []int64{-5289359443526028901, 393833754753102260, -7638034379124239610, 652641350602358032}},
	}

	for _, tt := range tests {
		kernels.BMulI64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{4, 5, 6}, make([]float32, 3), []float32{4, 10, 18}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{4, 0, 5, 0, 6}, make([]float32, 3), []float32{4, 10, 18}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30, 40, 50, 60}, make([]float32, 6), []float32{10, 40, 90, 160, 250, 360}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 40, 20, 50, 30, 60}, make([]float32, 6), []float32{10, 40, 90, 160, 250, 360}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{10, 40, 90, 160, 250, 360, 490, 640}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{10, 40, 90, 160, 250, 360, 490, 640}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMulStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{4, 5, 6}, make([]float64, 3), []float64{4, 10, 18}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{4, 0, 5, 0, 6}, make([]float64, 3), []float64{4, 10, 18}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 20, 30, 40, 50, 60}, make([]float64, 6), []float64{10, 40, 90, 160, 250, 360}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 40, 20, 50, 30, 60}, make([]float64, 6), []float64{10, 40, 90, 160, 250, 360}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{10, 40, 90, 160, 250, 360, 490, 640}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{10, 40, 90, 160, 250, 360, 490, 640}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMulStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulStridedU8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{2, 3, 4}, []uint8{5, 6, 7}, make([]uint8, 3), []uint8{10, 18, 28}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{2, 3, 4}, []uint8{5, 0, 6, 0, 7}, make([]uint8, 3), []uint8{10, 18, 28}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{1, 2, 3, 4, 5, 6}, []uint8{10, 20, 30, 40, 50, 60}, make([]uint8, 6), []uint8{10, 40, 90, 160, 250, 104}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{1, 2, 3, 4, 5, 6}, []uint8{10, 40, 20, 50, 30, 60}, make([]uint8, 6), []uint8{10, 40, 90, 160, 250, 104}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{1, 2, 3, 4, 5, 6, 7, 8}, []uint8{10, 20, 30, 40, 50, 60, 70, 80}, make([]uint8, 8), []uint8{10, 40, 90, 160, 250, 104, 234, 128}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{1, 3, 2, 4, 5, 7, 6, 8}, []uint8{10, 20, 30, 40, 50, 60, 70, 80}, make([]uint8, 8), []uint8{10, 40, 90, 160, 250, 104, 234, 128}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
		// Random 2D contiguous with overflow
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{102, 179, 92, 14, 106, 71}, []uint8{188, 20, 102, 121, 210, 214}, make([]uint8, 6), []uint8{232, 252, 168, 158, 244, 90}},
		// Random 2D x2 strided with overflow
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{102, 179, 92, 14, 106, 71}, []uint8{188, 121, 20, 210, 102, 214}, make([]uint8, 6), []uint8{232, 252, 168, 158, 244, 90}},
	}

	for _, tt := range tests {
		kernels.BMulStridedU8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulStridedU32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint32
		want         []uint32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{2, 3, 4}, []uint32{5, 6, 7}, make([]uint32, 3), []uint32{10, 18, 28}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{2, 3, 4}, []uint32{5, 0, 6, 0, 7}, make([]uint32, 3), []uint32{10, 18, 28}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{1, 2, 3, 4, 5, 6}, []uint32{10, 20, 30, 40, 50, 60}, make([]uint32, 6), []uint32{10, 40, 90, 160, 250, 360}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{1, 2, 3, 4, 5, 6}, []uint32{10, 40, 20, 50, 30, 60}, make([]uint32, 6), []uint32{10, 40, 90, 160, 250, 360}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1, 2, 3, 4, 5, 6, 7, 8}, []uint32{10, 20, 30, 40, 50, 60, 70, 80}, make([]uint32, 8), []uint32{10, 40, 90, 160, 250, 360, 490, 640}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1, 3, 2, 4, 5, 7, 6, 8}, []uint32{10, 20, 30, 40, 50, 60, 70, 80}, make([]uint32, 8), []uint32{10, 40, 90, 160, 250, 360, 490, 640}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
		// Random 2D contiguous with overflow
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{2134003008, 442015537, 638974010, 739303731, 3483374779, 321011650}, []uint32{193500574, 2629087552, 2554769279, 1076363643, 632939609, 48473455}, make([]uint32, 6), []uint32{2449599872, 2796334400, 963758790, 754280065, 3275800835, 1811394334}},
		// Random 2D x2 strided with overflow
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{2134003008, 442015537, 638974010, 739303731, 3483374779, 321011650}, []uint32{193500574, 1076363643, 2629087552, 632939609, 2554769279, 48473455}, make([]uint32, 6), []uint32{2449599872, 2796334400, 963758790, 754280065, 3275800835, 1811394334}},
	}

	for _, tt := range tests {
		kernels.BMulStridedU32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMulStridedI64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []int64
		want         []int64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{2, 3, 4}, []int64{5, 6, 7}, make([]int64, 3), []int64{10, 18, 28}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{2, 3, 4}, []int64{5, 0, 6, 0, 7}, make([]int64, 3), []int64{10, 18, 28}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{1, 2, 3, 4, 5, 6}, []int64{10, 20, 30, 40, 50, 60}, make([]int64, 6), []int64{10, 40, 90, 160, 250, 360}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{1, 2, 3, 4, 5, 6}, []int64{10, 40, 20, 50, 30, 60}, make([]int64, 6), []int64{10, 40, 90, 160, 250, 360}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{1, 2, 3, 4, 5, 6, 7, 8}, []int64{10, 20, 30, 40, 50, 60, 70, 80}, make([]int64, 8), []int64{10, 40, 90, 160, 250, 360, 490, 640}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{1, 3, 2, 4, 5, 7, 6, 8}, []int64{10, 20, 30, 40, 50, 60, 70, 80}, make([]int64, 8), []int64{10, 40, 90, 160, 250, 360, 490, 640}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
		// Random 2D contiguous with large values
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{1565061431407873443, -2678539032410823507, -98865630446694525, 2038380621796347988, 2016429022952508143, -2596593559401881480}, []int64{2652264590214140215, 2934371780109131225, -4429018912202019129, -735139782744127261, 3883523338859666541, 3811486700684880484}, make([]int64, 6), []int64{-5566792756888190715, 2088041230486319013, -2415991526684174123, 326214023187274364, 3487063936994757571, 6363310570973585120}},
		// Random 2D x2 strided with large values
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{1565061431407873443, -2678539032410823507, -98865630446694525, 2038380621796347988, 2016429022952508143, -2596593559401881480}, []int64{2652264590214140215, -735139782744127261, 2934371780109131225, 3883523338859666541, -4429018912202019129, 3811486700684880484}, make([]int64, 6), []int64{-5566792756888190715, 2088041230486319013, -2415991526684174123, 326214023187274364, 3487063936994757571, 6363310570973585120}},
	}

	for _, tt := range tests {
		kernels.BMulStridedI64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{4, 10, 18}, []float32{4, 5, 6}, make([]float32, 3), []float32{1, 2, 3}},
		{0, nil, nil, nil, nil},
		{1, []float32{50}, []float32{10}, []float32{0}, []float32{5}},
		{4, []float32{-1, -4, -9, -16}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{-1, -2, 3, 4}},
	}

	for _, tt := range tests {
		kernels.BDivF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{4, 10, 18}, []float64{4, 5, 6}, make([]float64, 3), []float64{1, 2, 3}},
		{0, nil, nil, nil, nil},
		{1, []float64{50}, []float64{10}, []float64{0}, []float64{5}},
		{4, []float64{-1, -4, -9, -16}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{-1, -2, 3, 4}},
	}

	for _, tt := range tests {
		kernels.BDivF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivU8(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint8
		want      []uint8
	}{
		{3, []uint8{4, 6, 8}, []uint8{2, 3, 4}, make([]uint8, 3), []uint8{2, 2, 2}},
		{0, nil, nil, nil, nil},
		{1, []uint8{10}, []uint8{2}, make([]uint8, 1), []uint8{5}},
		{4, []uint8{255, 0, 1, 9}, []uint8{1, 5, 0, 3}, make([]uint8, 4), []uint8{255, 0, 0, 3}},
		{2, []uint8{0, 5}, []uint8{0, 0}, make([]uint8, 2), []uint8{0, 0}},
	}
	for _, tt := range tests {
		kernels.BDivU8(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivU32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{3, []uint32{4, 6, 8}, []uint32{2, 3, 4}, make([]uint32, 3), []uint32{2, 2, 2}},
		{0, nil, nil, nil, nil},
		{1, []uint32{1000000000}, []uint32{1000}, make([]uint32, 1), []uint32{1000000}},
		{4, []uint32{4294967295, 0, 1, 9}, []uint32{1, 5, 0, 3}, make([]uint32, 4), []uint32{4294967295, 0, 0, 3}},
		{2, []uint32{0, 500000000}, []uint32{0, 0}, make([]uint32, 2), []uint32{0, 0}},
	}
	for _, tt := range tests {
		kernels.BDivU32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivI64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{3, []int64{4, 6, 8}, []int64{2, 3, 4}, make([]int64, 3), []int64{2, 2, 2}},
		{0, nil, nil, nil, nil},
		{1, []int64{-5}, []int64{3}, make([]int64, 1), []int64{-1}},
		{4, []int64{-5, 5, -5, 5}, []int64{3, 3, -3, -3}, make([]int64, 4), []int64{-1, 1, 1, -1}},
		{5, []int64{10, 0, -10, 1, -1}, []int64{2, 5, -2, 0, -1}, make([]int64, 5), []int64{5, 0, 5, 0, 1}},
		{2, []int64{0, 5}, []int64{0, 0}, make([]int64, 2), []int64{0, 0}},
	}
	for _, tt := range tests {
		kernels.BDivI64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{4, 10, 18}, []float32{4, 5, 6}, make([]float32, 3), []float32{1, 2, 3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{4, 10, 18}, []float32{4, 0, 5, 0, 6}, make([]float32, 3), []float32{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 10, 10, 10, 10, 10}, make([]float32, 6), []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 10, 10, 10, 10, 10}, make([]float32, 6), []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{10, 40, 90, 160, 250, 360, 490, 640}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{10, 90, 40, 160, 250, 490, 360, 640}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BDivStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{4, 10, 18}, []float64{4, 5, 6}, make([]float64, 3), []float64{1, 2, 3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{4, 10, 18}, []float64{4, 0, 5, 0, 6}, make([]float64, 3), []float64{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 10, 10, 10, 10, 10}, make([]float64, 6), []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 10, 10, 10, 10, 10}, make([]float64, 6), []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{10, 40, 90, 160, 250, 360, 490, 640}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{10, 90, 40, 160, 250, 490, 360, 640}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BDivStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivStridedU8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{10, 20, 30}, []uint8{1, 0, 3}, make([]uint8, 3), []uint8{10, 0, 10}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{10, 20, 30}, []uint8{1, 0, 0, 0, 3}, make([]uint8, 3), []uint8{10, 0, 10}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{10, 20, 30, 40, 50, 60}, []uint8{1, 0, 3, 4, 5, 6}, make([]uint8, 6), []uint8{10, 0, 10, 10, 10, 10}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{10, 20, 30, 40, 50, 60}, []uint8{1, 4, 0, 5, 3, 6}, make([]uint8, 6), []uint8{10, 0, 10, 10, 10, 10}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{10, 20, 30, 40, 50, 60, 70, 80}, []uint8{1, 0, 3, 4, 5, 6, 7, 8}, make([]uint8, 8), []uint8{10, 0, 10, 10, 10, 10, 10, 10}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{10, 30, 20, 40, 50, 70, 60, 80}, []uint8{1, 0, 3, 4, 5, 6, 7, 8}, make([]uint8, 8), []uint8{10, 0, 10, 10, 10, 10, 10, 10}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.BDivStridedU8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivStridedU32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint32
		want         []uint32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{10, 20, 30}, []uint32{1, 0, 3}, make([]uint32, 3), []uint32{10, 0, 10}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{10, 20, 30}, []uint32{1, 0, 0, 0, 3}, make([]uint32, 3), []uint32{10, 0, 10}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{10, 20, 30, 40, 50, 60}, []uint32{1, 0, 3, 4, 5, 6}, make([]uint32, 6), []uint32{10, 0, 10, 10, 10, 10}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{10, 20, 30, 40, 50, 60}, []uint32{1, 4, 0, 5, 3, 6}, make([]uint32, 6), []uint32{10, 0, 10, 10, 10, 10}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{10, 20, 30, 40, 50, 60, 70, 80}, []uint32{1, 0, 3, 4, 5, 6, 7, 8}, make([]uint32, 8), []uint32{10, 0, 10, 10, 10, 10, 10, 10}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{10, 30, 20, 40, 50, 70, 60, 80}, []uint32{1, 0, 3, 4, 5, 6, 7, 8}, make([]uint32, 8), []uint32{10, 0, 10, 10, 10, 10, 10, 10}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.BDivStridedU32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBDivStridedI64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []int64
		want         []int64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{4, -6, 9}, []int64{2, 0, -3}, make([]int64, 3), []int64{2, 0, -3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{4, -6, 9}, []int64{2, 0, 0, 0, -3}, make([]int64, 3), []int64{2, 0, -3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{4, -6, 9, -12, 15, -18}, []int64{2, 0, -3, 4, -5, 6}, make([]int64, 6), []int64{2, 0, -3, -3, -3, -3}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{4, -6, 9, -12, 15, -18}, []int64{2, 4, 0, -5, -3, 6}, make([]int64, 6), []int64{2, 0, -3, -3, -3, -3}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{4, -6, 9, -12, 15, -18, 21, -24}, []int64{2, 0, -3, 4, -5, 6, -7, 8}, make([]int64, 8), []int64{2, 0, -3, -3, -3, -3, -3, -3}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{4, 9, -6, -12, 15, 21, -18, -24}, []int64{2, 0, -3, 4, -5, 6, -7, 8}, make([]int64, 8), []int64{2, 0, -3, -3, -3, -3, -3, -3}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.BDivStridedI64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{1, 5, 3}, []float32{4, 2, 6}, make([]float32, 3), []float32{4, 5, 6}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{10}},
		{4, []float32{-1, -2, 3, 4}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{1, 2, 3, 4}},
	}

	for _, tt := range tests {
		kernels.BMaximumF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{1, 5, 3}, []float64{4, 2, 6}, make([]float64, 3), []float64{4, 5, 6}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{10}},
		{4, []float64{-1, -2, 3, 4}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{1, 2, 3, 4}},
	}

	for _, tt := range tests {
		kernels.BMaximumF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumU8(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint8
		want      []uint8
	}{
		{3, []uint8{10, 20, 30}, []uint8{2, 25, 25}, make([]uint8, 3), []uint8{10, 25, 30}},
		{0, nil, nil, nil, nil},
		{1, []uint8{255}, []uint8{1}, make([]uint8, 1), []uint8{255}},
		{2, []uint8{10, 20}, []uint8{30, 40}, make([]uint8, 2), []uint8{30, 40}},
		{1, []uint8{0}, []uint8{0}, make([]uint8, 1), []uint8{0}},
		{2, []uint8{1, 255}, []uint8{255, 3}, make([]uint8, 2), []uint8{255, 255}},
	}
	for _, tt := range tests {
		kernels.BMaximumU8(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumU32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{3, []uint32{10, 20, 30}, []uint32{2, 25, 25}, make([]uint32, 3), []uint32{10, 25, 30}},
		{0, nil, nil, nil, nil},
		{1, []uint32{4294967295}, []uint32{1}, make([]uint32, 1), []uint32{4294967295}},
		{1, []uint32{4294967295}, []uint32{2}, make([]uint32, 1), []uint32{4294967295}},
		{1, []uint32{0}, []uint32{0}, make([]uint32, 1), []uint32{0}},
		{2, []uint32{10, 20}, []uint32{0, 0}, make([]uint32, 2), []uint32{10, 20}},
		{2, []uint32{1, 4294967295}, []uint32{4294967295, 3}, make([]uint32, 2), []uint32{4294967295, 4294967295}},
	}
	for _, tt := range tests {
		kernels.BMaximumU32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumI64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{3, []int64{10, 20, 30}, []int64{2, 4, 5}, make([]int64, 3), []int64{10, 20, 30}},
		{0, nil, nil, nil, nil},
		{1, []int64{5}, []int64{-2}, make([]int64, 1), []int64{5}},
		{1, []int64{-5}, []int64{2}, make([]int64, 1), []int64{2}},
		{1, []int64{-5}, []int64{-2}, make([]int64, 1), []int64{-2}},
		{1, []int64{0}, []int64{0}, make([]int64, 1), []int64{0}},
		{2, []int64{10, 0}, []int64{0, 0}, make([]int64, 2), []int64{10, 0}},
		{4, []int64{-1, -2, 3, 4}, []int64{1, 2, -3, -4}, make([]int64, 4), []int64{1, 2, 3, 4}},
		{3, []int64{-9223372036854775808, 0, 9223372036854775807}, []int64{0, 9223372036854775807, -9223372036854775808}, make([]int64, 3), []int64{0, 9223372036854775807, 9223372036854775807}},
	}
	for _, tt := range tests {
		kernels.BMaximumI64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 5, 3}, []float32{4, 2, 6}, make([]float32, 3), []float32{4, 5, 6}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 5, 3}, []float32{4, 0, 2, 0, 6}, make([]float32, 3), []float32{4, 5, 6}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30, 40, 50, 60}, make([]float32, 6), []float32{10, 20, 30, 40, 50, 60}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 40, 20, 50, 30, 60}, make([]float32, 6), []float32{10, 20, 30, 40, 50, 60}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{10, 20, 30, 40, 50, 60, 70, 80}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{10, 20, 30, 40, 50, 60, 70, 80}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMaximumStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 5, 3}, []float64{4, 2, 6}, make([]float64, 3), []float64{4, 5, 6}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 5, 3}, []float64{4, 0, 2, 0, 6}, make([]float64, 3), []float64{4, 5, 6}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 20, 30, 40, 50, 60}, make([]float64, 6), []float64{10, 20, 30, 40, 50, 60}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 40, 20, 50, 30, 60}, make([]float64, 6), []float64{10, 20, 30, 40, 50, 60}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{10, 20, 30, 40, 50, 60, 70, 80}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{10, 20, 30, 40, 50, 60, 70, 80}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMaximumStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumStridedU8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint8
		want         []uint8
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{1, 5, 3}, []uint8{4, 2, 6}, make([]uint8, 3), []uint8{4, 5, 6}},
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{1, 5, 3}, []uint8{4, 0, 2, 0, 6}, make([]uint8, 3), []uint8{4, 5, 6}},
		{1, 1, []int{1}, []int{1}, []int{1}, []int{1}, []uint8{1}, []uint8{0}, make([]uint8, 1), []uint8{1}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{1, 5, 3, 4, 2, 6}, []uint8{4, 2, 6, 1, 5, 3}, make([]uint8, 6), []uint8{4, 5, 6, 4, 5, 6}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{1, 5, 3, 4, 2, 6}, []uint8{4, 1, 2, 5, 6, 3}, make([]uint8, 6), []uint8{4, 5, 6, 4, 5, 6}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{1, 5, 3, 4, 2, 6, 7, 0}, []uint8{4, 0, 6, 2, 5, 3, 1, 8}, make([]uint8, 8), []uint8{4, 5, 6, 4, 5, 6, 7, 8}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{1, 3, 5, 4, 2, 7, 6, 0}, []uint8{4, 0, 6, 2, 5, 3, 1, 8}, make([]uint8, 8), []uint8{4, 5, 6, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.BMaximumStridedU8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumStridedU32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint32
		want         []uint32
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{100000, 500000, 300000}, []uint32{400000, 200000, 600000}, make([]uint32, 3), []uint32{400000, 500000, 600000}},
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{100000, 500000, 300000}, []uint32{400000, 0, 200000, 0, 600000}, make([]uint32, 3), []uint32{400000, 500000, 600000}},
		{1, 1, []int{1}, []int{1}, []int{1}, []int{1}, []uint32{100000}, []uint32{0}, make([]uint32, 1), []uint32{100000}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{100000, 500000, 300000, 400000, 200000, 600000}, []uint32{400000, 200000, 600000, 100000, 500000, 300000}, make([]uint32, 6), []uint32{400000, 500000, 600000, 400000, 500000, 600000}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{100000, 500000, 300000, 400000, 200000, 600000}, []uint32{400000, 100000, 200000, 500000, 600000, 300000}, make([]uint32, 6), []uint32{400000, 500000, 600000, 400000, 500000, 600000}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{100000, 500000, 300000, 400000, 200000, 600000, 700000, 0}, []uint32{400000, 0, 600000, 200000, 500000, 300000, 100000, 800000}, make([]uint32, 8), []uint32{400000, 500000, 600000, 400000, 500000, 600000, 700000, 800000}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{100000, 300000, 500000, 400000, 200000, 700000, 600000, 0}, []uint32{400000, 0, 600000, 200000, 500000, 300000, 100000, 800000}, make([]uint32, 8), []uint32{400000, 500000, 600000, 400000, 500000, 600000, 700000, 800000}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.BMaximumStridedU32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMaximumStridedI64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []int64
		want         []int64
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{-5, -1, 4}, []int64{-6, 2, -3}, make([]int64, 3), []int64{-5, 2, 4}},
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{-5, -1, 4}, []int64{-6, 0, 2, 0, -3}, make([]int64, 3), []int64{-5, 2, 4}},
		{1, 1, []int{1}, []int{1}, []int{1}, []int{1}, []int64{-10}, []int64{-5}, make([]int64, 1), []int64{-5}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{-5, -1, 4, -6, 2, -3}, []int64{-4, -2, 3, 5, -7, 1}, make([]int64, 6), []int64{-4, -1, 4, 5, 2, 1}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{-5, -1, 4, -6, 2, -3}, []int64{-4, 5, -2, -7, 3, 1}, make([]int64, 6), []int64{-4, -1, 4, 5, 2, 1}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-1, 2, -3, 4, -5, 6, -7, 8}, []int64{0, -3, 1, -5, 2, -4, 3, -6}, make([]int64, 8), []int64{0, 2, 1, 4, 2, 6, 3, 8}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-1, -3, 2, 4, -5, -7, 6, 8}, []int64{0, -3, 1, -5, 2, -4, 3, -6}, make([]int64, 8), []int64{0, 2, 1, 4, 2, 6, 3, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.BMaximumStridedI64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumF32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{1, 5, 3}, []float32{4, 2, 6}, make([]float32, 3), []float32{1, 2, 3}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{5}},
		{4, []float32{-1, -2, 3, 4}, []float32{1, 2, -3, -4}, make([]float32, 4), []float32{-1, -2, -3, -4}},
	}

	for _, tt := range tests {
		kernels.BMinimumF32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumF64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{1, 5, 3}, []float64{4, 2, 6}, make([]float64, 3), []float64{1, 2, 3}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{5}},
		{4, []float64{-1, -2, 3, 4}, []float64{1, 2, -3, -4}, make([]float64, 4), []float64{-1, -2, -3, -4}},
	}

	for _, tt := range tests {
		kernels.BMinimumF64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumU8(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint8
		want      []uint8
	}{
		{3, []uint8{102, 179, 92}, []uint8{14, 106, 71}, make([]uint8, 3), []uint8{14, 106, 71}},
		{0, nil, nil, nil, nil},
		{1, []uint8{10}, []uint8{5}, []uint8{0}, []uint8{5}},
		{4, []uint8{188, 20, 102, 121}, []uint8{210, 214, 74, 202}, make([]uint8, 4), []uint8{188, 20, 74, 121}},
		{2, []uint8{0, 255}, []uint8{255, 0}, make([]uint8, 2), []uint8{0, 0}},
	}
	for _, tt := range tests {
		kernels.BMinimumU8(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestBMinimumU32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{3, []uint32{3421126067, 787846414, 3348747335}, []uint32{2563451924, 1914837113, 429389014}, make([]uint32, 3), []uint32{2563451924, 787846414, 429389014}},
		{0, nil, nil, nil, nil},
		{1, []uint32{10}, []uint32{5}, []uint32{0}, []uint32{5}},
		{4, []uint32{1972458954, 1433267572, 613608295, 2795544706}, []uint32{242285876, 3100961111, 4031053213, 3344769}, make([]uint32, 4), []uint32{242285876, 1433267572, 613608295, 3344769}},
		{2, []uint32{0, 4294967295}, []uint32{4294967295, 0}, make([]uint32, 2), []uint32{0, 0}},
	}
	for _, tt := range tests {
		kernels.BMinimumU32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestBMinimumI64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{3, []int64{77, 19, 7}, []int64{23, 43, -6}, make([]int64, 3), []int64{23, 19, -6}},
		{0, nil, nil, nil, nil},
		{1, []int64{10}, []int64{5}, []int64{0}, []int64{5}},
		{4, []int64{-7, -23, -30, 9}, []int64{70, 39, -14, -1}, make([]int64, 4), []int64{-7, -23, -30, -1}},
		{2, []int64{-9223372036854775808, 9223372036854775807}, []int64{0, 0}, make([]int64, 2), []int64{-9223372036854775808, 0}},
	}
	for _, tt := range tests {
		kernels.BMinimumI64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumStridedF32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 5, 3}, []float32{4, 2, 6}, make([]float32, 3), []float32{1, 2, 3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 5, 3}, []float32{4, 0, 2, 0, 6}, make([]float32, 3), []float32{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30, 40, 50, 60}, make([]float32, 6), []float32{1, 2, 3, 4, 5, 6}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 40, 20, 50, 30, 60}, make([]float32, 6), []float32{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{10, 20, 30, 40, 50, 60, 70, 80}, make([]float32, 8), []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMinimumStridedF32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumStridedF64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 5, 3}, []float64{4, 2, 6}, make([]float64, 3), []float64{1, 2, 3}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 5, 3}, []float64{4, 0, 2, 0, 6}, make([]float64, 3), []float64{1, 2, 3}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 20, 30, 40, 50, 60}, make([]float64, 6), []float64{1, 2, 3, 4, 5, 6}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{10, 40, 20, 50, 30, 60}, make([]float64, 6), []float64{1, 2, 3, 4, 5, 6}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{10, 20, 30, 40, 50, 60, 70, 80}, make([]float64, 8), []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.BMinimumStridedF64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumStridedU8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{2, 7, 16}, []uint8{14, 6, 15}, make([]uint8, 3), []uint8{2, 6, 15}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{2, 7, 16}, []uint8{14, 0, 6, 0, 15}, make([]uint8, 3), []uint8{2, 6, 15}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{0, 4, 10, 13, 18, 14}, []uint8{10, 14, 11, 12, 15, 15}, make([]uint8, 6), []uint8{0, 4, 10, 12, 15, 14}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{0, 4, 10, 13, 18, 14}, []uint8{10, 12, 14, 15, 11, 15}, make([]uint8, 6), []uint8{0, 4, 10, 12, 15, 14}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{7, 6, 9, 16, 13, 11, 19, 13}, []uint8{1, 9, 7, 19, 12, 0, 15, 9}, make([]uint8, 8), []uint8{1, 6, 7, 16, 12, 0, 15, 9}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{7, 9, 6, 16, 13, 19, 11, 13}, []uint8{1, 9, 7, 19, 12, 0, 15, 9}, make([]uint8, 8), []uint8{1, 6, 7, 16, 12, 0, 15, 9}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.BMinimumStridedU8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumStridedU32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint32
		want         []uint32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{42, 67, 76}, []uint32{14, 26, 35}, make([]uint32, 3), []uint32{14, 26, 35}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{42, 67, 76}, []uint32{14, 0, 26, 0, 35}, make([]uint32, 3), []uint32{14, 26, 35}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{20, 24, 50, 13, 78, 14}, []uint32{10, 54, 31, 72, 15, 95}, make([]uint32, 6), []uint32{10, 24, 31, 13, 15, 14}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{20, 24, 50, 13, 78, 14}, []uint32{10, 72, 54, 15, 31, 95}, make([]uint32, 6), []uint32{10, 24, 31, 13, 15, 14}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{67, 6, 49, 76, 73, 11, 99, 13}, []uint32{41, 69, 87, 19, 72, 80, 75, 29}, make([]uint32, 8), []uint32{41, 6, 49, 19, 72, 11, 75, 13}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{67, 49, 6, 76, 73, 99, 11, 13}, []uint32{41, 69, 87, 19, 72, 80, 75, 29}, make([]uint32, 8), []uint32{41, 6, 49, 19, 72, 11, 75, 13}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.BMinimumStridedU32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestBMinimumStridedI64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []int64
		want         []int64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{-8, 17, 26}, []int64{-36, -24, -15}, make([]int64, 3), []int64{-36, -24, -15}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{-8, 17, 26}, []int64{-36, 0, -24, 0, -15}, make([]int64, 3), []int64{-36, -24, -15}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{-30, -26, 0, -37, 28, -36}, []int64{-40, 4, -19, 22, -35, 45}, make([]int64, 6), []int64{-40, -26, -19, -37, -35, -36}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{-30, -26, 0, -37, 28, -36}, []int64{-40, 22, 4, -35, -19, 45}, make([]int64, 6), []int64{-40, -26, -19, -37, -35, -36}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{17, -44, -1, 26, 23, -39, 49, -37}, []int64{-9, 19, 37, -31, 22, 30, 25, -21}, make([]int64, 8), []int64{-9, -44, -1, -31, 22, -39, 25, -37}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{17, -1, -44, 26, 23, 49, -39, -37}, []int64{-9, 19, 37, -31, 22, 30, 25, -21}, make([]int64, 8), []int64{-9, -44, -1, -31, 22, -39, 25, -37}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.BMinimumStridedI64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// Comparison tests

// TestEqF32F32 tests y = (x1 == x2) ? 1 : 0 for float32 with float32 output
func TestEqF32F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []float32
		want   []float32
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{5}, []float32{0}, []float32{1}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]float32, 4), []float32{0, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.EqF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqF64F64 tests y = (x1 == x2) ? 1 : 0 for float64 with float64 output
func TestEqF64F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []float64
		want   []float64
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{5}, []float64{0}, []float64{1}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]float64, 4), []float64{0, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.EqF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestEqU32U32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{4, []uint32{3421126067, 787846414, 3348747335, 2563451924}, []uint32{1914837113, 429389014, 1972458954, 1433267572}, make([]uint32, 4), []uint32{0, 0, 0, 0}},
		{4, []uint32{0, 1, 4294967295, 0}, []uint32{0, 0, 4294967295, 1}, make([]uint32, 4), []uint32{1, 0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []uint32{5}, []uint32{5}, make([]uint32, 1), []uint32{1}},
		{1, []uint32{5}, []uint32{6}, make([]uint32, 1), []uint32{0}},
		{3, []uint32{1, 2, 3}, []uint32{1, 4, 3}, make([]uint32, 3), []uint32{1, 0, 1}},
	}
	for _, tt := range tests {
		kernels.EqU32U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestEqI64I64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{4, []int64{-1533875353, 648061058, -1905197772, 953477463}, []int64{1883569565, -2144138879, 2114032571, 504579232}, make([]int64, 4), []int64{0, 0, 0, 0}},
		{4, []int64{0, -1, 9223372036854775807, -9223372036854775808}, []int64{0, 1, 9223372036854775807, -9223372036854775808}, make([]int64, 4), []int64{1, 0, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []int64{5}, []int64{5}, make([]int64, 1), []int64{1}},
		{1, []int64{5}, []int64{6}, make([]int64, 1), []int64{0}},
		{3, []int64{-1, 0, 1}, []int64{-1, 2, 1}, make([]int64, 3), []int64{1, 0, 1}},
	}
	for _, tt := range tests {
		kernels.EqI64I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqU8F32 tests y = (x1 == x2) ? 1 : 0 for float32 with uint8 output
func TestEqU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{5}, []uint8{0}, []uint8{1}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.EqU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqU8F64 tests y = (x1 == x2) ? 1 : 0 for float64 with uint8 output
func TestEqU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{5}, []uint8{0}, []uint8{1}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.EqU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestEqU8U8(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint8
		want      []uint8
	}{
		{3, []uint8{254, 109, 126}, []uint8{254, 0, 126}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []uint8{100}, []uint8{100}, make([]uint8, 1), []uint8{1}},
		{1, []uint8{100}, []uint8{200}, make([]uint8, 1), []uint8{0}},
		{4, []uint8{102, 179, 92, 14}, []uint8{0, 179, 0, 14}, make([]uint8, 4), []uint8{0, 1, 0, 1}},
	}
	for _, tt := range tests {
		kernels.EqU8U8(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestEqU8U32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []uint32
		y      []uint8
		want   []uint8
	}{
		{3, []uint32{3062119789, 1840268610, 2967327842}, []uint32{3062119789, 0, 2967327842}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []uint32{100}, []uint32{100}, make([]uint8, 1), []uint8{1}},
		{1, []uint32{100}, []uint32{200}, make([]uint8, 1), []uint8{0}},
		{4, []uint32{3421126067, 787846414, 3348747335, 2563451924}, []uint32{0, 787846414, 0, 2563451924}, make([]uint8, 4), []uint8{0, 1, 0, 1}},
	}
	for _, tt := range tests {
		kernels.EqU8U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestEqU8I64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []int64
		y      []uint8
		want   []uint8
	}{
		{3, []int64{-618, 789, 102}, []int64{-618, 790, 102}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []int64{-100}, []int64{-100}, make([]uint8, 1), []uint8{1}},
		{1, []int64{-100}, []int64{200}, make([]uint8, 1), []uint8{0}},
		{4, []int64{542, -933, -124, -586}, []int64{543, -933, -123, -586}, make([]uint8, 4), []uint8{0, 1, 0, 1}},
	}
	for _, tt := range tests {
		kernels.EqU8I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqStridedF32F32 tests y = (x1 == x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestEqStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{0, 1, 0, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{0, 1, 0, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 0, 0, 1, 0, 0, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 0, 0, 1, 0, 0, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.EqStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqStridedF64F64 tests y = (x1 == x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestEqStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{0, 1, 0, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{0, 1, 0, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 0, 0, 1, 0, 0, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 0, 0, 1, 0, 0, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.EqStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestEqStridedU32U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint32
		want         []uint32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{2126557045, 1834397869, 895778712}, []uint32{1839126634, 1834397869, 895778712}, make([]uint32, 3), []uint32{0, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{4003108267, 486203110, 609124268}, []uint32{2884108403, 0, 486203110, 0, 755665730}, make([]uint32, 3), []uint32{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{2880583284, 4088480131, 4180258368, 317106009, 918240654, 3716832583}, []uint32{3816268813, 4088480131, 4180258368, 4081842660, 3638153567, 3278158419}, make([]uint32, 6), []uint32{0, 1, 1, 0, 0, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{1026582254, 2018749264, 2384174224, 287335862, 494474456, 862704186}, []uint32{1026582254, 287335862, 2018749264, 494474456, 2373096598, 862704186}, make([]uint32, 6), []uint32{1, 1, 0, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{3152157506, 1498987913, 33989666, 2957457565, 2591022090, 154359954, 1280173249, 80744645}, []uint32{3152157506, 1498987913, 3259048631, 2042651245, 2591022090, 154359954, 1280173249, 1634462841}, make([]uint32, 8), []uint32{1, 1, 0, 0, 1, 1, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{946066739, 851557588, 1202199027, 3721937893, 1381452159, 2302951409, 2159998012, 3632269863}, []uint32{946066739, 3636306316, 151617426, 3721937893, 1381452159, 2159998012, 2302951409, 3632269863}, make([]uint32, 8), []uint32{1, 0, 0, 1, 1, 1, 1, 1}},
		// edge cases
		{2, 1, []int{2}, []int{1}, []int{1}, []int{1}, []uint32{0, 4294967295}, []uint32{0, 0}, make([]uint32, 2), []uint32{1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.EqStridedU32U32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestEqStridedI64I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []int64
		want         []int64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{2426855030617254776, 4443723097237118199, 1102234190063617506}, []int64{2426855030617254776, 4443723097237118199, 1342300408732833893}, make([]int64, 3), []int64{1, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{1495517653071156389, -3287320394382133119, -3617550421138469274}, []int64{1495517653071156389, 0, 2554836102665886274, 0, -2329159127626650103}, make([]int64, 3), []int64{1, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{2908798904562626183, 100662273930001233, 4574195048251468604, 1741386191184585168, -916849500775205952, 3406952018860578378}, []int64{2908798904562626183, 100662273930001233, 1171816278198958872, 1741386191184585168, 220613556256315140, 3406952018860578378}, make([]int64, 6), []int64{1, 1, 0, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{-142754869599927829, -802639448725699354, -1653522573124996610, 550062600445258214, -2106721954149271232, -3174101946600635807}, []int64{-2715119379786355077, 550062600445258214, 2083201729658933711, -1181871101059649593, -1653522573124996610, -3174101946600635807}, make([]int64, 6), []int64{0, 0, 1, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-2296993547622599243, -311157209004279546, 4028107944763977147, 928417649567581196, 1419899544950059567, 485700680921939111, 2288101936363760824, -1740430909613472563}, []int64{-2296993547622599243, -1619900997978478914, 4028107944763977147, -3199531090132501809, 1368777444561164654, 2166253279062886067, 2288101936363760824, 1265777552855765713}, make([]int64, 8), []int64{1, 0, 1, 0, 0, 0, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-4104559282886758677, -1020394905601866606, -1472961792803716051, 1363328727782361153, 648093339091841071, 4352403941103854575, 1363598084345811991, 170833930636849749}, []int64{-4104559282886758677, -1472961792803716051, -1020394905601866606, 709257369102675676, 648093339091841071, 1363598084345811991, 4352403941103854575, 2300293162856998919}, make([]int64, 8), []int64{1, 1, 1, 0, 1, 1, 1, 0}},
		// edge cases
		{2, 1, []int{2}, []int{1}, []int{1}, []int{1}, []int64{-9223372036854775808, 9223372036854775807}, []int64{-9223372036854775808, 9223372036854775806}, make([]int64, 2), []int64{1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.EqStridedI64I64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqStridedU8F32 tests y = (x1 == x2) ? 1 : 0 for float32 with strided memory and uint8 output
func TestEqStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 0, 1, 0, 0, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 0, 1, 0, 0, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.EqStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestEqStridedU8F64 tests y = (x1 == x2) ? 1 : 0 for float64 with strided memory and uint8 output
func TestEqStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 0, 1, 0, 0, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 0, 1, 0, 0, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.EqStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestEqStridedU8U8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []uint8
		y            []uint8
		want         []uint8
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{2, 7, 6}, []uint8{4, 6, 5}, make([]uint8, 3), []uint8{0, 0, 0}},
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{0, 4, 0}, []uint8{3, 8, 4, 0, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{1, 2, 5, 5, 7, 6}, []uint8{9, 6, 3, 1, 9, 3}, make([]uint8, 6), []uint8{0, 0, 0, 0, 0, 0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{1, 9, 7, 9, 2, 0}, []uint8{5, 9, 3, 4, 9, 6}, make([]uint8, 6), []uint8{0, 0, 0, 1, 0, 0}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{2, 0, 6, 2, 7, 9, 7, 3}, []uint8{3, 4, 3, 7, 0, 9, 0, 9}, make([]uint8, 8), []uint8{0, 0, 0, 0, 0, 1, 0, 0}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{6, 9, 5, 4, 8, 8, 6, 0}, []uint8{0, 0, 0, 1, 3, 0, 1, 1}, make([]uint8, 8), []uint8{0, 0, 0, 0, 0, 0, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.EqStridedU8U8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestEqStridedU8U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []uint32
		y            []uint8
		want         []uint8
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{7, 9, 4}, []uint32{3, 8, 9}, make([]uint8, 3), []uint8{0, 0, 0}},
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{3, 7, 8}, []uint32{1, 4, 1, 6, 3}, make([]uint8, 3), []uint8{0, 0, 0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{2, 0, 9, 8, 5, 3}, []uint32{7, 7, 5, 9, 1, 5}, make([]uint8, 6), []uint8{0, 0, 0, 0, 0, 0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{1, 9, 1, 4, 0, 3}, []uint32{7, 5, 7, 1, 5, 7}, make([]uint8, 6), []uint8{0, 0, 0, 0, 0, 0}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{5, 8, 5, 4, 1, 1, 0, 9}, []uint32{0, 9, 1, 8, 9, 6, 7, 6}, make([]uint8, 8), []uint8{0, 0, 0, 0, 0, 0, 0, 0}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{0, 9, 5, 2, 9, 1, 7, 8}, []uint32{6, 0, 6, 8, 6, 8, 0, 6}, make([]uint8, 8), []uint8{0, 0, 0, 0, 0, 0, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.EqStridedU8U32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestEqStridedU8I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []int64
		y            []uint8
		want         []uint8
	}{
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{-1, -4, 0}, []int64{5, -1, -2}, make([]uint8, 3), []uint8{0, 0, 0}},
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{-2, 5, 3}, []int64{0, 5, -3, 4, -4}, make([]uint8, 3), []uint8{0, 0, 0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{-1, -2, -5, -2, 1, 5}, []int64{4, -3, -5, 0, 5, 4}, make([]uint8, 6), []uint8{0, 0, 1, 0, 0, 0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{-4, 3, 1, 0, 1, 0}, []int64{-5, -4, -5, -1, -4, 5}, make([]uint8, 6), []uint8{0, 0, 0, 0, 0, 0}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{5, -3, 5, 3, -1, 0, -1, -4}, []int64{5, 3, 2, -5, 0, 0, 4, 3}, make([]uint8, 8), []uint8{1, 0, 0, 0, 0, 1, 0, 0}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{4, 1, 1, -4, 4, -2, -4, -1}, []int64{5, 2, -3, 4, 4, -3, -5, -4}, make([]uint8, 8), []uint8{0, 0, 0, 0, 1, 0, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.EqStridedU8I64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeF32F32 tests y = (x1 != x2) ? 1 : 0 for float32 with float32 output
func TestNeF32F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []float32
		want   []float32
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{5}, []float32{0}, []float32{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]float32, 4), []float32{1, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.NeF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeF64F64 tests y = (x1 != x2) ? 1 : 0 for float64 with float64 output
func TestNeF64F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []float64
		want   []float64
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{5}, []float64{0}, []float64{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]float64, 4), []float64{1, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.NeF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestNeU32U32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{4, []uint32{383329928, 3324115917, 2811363265, 1884968545}, []uint32{1859786276, 3687649986, 369133709, 2995172878}, make([]uint32, 4), []uint32{1, 1, 1, 1}},
		{4, []uint32{0, 1, 4294967295, 0}, []uint32{0, 0, 4294967295, 1}, make([]uint32, 4), []uint32{0, 1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []uint32{5}, []uint32{5}, make([]uint32, 1), []uint32{0}},
		{1, []uint32{5}, []uint32{6}, make([]uint32, 1), []uint32{1}},
		{3, []uint32{1, 2, 3}, []uint32{1, 4, 3}, make([]uint32, 3), []uint32{0, 1, 0}},
	}
	for _, tt := range tests {
		kernels.NeU32U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestNeI64I64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{4, []int64{-7486106602830593557, 8773683796379128716, 4817177250100823153, 5276955028067489600}, []int64{-6860092642535747309, -915217906097603218, -2383355780799197143, 7872424528909669039}, make([]int64, 4), []int64{1, 1, 1, 1}},
		{4, []int64{0, -1, 9223372036854775807, -9223372036854775808}, []int64{0, 1, 9223372036854775807, -9223372036854775808}, make([]int64, 4), []int64{0, 1, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []int64{5}, []int64{5}, make([]int64, 1), []int64{0}},
		{1, []int64{5}, []int64{6}, make([]int64, 1), []int64{1}},
		{3, []int64{-1, 0, 1}, []int64{-1, 2, 1}, make([]int64, 3), []int64{0, 1, 0}},
	}
	for _, tt := range tests {
		kernels.NeI64I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeU8F32 tests y = (x1 != x2) ? 1 : 0 for float32 with uint8 output
func TestNeU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{5}, []uint8{0}, []uint8{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.NeU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeU8F64 tests y = (x1 != x2) ? 1 : 0 for float64 with uint8 output
func TestNeU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{5}, []uint8{0}, []uint8{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.NeU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestNeU8U8(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint8
		want      []uint8
	}{
		{3, []uint8{102, 179, 92}, []uint8{102, 106, 71}, make([]uint8, 3), []uint8{0, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []uint8{188}, []uint8{188}, make([]uint8, 1), []uint8{0}},
		{1, []uint8{20}, []uint8{102}, make([]uint8, 1), []uint8{1}},
		{4, []uint8{121, 210, 214, 74}, []uint8{202, 210, 116, 74}, make([]uint8, 4), []uint8{1, 0, 1, 0}},
	}
	for _, tt := range tests {
		kernels.NeU8U8(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestNeU8U32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []uint32
		y      []uint8
		want   []uint8
	}{
		{3, []uint32{3041148567, 88409749, 4165731073}, []uint32{3041148567, 911989541, 780932287}, make([]uint8, 3), []uint8{0, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []uint32{787716372}, []uint32{787716372}, make([]uint8, 1), []uint8{0}},
		{1, []uint32{1306710475}, []uint32{2253811733}, make([]uint8, 1), []uint8{1}},
		{4, []uint32{1855189739, 1250819632, 2627888186, 599121577}, []uint32{1254751707, 1250819632, 1958805693, 599121577}, make([]uint8, 4), []uint8{1, 0, 1, 0}},
	}
	for _, tt := range tests {
		kernels.NeU8U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestNeU8I64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []int64
		y      []uint8
		want   []uint8
	}{
		{3, []int64{-1289891278, 61136438, 396917567}, []int64{-1947980670, 461901618, 396917567}, make([]uint8, 3), []uint8{1, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []int64{-1868089178}, []int64{-1868089178}, make([]uint8, 1), []uint8{0}},
		{1, []int64{1927948675}, []int64{1999874363}, make([]uint8, 1), []uint8{1}},
		{4, []int64{1324556529, -839177464, -1727985100, 791274835}, []int64{1324556529, -1623333434, -1727985100, -1999786066}, make([]uint8, 4), []uint8{0, 1, 0, 1}},
	}
	for _, tt := range tests {
		kernels.NeU8I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeStridedF32F32 tests y = (x1 != x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestNeStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{1, 0, 1, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{1, 0, 1, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 1, 1, 0, 1, 1, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 1, 1, 0, 1, 1, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.NeStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeStridedF64F64 tests y = (x1 != x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestNeStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{1, 0, 1, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{1, 0, 1, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 1, 1, 0, 1, 1, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 1, 1, 0, 1, 1, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.NeStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestNeStridedU32U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint32
		want         []uint32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{900429414, 3232002110, 708846528}, []uint32{900429414, 3232002110, 708846528}, make([]uint32, 3), []uint32{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{1289644851, 1752121911, 2040622706}, []uint32{1401770077, 0, 1752121911, 0, 1688769722}, make([]uint32, 3), []uint32{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{3646282213, 1328893999, 2139682948, 2538323506, 766400809, 134368901}, []uint32{3646282213, 1328893999, 2638005529, 3477324106, 766400809, 3484276841}, make([]uint32, 6), []uint32{0, 0, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{2070966294, 406664826, 2049896683, 2132026050, 666097320, 944825023}, []uint32{2070966294, 1146635320, 2716736922, 1064798458, 1405245253, 3208528052}, make([]uint32, 6), []uint32{0, 1, 1, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1973768503, 795804453, 2054183477, 2061341494, 2951159782, 1233892470, 4180920497, 2900888898}, []uint32{1973768503, 795804453, 3270912278, 2061341494, 2951159782, 2054984526, 4180920497, 4227802925}, make([]uint32, 8), []uint32{0, 0, 1, 0, 0, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{2845562029, 1238253079, 3160027610, 1130775036, 663064023, 3041699049, 163531492, 2851006694}, []uint32{3621539890, 2288509356, 270468823, 4003060513, 663064023, 3767509456, 3041699049, 1753116438}, make([]uint32, 8), []uint32{1, 1, 1, 1, 0, 1, 0, 1}},
		// edge cases
		{2, 1, []int{2}, []int{1}, []int{1}, []int{1}, []uint32{0, 4294967295}, []uint32{4294967295, 0}, make([]uint32, 2), []uint32{1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.NeStridedU32U32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestNeStridedI64I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []int64
		want         []int64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{17994736556943773, -493955476421690037, -180167959395660532}, []int64{17994736556943773, -493955476421690037, -180167959395660532}, make([]int64, 3), []int64{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{-541123659654036322, 900386063286981125, -36850228479712439}, []int64{-919222101960940066, 0, 900386063286981125, 0, 534838033964238896}, make([]int64, 3), []int64{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{-744377485120088590, 654198222276916213, 500623000578120310, -191374620327099617, -678471367157836850, -1086168265410491672}, []int64{-912758823551750356, 654198222276916213, -189541263933704155, 1014333205797641628, 818630521155988585, -337948028678787652}, make([]int64, 6), []int64{1, 0, 1, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{1034747202085545007, -268853393181716948, -46628734738269821, -305907691560610006, -265266416153174630, 370232107262077558}, []int64{1034747202085545007, -192803477060577297, -268853393181716948, -610474136442331882, -46628734738269821, 802568316623985090}, make([]int64, 6), []int64{0, 0, 0, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-835784191235372052, 1110818848099073978, 538651322926709229, 649593396898511207, 889172349546135377, -1084606807721498612, -348578951862595560, -605985268888645595}, []int64{361697410097073971, 1110818848099073978, 538651322926709229, 649593396898511207, 889172349546135377, -1084606807721498612, -348578951862595560, -399202919902164226}, make([]int64, 8), []int64{1, 0, 0, 0, 0, 0, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{705586963864918941, -638136030611491076, -412310342938282844, 847023438678417088, -733478969824858391, -498863130296061712, 14110212593324951, -183597649455496375}, []int64{705586963864918941, -412310342938282844, -638136030611491076, 41392780576654060, -733478969824858391, 14110212593324951, -498863130296061712, -183597649455496375}, make([]int64, 8), []int64{0, 0, 0, 1, 0, 0, 0, 0}},
		// edge cases
		{2, 1, []int{2}, []int{1}, []int{1}, []int{1}, []int64{-9223372036854775808, 9223372036854775807}, []int64{9223372036854775807, -9223372036854775808}, make([]int64, 2), []int64{1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.NeStridedI64I64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeStridedU8F32 tests y = (x1 != x2) ? 1 : 0 for float32 with strided memory and uint8 output
func TestNeStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 1, 0, 1, 1, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 1, 0, 1, 1, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.NeStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestNeStridedU8F64 tests y = (x1 != x2) ? 1 : 0 for float64 with strided memory and uint8 output
func TestNeStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{1, 0, 1, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 1, 0, 1, 1, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 1, 1, 0, 1, 1, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.NeStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestNeStridedU8U8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []uint8
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{2, 7, 6}, []uint8{4, 6, 5}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{4, 0, 4}, []uint8{0, 4, 0, 3, 8}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{1, 2, 5, 5, 7, 6}, []uint8{9, 6, 3, 1, 9, 3}, make([]uint8, 6), []uint8{1, 1, 1, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{5, 9, 3, 4, 9, 6}, []uint8{1, 9, 7, 9, 2, 0}, make([]uint8, 6), []uint8{1, 1, 1, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{2, 0, 6, 2, 7, 9, 7, 3}, []uint8{3, 4, 3, 7, 0, 9, 0, 9}, make([]uint8, 8), []uint8{1, 1, 1, 1, 1, 0, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{6, 9, 5, 4, 8, 8, 6, 0}, []uint8{0, 0, 0, 1, 3, 0, 1, 1}, make([]uint8, 8), []uint8{1, 1, 1, 1, 1, 1, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.NeStridedU8U8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestNeStridedU8U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []uint32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{7, 9, 4}, []uint32{3, 8, 9}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{1, 6, 3}, []uint32{3, 7, 8, 1, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{2, 0, 9, 8, 5, 3}, []uint32{7, 7, 5, 9, 1, 5}, make([]uint8, 6), []uint8{1, 1, 1, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{7, 5, 7, 1, 5, 7}, []uint32{1, 9, 1, 4, 0, 3}, make([]uint8, 6), []uint8{1, 1, 1, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{5, 8, 5, 4, 1, 1, 0, 9}, []uint32{0, 9, 1, 8, 9, 6, 7, 6}, make([]uint8, 8), []uint8{1, 1, 1, 1, 1, 1, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{0, 9, 5, 2, 9, 1, 7, 8}, []uint32{6, 0, 6, 8, 6, 8, 0, 6}, make([]uint8, 8), []uint8{1, 1, 1, 1, 1, 1, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.NeStridedU8U32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestNeStridedU8I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []int64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{-1, -4, 0}, []int64{5, -1, -2}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{-3, 4, -4}, []int64{-2, 5, 3, 0, 5}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{-1, -2, -5, -2, 1, 5}, []int64{4, -3, -5, 0, 5, 4}, make([]uint8, 6), []uint8{1, 1, 0, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{-5, -4, -5, -1, -4, 5}, []int64{-4, 3, 1, 0, 1, 0}, make([]uint8, 6), []uint8{1, 1, 1, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{5, -3, 5, 3, -1, 0, -1, -4}, []int64{5, 3, 2, -5, 0, 0, 4, 3}, make([]uint8, 8), []uint8{0, 1, 1, 1, 1, 0, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{4, 1, 1, -4, 4, -2, -4, -1}, []int64{5, 2, -3, 4, 4, -3, -5, -4}, make([]uint8, 8), []uint8{1, 1, 1, 1, 0, 1, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.NeStridedU8I64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtF32F32 tests y = (x1 < x2) ? 1 : 0 for float32 with float32 output
func TestLtF32F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []float32
		want   []float32
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{1}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]float32, 4), []float32{1, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.LtF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtF64F64 tests y = (x1 < x2) ? 1 : 0 for float64 with float64 output
func TestLtF64F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []float64
		want   []float64
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{1}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]float64, 4), []float64{1, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.LtF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLtU32U32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{4, []uint32{3508392673, 3770518799, 1703085865, 2439179832}, []uint32{3739910397, 2147398764, 3442827927, 1965580511}, make([]uint32, 4), []uint32{1, 0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []uint32{5}, []uint32{6}, make([]uint32, 1), []uint32{1}},
		{1, []uint32{6}, []uint32{5}, make([]uint32, 1), []uint32{0}},
		{1, []uint32{5}, []uint32{5}, make([]uint32, 1), []uint32{0}},
		{3, []uint32{1, 2, 3}, []uint32{2, 2, 2}, make([]uint32, 3), []uint32{1, 0, 0}},
		{2, []uint32{4294967295, 0}, []uint32{0, 4294967295}, make([]uint32, 2), []uint32{0, 1}},
	}
	for _, tt := range tests {
		kernels.LtU32U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestLtI64I64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{4, []int64{788014010005370311, -3329029517141677617, -6076130490227969037, -8904874004283724123}, []int64{-7383409719100804064, 5542985121775927968, 7965646558203615215, 8067512690384191690}, make([]int64, 4), []int64{0, 1, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []int64{5}, []int64{6}, make([]int64, 1), []int64{1}},
		{1, []int64{6}, []int64{5}, make([]int64, 1), []int64{0}},
		{1, []int64{5}, []int64{5}, make([]int64, 1), []int64{0}},
		{1, []int64{-5}, []int64{0}, make([]int64, 1), []int64{1}},
		{1, []int64{0}, []int64{-5}, make([]int64, 1), []int64{0}},
		{3, []int64{-9223372036854775808, 0, 9223372036854775807}, []int64{-9223372036854775807, 9223372036854775807, 0}, make([]int64, 3), []int64{1, 1, 0}},
	}
	for _, tt := range tests {
		kernels.LtI64I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtU8F32 tests y = (x1 < x2) ? 1 : 0 for float32 with uint8 output
func TestLtU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []uint8{0}, []uint8{1}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.LtU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtU8F64 tests y = (x1 < x2) ? 1 : 0 for float64 with uint8 output
func TestLtU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []uint8{0}, []uint8{1}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 0, 0, 0}},
	}

	for _, tt := range tests {
		kernels.LtU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLtU8U8(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []uint8
		y      []uint8
		want   []uint8
	}{
		{3, []uint8{102, 179, 92}, []uint8{14, 106, 71}, make([]uint8, 3), []uint8{0, 0, 0}},
		{1, []uint8{188}, []uint8{20}, make([]uint8, 1), []uint8{0}},
		{4, []uint8{102, 121, 210, 214}, []uint8{74, 202, 87, 116}, make([]uint8, 4), []uint8{0, 1, 0, 0}},
		{0, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LtU8U8(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLtU8U32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []uint32
		y      []uint8
		want   []uint8
	}{
		{3, []uint32{613608295, 2795544706, 242285876}, []uint32{3100961111, 4031053213, 3344769}, make([]uint8, 3), []uint8{1, 1, 0}},
		{1, []uint32{4261516219}, []uint32{2652062880}, make([]uint8, 1), []uint8{0}},
		{4, []uint32{2627030329, 30349564, 99052376, 2253890010}, []uint32{1717389822, 200427519, 4182248123, 999745294}, make([]uint8, 4), []uint8{0, 1, 1, 0}},
		{0, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LtU8U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLtU8I64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []int64
		y      []uint8
		want   []uint8
	}{
		{3, []int64{-1758331971, 508464061, -504821909}, []int64{2075460851, -142752264, 1545932260}, make([]uint8, 3), []uint8{1, 0, 1}},
		{1, []int64{774414982}, []int64{-212604088}, make([]uint8, 1), []uint8{0}},
		{4, []int64{-2090511087, 1899242072, 271820813, -492132359}, []int64{-2078909095, -1155802239, -1112287141, 787110843}, make([]uint8, 4), []uint8{1, 0, 0, 1}},
		{0, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LtU8I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtStridedF32F32 tests y = (x1 < x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestLtStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{1, 0, 1, 0, 0, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{1, 0, 1, 0, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 1, 0, 0, 1, 1, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 1, 0, 0, 1, 1, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LtStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLtStridedF64F64 tests y = (x1 < x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestLtStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{1, 0, 1, 0, 0, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{1, 0, 1, 0, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 1, 0, 0, 1, 1, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 1, 0, 0, 1, 1, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LtStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLtStridedU8U8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []uint8
		y            []uint8
		want         []uint8
	}{
		{1, 1, []int{1}, []int{1}, []int{1}, []int{1}, []uint8{102}, []uint8{179}, make([]uint8, 1), []uint8{1}},
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{68, 64, 255}, []uint8{49, 21, 58}, make([]uint8, 3), []uint8{0, 0, 0}},
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{20, 163, 241}, []uint8{173, 0, 59, 0, 131}, make([]uint8, 3), []uint8{1, 0, 0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{203, 158, 131, 124, 32, 95}, []uint8{189, 213, 163, 68, 121, 120}, make([]uint8, 6), []uint8{0, 1, 1, 0, 1, 1}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{189, 69, 40, 116, 186, 147}, []uint8{146, 60, 203, 155, 93, 208}, make([]uint8, 6), []uint8{0, 1, 1, 0, 0, 1}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{135, 134, 71, 8, 72, 179, 80, 23}, []uint8{59, 208, 7, 71, 173, 113, 55, 50}, make([]uint8, 8), []uint8{0, 1, 0, 1, 1, 0, 0, 1}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{0, 81, 51, 196, 91, 64, 176, 198}, []uint8{198, 34, 92, 208, 236, 157, 6, 214}, make([]uint8, 8), []uint8{1, 0, 1, 1, 1, 0, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LtStridedU8U8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestLtStridedU8U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []uint32
		y            []uint8
		want         []uint8
	}{
		{1, 1, []int{1}, []int{1}, []int{1}, []int{1}, []uint32{3421126067}, []uint32{787846414}, make([]uint8, 1), []uint8{0}},
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{2134003008, 442015537, 638974010}, []uint32{739303731, 3483374779, 321011650}, make([]uint8, 3), []uint8{0, 1, 0}},
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{1857752483, 136471725, 2696597379}, []uint32{792072276, 0, 1272924911, 0, 1103667320}, make([]uint8, 3), []uint8{0, 1, 0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{1307974046, 2883983228, 1804481119, 4111367893, 3928351300, 2065337976}, []uint32{3090989633, 1569240373, 245833590, 939244759, 1361033480, 3975724940}, make([]uint8, 6), []uint8{1, 0, 0, 0, 0, 1}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{1108302661, 2938690932, 3939944851, 880185035, 4250670396, 3087293392}, []uint32{375990914, 985473634, 1692484128, 783577423, 215003959, 3357252962}, make([]uint8, 6), []uint8{0, 0, 0, 1, 0, 1}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{3658270598, 3907005704, 3536609971, 1598000151, 4201593552, 4007880775, 1698958449, 1316356658}, []uint32{990990890, 2724714561, 2207650171, 1716241992, 1522299378, 3275389249, 221686320, 3150046940}, make([]uint8, 8), []uint8{0, 0, 0, 1, 0, 0, 0, 1}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1831733811, 2979748784, 1144485316, 464765894, 4137477410, 2059728541, 1702226128, 3208537302}, []uint32{1941163822, 2797350619, 3759332583, 3149940328, 1414277662, 3287864850, 2141097490, 3682407445}, make([]uint8, 8), []uint8{1, 1, 1, 1, 0, 1, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LtStridedU8U32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestLtStridedU8I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []int64
		y            []uint8
		want         []uint8
	}{
		{1, 1, []int{1}, []int{1}, []int{1}, []int{1}, []int64{1273642419}, []int64{-1359637234}, make([]uint8, 1), []uint8{0}},
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{-13480640, -1705468111, -1508509638}, []int64{-1408179917, 1335891131, -1826471998}, make([]uint8, 3), []uint8{0, 1, 0}},
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{-289731165, -2011011923, 549113731}, []int64{-1355411372, 0, -874558737, 0, -1043816328}, make([]uint8, 3), []uint8{0, 1, 0}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{-839509602, 736499580, -343002529, 1963884245, 1780867652, -82145672}, []int64{943505985, -578243275, -1901650058, -1208238889, -786450168, 1828241292}, make([]uint8, 6), []uint8{1, 0, 0, 0, 0, 1}},
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{-1039180987, 791207284, 1792461203, -1267298613, 2103186748, 939809744}, []int64{-1771492734, -1162010014, -454999520, -1363906225, -1932479689, 1209769314}, make([]uint8, 6), []uint8{0, 0, 0, 1, 0, 1}},
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{1510786950, 1759522056, 1389126323, -549483497, 2054109904, 1860397127, -448525199, -831126990}, []int64{-1157039758, 577230913, 60166523, -431241656, -625184270, 1127905601, -1925797328, 1002563292}, make([]uint8, 8), []uint8{0, 0, 0, 1, 0, 0, 0, 1}},
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-315749837, 832265136, -1002998332, -1682717754, 1989993762, -87755107, -445257520, 1061053654}, []int64{-206319826, 649866971, 1611848935, 1002456680, -733205986, 1140381202, -6386158, 1534923797}, make([]uint8, 8), []uint8{1, 1, 1, 1, 0, 1, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LtStridedU8I64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLeF32F32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float32
		want      []float32
	}{
		{3, []float32{0.33669036626815796, 0.12880940735340118, 0.23446236550807953}, []float32{0.23033303022384644, -1.1228563785552979, -0.18632829189300537}, make([]float32, 3), []float32{0.0, 0.0, 0.0}},
		{0, nil, nil, nil, nil},
		{1, []float32{0.5}, []float32{0.5}, make([]float32, 1), []float32{1.0}},
		{1, []float32{1.0}, []float32{0.0}, make([]float32, 1), []float32{0.0}},
		{4, []float32{0.46165722608566284, 0.2673508822917938, 0.5349046587944031, 0.809357225894928}, []float32{1.110290288925171, -1.6897989511489868, -0.9889599084854126, 0.9579718112945557}, make([]float32, 4), []float32{1.0, 0.0, 0.0, 1.0}},
	}
	for _, tt := range tests {
		kernels.LeF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestLeF64F64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []float64
		want      []float64
	}{
		{3, []float64{1.3221351399922163, 0.8171897708378104, -0.7658385999620981}, []float64{-0.7506223399995002, 1.3525477542195647, 0.6863218849828586}, make([]float64, 3), []float64{0.0, 1.0, 1.0}},
		{0, nil, nil, nil, nil},
		{1, []float64{0.5}, []float64{0.5}, make([]float64, 1), []float64{1.0}},
		{1, []float64{1.0}, []float64{0.0}, make([]float64, 1), []float64{0.0}},
		{4, []float64{0.28151956559794167, 0.05616354046869207, 0.5227160560244443, -0.23835686682577745}, []float64{-0.04990334692036799, 0.5263369393957674, -0.008498823645732559, 0.7290606194273371}, make([]float64, 4), []float64{0.0, 1.0, 0.0, 1.0}},
	}
	for _, tt := range tests {
		kernels.LeF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLeU32U32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{3, []uint32{4159606953, 2317794028, 2031764889}, []uint32{4008378212, 872681273, 4105569283}, make([]uint32, 3), []uint32{0, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []uint32{936352740}, []uint32{1429249040}, make([]uint32, 1), []uint32{1}},
		{4, []uint32{192544988, 28411769, 3274634121, 2577369019}, []uint32{918849194, 2965789985, 3976312647, 390484377}, make([]uint32, 4), []uint32{1, 1, 1, 0}},
		{2, []uint32{0, 4294967295}, []uint32{4294967295, 0}, make([]uint32, 2), []uint32{1, 0}},
	}
	for _, tt := range tests {
		kernels.LeU32U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestLeI64I64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{3, []int64{6058222662777746728, -192207805627430832, 4805119135753249703}, []int64{-2799311294657531339, -7104198915102372426, -5478619313638470920}, make([]int64, 3), []int64{0, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []int64{-7954502854394615809}, []int64{3467916982582106879}, make([]int64, 1), []int64{1}},
		{4, []int64{7964908024666093872, -4781125400480466666, -2924856690423215831, 4424952669103715920}, []int64{661280560519328075, 606336261304181445, -7333530199904528139, -5675857819113078156}, make([]int64, 4), []int64{0, 1, 0, 0}},
		{3, []int64{-9223372036854775808, 9223372036854775807, 0}, []int64{0, 0, 0}, make([]int64, 3), []int64{1, 0, 1}},
	}
	for _, tt := range tests {
		kernels.LeI64I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeU8F32 tests y = (x1 <= x2) ? 1 : 0 for float32 with uint8 output
func TestLeU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []uint8{0}, []uint8{1}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.LeU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeU8F64 tests y = (x1 <= x2) ? 1 : 0 for float64 with uint8 output
func TestLeU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []uint8{0}, []uint8{1}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{1, 1, 1, 0}},
	}

	for _, tt := range tests {
		kernels.LeU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLeU8U8(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []uint8
		y      []uint8
		want   []uint8
	}{
		{3, []uint8{102, 179, 92}, []uint8{14, 106, 71}, make([]uint8, 3), []uint8{0, 0, 0}},
		{1, []uint8{100}, []uint8{100}, make([]uint8, 1), []uint8{1}},
		{4, []uint8{188, 20, 102, 121}, []uint8{210, 214, 74, 202}, make([]uint8, 4), []uint8{1, 1, 0, 1}},
		{0, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LeU8U8(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLeU8U32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []uint32
		y      []uint8
		want   []uint8
	}{
		{3, []uint32{1433267572, 613608295, 2795544706}, []uint32{242285876, 3100961111, 4031053213}, make([]uint8, 3), []uint8{0, 1, 1}},
		{1, []uint32{4294967295}, []uint32{4294967295}, make([]uint8, 1), []uint8{1}},
		{4, []uint32{3344769, 4261516219, 2652062880, 2627030329}, []uint32{30349564, 99052376, 2253890010, 1717389822}, make([]uint8, 4), []uint8{1, 0, 0, 0}},
		{0, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LeU8U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLeU8I64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []int64
		y      []uint8
		want   []uint8
	}{
		{3, []int64{-2038478438684014593, 777431531920034491, 2146497176616232718}, []int64{3801320372615379901, 648881935158774717, -928354834335594645}, make([]uint8, 3), []uint8{1, 0, 0}},
		{1, []int64{0}, []int64{0}, make([]uint8, 1), []uint8{1}},
		{4, []int64{-4349107012400511757, -2906938046949767688, -3754827248769364508, -2627833672226004346}, []int64{-1466071124456248504, -3411695907037162223, 3668790493110870616, 3977708969185749005}, make([]uint8, 4), []uint8{1, 0, 1, 1}},
		{0, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LeU8I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeStridedF32F32 tests y = (x1 <= x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestLeStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{1, 1, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{1, 1, 1, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 1, 0, 1, 1, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 1, 0, 1, 1, 1, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LeStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeStridedF64F64 tests y = (x1 <= x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestLeStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{1, 1, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{1, 1, 1, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 1, 0, 1, 1, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 1, 0, 1, 1, 1, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LeStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLeStridedU32U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint32
		want         []uint32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{1987027793, 1396114773, 3128015259}, []uint32{586997441, 3661102379, 2881765476}, make([]uint32, 3), []uint32{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{3088543231, 3570295494, 3528397010}, []uint32{146417929, 0, 3223485904, 0, 2376322427}, make([]uint32, 3), []uint32{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{3245496387, 2548802026, 96887398, 1443742607, 2456379393, 1723756558}, []uint32{2416541589, 3822918813, 1890540130, 1340764494, 1157334943, 2961268645}, make([]uint32, 6), []uint32{0, 1, 1, 0, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{2858071459, 3966506315, 1625578035, 2026672759, 483407054, 3771751422}, []uint32{2437471775, 267877816, 640081365, 3901036409, 122872954, 607136517}, make([]uint32, 6), []uint32{0, 0, 0, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{189543867, 1345241285, 2442403340, 1402967118, 1639198227, 2942930855, 630456115, 290502620}, []uint32{2536163495, 1218188447, 871332263, 1528477630, 3651009455, 898629607, 1810766838, 581146142}, make([]uint32, 8), []uint32{1, 0, 0, 1, 1, 0, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{2751822346, 1109663605, 3735583568, 829507288, 1852020229, 1682794058, 675234332, 554301403}, []uint32{1700050363, 2465163660, 2929755768, 1364749070, 1801511990, 2621452091, 3473202121, 1199375529}, make([]uint32, 8), []uint32{0, 0, 1, 1, 0, 1, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LeStridedU32U32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestLeStridedI64I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []int64
		want         []int64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{-2311289012512574652, -706231199052643622, -2994973389538788661}, []int64{-3168036758871418134, -454389558966480343, 5881615638382970617}, make([]int64, 3), []int64{0, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{7080287069857782902, 8202123213262345961, 7064264001406805523}, []int64{-2145556009706008462, 0, -3608086333303197335, 0, -7049908448927708800}, make([]int64, 3), []int64{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{-5147365775237469545, -3043852911393359879, 1211328845771149786, -3714536849096779900, -5189001053069710748, 5798491805091915836}, []int64{-4379538423873055411, 6616145871534770062, -8311599412615137569, -2978795474390505020, 4417689713836322977, -2172376730624174158}, make([]int64, 6), []int64{1, 1, 0, 1, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{6985625873891438540, 4474337588313543237, 3044772429905616237, 4781458234272044661, -4103021852913179254, 8083044495031910748}, []int64{2090510781865493888, 4959627530181964643, -6588875529536773983, 4482927003154186587, 5656596229435965661, -3681541281104039155}, make([]int64, 6), []int64{0, 0, 1, 1, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-6522515712204673939, 2821542905755227005, -7764508551129054141, 7335070883523507838, 7611975408007744698, 5209913254391211726, -5877082918766420143, 7447528169529702781}, []int64{6357182898760313285, 3263266695484014930, 129612659705337261, 4898444054038229247, 6366993524662119794, -3032351972007011650, 1720312414492489038, 7442816899691202573}, make([]int64, 8), []int64{1, 1, 1, 0, 0, 0, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-870018474188577655, -9182307169143139314, 3244998173042736637, 7297192449654769236, -4295505173345312231, 8716687907521994538, 8482332282235864874, 2347545848738781120}, []int64{-3348880697434124872, -8456883959388288889, 4684916046624055646, -4088758755670883926, 2313898552259224700, 6668791938498129628, 3393128188455010113, 5572193449973479962}, make([]int64, 8), []int64{0, 0, 1, 0, 1, 0, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LeStridedI64I64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeStridedU8F32 tests y = (x1 <= x2) ? 1 : 0 for float32 with strided memory and uint8 output
func TestLeStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{1, 1, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{1, 1, 1, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 1, 0, 1, 1, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 1, 0, 1, 1, 1, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LeStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestLeStridedU8F64 tests y = (x1 <= x2) ? 1 : 0 for float64 with strided memory and uint8 output
func TestLeStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{1, 1, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{1, 1, 1, 1, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 1, 0, 1, 1, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 1, 0, 1, 1, 1, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.LeStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLeStridedU8U8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []uint8
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{2, 7, 6}, []uint8{4, 6, 5}, make([]uint8, 3), []uint8{1, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{0, 4, 0}, []uint8{3, 0, 8, 0, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{0, 4, 1, 2, 5, 5}, []uint8{7, 6, 9, 6, 3, 1}, make([]uint8, 6), []uint8{1, 1, 1, 1, 0, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{0, 4, 1, 2, 5, 5}, []uint8{7, 6, 6, 3, 9, 1}, make([]uint8, 6), []uint8{1, 1, 1, 1, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{9, 6, 2, 0, 6, 2, 7, 9}, []uint8{7, 3, 3, 4, 3, 7, 0, 9}, make([]uint8, 8), []uint8{0, 0, 1, 1, 0, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{9, 2, 6, 0, 6, 7, 2, 9}, []uint8{7, 3, 3, 4, 3, 7, 0, 9}, make([]uint8, 8), []uint8{0, 0, 1, 1, 0, 1, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LeStridedU8U8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLeStridedU8U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []uint32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{42, 67, 76}, []uint32{14, 26, 35}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{20, 24, 50}, []uint32{13, 0, 78, 0, 14}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{10, 54, 31, 72, 15, 95}, []uint32{67, 6, 49, 76, 73, 11}, make([]uint8, 6), []uint8{1, 0, 1, 1, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{99, 13, 41, 69, 87, 19}, []uint32{72, 29, 80, 33, 75, 64}, make([]uint8, 6), []uint8{0, 1, 1, 0, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{39, 76, 32, 10, 86, 22, 77, 19}, []uint32{7, 23, 43, 94, 93, 77, 70, 9}, make([]uint8, 8), []uint8{0, 0, 1, 1, 1, 1, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{70, 86, 39, 99, 15, 78, 84, 8}, []uint32{66, 30, 40, 60, 70, 61, 23, 20}, make([]uint8, 8), []uint8{0, 0, 0, 0, 1, 0, 0, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LeStridedU8U32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestLeStridedU8I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []int64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{-8, -3, 6}, []int64{4, -4, 5}, make([]uint8, 3), []uint8{1, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{-10, -6, 0}, []int64{3, 0, 8, 0, 4}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{0, 4, 1, 2, 5, 5}, []int64{-3, -4, -1, 6, 3, 1}, make([]uint8, 6), []uint8{0, 0, 0, 1, 0, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{9, 3, -9, -1, -3, 9}, []int64{2, -1, -10, 3, 5, -6}, make([]uint8, 6), []uint8{0, 0, 1, 1, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{9, 6, 2, 0, -4, -8, 7, 9}, []int64{-3, -7, -7, 4, 3, 7, 0, -1}, make([]uint8, 8), []uint8{0, 0, 0, 1, 1, 1, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{9, 2, 6, 0, -4, 7, -8, 9}, []int64{-3, -7, -7, 4, 3, 7, 0, -1}, make([]uint8, 8), []uint8{0, 0, 0, 1, 1, 1, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.LeStridedU8I64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtF32F32 tests y = (x1 > x2) ? 1 : 0 for float32 with float32 output
func TestGtF32F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []float32
		want   []float32
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]float32, 4), []float32{0, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.GtF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtF64F64 tests y = (x1 > x2) ? 1 : 0 for float64 with float64 output
func TestGtF64F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []float64
		want   []float64
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]float64, 4), []float64{0, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.GtF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGtU32U32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{3, []uint32{1608637542, 3421126067, 4083286876}, []uint32{787846414, 3143890026, 3348747335}, make([]uint32, 3), []uint32{1, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []uint32{2571218620}, []uint32{2563451924}, make([]uint32, 1), []uint32{1}},
		{4, []uint32{670094950, 1914837113, 669991378, 429389014}, []uint32{249467210, 1972458954, 3720198231, 1433267572}, make([]uint32, 4), []uint32{1, 0, 0, 0}},
		{2, []uint32{0, 4294967295}, []uint32{4294967295, 0}, make([]uint32, 2), []uint32{0, 1}},
	}
	for _, tt := range tests {
		kernels.GtU32U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestGtI64I64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{3, []int64{1865242737500154727, 3838261603483033730, -8843655056009921228}, []int64{8668306688712173911, 6132484236315524509, -5306406783962379903}, make([]int64, 3), []int64{0, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []int64{-5869293399537773637}, []int64{-5840155977938942816}, make([]int64, 1), []int64{0}},
		{4, []int64{-3611093278762119879, 456675647751657724, -1255392779875947688, -3851142621966130726}, []int64{2063321781277379070, -6650164457111402497, -3834254486507353413, -2465188841811155186}, make([]int64, 4), []int64{0, 1, 1, 0}},
		{3, []int64{-9223372036854775808, 9223372036854775807, 0}, []int64{0, 0, -1}, make([]int64, 3), []int64{0, 1, 1}},
	}
	for _, tt := range tests {
		kernels.GtI64I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtU8F32 tests y = (x1 > x2) ? 1 : 0 for float32 with uint8 output
func TestGtU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []uint8{0}, []uint8{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.GtU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtU8F64 tests y = (x1 > x2) ? 1 : 0 for float64 with uint8 output
func TestGtU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []uint8{0}, []uint8{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 0, 0, 1}},
	}

	for _, tt := range tests {
		kernels.GtU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGtU8U8(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint8
		want      []uint8
	}{
		{3, []uint8{165, 4, 192}, []uint8{241, 141, 86}, make([]uint8, 3), []uint8{0, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []uint8{188}, []uint8{20}, make([]uint8, 1), []uint8{1}},
		{3, []uint8{0, 255, 128}, []uint8{255, 0, 128}, make([]uint8, 3), []uint8{0, 1, 0}},
		{4, []uint8{236, 104, 216, 196}, []uint8{163, 57, 142, 156}, make([]uint8, 4), []uint8{1, 1, 1, 1}},
	}
	for _, tt := range tests {
		kernels.GtU8U8(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGtU8U32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []uint32
		y      []uint8
		want   []uint8
	}{
		{3, []uint32{1907596332, 2049465263, 1968094872}, []uint32{1704462728, 1597038692, 476066397}, make([]uint8, 3), []uint8{1, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []uint32{2470827871}, []uint32{4233769675}, make([]uint8, 1), []uint8{0}},
		{3, []uint32{0, 4294967295, 2147483648}, []uint32{4294967295, 0, 2147483648}, make([]uint8, 3), []uint8{0, 1, 0}},
		{4, []uint32{2978241676, 2438106638, 1986361829, 976665734}, []uint32{688232577, 2018991982, 87025424, 1003642970}, make([]uint8, 4), []uint8{1, 1, 1, 0}},
	}
	for _, tt := range tests {
		kernels.GtU8U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGtU8I64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []int64
		y      []uint8
		want   []uint8
	}{
		{3, []int64{4535311451329819337, -2965113260396281154, 122983201612814733}, []int64{12717166372624592, -1669131881297481177, -376140319007160991}, make([]uint8, 3), []uint8{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []int64{-1123604296088030145}, []int64{-863514995561078673}, make([]uint8, 1), []uint8{0}},
		{4, []int64{-9223372036854775808, 9223372036854775807, 0, -1}, []int64{9223372036854775807, -9223372036854775808, 0, 1}, make([]uint8, 4), []uint8{0, 1, 0, 0}},
		{4, []int64{-3951507253728956227, 2071067873325136760, -1022218578877308949, -2928433995359375012}, []int64{-1051847249984543907, 3757184644264627456, 2180670305138327494, -1349617033920322326}, make([]uint8, 4), []uint8{0, 0, 0, 0}},
	}
	for _, tt := range tests {
		kernels.GtU8I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtStridedF32F32 tests y = (x1 > x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestGtStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{0, 0, 0, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{0, 0, 0, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 0, 1, 0, 0, 0, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{0, 0, 1, 0, 0, 0, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GtStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtStridedF64F64 tests y = (x1 > x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestGtStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{0, 0, 0, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{0, 0, 0, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 0, 1, 0, 0, 0, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{0, 0, 1, 0, 0, 0, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GtStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGtStridedU32U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint32
		want         []uint32
	}{
		{1, 1, []int{1}, []int{1}, []int{1}, []int{1}, []uint32{15}, []uint32{2}, make([]uint32, 1), []uint32{1}},
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{4, 19, 18}, []uint32{17, 19, 5}, make([]uint32, 3), []uint32{0, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{7, 14, 7}, []uint32{19, 0, 20, 0, 18}, make([]uint32, 3), []uint32{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{15, 16, 20, 13, 9, 10}, []uint32{1, 14, 7, 17, 17, 19}, make([]uint32, 6), []uint32{1, 1, 1, 0, 0, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{8, 15, 16, 16, 9, 12}, []uint32{7, 7, 7, 11, 5, 17}, make([]uint32, 6), []uint32{1, 1, 1, 1, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{13, 15, 14, 15, 19, 12, 6, 11}, []uint32{14, 16, 12, 5, 15, 9, 19, 12}, make([]uint32, 8), []uint32{0, 0, 1, 1, 1, 1, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1, 13, 4, 5, 17, 5, 0, 0}, []uint32{2, 19, 12, 2, 18, 16, 13, 7}, make([]uint32, 8), []uint32{0, 0, 1, 1, 0, 0, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.GtStridedU32U32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGtStridedI64I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []int64
		want         []int64
	}{
		{1, 1, []int{1}, []int{1}, []int{1}, []int{1}, []int64{5}, []int64{-8}, make([]int64, 1), []int64{1}},
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{-6, 9, 8}, []int64{7, 9, -5}, make([]int64, 3), []int64{0, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{-3, 4, -3}, []int64{9, 0, 10, 0, 8}, make([]int64, 3), []int64{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{5, 6, 10, 3, -1, 0}, []int64{-9, 4, -3, 7, 7, 9}, make([]int64, 6), []int64{1, 1, 1, 0, 0, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{-2, 5, 6, 6, -1, 2}, []int64{-3, -3, -3, 1, -5, 7}, make([]int64, 6), []int64{1, 1, 1, 1, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{3, 5, 4, 5, 9, 2, -4, 1}, []int64{4, 6, 2, -5, 5, -1, 9, 2}, make([]int64, 8), []int64{0, 0, 1, 1, 1, 1, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-9, 3, -6, -5, 7, -5, -10, -10}, []int64{-8, 9, 2, -8, 8, 6, 3, -3}, make([]int64, 8), []int64{0, 0, 1, 1, 0, 0, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.GtStridedI64I64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtStridedU8F32 tests y = (x1 > x2) ? 1 : 0 for float32 with strided memory and uint8 output
func TestGtStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 0, 0, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 0, 0, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 0, 1, 0, 0, 0, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 0, 1, 0, 0, 0, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GtStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGtStridedU8F64 tests y = (x1 > x2) ? 1 : 0 for float64 with strided memory and uint8 output
func TestGtStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 0, 0, 0, 1, 0}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 0, 0, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 0, 1, 0, 0, 0, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{0, 0, 1, 0, 0, 0, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GtStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGtStridedU8U8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []uint8
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{4, 175, 21}, []uint8{76, 222, 228}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{68, 75, 135}, []uint8{220, 0, 47, 0, 47}, make([]uint8, 3), []uint8{0, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{49, 142, 75, 209, 14, 152}, []uint8{4, 243, 39, 192, 93, 54}, make([]uint8, 6), []uint8{1, 0, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{139, 95, 172, 61, 193, 29}, []uint8{10, 250, 171, 143, 7, 188}, make([]uint8, 6), []uint8{1, 0, 1, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{19, 228, 10, 3, 76, 243, 111, 74}, []uint8{87, 149, 96, 237, 105, 240, 157, 202}, make([]uint8, 8), []uint8{0, 1, 0, 0, 0, 1, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{1, 57, 12, 183, 200, 223, 178, 31}, []uint8{204, 134, 141, 47, 211, 163, 142, 245}, make([]uint8, 8), []uint8{0, 0, 0, 1, 0, 1, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.GtStridedU8U8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGtStridedU8U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []uint32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{170, 2871, 1275}, []uint32{1919, 2266, 2421}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{3379, 1446, 1126}, []uint32{736, 0, 2889, 0, 2947}, make([]uint8, 3), []uint8{1, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{1168, 3509, 2846, 3393, 603, 3939}, []uint32{2292, 1, 1157, 166, 3033, 675}, make([]uint8, 6), []uint8{0, 1, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{293, 812, 1262, 759, 3204, 1056}, []uint32{3348, 2759, 1266, 3815, 972, 3462}, make([]uint8, 6), []uint8{0, 0, 1, 0, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{2595, 3390, 3053, 2824, 2051, 3695, 930, 1345}, []uint32{1818, 3762, 237, 1561, 1569, 2159, 2811, 76}, make([]uint8, 8), []uint8{1, 0, 1, 1, 1, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{3880, 248, 3712, 3902, 2932, 3751, 2594, 1843}, []uint32{408, 3598, 1751, 350, 1087, 431, 3682, 1992}, make([]uint8, 8), []uint8{1, 1, 0, 1, 1, 1, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.GtStridedU8U32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGtStridedU8I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []int64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{-22, -1253, 1109}, []int64{-831, 256, -1381}, make([]uint8, 3), []uint8{1, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{769, 309, 1559}, []int64{1454, 0, 721, 0, -183}, make([]uint8, 3), []uint8{0, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{-1244, -487, 184, -182, -84, -232}, []int64{1219, -635, 72, -865, -966, -1645}, make([]uint8, 6), []uint8{0, 1, 1, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{-1907, 1510, 687, 16, 1963, -689}, []int64{-1068, 37, 1429, 1632, 757, 3}, make([]uint8, 6), []uint8{0, 1, 0, 0, 1, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-727, 1013, 1069, 522, 1558, -706, 819, 1698}, []int64{1250, -294, -1838, -1970, 1647, -425, 1445, -787}, make([]uint8, 8), []uint8{0, 1, 1, 1, 0, 0, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-1196, 1742, 219, 537, -692, -1194, -50, -1635}, []int64{1230, -804, -1690, -1161, 911, -1972, 825, 863}, make([]uint8, 8), []uint8{0, 1, 1, 1, 0, 1, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.GtStridedU8I64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeF32F32 tests y = (x1 >= x2) ? 1 : 0 for float32 with float32 output
func TestGeF32F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []float32
		want   []float32
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []float32{0}, []float32{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]float32, 4), []float32{0, 1, 1, 1}},
	}

	for _, tt := range tests {
		kernels.GeF32F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeF64F64 tests y = (x1 >= x2) ? 1 : 0 for float64 with float64 output
func TestGeF64F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []float64
		want   []float64
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []float64{0}, []float64{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]float64, 4), []float64{0, 1, 1, 1}},
	}

	for _, tt := range tests {
		kernels.GeF64F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGeU32U32(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint32
		want      []uint32
	}{
		{3, []uint32{3398118020, 2458429517, 3145758303}, []uint32{3132035640, 4012788575, 396057066}, make([]uint32, 3), []uint32{1, 0, 1}},
		{0, nil, nil, nil, nil},
		{1, []uint32{2507212498}, []uint32{3553093513}, []uint32{0}, []uint32{0}},
		{4, []uint32{1065086761, 2399199316, 3027090233, 2911369228}, []uint32{2712333325, 2181360746, 1297391387, 3161396465}, make([]uint32, 4), []uint32{0, 1, 1, 0}},
		{2, []uint32{0, 4294967295}, []uint32{4294967295, 0}, make([]uint32, 2), []uint32{0, 1}},
	}
	for _, tt := range tests {
		kernels.GeU32U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestGeI64I64(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []int64
		want      []int64
	}{
		{3, []int64{-1208921175532042145, -3549747846638858119, 192536147187665350}, []int64{932739136271503278, -3522088855826898112, 3638363212974966249}, make([]int64, 3), []int64{0, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []int64{7990265759796603468}, []int64{2748244299973925547}, []int64{0}, []int64{1}},
		{4, []int64{8300403343599976176, 8241706798601227680, 806251621251122457, -1157449544568945578}, []int64{3305579448340966237, 5797096463804156392, 2693885526871483195, -1718497381336334495}, make([]int64, 4), []int64{1, 1, 0, 1}},
		{3, []int64{-9223372036854775808, 9223372036854775807, 0}, []int64{0, 0, -1}, make([]int64, 3), []int64{0, 1, 1}},
	}
	for _, tt := range tests {
		kernels.GeI64I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeU8F32 tests y = (x1 >= x2) ? 1 : 0 for float32 with uint8 output
func TestGeU8F32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float32
		y      []uint8
		want   []uint8
	}{
		{3, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float32{5}, []float32{10}, []uint8{0}, []uint8{0}},
		{4, []float32{-1, 2, 3, 4}, []float32{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 1, 1, 1}},
	}

	for _, tt := range tests {
		kernels.GeU8F32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeU8F64 tests y = (x1 >= x2) ? 1 : 0 for float64 with uint8 output
func TestGeU8F64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []float64
		y      []uint8
		want   []uint8
	}{
		{3, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		{0, nil, nil, nil, nil},
		{1, []float64{5}, []float64{10}, []uint8{0}, []uint8{0}},
		{4, []float64{-1, 2, 3, 4}, []float64{1, 2, 3, -4}, make([]uint8, 4), []uint8{0, 1, 1, 1}},
	}

	for _, tt := range tests {
		kernels.GeU8F64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGeU8U8(t *testing.T) {
	tests := []struct {
		numel     int
		x1, x2, y []uint8
		want      []uint8
	}{
		{3, []uint8{7, 246, 189}, []uint8{184, 153, 77}, make([]uint8, 3), []uint8{0, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []uint8{5}, []uint8{4}, make([]uint8, 1), []uint8{1}},
		{1, []uint8{5}, []uint8{5}, make([]uint8, 1), []uint8{1}},
		{1, []uint8{5}, []uint8{6}, make([]uint8, 1), []uint8{0}},
		{3, []uint8{0, 255, 128}, []uint8{255, 0, 128}, make([]uint8, 3), []uint8{0, 1, 1}},
	}
	for _, tt := range tests {
		kernels.GeU8U8(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestGeU8U32(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []uint32
		y      []uint8
		want   []uint8
	}{
		{3, []uint32{3381838898, 498283919, 415854238}, []uint32{2314506074, 3338958769, 1164971629}, make([]uint8, 3), []uint8{1, 0, 0}},
		{0, nil, nil, nil, nil},
		{1, []uint32{5}, []uint32{4}, make([]uint8, 1), []uint8{1}},
		{1, []uint32{5}, []uint32{5}, make([]uint8, 1), []uint8{1}},
		{1, []uint32{5}, []uint32{6}, make([]uint8, 1), []uint8{0}},
		{3, []uint32{0, 4294967295, 2147483648}, []uint32{4294967295, 0, 2147483648}, make([]uint8, 3), []uint8{0, 1, 1}},
	}
	for _, tt := range tests {
		kernels.GeU8U32(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestGeU8I64(t *testing.T) {
	tests := []struct {
		numel  int
		x1, x2 []int64
		y      []uint8
		want   []uint8
	}{
		{3, []int64{-4101946293123417127, 2201093810211456767, 3688546076359399250}, []int64{1302881648589687696, -1372162491173550838, 3082393301681482502}, make([]uint8, 3), []uint8{0, 1, 1}},
		{0, nil, nil, nil, nil},
		{1, []int64{5}, []int64{4}, make([]uint8, 1), []uint8{1}},
		{1, []int64{5}, []int64{5}, make([]uint8, 1), []uint8{1}},
		{1, []int64{5}, []int64{6}, make([]uint8, 1), []uint8{0}},
		{1, []int64{-5}, []int64{-6}, make([]uint8, 1), []uint8{1}},
		{1, []int64{-5}, []int64{-5}, make([]uint8, 1), []uint8{1}},
		{1, []int64{-5}, []int64{-4}, make([]uint8, 1), []uint8{0}},
		{4, []int64{-9223372036854775808, 9223372036854775807, 0, -1}, []int64{9223372036854775807, -9223372036854775808, 0, 1}, make([]uint8, 4), []uint8{0, 1, 1, 0}},
	}
	for _, tt := range tests {
		kernels.GeU8I64(tt.numel, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeStridedF32F32 tests y = (x1 >= x2) ? 1 : 0 for float32 with strided memory and float32 output
func TestGeStridedF32F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []float32
		want         []float32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]float32, 3), []float32{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]float32, 3), []float32{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]float32, 6), []float32{0, 1, 0, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]float32, 6), []float32{0, 1, 0, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 0, 1, 1, 0, 0, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]float32, 8), []float32{1, 0, 1, 1, 0, 0, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GeStridedF32F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeStridedF64F64 tests y = (x1 >= x2) ? 1 : 0 for float64 with strided memory and float64 output
func TestGeStridedF64F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []float64
		want         []float64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]float64, 3), []float64{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]float64, 3), []float64{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]float64, 6), []float64{0, 1, 0, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]float64, 6), []float64{0, 1, 0, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 0, 1, 1, 0, 0, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]float64, 8), []float64{1, 0, 1, 1, 0, 0, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GeStridedF64F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGeStridedU32U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint32
		want         []uint32
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{290349879, 1660869044, 1355315501}, []uint32{1159479630, 95752500, 516301619}, make([]uint32, 3), []uint32{0, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{581955479, 806146010, 1581310230}, []uint32{2036811792, 0, 1780108929, 0, 1709159975}, make([]uint32, 3), []uint32{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{678783860, 319831706, 1975992385, 1260654248, 548601600, 1777131075}, []uint32{642877838, 556063238, 324571958, 227533496, 1939096242, 1389617799}, make([]uint32, 6), []uint32{1, 0, 1, 1, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{2041150932, 1106616197, 1285826611, 153509182, 1342291766, 834649823}, []uint32{1451798735, 2076879066, 431838729, 864215405, 2028230498, 790727461}, make([]uint32, 6), []uint32{1, 1, 0, 0, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{475685035, 1358535746, 1634619920, 1972794853, 1465118510, 1849130508, 563386099, 49430996}, []uint32{1090064182, 1502264521, 2103643376, 1416595771, 1386283040, 2120849484, 1422038144, 203980748}, make([]uint32, 8), []uint32{0, 0, 0, 1, 1, 0, 0, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{1850762374, 1495575136, 357118775, 1510394691, 159939730, 417305550, 1159352856, 498954557}, []uint32{1780245493, 1210011423, 736076583, 350644296, 269146693, 1305607844, 1847024037, 250546693}, make([]uint32, 8), []uint32{1, 0, 1, 1, 0, 0, 0, 1}},
		{1, 1, []int{1}, []int{1}, []int{1}, []int{1}, []uint32{2059423207}, []uint32{1806855247}, make([]uint32, 1), []uint32{1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.GeStridedU32U32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestGeStridedI64I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []int64
		want         []int64
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{982321620743036456, 348494890284607743, -674573945899787158}, []int64{-371553049398526194, 368092188435773262, 711497124991148898}, make([]int64, 3), []int64{1, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{126805736102839286, 129362934629635077, 918405467898276506}, []int64{47168398422626764, 0, 713560230852928730, 0, -915998492452049098}, make([]int64, 3), []int64{1, 0, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{-479612519735594905, -715327962285267549, -877072407154974098, 191957911976076286, 552421920617625415, 250563516740364029}, []int64{165605611173858180, -782284495676159080, 992165222303981153, -230126443612105070, -590243243595693620, -795883582731660402}, make([]int64, 6), []int64{0, 1, 0, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{-1115750404015922088, -455376640429285568, -144155892797018502, 413831976192548301, 1083109020184373132, 1088118588429369379}, []int64{-307667570427489871, 922551753833722235, 364557189315593357, 1094142582013650615, 1071631137345225051, -846601827742976500}, make([]int64, 6), []int64{0, 0, 0, 0, 0, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{1034058892265641035, -606174257671265770, 217418143295032413, -947416899500889301, -631496802582056592, 870636980087233001, -467249916554504739, 644339135064296972}, []int64{162465433640526618, -617514653192990386, -532141638715257766, -236123375991250507, -1111466637718177433, -598819445591705759, 277684862634668370, 452444210915188910}, make([]int64, 8), []int64{1, 1, 1, 0, 1, 1, 0, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{194072402757071509, 4462297609349987, 1128802159778264913, 848746786302265805, -722574075528234284, 478380370574607244, -635558523678998064, -833683217245669931}, []int64{544413782465787560, -98909297745039408, -274632735308586152, -375681221336717667, -735851568039702778, -1087637896183136447, 266826565754179891, 1079691852565093592}, make([]int64, 8), []int64{0, 1, 1, 1, 1, 1, 1, 0}},
		{1, 1, []int{1}, []int{1}, []int{1}, []int{1}, []int64{696804396753317375}, []int64{-224115416841344833}, make([]int64, 1), []int64{1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.GeStridedI64I64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeStridedU8F32 tests y = (x1 >= x2) ? 1 : 0 for float32 with strided memory and uint8 output
func TestGeStridedU8F32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float32{1, 2, 3}, []float32{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float32{1, 2, 3}, []float32{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float32{1, 2, 3, 4, 5, 6}, []float32{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 1, 1, 0, 0, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float32{1, 3, 2, 4, 5, 7, 6, 8}, []float32{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 1, 1, 0, 0, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GeStridedU8F32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

// TestGeStridedU8F64 tests y = (x1 >= x2) ? 1 : 0 for float64 with strided memory and uint8 output
func TestGeStridedU8F64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []float64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []float64{1, 2, 3}, []float64{2, 2, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []float64{1, 2, 3}, []float64{2, 0, 2, 0, 4}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 2, 4, 4, 3, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []float64{1, 2, 3, 4, 5, 6}, []float64{2, 4, 2, 3, 4, 6}, make([]uint8, 6), []uint8{0, 1, 0, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 1, 1, 0, 0, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []float64{1, 3, 2, 4, 5, 7, 6, 8}, []float64{1, 3, 2, 4, 6, 7, 5, 8}, make([]uint8, 8), []uint8{1, 0, 1, 1, 0, 0, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}

	for _, tt := range tests {
		kernels.GeStridedU8F64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}
func TestGeStridedU8U8(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2, y    []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint8{102, 179, 92}, []uint8{14, 106, 71}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint8{68, 64, 255}, []uint8{49, 21, 58, 16, 51}, make([]uint8, 3), []uint8{1, 1, 1}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint8{20, 163, 241, 173, 59, 131}, []uint8{96, 84, 131, 239, 234, 120}, make([]uint8, 6), []uint8{0, 1, 1, 0, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint8{203, 158, 131, 124, 32, 95}, []uint8{189, 213, 163, 68, 121, 120}, make([]uint8, 6), []uint8{1, 0, 1, 0, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{189, 69, 40, 116, 186, 147, 146, 203}, []uint8{93, 60, 155, 208, 235, 130, 72, 32}, make([]uint8, 8), []uint8{1, 1, 0, 0, 0, 1, 1, 1}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint8{135, 134, 71, 8, 72, 179, 80, 23}, []uint8{59, 208, 7, 71, 173, 113, 55, 50}, make([]uint8, 8), []uint8{1, 0, 1, 0, 0, 0, 1, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.GeStridedU8U8(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGeStridedU8U32(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []uint32
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []uint32{3536365212, 628153869, 1806825558}, []uint32{3555566071, 2728220613, 2032962443}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []uint32{2915230493, 3634682277, 145968317}, []uint32{3325727220, 3110557567, 1945247576, 2098791564, 3747771194}, make([]uint8, 3), []uint8{0, 1, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []uint32{832937285, 10809103, 4258492129, 1014816056, 932136977, 1042937350}, []uint32{1908321527, 2139819252, 1635849229, 1476425686, 150444094, 953790822}, make([]uint8, 6), []uint8{0, 0, 1, 0, 1, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []uint32{2985300634, 4203824072, 1064958493, 4224067557, 3106928677, 2265948833}, []uint32{1338545520, 936173230, 2941254461, 1330651742, 1977267096, 1669939143}, make([]uint8, 6), []uint8{1, 1, 0, 1, 1, 1}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{412060132, 3572632768, 1773823211, 2574856846, 780612603, 1153845270, 4114990521, 413150519}, []uint32{2424549745, 3478346892, 3024892158, 3558411, 223549520, 96863974, 2992176729, 3631839519}, make([]uint8, 8), []uint8{0, 1, 0, 1, 1, 1, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []uint32{2628267734, 2825499030, 4167846520, 3016972984, 3462712017, 4024957433, 2090614216, 1937892312}, []uint32{2819594042, 1434852993, 2753364656, 3493985990, 1398851589, 1062288310, 121847438, 1086710888}, make([]uint8, 8), []uint8{0, 1, 1, 0, 1, 1, 1, 1}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.GeStridedU8U32(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

func TestGeStridedU8I64(t *testing.T) {
	tests := []struct {
		numel, ndims int
		dims         []int
		stridesX1    []int
		stridesX2    []int
		stridesY     []int
		x1, x2       []int64
		y            []uint8
		want         []uint8
	}{
		// 1D contiguous
		{3, 1, []int{3}, []int{1}, []int{1}, []int{1}, []int64{181681176, -142237815, -207450869}, []int64{898182325, 1659636068, -932757587}, make([]uint8, 3), []uint8{0, 0, 1}},
		// 1D x2 strided
		{3, 1, []int{3}, []int{1}, []int{2}, []int{1}, []int64{-60019596, -2070518569, -1839660768}, []int64{1382223104, 1061432667, 1226426054, 1049422972, 1037625619}, make([]uint8, 3), []uint8{0, 0, 0}},
		// 2D contiguous
		{6, 2, []int{2, 3}, []int{3, 1}, []int{3, 1}, []int{3, 1}, []int64{1167676390, 222530726, 1557570545, -751788237, 495793200, 1562373496}, []int64{933954678, 1868039048, 1425988624, 97242078, 1949140546, 281738004}, make([]uint8, 6), []uint8{1, 0, 1, 0, 0, 1}},
		// 2D x2 strided
		{6, 2, []int{2, 3}, []int{3, 1}, []int{1, 2}, []int{3, 1}, []int64{1585719413, 707199272, -532984505, -343758883, 302206310, 609867624}, []int64{-1643325343, 1949465720, -182514703, 1159166700, -2009804809, 619581338}, make([]uint8, 6), []uint8{1, 1, 1, 0, 0, 0}},
		// 3D contiguous
		{8, 3, []int{2, 2, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{-1308236996, -980279245, -1078540595, -743254182, 1738694963, 1810899917, 1932166439, -1367416579}, []int64{-2115063781, 770735590, -1543979873, 247708490, -776335662, -1882772990, -1175945696, -953871092}, make([]uint8, 8), []uint8{1, 0, 1, 0, 1, 1, 1, 0}},
		// 3D x1 strided (transposed dims 1 and 2)
		{8, 3, []int{2, 2, 2}, []int{4, 1, 2}, []int{4, 2, 1}, []int{4, 2, 1}, []int64{1079857973, 1938546117, 519732185, 1459454811, 1246958614, -1509965011, 1100092695, 1798780341}, []int64{1414155870, 63965668, -1348029129, -140372051, -1505228502, 128332826, 342034999, 2134716798}, make([]uint8, 8), []uint8{0, 1, 1, 1, 1, 1, 0, 0}},
		{0, 0, nil, nil, nil, nil, nil, nil, nil, nil},
	}
	for _, tt := range tests {
		kernels.GeStridedU8I64(tt.numel, tt.ndims, tt.dims, tt.stridesX1, tt.stridesX2, tt.stridesY, tt.x1, tt.x2, tt.y)
		if !slices.Equal(tt.y, tt.want) {
			t.Errorf("got %v, want %v", tt.y, tt.want)
		}
	}
}

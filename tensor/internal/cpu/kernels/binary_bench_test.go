package kernels_test

import (
	"fmt"
	"testing"

	"github.com/gocnn/candy/tensor/internal/cpu/kernels"
)

func BenchmarkBAddF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.BAddF32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBAddF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.BAddF64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBAddU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.BAddU8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBAddU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.BAddU32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBAddI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.BAddI64(numel, x1, x2, y)
			}
		})
	}
}
func BenchmarkBAddStridedF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.BAddStridedF32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBAddStridedF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.BAddStridedF64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBAddStridedU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.BAddStridedU8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBAddStridedU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.BAddStridedU32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBAddStridedI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.BAddStridedI64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}
func BenchmarkBSubF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i + 2)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.BSubF32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBSubF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i + 2)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.BSubF64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBSubU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8((i + 2) % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.BSubU8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBSubU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i + 2)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.BSubU32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBSubI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i + 2)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.BSubI64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBSubStridedF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i + 2)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.BSubStridedF32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBSubStridedF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i + 2)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.BSubStridedF64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBSubStridedU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8((i + 2) % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.BSubStridedU8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBSubStridedU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i + 2)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.BSubStridedU32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBSubStridedI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i + 2)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.BSubStridedI64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}
func BenchmarkBMulF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i) + 1
				x2[i] = float32(i) + 2
			}
			for b.Loop() {
				kernels.BMulF32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMulF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i) + 1
				x2[i] = float64(i) + 2
			}
			for b.Loop() {
				kernels.BMulF64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMulU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8((i + 1) % 256)
				x2[i] = uint8((i + 2) % 256)
			}
			for b.Loop() {
				kernels.BMulU8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMulU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i) + 1
				x2[i] = uint32(i) + 2
			}
			for b.Loop() {
				kernels.BMulU32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMulI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i) + 1
				x2[i] = int64(i) + 2
			}
			for b.Loop() {
				kernels.BMulI64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMulStridedF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i) + 1
				x2[i] = float32(i) + 2
			}
			for b.Loop() {
				kernels.BMulStridedF32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMulStridedF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i) + 1
				x2[i] = float64(i) + 2
			}
			for b.Loop() {
				kernels.BMulStridedF64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMulStridedU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8((i + 1) % 256)
				x2[i] = uint8((i + 2) % 256)
			}
			for b.Loop() {
				kernels.BMulStridedU8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMulStridedU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i) + 1
				x2[i] = uint32(i) + 2
			}
			for b.Loop() {
				kernels.BMulStridedU32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMulStridedI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i) + 1
				x2[i] = int64(i) + 2
			}
			for b.Loop() {
				kernels.BMulStridedI64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBDivF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i) + 1
				x2[i] = float32(i) + 2
			}
			for b.Loop() {
				kernels.BDivF32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBDivF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i) + 1
				x2[i] = float64(i) + 2
			}
			for b.Loop() {
				kernels.BDivF64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBDivU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8((i + 1) % 256)
				x2[i] = uint8((i + 2) % 256)
			}
			for b.Loop() {
				kernels.BDivU8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBDivU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i) + 1
				x2[i] = uint32(i) + 2
			}
			for b.Loop() {
				kernels.BDivU32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBDivI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i) + 1
				x2[i] = int64(i) + 2
			}
			for b.Loop() {
				kernels.BDivI64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBDivStridedF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i) + 1
				x2[i] = float32(i) + 2
			}
			for b.Loop() {
				kernels.BDivStridedF32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBDivStridedF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i) + 1
				x2[i] = float64(i) + 2
			}
			for b.Loop() {
				kernels.BDivStridedF64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBDivStridedU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8((i + 1) % 256)
				x2[i] = uint8((i + 2) % 256)
			}
			for b.Loop() {
				kernels.BDivStridedU8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBDivStridedU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i) + 1
				x2[i] = uint32(i) + 2
			}
			for b.Loop() {
				kernels.BDivStridedU32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBDivStridedI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i) + 1
				x2[i] = int64(i) + 2
			}
			for b.Loop() {
				kernels.BDivStridedI64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMaximumF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.BMaximumF32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMaximumF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.BMaximumF64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMaximumU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.BMaximumU8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMaximumU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.BMaximumU32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMaximumI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.BMaximumI64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMaximumStridedF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.BMaximumStridedF32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMaximumStridedF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.BMaximumStridedF64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMaximumStridedU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.BMaximumStridedU8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMaximumStridedU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.BMaximumStridedU32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMaximumStridedI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.BMaximumStridedI64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMinimumF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.BMinimumF32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMinimumF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.BMinimumF64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMinimumU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.BMinimumU8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMinimumU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.BMinimumU32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMinimumI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.BMinimumI64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMinimumStridedF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.BMinimumStridedF32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMinimumStridedF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.BMinimumStridedF64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMinimumStridedU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.BMinimumStridedU8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMinimumStridedU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.BMinimumStridedU32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkBMinimumStridedI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.BMinimumStridedI64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.EqF32F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.EqF64F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.EqU32U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.EqI64I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.EqU8F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.EqU8F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.EqU8U8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.EqU8U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.EqU8I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqStridedF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.EqStridedF32F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqStridedF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.EqStridedF64F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqStridedU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.EqStridedU32U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqStridedI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.EqStridedI64I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqStridedU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.EqStridedU8F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqStridedU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.EqStridedU8F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqStridedU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.EqStridedU8U8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqStridedU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.EqStridedU8U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkEqStridedU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.EqStridedU8I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.NeF32F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.NeF64F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.NeU32U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.NeI64I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.NeU8F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.NeU8F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.NeU8U8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.NeU8U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.NeU8I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeStridedF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.NeStridedF32F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeStridedF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.NeStridedF64F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeStridedU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.NeStridedU32U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeStridedI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.NeStridedI64I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeStridedU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.NeStridedU8F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeStridedU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.NeStridedU8F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeStridedU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.NeStridedU8U8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeStridedU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.NeStridedU8U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkNeStridedU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.NeStridedU8I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.LtF32F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.LtF64F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.LtU32U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.LtI64I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.LtU8F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.LtU8F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.LtU8U8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.LtU8U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.LtU8I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtStridedF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.LtStridedF32F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtStridedF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.LtStridedF64F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtStridedU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.LtStridedU32U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtStridedI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.LtStridedI64I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtStridedU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.LtStridedU8F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtStridedU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.LtStridedU8F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtStridedU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.LtStridedU8U8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtStridedU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.LtStridedU8U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLtStridedU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.LtStridedU8I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.LeF32F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.LeF64F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.LeU32U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.LeI64I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.LeU8F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.LeU8F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.LeU8U8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.LeU8U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.LeU8I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeStridedF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.LeStridedF32F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeStridedF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.LeStridedF64F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeStridedU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.LeStridedU32U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeStridedI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.LeStridedI64I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeStridedU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.LeStridedU8F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeStridedU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.LeStridedU8F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeStridedU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.LeStridedU8U8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeStridedU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.LeStridedU8U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkLeStridedU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.LeStridedU8I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.GtF32F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.GtF64F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.GtU32U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.GtI64I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.GtU8F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.GtU8F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.GtU8U8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.GtU8U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.GtU8I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtStridedF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.GtStridedF32F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtStridedF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.GtStridedF64F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtStridedU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.GtStridedU32U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtStridedI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.GtStridedI64I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtStridedU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.GtStridedU8F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtStridedU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.GtStridedU8F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtStridedU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.GtStridedU8U8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtStridedU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.GtStridedU8U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGtStridedU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.GtStridedU8I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.GeF32F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.GeF64F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.GeU32U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.GeI64I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float32, numel)
			x2 := make([]float32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.GeU8F32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]float64, numel)
			x2 := make([]float64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.GeU8F64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint8, numel)
			x2 := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.GeU8U8(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]uint32, numel)
			x2 := make([]uint32, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.GeU8U32(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			x1 := make([]int64, numel)
			x2 := make([]int64, numel)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.GeU8I64(numel, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeStridedF32F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.GeStridedF32F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeStridedF64F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.GeStridedF64F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeStridedU32U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.GeStridedU32U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeStridedI64I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.GeStridedI64I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeStridedU8F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float32, xLen)
			x2 := make([]float32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float32(i)
				x2[i] = float32(i + 1)
			}
			for b.Loop() {
				kernels.GeStridedU8F32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeStridedU8F64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]float64, xLen)
			x2 := make([]float64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = float64(i)
				x2[i] = float64(i + 1)
			}
			for b.Loop() {
				kernels.GeStridedU8F64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeStridedU8U8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint8, xLen)
			x2 := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint8(i % 256)
				x2[i] = uint8((i + 1) % 256)
			}
			for b.Loop() {
				kernels.GeStridedU8U8(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeStridedU8U32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]uint32, xLen)
			x2 := make([]uint32, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = uint32(i)
				x2[i] = uint32(i + 1)
			}
			for b.Loop() {
				kernels.GeStridedU8U32(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

func BenchmarkGeStridedU8I64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			ndims := 1
			dims := []int{numel}
			stridesX1 := []int{2}
			stridesX2 := []int{2}
			stridesY := []int{1}
			xLen := 2 * numel
			x1 := make([]int64, xLen)
			x2 := make([]int64, xLen)
			y := make([]uint8, numel)
			for i := range x1 {
				x1[i] = int64(i)
				x2[i] = int64(i + 1)
			}
			for b.Loop() {
				kernels.GeStridedU8I64(numel, ndims, dims, stridesX1, stridesX2, stridesY, x1, x2, y)
			}
		})
	}
}

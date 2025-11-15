package kernels_test

import (
	"fmt"
	"testing"

	"github.com/gocnn/candy/tensor/internal/cpu/kernels"
)

func BenchmarkAffineF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			a, bb := float32(2.0), float32(3.0)
			x := make([]float32, numel)
			y := make([]float32, numel)
			for i := range x {
				x[i] = float32(i)
			}
			for b.Loop() {
				kernels.AffineF32(numel, a, bb, x, y)
			}
		})
	}
}

func BenchmarkAffineF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			a, bb := float64(2.0), float64(3.0)
			x := make([]float64, numel)
			y := make([]float64, numel)
			for i := range x {
				x[i] = float64(i)
			}
			for b.Loop() {
				kernels.AffineF64(numel, a, bb, x, y)
			}
		})
	}
}

func BenchmarkAffineU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			a, bb := uint8(2), uint8(3)
			x := make([]uint8, numel)
			y := make([]uint8, numel)
			for i := range x {
				x[i] = uint8(i % 256)
			}
			for b.Loop() {
				kernels.AffineU8(numel, a, bb, x, y)
			}
		})
	}
}

func BenchmarkAffineU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			a, bb := uint32(2), uint32(3)
			x := make([]uint32, numel)
			y := make([]uint32, numel)
			for i := range x {
				x[i] = uint32(i)
			}
			for b.Loop() {
				kernels.AffineU32(numel, a, bb, x, y)
			}
		})
	}
}

func BenchmarkAffineI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			a, bb := int64(2), int64(3)
			x := make([]int64, numel)
			y := make([]int64, numel)
			for i := range x {
				x[i] = int64(i)
			}
			for b.Loop() {
				kernels.AffineI64(numel, a, bb, x, y)
			}
		})
	}
}

func BenchmarkAffineStridedF32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			a, bb := float32(2.0), float32(3.0)
			ndims := 1
			dims := []int{numel}
			strides := []int{2}
			xLen := 2 * numel
			x := make([]float32, xLen)
			y := make([]float32, numel)
			for i := range x {
				x[i] = float32(i)
			}
			for b.Loop() {
				kernels.AffineStridedF32(numel, ndims, dims, strides, a, bb, x, y)
			}
		})
	}
}

func BenchmarkAffineStridedF64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			a, bb := float64(2.0), float64(3.0)
			ndims := 1
			dims := []int{numel}
			strides := []int{2}
			xLen := 2 * numel
			x := make([]float64, xLen)
			y := make([]float64, numel)
			for i := range x {
				x[i] = float64(i)
			}
			for b.Loop() {
				kernels.AffineStridedF64(numel, ndims, dims, strides, a, bb, x, y)
			}
		})
	}
}

func BenchmarkAffineStridedU8(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			a, bb := uint8(2), uint8(3)
			ndims := 1
			dims := []int{numel}
			strides := []int{2}
			xLen := 2 * numel
			x := make([]uint8, xLen)
			y := make([]uint8, numel)
			for i := range x {
				x[i] = uint8(i % 256)
			}
			for b.Loop() {
				kernels.AffineStridedU8(numel, ndims, dims, strides, a, bb, x, y)
			}
		})
	}
}

func BenchmarkAffineStridedU32(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			a, bb := uint32(2), uint32(3)
			ndims := 1
			dims := []int{numel}
			strides := []int{2}
			xLen := 2 * numel
			x := make([]uint32, xLen)
			y := make([]uint32, numel)
			for i := range x {
				x[i] = uint32(i)
			}
			for b.Loop() {
				kernels.AffineStridedU32(numel, ndims, dims, strides, a, bb, x, y)
			}
		})
	}
}

func BenchmarkAffineStridedI64(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	for _, numel := range sizes {
		b.Run(fmt.Sprintf("numel=%d", numel), func(b *testing.B) {
			a, bb := int64(2), int64(3)
			ndims := 1
			dims := []int{numel}
			strides := []int{2}
			xLen := 2 * numel
			x := make([]int64, xLen)
			y := make([]int64, numel)
			for i := range x {
				x[i] = int64(i)
			}
			for b.Loop() {
				kernels.AffineStridedI64(numel, ndims, dims, strides, a, bb, x, y)
			}
		})
	}
}

package f32_test

import (
	"fmt"
	"testing"

	"github.com/qntx/goml/internal/mat/f32"
)

const (
	// A large enough slice length to accommodate all benchmark cases.
	benchLen = 1000000
	// A constant alpha scalar value used in Axpy operations.
	alpha = 2.0
)

var (
	// Global slices are used to avoid reallocation overhead in each benchmark.
	x = make([]float32, benchLen)
	y = make([]float32, benchLen)
	z = make([]float32, benchLen)

	// Global sinks to store function results, preventing the compiler from
	// optimizing away the calls.
	benchSink   float32
	benchSink64 float64
)

// init populates the global slices with initial data.
func init() {
	for i := range x {
		x[i] = float32(i)
		y[i] = float32(i)
	}
}

func axpyUnitaryGo(a float32, x, y []float32) {
	for i, v := range x {
		y[i] += a * v
	}
}

func axpyUnitaryToGo(dst []float32, a float32, x, y []float32) {
	for i, v := range x {
		dst[i] = y[i] + a*v
	}
}

func axpyIncGo(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	for i := 0; i < int(n); i++ {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
	}
}

func axpyIncToGo(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	for i := 0; i < int(n); i++ {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
	}
}

func dotUnitaryGo(x, y []float32) float32 {
	var sum float32
	for i, v := range x {
		sum += v * y[i]
	}
	return sum
}

func ddotUnitaryGo(x, y []float32) float64 {
	var sum float64
	for i, v := range x {
		sum += float64(v) * float64(y[i])
	}
	return sum
}

func dotIncGo(x, y []float32, n, incX, incY, ix, iy uintptr) float32 {
	var sum float32
	for i := 0; i < int(n); i++ {
		sum += x[ix] * y[iy]
		ix += incX
		iy += incY
	}
	return sum
}

func ddotIncGo(x, y []float32, n, incX, incY, ix, iy uintptr) float64 {
	var sum float64
	for i := 0; i < int(n); i++ {
		sum += float64(x[ix]) * float64(y[iy])
		ix += incX
		iy += incY
	}
	return sum
}

// Common vector lengths for unitary benchmarks.
var vectorLens = []int{1, 2, 3, 4, 5, 10, 100, 1000, 5000, 10000, 50000}

// Common (length, increment) pairs for non-unitary benchmarks.
var incCases = []struct{ n, inc int }{
	{1, 1},
	{2, 1}, {2, 2}, {2, 4}, {2, 10},
	{3, 1}, {3, 2}, {3, 4}, {3, 10},
	{4, 1}, {4, 2}, {4, 4}, {4, 10},
	{10, 1}, {10, 2}, {10, 4}, {10, 10},
	{1000, 1}, {1000, 2}, {1000, 4}, {1000, 10},
	{100000, 1}, {100000, 2}, {100000, 4}, {100000, 10},
	{100000, -1}, {100000, -2}, {100000, -4}, {100000, -10},
}

// getIncStartIdx calculates the starting index for slices with negative increments.
func getIncStartIdx(n, inc int) int {
	if inc < 0 {
		return (-n + 1) * inc
	}
	return 0
}

func BenchmarkAxpyUnitary(b *testing.B) {
	b.Run("Asm", func(b *testing.B) {
		for _, n := range vectorLens {
			b.Run(fmt.Sprintf("len=%d", n), func(b *testing.B) {
				xs, ys := x[:n], y[:n]
				b.ResetTimer()
				for b.Loop() {
					f32.AxpyUnitary(alpha, xs, ys)
				}
			})
		}
	})
	b.Run("Go", func(b *testing.B) {
		for _, n := range vectorLens {
			b.Run(fmt.Sprintf("len=%d", n), func(b *testing.B) {
				xs, ys := x[:n], y[:n]
				b.ResetTimer()
				for b.Loop() {
					axpyUnitaryGo(alpha, xs, ys)
				}
			})
		}
	})
}

func BenchmarkAxpyUnitaryTo(b *testing.B) {
	b.Run("Asm", func(b *testing.B) {
		for _, n := range vectorLens {
			b.Run(fmt.Sprintf("len=%d", n), func(b *testing.B) {
				xs, ys, zs := x[:n], y[:n], z[:n]
				b.ResetTimer()
				for b.Loop() {
					f32.AxpyUnitaryTo(zs, alpha, xs, ys)
				}
			})
		}
	})
	b.Run("Go", func(b *testing.B) {
		for _, n := range vectorLens {
			b.Run(fmt.Sprintf("len=%d", n), func(b *testing.B) {
				xs, ys, zs := x[:n], y[:n], z[:n]
				b.ResetTimer()
				for b.Loop() {
					axpyUnitaryToGo(zs, alpha, xs, ys)
				}
			})
		}
	})
}

func BenchmarkAxpyInc(b *testing.B) {
	run := func(b *testing.B, f func(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr)) {
		for _, tc := range incCases {
			b.Run(fmt.Sprintf("len=%d/inc=%d", tc.n, tc.inc), func(b *testing.B) {
				idx := getIncStartIdx(tc.n, tc.inc)
				n, inc, ix, iy := uintptr(tc.n), uintptr(tc.inc), uintptr(idx), uintptr(idx)
				b.ResetTimer()
				for b.Loop() {
					f(alpha, x, y, n, inc, inc, ix, iy)
				}
			})
		}
	}
	b.Run("Asm", func(b *testing.B) { run(b, f32.AxpyInc) })
	b.Run("Go", func(b *testing.B) { run(b, axpyIncGo) })
}

func BenchmarkAxpyIncTo(b *testing.B) {
	run := func(b *testing.B, f func(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr)) {
		for _, tc := range incCases {
			b.Run(fmt.Sprintf("len=%d/inc=%d", tc.n, tc.inc), func(b *testing.B) {
				idx := getIncStartIdx(tc.n, tc.inc)
				n, inc, idst := uintptr(tc.n), uintptr(tc.inc), uintptr(idx)
				ix, iy := uintptr(idx), uintptr(idx)
				b.ResetTimer()
				for b.Loop() {
					f(z, inc, idst, alpha, x, y, n, inc, inc, ix, iy)
				}
			})
		}
	}
	b.Run("Asm", func(b *testing.B) { run(b, f32.AxpyIncTo) })
	b.Run("Go", func(b *testing.B) { run(b, axpyIncToGo) })
}

func BenchmarkDotUnitary(b *testing.B) {
	run := func(b *testing.B, f func(x, y []float32) float32) {
		for _, n := range vectorLens {
			b.Run(fmt.Sprintf("len=%d", n), func(b *testing.B) {
				xs, ys := x[:n], y[:n]
				b.SetBytes(int64(n * 8)) // 2 slices of float32
				b.ResetTimer()
				for b.Loop() {
					benchSink = f(xs, ys)
				}
			})
		}
	}
	b.Run("Asm", func(b *testing.B) { run(b, f32.DotUnitary) })
	b.Run("Go", func(b *testing.B) { run(b, dotUnitaryGo) })
}

func BenchmarkDdotUnitary(b *testing.B) {
	run := func(b *testing.B, f func(x, y []float32) float64) {
		for _, n := range vectorLens {
			b.Run(fmt.Sprintf("len=%d", n), func(b *testing.B) {
				xs, ys := x[:n], y[:n]
				b.SetBytes(int64(n * 8)) // 2 slices of float32
				b.ResetTimer()
				for b.Loop() {
					benchSink64 = f(xs, ys)
				}
			})
		}
	}
	b.Run("Asm", func(b *testing.B) { run(b, f32.DdotUnitary) })
	b.Run("Go", func(b *testing.B) { run(b, ddotUnitaryGo) })
}

func BenchmarkDotInc(b *testing.B) {
	run := func(b *testing.B, f func(x, y []float32, n, incX, incY, ix, iy uintptr) float32) {
		for _, tc := range incCases {
			b.Run(fmt.Sprintf("len=%d/inc=%d", tc.n, tc.inc), func(b *testing.B) {
				idx := getIncStartIdx(tc.n, tc.inc)
				n, inc, ix, iy := uintptr(tc.n), uintptr(tc.inc), uintptr(idx), uintptr(idx)
				b.SetBytes(int64(tc.n * 8))
				b.ResetTimer()
				for b.Loop() {
					benchSink = f(x, y, n, inc, inc, ix, iy)
				}
			})
		}
	}
	b.Run("Asm", func(b *testing.B) { run(b, f32.DotInc) })
	b.Run("Go", func(b *testing.B) { run(b, dotIncGo) })
}

func BenchmarkDdotInc(b *testing.B) {
	run := func(b *testing.B, f func(x, y []float32, n, incX, incY, ix, iy uintptr) float64) {
		for _, tc := range incCases {
			b.Run(fmt.Sprintf("len=%d/inc=%d", tc.n, tc.inc), func(b *testing.B) {
				idx := getIncStartIdx(tc.n, tc.inc)
				n, inc, ix, iy := uintptr(tc.n), uintptr(tc.inc), uintptr(idx), uintptr(idx)
				b.SetBytes(int64(tc.n * 8))
				b.ResetTimer()
				for b.Loop() {
					benchSink64 = f(x, y, n, inc, inc, ix, iy)
				}
			})
		}
	}
	b.Run("Asm", func(b *testing.B) { run(b, f32.DdotInc) })
	b.Run("Go", func(b *testing.B) { run(b, ddotIncGo) })
}

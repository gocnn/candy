package mat_test

import (
	"math/rand/v2"
	"testing"

	"github.com/qntx/spark/internal/mat"
)

const size = 1 << 8 // 256

type matrix2D [][]float64

func matmul2D(a, b [][]float64) [][]float64 {
	n, m, p := len(a), len(b), len(b[0])

	out := make([][]float64, n)
	for i := range out {
		out[i] = make([]float64, p)
	}

	for i := range n {
		for k := range m {
			aik := a[i][k]
			for j := range p {
				out[i][j] += aik * b[k][j]
			}
		}
	}

	return out
}

func genMatrix2D(rows, cols int) matrix2D {
	out := make(matrix2D, rows)
	for i := range out {
		out[i] = make([]float64, cols)
	}

	for i := range rows {
		for j := range cols {
			out[i][j] = rand.Float64()
		}
	}

	return out
}

func BenchmarkMatMul2D(b *testing.B) {
	m := genMatrix2D(size, size)
	n := genMatrix2D(size, size)

	b.ResetTimer()
	for b.Loop() {
		_ = matmul2D(m, n)
	}
}

func BenchmarkMatMul1D(b *testing.B) {
	m := mat.Rand(size, size)
	n := mat.Rand(size, size)

	b.ResetTimer()
	for b.Loop() {
		_ = mat.MatMul(m, n)
	}
}

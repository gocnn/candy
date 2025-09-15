//go:build !noasm && !gccgo && !safe

package f32

// Ger performs the rank-one operation
//
//	A += alpha * x * yᵀ
//
// where A is an m×n Tensor matrix, x and y are vectors, and alpha is a scalar.
func Ger(m, n uintptr, alpha float32,
	x []float32, incX uintptr,
	y []float32, incY uintptr,
	a []float32, lda uintptr)

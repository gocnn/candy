//go:build !noasm && !gccgo && !safe

package f64

// Ger performs the rank-one operation
//
//	A += alpha * x * yᵀ
//
// where A is an m×n Tensor matrix, x and y are vectors, and alpha is a scalar.
func Ger(m, n uintptr, alpha float64, x []float64, incX uintptr, y []float64, incY uintptr, a []float64, lda uintptr)

//go:build !noasm && !gccgo && !safe

package f64

// GemvN is
//
//	for i := 0; i < int(m); i++ {
//		var sum float64
//		for j := 0; j < int(n); j++ {
//			sum += a[i*int(lda)+j] * x[j*int(incX)]
//		}
//		y[i*int(incY)] += alpha * sum
//	}
func GemvN(m, n uintptr, alpha float64,
	x []float64, incX uintptr,
	y []float64, incY uintptr,
	a []float64, lda uintptr)

// GemvT is
//
//	for i := 0; i < int(m); i++ {
//		var sum float64
//		for j := 0; j < int(n); j++ {
//			sum += a[j*int(lda)+i] * x[j*int(incX)]
//		}
//		y[i*int(incY)] += alpha * sum
//	}
func GemvT(m, n uintptr, alpha float64,
	x []float64, incX uintptr,
	y []float64, incY uintptr,
	a []float64, lda uintptr)

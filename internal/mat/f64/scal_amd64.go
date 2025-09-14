//go:build !noasm && !gccgo && !safe

package f64

// ScalUnitary is
//
//	for i := range x {
//		x[i] *= alpha
//	}
func ScalUnitary(alpha float64, x []float64)

// ScalUnitaryTo is
//
//	for i, v := range x {
//		dst[i] = alpha * v
//	}
func ScalUnitaryTo(dst []float64, alpha float64, x []float64)

// ScalInc is
//
//	for i := 0; i < int(n); i++ {
//		x[ix] *= alpha
//		ix += incX
//	}
func ScalInc(alpha float64, x []float64, n, incX, ix uintptr)

// ScalIncTo is
//
//	for i := 0; i < int(n); i++ {
//		dst[idst] = alpha * x[ix]
//		ix += incX
//		idst += incDst
//	}
func ScalIncTo(dst []float64, incDst, idst uintptr, alpha float64, x []float64, n, incX, ix uintptr)

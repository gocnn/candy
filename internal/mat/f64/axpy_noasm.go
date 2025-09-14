//go:build !amd64 || noasm || gccgo || safe

package f64

// AxpyUnitary is
//
//	for i, v := range x {
//		y[i] += alpha * v
//	}
func AxpyUnitary(alpha float64, x, y []float64) {
	for i, v := range x {
		y[i] += alpha * v
	}
}

// AxpyUnitaryTo is
//
//	for i, v := range x {
//		dst[i] = alpha*v + y[i]
//	}
func AxpyUnitaryTo(dst []float64, alpha float64, x, y []float64) {
	for i, v := range x {
		dst[i] = alpha*v + y[i]
	}
}

// AxpyInc is
//
//	for i := 0; i < int(n); i++ {
//		y[iy] += alpha * x[ix]
//		ix += incX
//		iy += incY
//	}
func AxpyInc(alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr) {
	for i := 0; i < int(n); i++ {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
	}
}

// AxpyIncTo is
//
//	for i := 0; i < int(n); i++ {
//		dst[idst] = alpha*x[ix] + y[iy]
//		ix += incX
//		iy += incY
//		idst += incDst
//	}
func AxpyIncTo(dst []float64, incDst, idst uintptr, alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr) {
	for i := 0; i < int(n); i++ {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
	}
}

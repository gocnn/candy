//go:build !amd64 || noasm || gccgo || safe

package f32

// DotUnitary is
//
//	for i, v := range x {
//		sum += y[i] * v
//	}
//	return sum
func DotUnitary(x, y []float32) (sum float32) {
	for i, v := range x {
		sum += y[i] * v
	}
	return sum
}

// DotInc is
//
//	for i := 0; i < int(n); i++ {
//		sum += y[iy] * x[ix]
//		ix += incX
//		iy += incY
//	}
//	return sum
func DotInc(x, y []float32, n, incX, incY, ix, iy uintptr) (sum float32) {
	for i := 0; i < int(n); i++ {
		sum += y[iy] * x[ix]
		ix += incX
		iy += incY
	}
	return sum
}

// DdotUnitary is
//
//	for i, v := range x {
//		sum += float64(y[i]) * float64(v)
//	}
//	return
func DdotUnitary(x, y []float32) (sum float64) {
	for i, v := range x {
		sum += float64(y[i]) * float64(v)
	}
	return
}

// DdotInc is
//
//	for i := 0; i < int(n); i++ {
//		sum += float64(y[iy]) * float64(x[ix])
//		ix += incX
//		iy += incY
//	}
//	return
func DdotInc(x, y []float32, n, incX, incY, ix, iy uintptr) (sum float64) {
	for i := 0; i < int(n); i++ {
		sum += float64(y[iy]) * float64(x[ix])
		ix += incX
		iy += incY
	}
	return
}

//go:build !amd64 || noasm || gccgo || safe

package f64

// GemvN computes
//
//	y = alpha * A * x + beta * y
//
// where A is an m×n Tensor matrix, x and y are vectors, and alpha and beta are scalars.
func GemvN(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	var kx, ky, i uintptr
	if int(incX) < 0 {
		kx = uintptr(-int(n-1) * int(incX))
	}
	if int(incY) < 0 {
		ky = uintptr(-int(m-1) * int(incY))
	}

	if incX == 1 && incY == 1 {
		if beta == 0 {
			for i = 0; i < m; i++ {
				y[i] = alpha * DotUnitary(a[lda*i:lda*i+n], x)
			}
			return
		}
		for i = 0; i < m; i++ {
			y[i] = y[i]*beta + alpha*DotUnitary(a[lda*i:lda*i+n], x)
		}
		return
	}
	iy := ky
	if beta == 0 {
		for i = 0; i < m; i++ {
			y[iy] = alpha * DotInc(x, a[lda*i:lda*i+n], n, incX, 1, kx, 0)
			iy += incY
		}
		return
	}
	for i = 0; i < m; i++ {
		y[iy] = y[iy]*beta + alpha*DotInc(x, a[lda*i:lda*i+n], n, incX, 1, kx, 0)
		iy += incY
	}
}

// GemvT computes
//
//	y = alpha * Aᵀ * x + beta * y
//
// where A is an m×n Tensor matrix, x and y are vectors, and alpha and beta are scalars.
func GemvT(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	var kx, ky, i uintptr
	if int(incX) < 0 {
		kx = uintptr(-int(m-1) * int(incX))
	}
	if int(incY) < 0 {
		ky = uintptr(-int(n-1) * int(incY))
	}
	switch {
	case beta == 0: // beta == 0 is special-cased to memclear
		if incY == 1 {
			for i := range y {
				y[i] = 0
			}
		} else {
			iy := ky
			for i := 0; i < int(n); i++ {
				y[iy] = 0
				iy += incY
			}
		}
	case int(incY) < 0:
		ScalInc(beta, y, n, uintptr(int(-incY)))
	case incY == 1:
		ScalUnitary(beta, y[:n])
	default:
		ScalInc(beta, y, n, incY)
	}

	if incX == 1 && incY == 1 {
		for i = 0; i < m; i++ {
			AxpyUnitaryTo(y, alpha*x[i], a[lda*i:lda*i+n], y)
		}
		return
	}
	ix := kx
	for i = 0; i < m; i++ {
		AxpyInc(alpha*x[ix], a[lda*i:lda*i+n], y, n, 1, incY, 0, ky)
		ix += incX
	}
}

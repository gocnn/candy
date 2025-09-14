//go:build !amd64 || noasm || gccgo || safe

package f64

import "math"

// L1Dist is
//
//	var norm float64
//	for i, v := range s {
//		norm += math.Abs(t[i] - v)
//	}
//	return norm
func L1Dist(s, t []float64) float64 {
	var norm float64
	for i, v := range s {
		norm += math.Abs(t[i] - v)
	}
	return norm
}

// L1Norm is
//
//	for _, v := range x {
//		sum += math.Abs(v)
//	}
//	return sum
func L1Norm(x []float64) (sum float64) {
	for _, v := range x {
		sum += math.Abs(v)
	}
	return sum
}

// L1NormInc is
//
//	for i := 0; i < n*incX; i += incX {
//		sum += math.Abs(x[i])
//	}
//	return sum
func L1NormInc(x []float64, n, incX int) (sum float64) {
	for i := 0; i < n*incX; i += incX {
		sum += math.Abs(x[i])
	}
	return sum
}

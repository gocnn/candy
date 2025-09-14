//go:build !noasm && !gccgo && !safe

package f64

// CumSum is
//
//	if len(s) == 0 {
//		return dst
//	}
//	dst[0] = s[0]
//	for i, v := range s[1:] {
//		dst[i+1] = dst[i] + v
//	}
//	return dst
func CumSum(dst, s []float64) []float64

// CumProd is
//
//	if len(s) == 0 {
//		return dst
//	}
//	dst[0] = s[0]
//	for i, v := range s[1:] {
//		dst[i+1] = dst[i] * v
//	}
//	return dst
func CumProd(dst, s []float64) []float64

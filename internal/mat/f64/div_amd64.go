//go:build !noasm && !gccgo && !safe

package f64

// Div is
//
//	for i, v := range s {
//		dst[i] /= v
//	}
func Div(dst, s []float64)

// DivTo is
//
//	for i, v := range s {
//		dst[i] = v / t[i]
//	}
//	return dst
func DivTo(dst, x, y []float64) []float64

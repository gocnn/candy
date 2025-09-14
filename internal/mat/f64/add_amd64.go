//go:build !noasm && !gccgo && !safe

package f64

// Add is
//
//	for i, v := range s {
//		dst[i] += v
//	}
func Add(dst, s []float64)

// AddConst is
//
//	for i := range x {
//		x[i] += alpha
//	}
func AddConst(alpha float64, x []float64)

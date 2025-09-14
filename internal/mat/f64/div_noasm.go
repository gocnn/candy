//go:build !amd64 || noasm || gccgo || safe

package f64

// Div is
//
//	for i, v := range s {
//		dst[i] /= v
//	}
func Div(dst, s []float64) {
	for i, v := range s {
		dst[i] /= v
	}
}

// DivTo is
//
//	for i, v := range s {
//		dst[i] = v / t[i]
//	}
//	return dst
func DivTo(dst, s, t []float64) []float64 {
	for i, v := range s {
		dst[i] = v / t[i]
	}
	return dst
}

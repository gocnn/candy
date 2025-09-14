//go:build !noasm && !gccgo && !safe

package f64

// LinfDist is
//
//	var norm float64
//	if len(s) == 0 {
//		return 0
//	}
//	norm = math.Abs(t[0] - s[0])
//	for i, v := range s[1:] {
//		absDiff := math.Abs(t[i+1] - v)
//		if absDiff > norm || math.IsNaN(norm) {
//			norm = absDiff
//		}
//	}
//	return norm
func LinfDist(s, t []float64) float64

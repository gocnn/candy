//go:build !noasm && !gccgo && !safe

package f64

// Sum is
//
//	var sum float64
//	for i := range x {
//	    sum += x[i]
//	}
func Sum(x []float64) float64

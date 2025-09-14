//go:build !amd64 || noasm || gccgo || safe

package f64

// Sum is
//
//	var sum float64
//	for i := range x {
//	    sum += x[i]
//	}
func Sum(x []float64) float64 {
	var sum float64
	for _, v := range x {
		sum += v
	}
	return sum
}

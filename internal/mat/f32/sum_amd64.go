//go:build !noasm && !gccgo && !safe

package f32

// Sum is
//
//	 var sum float32
//	 for _, v := range x {
//			sum += v
//	 }
//	 return sum
func Sum(x []float32) float32

//go:build !amd64 || !gc || noasm

package mat

// Add32 adds x1 and x2 element-wise, storing the result in y (32 bits).
func Add32(x1, x2, y []float32) {
	add(x1, x2, y)
}

// Add64 adds x1 and x2 element-wise, storing the result in y (64 bits).
func Add64(x1, x2, y []float64) {
	add(x1, x2, y)
}

func add[F float32 | float64](x1, x2, y []F) {
	if len(x1) == 0 {
		return
	}
	_ = y[len(x1)-1]
	_ = x2[len(x1)-1]
	for i, x1v := range x1 {
		y[i] = x1v + x2[i]
	}
}

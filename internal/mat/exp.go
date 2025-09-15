package mat

import "math"

func exp[F float32 | float64](x, y []F) {
	if len(x) == 0 {
		return
	}
	_ = y[len(x)-1]
	for i, xv := range x {
		y[i] = F(math.Exp(float64(xv)))
	}
}

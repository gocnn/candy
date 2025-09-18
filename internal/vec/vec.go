package vec

import "math"

func ZeroLike(v []float64) []float64 {
	return make([]float64, len(v))
}

func OneLike(v []float64) []float64 {
	return AddC(1.0, ZeroLike(v))
}

func Shape(v []float64) []int {
	return []int{1, len(v)}
}

func Int(v []float64) []int {
	out := make([]int, len(v))
	for i := range v {
		out[i] = int(v[i])
	}

	return out
}

func AddC(c float64, v []float64) []float64 {
	return F(v, func(a float64) float64 { return c + a })
}

// SubC returns c - v
func SubC(c float64, v []float64) []float64 {
	return F(v, func(a float64) float64 { return c - a })
}

func MulC(c float64, v []float64) []float64 {
	return F(v, func(a float64) float64 { return c * a })
}

func Exp(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Exp(a) })
}

func Log(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Log(a) })
}

func Sin(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Sin(a) })
}

func Cos(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Cos(a) })
}

func Tanh(v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Tanh(a) })
}

func Pow(c float64, v []float64) []float64 {
	return F(v, func(a float64) float64 { return math.Pow(a, c) })
}

func Add(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a + b })
}

// Sub returns v - w
func Sub(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a - b })
}

func Mul(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a * b })
}

// Div returns v / w
func Div(v, w []float64) []float64 {
	return F2(v, w, func(a, b float64) float64 { return a / b })
}

func BroadcastTo(shape []int, v []float64) []float64 {
	// NOTE: v is vector, shape[0] has no effect. shape[0] is always 1.
	out, _ := Broadcast(v, make([]float64, shape[1]))
	return out
}

func Broadcast(v, w []float64) ([]float64, []float64) {
	if len(v) == 1 {
		out := ZeroLike(w)
		for i := range w {
			out[i] = v[0]
		}

		return out, w
	}

	if len(w) == 1 {
		out := ZeroLike(v)
		for i := range v {
			out[i] = w[0]
		}

		return v, out
	}

	return v, w
}

func SumTo(shape []int, v []float64) float64 {
	// NOTE: v is vector, shape has no effect.
	return Sum(v)
}

func Sum(v []float64) float64 {
	var sum float64
	for i := range v {
		sum += v[i]
	}

	return sum
}

func Transpose(v []float64) [][]float64 {
	out := make([][]float64, len(v))
	for i := range v {
		out[i] = []float64{v[i]}
	}

	return out
}

func F(v []float64, f func(a float64) float64) []float64 {
	out := ZeroLike(v)
	for i := range v {
		out[i] = f(v[i])
	}

	return out
}

func F2(v, w []float64, f func(a, b float64) float64) []float64 {
	x, y := Broadcast(v, w)

	out := ZeroLike(x)
	for i := range x {
		out[i] = f(x[i], y[i])
	}

	return out
}

func Equal(v, w []int) bool {
	if len(v) != len(w) {
		return false
	}

	for i := range v {
		if v[i] != w[i] {
			return false
		}
	}

	return true
}

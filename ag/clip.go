package ag

import "github.com/qntx/spark/internal/mat"

func Clip(min, max float64) func(x ...*Var) *Var {
	return (&Operator{
		Op: &ClipT{
			Min: min,
			Max: max,
		},
	}).First
}

type ClipT struct {
	Min, Max float64
	x        *Var
}

func (f *ClipT) Forward(x ...*Var) []*Var {
	f.x = x[0]

	y := mat.Clip(x[0].Data, f.Min, f.Max)
	return []*Var{
		NewFrom(y),
	}
}

func (f *ClipT) Backward(gy ...*Var) []*Var {
	mask := mat.Mask(f.x.Data, clip(f.Min, f.Max))
	return []*Var{
		Mul(gy[0], NewFrom(mask)), // gy * mask
	}
}

func clip(min, max float64) func(v float64) bool {
	return func(v float64) bool {
		return min <= v && v <= max
	}
}

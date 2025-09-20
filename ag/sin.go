package ag

import "github.com/qntx/spark/internal/mat"

func Sin(x ...*Var) *Var {
	return (&Operator{
		Op: &SinT{},
	}).First(x...)
}

type SinT struct {
	x *Var
}

func (f *SinT) Forward(x ...*Var) []*Var {
	f.x = x[0]

	y := mat.Sin(x[0].Data)
	return []*Var{
		NewFrom(y),
	}
}

func (f *SinT) Backward(gy ...*Var) []*Var {
	return []*Var{
		Mul(Cos(f.x), gy[0]), // cos(x) * gy
	}
}

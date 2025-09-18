package ag

import "github.com/qntx/spark/internal/mat"

func Sin(x ...*Variable) *Variable {
	return (&Operator{
		Op: &SinT{},
	}).First(x...)
}

type SinT struct {
	x *Variable
}

func (f *SinT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := mat.Sin(x[0].Data)
	return []*Variable{
		NewFrom(y),
	}
}

func (f *SinT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(Cos(f.x), gy[0]), // cos(x) * gy
	}
}

package tensor

import "github.com/qntx/spark/internal/mat"

func Cos(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &CosT{},
	}).First(x...)
}

type CosT struct {
	x *Variable
}

func (f *CosT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := mat.Cos(x[0].Data)
	return []*Variable{
		NewFrom(y),
	}
}

func (f *CosT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(Neg(Sin(f.x)), gy[0]), // -1.0 * sin(x) * gy
	}
}

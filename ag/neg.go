package ag

import "github.com/qntx/spark/internal/mat"

func Neg(x ...*Variable) *Variable {
	return (&Operator{
		Op: &NegT{},
	}).First(x...)
}

type NegT struct{}

func (f *NegT) Forward(x ...*Variable) []*Variable {
	y := mat.MulC(-1.0, x[0].Data)
	return []*Variable{
		NewFrom(y),
	}
}

func (f *NegT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Neg(gy[0]),
	}
}

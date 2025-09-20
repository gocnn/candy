package ag

import "github.com/qntx/spark/internal/mat"

func Neg(x ...*Var) *Var {
	return (&Operator{
		Op: &NegT{},
	}).First(x...)
}

type NegT struct{}

func (f *NegT) Forward(x ...*Var) []*Var {
	y := mat.MulC(-1.0, x[0].Data)
	return []*Var{
		NewFrom(y),
	}
}

func (f *NegT) Backward(gy ...*Var) []*Var {
	return []*Var{
		Neg(gy[0]),
	}
}

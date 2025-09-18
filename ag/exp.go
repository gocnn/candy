package ag

import "github.com/qntx/spark/internal/mat"

func Exp(x ...*Variable) *Variable {
	return (&Operator{
		Op: &ExpT{},
	}).First(x...)
}

type ExpT struct {
	y *Variable
}

func (f *ExpT) Forward(x ...*Variable) []*Variable {
	y := mat.Exp(x[0].Data)
	f.y = NewFrom(y)

	return []*Variable{
		f.y,
	}
}

func (f *ExpT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Mul(gy[0], f.y), // gy * y
	}
}

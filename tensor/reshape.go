package tensor

import "github.com/qntx/spark/internal/mat"

func Reshape(shape ...int) func(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &ReshapeT{
			Shape: shape,
		},
	}).First
}

type ReshapeT struct {
	Shape, xShape []int
}

func (f *ReshapeT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	y := mat.Reshape(f.Shape, x[0].Data)
	return []*Variable{
		NewFrom(y),
	}
}

func (f *ReshapeT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Reshape(f.xShape...)(gy[0]),
	}
}

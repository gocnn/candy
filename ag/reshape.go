package ag

import "github.com/qntx/spark/internal/mat"

func Reshape(shape ...int) func(x ...*Var) *Var {
	return (&Operator{
		Op: &ReshapeT{
			Shape: shape,
		},
	}).First
}

type ReshapeT struct {
	Shape, xShape []int
}

func (f *ReshapeT) Forward(x ...*Var) []*Var {
	f.xShape = x[0].Shape()

	y := mat.Reshape(f.Shape, x[0].Data)
	return []*Var{
		NewFrom(y),
	}
}

func (f *ReshapeT) Backward(gy ...*Var) []*Var {
	return []*Var{
		Reshape(f.xShape...)(gy[0]),
	}
}

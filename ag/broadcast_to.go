package ag

import "github.com/qntx/spark/internal/mat"

func BroadcastTo(shape ...int) func(x ...*Var) *Var {
	return (&Operator{
		Op: &BroadcastToT{
			Shape: shape,
		},
	}).First
}

type BroadcastToT struct {
	Shape, xShape []int
}

func (f *BroadcastToT) Forward(x ...*Var) []*Var {
	f.xShape = x[0].Shape()

	y := mat.BroadcastTo(f.Shape, x[0].Data)
	return []*Var{
		NewFrom(y),
	}
}

func (f *BroadcastToT) Backward(gy ...*Var) []*Var {
	return []*Var{
		SumTo(f.xShape...)(gy[0]),
	}
}

package ad

import "github.com/qntx/spark/internal/mat"

func BroadcastTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Operator{
		Op: &BroadcastToT{
			Shape: shape,
		},
	}).First
}

type BroadcastToT struct {
	Shape, xShape []int
}

func (f *BroadcastToT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	y := mat.BroadcastTo(f.Shape, x[0].Data)
	return []*Variable{
		NewFrom(y),
	}
}

func (f *BroadcastToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		SumTo(f.xShape...)(gy[0]),
	}
}

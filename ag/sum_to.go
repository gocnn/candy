package ag

import "github.com/qntx/spark/internal/mat"

func SumTo(shape ...int) func(x ...*Variable) *Variable {
	return (&Operator{
		Op: &SumToT{
			Shape: shape,
		},
	}).First
}

type SumToT struct {
	Shape, xShape []int
}

func (f *SumToT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	y := mat.SumTo(f.Shape, x[0].Data)
	return []*Variable{
		NewFrom(y),
	}
}

func (f *SumToT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		BroadcastTo(f.xShape...)(gy[0]),
	}
}

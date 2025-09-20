package ag

import "github.com/qntx/spark/internal/mat"

func SumTo(shape ...int) func(x ...*Var) *Var {
	return (&Operator{
		Op: &SumToT{
			Shape: shape,
		},
	}).First
}

type SumToT struct {
	Shape, xShape []int
}

func (f *SumToT) Forward(x ...*Var) []*Var {
	f.xShape = x[0].Shape()

	y := mat.SumTo(f.Shape, x[0].Data)
	return []*Var{
		NewFrom(y),
	}
}

func (f *SumToT) Backward(gy ...*Var) []*Var {
	return []*Var{
		BroadcastTo(f.xShape...)(gy[0]),
	}
}

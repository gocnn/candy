package ad

import "github.com/qntx/spark/internal/mat"

func Sum(x ...*Variable) *Variable {
	return (&Operator{
		Op: &SumT{},
	}).First(x...)
}

type SumT struct {
	xShape []int
}

func (f *SumT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	y := mat.Sum(x[0].Data)
	return []*Variable{
		New(y),
	}
}

func (f *SumT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		BroadcastTo(f.xShape...)(gy[0]),
	}
}

package ag

import "github.com/qntx/spark/internal/mat"

func Sum(x ...*Var) *Var {
	return (&Operator{
		Op: &SumT{},
	}).First(x...)
}

type SumT struct {
	xShape []int
}

func (f *SumT) Forward(x ...*Var) []*Var {
	f.xShape = x[0].Shape()

	y := mat.Sum(x[0].Data)
	return []*Var{
		New(y),
	}
}

func (f *SumT) Backward(gy ...*Var) []*Var {
	return []*Var{
		BroadcastTo(f.xShape...)(gy[0]),
	}
}

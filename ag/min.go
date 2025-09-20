package ag

import "github.com/qntx/spark/internal/mat"

func Min(x ...*Var) *Var {
	return (&Operator{
		Op: &MinT{},
	}).First(x...)
}

type MinT struct {
	MaxT
}

func (f *MinT) Forward(x ...*Var) []*Var {
	f.x = x[0]
	f.y = New(mat.Min(x[0].Data))

	return []*Var{
		f.y,
	}
}

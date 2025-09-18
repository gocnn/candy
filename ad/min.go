package ad

import "github.com/qntx/spark/internal/mat"

func Min(x ...*Variable) *Variable {
	return (&Operator{
		Op: &MinT{},
	}).First(x...)
}

type MinT struct {
	MaxT
}

func (f *MinT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]
	f.y = New(mat.Min(x[0].Data))

	return []*Variable{
		f.y,
	}
}

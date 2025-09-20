package ag

import "github.com/qntx/spark/internal/mat"

func Transpose(x ...*Var) *Var {
	return (&Operator{Op: &TransposeT{}}).First(x...)
}

type TransposeT struct{}

func (f *TransposeT) Forward(x ...*Var) []*Var {
	y := mat.Transpose(x[0].Data)
	return []*Var{
		NewFrom(y),
	}
}

func (f *TransposeT) Backward(gy ...*Var) []*Var {
	return []*Var{
		Transpose(gy[0]),
	}
}

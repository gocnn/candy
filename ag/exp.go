package ag

import "github.com/qntx/spark/internal/mat"

func Exp(x ...*Var) *Var {
	return (&Operator{
		Op: &ExpT{},
	}).First(x...)
}

type ExpT struct {
	y *Var
}

func (f *ExpT) Forward(x ...*Var) []*Var {
	y := mat.Exp(x[0].Data)
	f.y = NewFrom(y)

	return []*Var{
		f.y,
	}
}

func (f *ExpT) Backward(gy ...*Var) []*Var {
	return []*Var{
		Mul(gy[0], f.y), // gy * y
	}
}

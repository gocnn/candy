package ag

import "github.com/qntx/spark/internal/mat"

func Cos(x ...*Var) *Var {
	return (&Operator{
		Op: &CosT{},
	}).First(x...)
}

type CosT struct {
	x *Var
}

func (f *CosT) Forward(x ...*Var) []*Var {
	f.x = x[0]

	y := mat.Cos(x[0].Data)
	return []*Var{
		NewFrom(y),
	}
}

func (f *CosT) Backward(gy ...*Var) []*Var {
	return []*Var{
		Mul(Neg(Sin(f.x)), gy[0]), // -1.0 * sin(x) * gy
	}
}

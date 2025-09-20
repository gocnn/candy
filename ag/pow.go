package ag

import "github.com/qntx/spark/internal/mat"

func Pow(p float64) func(x ...*Var) *Var {
	return (&Operator{
		Op: &PowT{
			P: p,
		},
	}).First
}

type PowT struct {
	P float64
	x *Var
}

func (f *PowT) Forward(x ...*Var) []*Var {
	f.x = x[0]

	y := mat.Pow(f.P, x[0].Data)
	return []*Var{
		NewFrom(y),
	}
}

func (f *PowT) Backward(gy ...*Var) []*Var {
	return []*Var{
		Mul(gy[0], MulC(f.P, Pow(f.P-1)(f.x))), // gy * c * x^(c-1)
	}
}

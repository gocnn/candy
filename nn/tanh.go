package nn

import (
	"github.com/qntx/spark/ag"
	"github.com/qntx/spark/internal/mat"
)

func Tanh(x ...*ag.Var) *ag.Var {
	return (&ag.Operator{
		Op: &TanhT{},
	}).First(x...)
}

type TanhT struct {
	y *ag.Var
}

func (f *TanhT) Forward(x ...*ag.Var) []*ag.Var {
	y := mat.Tanh(x[0].Data)
	f.y = ag.NewFrom(y)

	return []*ag.Var{
		f.y,
	}
}

func (f *TanhT) Backward(gy ...*ag.Var) []*ag.Var {
	return []*ag.Var{
		ag.Mul(gy[0], ag.SubC(1.0, ag.Mul(f.y, f.y))), // gy * (1-y^2)
	}
}

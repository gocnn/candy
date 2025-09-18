package nn

import (
	"github.com/qntx/spark/ag"
	"github.com/qntx/spark/internal/mat"
)

func Tanh(x ...*ag.Variable) *ag.Variable {
	return (&ag.Operator{
		Op: &TanhT{},
	}).First(x...)
}

type TanhT struct {
	y *ag.Variable
}

func (f *TanhT) Forward(x ...*ag.Variable) []*ag.Variable {
	y := mat.Tanh(x[0].Data)
	f.y = ag.NewFrom(y)

	return []*ag.Variable{
		f.y,
	}
}

func (f *TanhT) Backward(gy ...*ag.Variable) []*ag.Variable {
	return []*ag.Variable{
		ag.Mul(gy[0], ag.SubC(1.0, ag.Mul(f.y, f.y))), // gy * (1-y^2)
	}
}

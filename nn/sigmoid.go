package nn

import (
	"github.com/qntx/spark/ag"
	"github.com/qntx/spark/internal/mat"
)

func Sigmoid(x ...*ag.Var) *ag.Var {
	return (&ag.Operator{Op: &SigmoidT{}}).First(x...)
}

type SigmoidT struct {
	y *ag.Var
}

func (f *SigmoidT) Forward(x ...*ag.Var) []*ag.Var {
	tanh := mat.Tanh(mat.MulC(0.5, x[0].Data)) // tanh(0.5 * x)
	y := mat.AddC(0.5, mat.MulC(0.5, tanh))    // 0.5 + 0.5 * tanh(0.5 * x)

	f.y = ag.NewFrom(y)
	return []*ag.Var{
		f.y,
	}
}

func (f *SigmoidT) Backward(gy ...*ag.Var) []*ag.Var {
	return []*ag.Var{
		ag.Mul(gy[0], ag.Mul(f.y, ag.SubC(1.0, f.y))), // gy * y * (1 - y)
	}
}

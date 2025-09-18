package nn

import (
	"github.com/qntx/spark/ag"
	"github.com/qntx/spark/internal/mat"
)

func Sigmoid(x ...*ag.Variable) *ag.Variable {
	return (&ag.Operator{Op: &SigmoidT{}}).First(x...)
}

type SigmoidT struct {
	y *ag.Variable
}

func (f *SigmoidT) Forward(x ...*ag.Variable) []*ag.Variable {
	tanh := mat.Tanh(mat.MulC(0.5, x[0].Data)) // tanh(0.5 * x)
	y := mat.AddC(0.5, mat.MulC(0.5, tanh))    // 0.5 + 0.5 * tanh(0.5 * x)

	f.y = ag.NewFrom(y)
	return []*ag.Variable{
		f.y,
	}
}

func (f *SigmoidT) Backward(gy ...*ag.Variable) []*ag.Variable {
	return []*ag.Variable{
		ag.Mul(gy[0], ag.Mul(f.y, ag.SubC(1.0, f.y))), // gy * y * (1 - y)
	}
}

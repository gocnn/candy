package nn

import (
	"github.com/qntx/spark/internal/mat"
	"github.com/qntx/spark/tensor"
)

func Sigmoid(x ...*tensor.Variable) *tensor.Variable {
	return (&tensor.Function{Forwarder: &SigmoidT{}}).First(x...)
}

type SigmoidT struct {
	y *tensor.Variable
}

func (f *SigmoidT) Forward(x ...*tensor.Variable) []*tensor.Variable {
	tanh := mat.Tanh(mat.MulC(0.5, x[0].Data)) // tanh(0.5 * x)
	y := mat.AddC(0.5, mat.MulC(0.5, tanh))    // 0.5 + 0.5 * tanh(0.5 * x)

	f.y = tensor.NewFrom(y)
	return []*tensor.Variable{
		f.y,
	}
}

func (f *SigmoidT) Backward(gy ...*tensor.Variable) []*tensor.Variable {
	return []*tensor.Variable{
		Mul(gy[0], Mul(f.y, SubC(1.0, f.y))), // gy * y * (1 - y)
	}
}

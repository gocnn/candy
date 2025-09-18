package nn

import (
	"github.com/qntx/spark/ad"
	"github.com/qntx/spark/internal/mat"
)

func Sigmoid(x ...*ad.Variable) *ad.Variable {
	return (&ad.Operator{Op: &SigmoidT{}}).First(x...)
}

type SigmoidT struct {
	y *ad.Variable
}

func (f *SigmoidT) Forward(x ...*ad.Variable) []*ad.Variable {
	tanh := mat.Tanh(mat.MulC(0.5, x[0].Data)) // tanh(0.5 * x)
	y := mat.AddC(0.5, mat.MulC(0.5, tanh))    // 0.5 + 0.5 * tanh(0.5 * x)

	f.y = ad.NewFrom(y)
	return []*ad.Variable{
		f.y,
	}
}

func (f *SigmoidT) Backward(gy ...*ad.Variable) []*ad.Variable {
	return []*ad.Variable{
		ad.Mul(gy[0], ad.Mul(f.y, ad.SubC(1.0, f.y))), // gy * y * (1 - y)
	}
}

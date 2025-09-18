package nn

import (
	"github.com/qntx/spark/ad"
	"github.com/qntx/spark/internal/mat"
)

func Tanh(x ...*ad.Variable) *ad.Variable {
	return (&ad.Operator{
		Op: &TanhT{},
	}).First(x...)
}

type TanhT struct {
	y *ad.Variable
}

func (f *TanhT) Forward(x ...*ad.Variable) []*ad.Variable {
	y := mat.Tanh(x[0].Data)
	f.y = ad.NewFrom(y)

	return []*ad.Variable{
		f.y,
	}
}

func (f *TanhT) Backward(gy ...*ad.Variable) []*ad.Variable {
	return []*ad.Variable{
		ad.Mul(gy[0], ad.SubC(1.0, ad.Mul(f.y, f.y))), // gy * (1-y^2)
	}
}

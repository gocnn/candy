package nn

import (
	"github.com/qntx/spark/ad"
	"github.com/qntx/spark/internal/mat"
)

func Linear(x ...*ad.Variable) *ad.Variable {
	return (&ad.Operator{Op: &LinearT{}}).First(x...)
}

type LinearT struct {
	x, w, b *ad.Variable
}

func (f *LinearT) Forward(x ...*ad.Variable) []*ad.Variable {
	f.x, f.w = x[0], x[1]
	y := mat.MatMul(x[0].Data, x[1].Data)

	if len(x) < 3 {
		// no bias
		return []*ad.Variable{
			ad.NewFrom(y),
		}
	}

	// add bias
	f.b, y = x[2], mat.Add(y, x[2].Data)

	return []*	ad.Variable{
		ad.NewFrom(y),
	}
}

func (f *LinearT) Backward(gy ...*ad.Variable) []*ad.Variable {
	gxs := []*ad.Variable{
		ad.MatMul(gy[0], ad.Transpose(f.w)), // gy * w.T
		ad.MatMul(ad.Transpose(f.x), gy[0]), // x.T * gy
	}

	if f.b == nil {
		// no bias
		return gxs
	}

	// add bias
	gb := ad.SumTo(f.b.Shape()...)(gy[0])
	return append(gxs, gb)
}

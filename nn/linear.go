package nn

import (
	"github.com/qntx/spark/internal/mat"
	"github.com/qntx/spark/tensor"
)

func Linear(x ...*tensor.Variable) *tensor.Variable {
	return (&tensor.Function{Forwarder: &LinearT{}}).First(x...)
}

type LinearT struct {
	x, w, b *tensor.Variable
}

func (f *LinearT) Forward(x ...*tensor.Variable) []*tensor.Variable {
	f.x, f.w = x[0], x[1]
	y := mat.MatMul(x[0].Data, x[1].Data)

	if len(x) < 3 {
		// no bias
		return []*tensor.Variable{
			tensor.NewFrom(y),
		}
	}

	// add bias
	f.b, y = x[2], mat.Add(y, x[2].Data)

	return []*tensor.Variable{
		tensor.NewFrom(y),
	}
}

func (f *LinearT) Backward(gy ...*tensor.Variable) []*tensor.Variable {
	gxs := []*tensor.Variable{
		MatMul(gy[0], Transpose(f.w)), // gy * w.T
		MatMul(Transpose(f.x), gy[0]), // x.T * gy
	}

	if f.b == nil {
		// no bias
		return gxs
	}

	// add bias
	gb := SumTo(f.b.Shape()...)(gy[0])
	return append(gxs, gb)
}

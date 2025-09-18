package nn

import (
	"github.com/qntx/spark/ag"
	"github.com/qntx/spark/internal/mat"
)

func Linear(x ...*ag.Variable) *ag.Variable {
	return (&ag.Operator{Op: &LinearT{}}).First(x...)
}

type LinearT struct {
	x, w, b *ag.Variable
}

func (f *LinearT) Forward(x ...*ag.Variable) []*ag.Variable {
	f.x, f.w = x[0], x[1]
	y := mat.MatMul(x[0].Data, x[1].Data)

	if len(x) < 3 {
		// no bias
		return []*ag.Variable{
			ag.NewFrom(y),
		}
	}

	// add bias
	f.b, y = x[2], mat.Add(y, x[2].Data)

	return []*ag.Variable{
		ag.NewFrom(y),
	}
}

func (f *LinearT) Backward(gy ...*ag.Variable) []*ag.Variable {
	gxs := []*ag.Variable{
		ag.MatMul(gy[0], ag.Transpose(f.w)), // gy * w.T
		ag.MatMul(ag.Transpose(f.x), gy[0]), // x.T * gy
	}

	if f.b == nil {
		// no bias
		return gxs
	}

	// add bias
	gb := ag.SumTo(f.b.Shape()...)(gy[0])
	return append(gxs, gb)
}

package nn

import (
	"github.com/qntx/spark/ag"
	"github.com/qntx/spark/internal/mat"
)

func Linear(x ...*ag.Var) *ag.Var {
	return (&ag.Operator{Op: &LinearT{}}).First(x...)
}

type LinearT struct {
	x, w, b *ag.Var
}

func (f *LinearT) Forward(x ...*ag.Var) []*ag.Var {
	f.x, f.w = x[0], x[1]
	y := mat.MatMul(x[0].Data, x[1].Data)

	if len(x) < 3 {
		// no bias
		return []*ag.Var{
			ag.NewFrom(y),
		}
	}

	// add bias
	f.b, y = x[2], mat.Add(y, x[2].Data)

	return []*ag.Var{
		ag.NewFrom(y),
	}
}

func (f *LinearT) Backward(gy ...*ag.Var) []*ag.Var {
	gxs := []*ag.Var{
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

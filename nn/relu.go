package nn

import (
	"math"

	"github.com/qntx/spark/ag"
	"github.com/qntx/spark/internal/mat"
)

func ReLU(x ...*ag.Var) *ag.Var {
	return (&ag.Operator{Op: &ReLUT{}}).First(x...)
}

type ReLUT struct {
	x *ag.Var
}

func (f *ReLUT) Forward(x ...*ag.Var) []*ag.Var {
	f.x = x[0]
	y := mat.F(x[0].Data, maximum)

	return []*ag.Var{
		ag.NewFrom(y),
	}
}

func (f *ReLUT) Backward(gy ...*ag.Var) []*ag.Var {
	mask := mat.Mask(f.x.Data, relu)

	return []*ag.Var{
		ag.Mul(gy[0], ag.NewFrom(mask)), // gy * mask
	}
}

func maximum(v float64) float64 { return math.Max(v, 0.0) }

func relu(v float64) bool { return v > 0 }

package nn

import (
	"math"

	"github.com/qntx/spark/ag"
	"github.com/qntx/spark/internal/mat"
)

func ReLU(x ...*ag.Variable) *ag.Variable {
	return (&ag.Operator{Op: &ReLUT{}}).First(x...)
}

type ReLUT struct {
	x *ag.Variable
}

func (f *ReLUT) Forward(x ...*ag.Variable) []*ag.Variable {
	f.x = x[0]
	y := mat.F(x[0].Data, maximum)

	return []*ag.Variable{
		ag.NewFrom(y),
	}
}

func (f *ReLUT) Backward(gy ...*ag.Variable) []*ag.Variable {
	mask := mat.Mask(f.x.Data, relu)

	return []*ag.Variable{
		ag.Mul(gy[0], ag.NewFrom(mask)), // gy * mask
	}
}

func maximum(v float64) float64 { return math.Max(v, 0.0) }

func relu(v float64) bool { return v > 0 }

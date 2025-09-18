package nn

import (
	"math"

	"github.com/qntx/spark/ad"
	"github.com/qntx/spark/internal/mat"
)

func ReLU(x ...*ad.Variable) *ad.Variable {
	return (&ad.Operator{Op: &ReLUT{}}).First(x...)
}

type ReLUT struct {
	x *ad.Variable
}

func (f *ReLUT) Forward(x ...*ad.Variable) []*ad.Variable {
	f.x = x[0]
	y := mat.F(x[0].Data, maximum)

	return []*ad.Variable{
		ad.NewFrom(y),
	}
}

func (f *ReLUT) Backward(gy ...*ad.Variable) []*ad.Variable {
	mask := mat.Mask(f.x.Data, relu)

	return []*ad.Variable{
		ad.Mul(gy[0], ad.NewFrom(mask)), // gy * mask
	}
}

func maximum(v float64) float64 { return math.Max(v, 0.0) }

func relu(v float64) bool { return v > 0 }

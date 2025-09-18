package nn

import (
	"math"

	"github.com/qntx/spark/internal/mat"
	"github.com/qntx/spark/tensor"
)

func ReLU(x ...*tensor.Variable) *tensor.Variable {
	return (&tensor.Function{Forwarder: &ReLUT{}}).First(x...)
}

type ReLUT struct {
	x *tensor.Variable
}

func (f *ReLUT) Forward(x ...*tensor.Variable) []*tensor.Variable {
	f.x = x[0]
	y := mat.F(x[0].Data, maximum)

	return []*tensor.Variable{
		tensor.NewFrom(y),
	}
}

func (f *ReLUT) Backward(gy ...*tensor.Variable) []*tensor.Variable {
	mask := mat.Mask(f.x.Data, relu)

	return []*tensor.Variable{
		Mul(gy[0], tensor.NewFrom(mask)), // gy * mask
	}
}

func maximum(v float64) float64 { return math.Max(v, 0.0) }

func relu(v float64) bool { return v > 0 }

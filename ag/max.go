package ag

import (
	"math"

	"github.com/qntx/spark/internal/mat"
)

func Max(x ...*Var) *Var {
	return (&Operator{
		Op: &MaxT{},
	}).First(x...)
}

type MaxT struct {
	x, y *Var
}

func (f *MaxT) Forward(x ...*Var) []*Var {
	f.x = x[0]
	f.y = New(mat.Max(x[0].Data))

	return []*Var{
		f.y,
	}
}

func (f *MaxT) Backward(gy ...*Var) []*Var {
	mask := NewFrom(mat.F2(f.x.Data, f.y.Data, IsClose))
	return []*Var{
		Mul(gy[0], mask),
	}
}

func IsClose(a, b float64) float64 {
	atol, rtol := 1e-08, 1e-05
	if math.Abs(a-b) < atol+rtol*math.Abs(b) {
		return 1
	}

	return 0
}

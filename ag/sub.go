package ag

import (
	"github.com/qntx/spark/internal/mat"
	"github.com/qntx/spark/internal/vec"
)

// SubC returns a variable that c - x[0].
func SubC(c float64, x ...*Var) *Var {
	return (&Operator{
		Op: &SubT{},
	}).First(New(c), x[0])
}

// Sub returns a variable that x[0] - x[1].
func Sub(x ...*Var) *Var {
	return (&Operator{
		Op: &SubT{},
	}).First(x...)
}

type SubT struct {
	x0Shape, x1Shape []int
}

func (f *SubT) Forward(x ...*Var) []*Var {
	f.x0Shape, f.x1Shape = x[0].Shape(), x[1].Shape()

	y := mat.Sub(x[0].Data, x[1].Data)
	return []*Var{
		NewFrom(y),
	}
}

func (f *SubT) Backward(gy ...*Var) []*Var {
	gx0 := gy[0]
	gx1 := Neg(gy[0]) // -1.0 * gy

	if vec.Equal(f.x0Shape, f.x1Shape) {
		return []*Var{
			gx0,
			gx1,
		}
	}

	return []*Var{
		SumTo(f.x0Shape...)(gx0),
		SumTo(f.x1Shape...)(gx1),
	}
}

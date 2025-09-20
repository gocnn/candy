package ag

import (
	"github.com/qntx/spark/internal/mat"
	"github.com/qntx/spark/internal/vec"
)

func MulC(c float64, x ...*Var) *Var {
	return (&Operator{Op: &MulT{}}).First(New(c), x[0])
}

func Mul(x ...*Var) *Var {
	return (&Operator{Op: &MulT{}}).First(x...)
}

type MulT struct {
	x0, x1           *Var
	x0Shape, x1Shape []int
}

func (f *MulT) Forward(x ...*Var) []*Var {
	f.x0, f.x1 = x[0], x[1]
	f.x0Shape, f.x1Shape = x[0].Shape(), x[1].Shape()

	y := mat.Mul(x[0].Data, x[1].Data)
	return []*Var{
		NewFrom(y),
	}
}

func (f *MulT) Backward(gy ...*Var) []*Var {
	gx0 := Mul(gy[0], f.x1) // gy * x1
	gx1 := Mul(gy[0], f.x0) // gy * x0

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

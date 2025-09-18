package ad

import (
	"github.com/qntx/spark/internal/mat"
	"github.com/qntx/spark/internal/vec"
)

func DivC(c float64, x ...*Variable) *Variable {
	return (&Operator{
		Op: &DivT{},
	}).First(New(c), x[0])
}

func Div(x ...*Variable) *Variable {
	return (&Operator{
		Op: &DivT{},
	}).First(x...)
}

type DivT struct {
	x0, x1           *Variable
	x0Shape, x1Shape []int
}

func (f *DivT) Forward(x ...*Variable) []*Variable {
	f.x0, f.x1 = x[0], x[1]
	f.x0Shape, f.x1Shape = x[0].Shape(), x[1].Shape()

	y := mat.Div(x[0].Data, x[1].Data)
	return []*Variable{
		NewFrom(y),
	}
}

func (f *DivT) Backward(gy ...*Variable) []*Variable {
	gx0 := Div(gy[0], f.x1)
	gx1 := Mul(gy[0], Div(Neg(f.x0), Mul(f.x1, f.x1))) // gy * (-x0 / x1^2)

	if vec.Equal(f.x0Shape, f.x1Shape) {
		return []*Variable{
			gx0,
			gx1,
		}
	}

	return []*Variable{
		SumTo(f.x0Shape...)(gx0),
		SumTo(f.x1Shape...)(gx1),
	}
}

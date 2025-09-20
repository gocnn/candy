package ag

import "github.com/qntx/spark/internal/mat"

func Log(x ...*Var) *Var {
	return (&Operator{
		Op: &LogT{},
	}).First(x...)
}

type LogT struct {
	x *Var
}

func (f *LogT) Forward(x ...*Var) []*Var {
	f.x = x[0]

	y := mat.Log(x[0].Data)
	return []*Var{
		NewFrom(y),
	}
}

func (f *LogT) Backward(gy ...*Var) []*Var {
	return []*Var{
		Div(gy[0], f.x), // gy / x
	}
}

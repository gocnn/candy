package ag

import "github.com/qntx/spark/internal/mat"

func Log(x ...*Variable) *Variable {
	return (&Operator{
		Op: &LogT{},
	}).First(x...)
}

type LogT struct {
	x *Variable
}

func (f *LogT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := mat.Log(x[0].Data)
	return []*Variable{
		NewFrom(y),
	}
}

func (f *LogT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Div(gy[0], f.x), // gy / x
	}
}

package ag

import "github.com/qntx/spark/internal/mat"

func MatMul(x ...*Variable) *Variable {
	return (&Operator{
		Op: &MatMulT{},
	}).First(x...)
}

type MatMulT struct {
	x, w *Variable
}

func (f *MatMulT) Forward(x ...*Variable) []*Variable {
	f.x, f.w = x[0], x[1]

	y := mat.MatMul(x[0].Data, x[1].Data)
	return []*Variable{
		NewFrom(y),
	}
}

func (f *MatMulT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		MatMul(gy[0], Transpose(f.w)), // gy * w.T
		MatMul(Transpose(f.x), gy[0]), // x.T * gy
	}
}

package nn

import (
	"github.com/qntx/spark/ag"
	"github.com/qntx/spark/internal/mat"
)

func Softmax(x ...*ag.Var) *ag.Var {
	return (&ag.Operator{Op: &SoftmaxT{}}).First(x...)
}

type SoftmaxT struct {
	y *ag.Var
}

func (f *SoftmaxT) Forward(x ...*ag.Var) []*ag.Var {
	max := mat.MaxAxis1(x[0].Data)           // max(x, axis=1)
	expy := mat.Exp(mat.Sub(x[0].Data, max)) // expy = exp(x - max)
	sumy := mat.SumAxis1(expy)               // sumy = sum(expy, axis=1)
	y := mat.Div(expy, sumy)                 // y = expy / sumy

	f.y = ag.NewFrom(y)
	return []*ag.Var{
		f.y,
	}
}

func (f *SoftmaxT) Backward(gy ...*ag.Var) []*ag.Var {
	gyy := ag.Mul(gy[0], f.y) // gyy = gy * y
	N := gyy.Shape()[0]
	sum := ag.SumTo(N, 1)(gyy) // sum = sum(gx, axis=1)

	return []*ag.Var{
		ag.Sub(gyy, ag.Mul(f.y, sum)), // gyy - y * sum
	}
}

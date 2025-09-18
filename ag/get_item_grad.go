package ag

import (
	"github.com/qntx/spark/internal/mat"
	"github.com/qntx/spark/internal/vec"
)

func GetItemGrad(slices, inShape []int) func(x ...*Variable) *Variable {
	return (&Operator{
		Op: &GetItemGradT{
			Slices:  slices,
			InShape: inShape,
		},
	}).First
}

type GetItemGradT struct {
	Slices  []int
	InShape []int
}

func (f *GetItemGradT) Forward(gy ...*Variable) []*Variable {
	gx := mat.Zero(f.InShape[0], f.InShape[1])
	for i, idx := range f.Slices {
		gx.SetRow(idx, vec.Add(gx.Row(idx), gy[0].Data.Row(i)))
	}

	return []*Variable{
		NewFrom(gx),
	}
}

func (f *GetItemGradT) Backward(ggx ...*Variable) []*Variable {
	return []*Variable{
		GetItem(f.Slices)(ggx...),
	}
}

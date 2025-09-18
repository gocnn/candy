package ad

func GetItem(slices []int) func(x ...*Variable) *Variable {
	return (&Operator{
		Op: &GetItemT{
			Slices: slices,
		},
	}).First
}

type GetItemT struct {
	Slices []int
	xShape []int
}

func (f *GetItemT) Forward(x ...*Variable) []*Variable {
	f.xShape = x[0].Shape()

	y := make([][]float64, len(f.Slices))
	for i, idx := range f.Slices {
		y[i] = x[0].Data.Row(idx)
	}

	return []*Variable{
		NewOf(y...),
	}
}

func (f *GetItemT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		GetItemGrad(f.Slices, f.xShape)(gy...),
	}
}

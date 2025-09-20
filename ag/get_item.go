package ag

func GetItem(slices []int) func(x ...*Var) *Var {
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

func (f *GetItemT) Forward(x ...*Var) []*Var {
	f.xShape = x[0].Shape()

	y := make([][]float64, len(f.Slices))
	for i, idx := range f.Slices {
		y[i] = x[0].Data.Row(idx)
	}

	return []*Var{
		NewOf(y...),
	}
}

func (f *GetItemT) Backward(gy ...*Var) []*Var {
	return []*Var{
		GetItemGrad(f.Slices, f.xShape)(gy...),
	}
}

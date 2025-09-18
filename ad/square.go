package ad

func Square(x ...*Variable) *Variable {
	return (&Operator{
		Op: &SquareT{
			PowT{
				P: 2.0,
			},
		},
	}).First(x...)
}

type SquareT struct {
	PowT
}

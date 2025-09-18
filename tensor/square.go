package tensor

func Square(x ...*Variable) *Variable {
	return (&Function{
		Forwarder: &SquareT{
			PowT{
				P: 2.0,
			},
		},
	}).First(x...)
}

type SquareT struct {
	PowT
}

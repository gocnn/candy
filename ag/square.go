package ag

func Square(x ...*Var) *Var {
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

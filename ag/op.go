package ag

import "fmt"

type Op interface {
	Forward(x ...*Var) []*Var
	Backward(gy ...*Var) []*Var
}

type Operator struct {
	Input, Output []*Var
	Generation    int
	Op
}

// First applies the function and returns the first output
func (f *Operator) First(x ...*Var) *Var {
	return f.Forward(x...)[0]
}

// Forward applies the function
func (f *Operator) Forward(x ...*Var) []*Var {
	y := f.Op.Forward(x...)
	if !Config.EnableBackprop {
		return y
	}

	f.Generation = maxgen(x...) //
	f.setCreator(y)             // set creator and increment generation
	f.Input, f.Output = x, y    //
	return y
}

func (f Operator) String() string {
	return fmt.Sprintf("%T%v", f.Op, f.Input)
}

func (f *Operator) setCreator(y []*Var) {
	for i := range y {
		y[i].SetCreator(f)
	}
}

func maxgen(x ...*Var) int {
	var max int
	for _, v := range x {
		if max < v.Generation {
			max = v.Generation
		}
	}

	return max
}

package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func Example_tanh() {
	// p249
	x := tensor.New(1.0)
	y := tensor.Tanh(x)
	y.Backward(tensor.Opts{CreateGraph: true})

	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward(tensor.Opts{CreateGraph: true})
	fmt.Println(x.Grad)

	gx = x.Grad
	x.Cleargrad()
	gx.Backward(tensor.Opts{CreateGraph: true})
	fmt.Println(x.Grad)

	gx = x.Grad
	x.Cleargrad()
	gx.Backward(tensor.Opts{CreateGraph: true})
	fmt.Println(x.Grad)

	// 1-tanh(1)^2                                   =  0.41997434161
	// −2*tanh(1)*(1−tanh^2(1))                      = -0.63970000844
	// 4*tanh^2(1)*sech^2(1) - 2*sech^4(1)           =  0.62162668077
	// 16*tanh(1) *sech^4(1) - 8*tanh^3(1)*sech^2(1) =  0.66509104475

	// Output:
	// variable(0.41997434161402614)
	// variable(-0.6397000084492246)
	// variable(0.6216266807712962)
	// variable(0.6650910447505024)
}

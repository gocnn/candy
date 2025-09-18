package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExamplePowT() {
	x := tensor.New(3.0)
	f := tensor.PowT{P: 4.0}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(tensor.OneLike(x)))

	// Output:
	// variable(3)
	// [variable(81)]
	// [variable(108)]
}

func ExamplePow() {
	x := tensor.New(2.0)
	y := tensor.Pow(3.0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(8)
	// variable(12)
}

func ExamplePow_double() {
	x := tensor.New(2.0)
	y := tensor.Pow(3.0)(x)
	y.Backward(tensor.Opts{CreateGraph: true})

	fmt.Println(y)
	fmt.Println(x.Grad)

	for i := 0; i < 3; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward(tensor.Opts{CreateGraph: true})
		fmt.Println(x.Grad)
	}

	// Output:
	// variable(8)
	// variable(12)
	// variable(12)
	// variable(6)
	// variable(0)
}

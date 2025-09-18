package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleSquare() {
	x := tensor.New(3.0)
	y := tensor.Square(x)
	y.Backward()

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(3)
	// variable(9)
	// variable(6)
}

func ExampleSquare_double() {
	x := tensor.New(3.0)
	y := tensor.Square(x)
	y.Backward(tensor.Opts{CreateGraph: true})

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x.Grad)

	for range 2 {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward(tensor.Opts{CreateGraph: true})
		fmt.Println(x.Grad)
	}

	// Output:
	// variable(3)
	// variable(9)
	// variable(6)
	// variable(2)
	// variable(0)
}

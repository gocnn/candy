package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleDiv() {
	a := tensor.New(10)
	b := tensor.New(2)
	y := tensor.Div(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable(5)
	// variable(0.5) variable(-2.5)
}

func ExampleDivC() {
	a := 10.0
	b := tensor.New(2)
	y := tensor.DivC(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(b.Grad)

	// Output:
	// variable(5)
	// variable(-2.5)
}

func ExampleDiv_broadcast() {
	// p305
	a := tensor.New(1, 2, 3, 4, 5)
	b := tensor.New(2)
	y := tensor.Div(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable[1 5]([0.5 1 1.5 2 2.5])
	// variable[1 5]([0.5 0.5 0.5 0.5 0.5]) variable(-3.75)
}

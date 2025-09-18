package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleSub() {
	a := tensor.New(3.0)
	b := tensor.New(2.0)
	y := tensor.Sub(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable(1)
	// variable(1) variable(-1)
}

func ExampleSubC() {
	x := tensor.New(3.0)
	y := tensor.SubC(10.0, x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(7)
	// variable(-1)
}

func ExampleSub_broadcast() {
	// p305
	a := tensor.New(1, 2, 3, 4, 5)
	b := tensor.New(1)
	y := tensor.Sub(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable[1 5]([0 1 2 3 4])
	// variable[1 5]([1 1 1 1 1]) variable(-5)
}

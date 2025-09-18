package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleAddT() {
	a := tensor.New(2, 3)
	b := tensor.New(3, 4)
	f := tensor.AddT{}

	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(f.Forward(a, b))
	fmt.Println(f.Backward(tensor.OneLike(a), tensor.OneLike(b)))

	// Output:
	// variable[1 2]([2 3])
	// variable[1 2]([3 4])
	// [variable[1 2]([5 7])]
	// [variable[1 2]([1 1]) variable[1 2]([1 1])]
}

func ExampleAdd() {
	a := tensor.New(2, 3)
	b := tensor.New(3, 4)
	y := tensor.Add(a, b)
	y.Backward()

	fmt.Println(a.Grad)
	fmt.Println(b.Grad)

	// Output:
	// variable[1 2]([1 1])
	// variable[1 2]([1 1])
}

func ExampleAddC() {
	x := tensor.New(3)
	y := tensor.AddC(10.0, x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(13)
	// variable(1)
}

func ExampleAdd_broadcast() {
	// p305
	a := tensor.New(1, 2, 3)
	b := tensor.New(10)
	y := tensor.Add(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable[1 3]([11 12 13])
	// variable[1 3]([1 1 1]) variable(3)
}

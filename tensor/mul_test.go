package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleMulT() {
	a := tensor.New(3.0)
	b := tensor.New(2.0)
	f := tensor.MulT{}

	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(f.Forward(a, b))
	fmt.Println(f.Backward(tensor.OneLike(a), tensor.OneLike(b)))

	// Output:
	// variable(3)
	// variable(2)
	// [variable(6)]
	// [variable(2) variable(3)]
}

func ExampleMul() {
	// p139
	a := tensor.New(3.0)
	b := tensor.New(2.0)
	c := tensor.New(1.0)
	y := tensor.Add(tensor.Mul(a, b), c)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable(7)
	// variable(2) variable(3)
}

func ExampleMul_broadcast() {
	// p305
	a := tensor.New(2, 2, 2, 2, 2)
	b := tensor.New(3.0)
	y := tensor.Mul(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable[1 5]([6 6 6 6 6])
	// variable[1 5]([3 3 3 3 3]) variable(10)
}

package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleExpT() {
	x := tensor.New(1, 2, 3, 4, 5)
	f := tensor.ExpT{}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(tensor.OneLike(x)))

	// Output:
	// variable[1 5]([1 2 3 4 5])
	// [variable[1 5]([2.718281828459045 7.38905609893065 20.085536923187668 54.598150033144236 148.4131591025766])]
	// [variable[1 5]([2.718281828459045 7.38905609893065 20.085536923187668 54.598150033144236 148.4131591025766])]
}

func ExampleExp() {
	v := tensor.New(1, 2, 3, 4, 5)
	y := tensor.Exp(v)
	y.Backward()

	fmt.Println(v.Grad)

	// Output:
	// variable[1 5]([2.718281828459045 7.38905609893065 20.085536923187668 54.598150033144236 148.4131591025766])
}

func ExampleExp_double() {
	x := tensor.New(2.0)
	y := tensor.Exp(x)
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
	// variable(7.38905609893065)
	// variable(7.38905609893065)
	// variable(7.38905609893065)
	// variable(7.38905609893065)
	// variable(7.38905609893065)
}

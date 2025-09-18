package tensor_test

import (
	"fmt"
	"math"

	"github.com/qntx/spark/tensor"
)

func ExampleCosT() {
	x := tensor.New(math.Pi / 4)
	f := tensor.CosT{}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(tensor.OneLike(x)))

	// Output:
	// variable(0.7853981633974483)
	// [variable(0.7071067811865476)]
	// [variable(-0.7071067811865475)]
}

func ExampleCos() {
	// p198
	x := tensor.New(math.Pi / 4)
	y := tensor.Cos(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println(1.0 / math.Sqrt2)

	// Output:
	// variable(0.7071067811865476)
	// variable(-0.7071067811865475)
	// 0.7071067811865476
}

func ExampleCos_double() {
	x := tensor.New(1.0)
	y := tensor.Cos(x)
	y.Backward(tensor.Opts{CreateGraph: true})

	fmt.Println(y)
	fmt.Println(x.Grad)

	for i := 0; i < 10; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward(tensor.Opts{CreateGraph: true})
		fmt.Println(x.Grad)
	}

	// Output:
	// variable(0.5403023058681398)
	// variable(-0.8414709848078965)
	// variable(-0.5403023058681398)
	// variable(0.8414709848078965)
	// variable(0.5403023058681398)
	// variable(-0.8414709848078965)
	// variable(-0.5403023058681398)
	// variable(0.8414709848078965)
	// variable(0.5403023058681398)
	// variable(-0.8414709848078965)
	// variable(-0.5403023058681398)
	// variable(0.8414709848078965)
}

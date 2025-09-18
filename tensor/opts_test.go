package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func Example_retaingrad() {
	// p121
	x0 := tensor.New(1.0)
	x1 := tensor.New(1.0)
	t := tensor.Add(x0, x1)
	y := tensor.Add(x0, t)
	y.Backward(tensor.Opts{RetainGrad: true})

	fmt.Println(y.Grad, t.Grad)
	fmt.Println(x0.Grad, x1.Grad)

	// Output:
	// variable(1) variable(1)
	// variable(2) variable(1)
}

func Example_retaingrad_false() {
	// p123
	x0 := tensor.New(1.0)
	x1 := tensor.New(1.0)
	t := tensor.Add(x0, x1)
	y := tensor.Add(x0, t)
	y.Backward()

	fmt.Println(y.Grad, t.Grad)
	fmt.Println(x0.Grad, x1.Grad)

	// Output:
	// <nil> <nil>
	// variable(2) variable(1)
}

func ExampleHasRetainGrad() {
	fmt.Println(tensor.HasRetainGrad())
	fmt.Println(tensor.HasRetainGrad(tensor.Opts{RetainGrad: false}))
	fmt.Println(tensor.HasRetainGrad(tensor.Opts{RetainGrad: true}))

	// Output:
	// false
	// false
	// true
}

func ExampleNoRetainGrad() {
	fmt.Println(tensor.NoRetainGrad())
	fmt.Println(tensor.NoRetainGrad(tensor.Opts{RetainGrad: false}))
	fmt.Println(tensor.NoRetainGrad(tensor.Opts{RetainGrad: true}))

	// Output:
	// true
	// true
	// false
}

func ExampleHasCreateGraph() {
	fmt.Println(tensor.HasCreateGraph())
	fmt.Println(tensor.HasCreateGraph(tensor.Opts{CreateGraph: false}))
	fmt.Println(tensor.HasCreateGraph(tensor.Opts{CreateGraph: true}))

	// Output:
	// false
	// false
	// true
}

func ExampleNoCreateGraph() {
	fmt.Println(tensor.NoCreateGraph())
	fmt.Println(tensor.NoCreateGraph(tensor.Opts{CreateGraph: false}))
	fmt.Println(tensor.NoCreateGraph(tensor.Opts{CreateGraph: true}))

	// Output:
	// true
	// true
	// false
}

package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleLogT() {
	x := tensor.New(1, 2, 3, 4, 5)
	f := tensor.LogT{}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(tensor.OneLike(x)))

	// Output:
	// variable[1 5]([1 2 3 4 5])
	// [variable[1 5]([0 0.6931471805599453 1.0986122886681096 1.3862943611198906 1.6094379124341003])]
	// [variable[1 5]([1 0.5 0.3333333333333333 0.25 0.2])]
}

func ExampleLog() {
	v := tensor.New(1, 2, 3, 4, 5)
	y := tensor.Log(v)
	y.Backward()

	fmt.Println(v.Grad)

	// Output:
	// variable[1 5]([1 0.5 0.3333333333333333 0.25 0.2])
}

func ExampleLog_double() {
	x := tensor.New(2)
	y := tensor.Log(x)
	y.Backward(tensor.Opts{CreateGraph: true})

	fmt.Println(y)
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

	// Output:
	// variable(0.6931471805599453)
	// variable(0.5)
	// variable(-0.25)
	// variable(0.25)
	// variable(-0.375)
}

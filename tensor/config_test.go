package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleNograd() {
	f := func() {
		x := tensor.New(3)
		y := tensor.Square(x)
		y.Backward()

		fmt.Println("gx: ", x.Grad)
		fmt.Println()
	}

	fmt.Println("backprop:", tensor.Config.EnableBackprop)
	f()

	func() {
		defer tensor.Nograd().End()

		fmt.Println("backprop:", tensor.Config.EnableBackprop)
		f()
	}()

	fmt.Println("backprop:", tensor.Config.EnableBackprop)
	f()

	// Output:
	// backprop: true
	// gx:  variable(6)
	//
	// backprop: false
	// gx:  <nil>
	//
	// backprop: true
	// gx:  variable(6)
}

func ExampleTestMode() {
	fmt.Println("train:", tensor.Config.Train)

	func() {
		defer tensor.TestMode().End()

		fmt.Println("train:", tensor.Config.Train)
	}()

	fmt.Println("train:", tensor.Config.Train)

	// Output:
	// train: true
	// train: false
	// train: true
}

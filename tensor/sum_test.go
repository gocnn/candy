package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleSum() {
	// p292
	x := tensor.New(1, 2, 3, 4, 5, 6)
	y := tensor.Sum(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(21)
	// variable[1 6]([1 1 1 1 1 1])
}

func ExampleSum_matrix() {
	// p293
	x := tensor.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	y := tensor.Sum(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(21)
	// variable[2 3]([[1 1 1] [1 1 1]])
}

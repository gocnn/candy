package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleSumTo() {
	// p301
	x := tensor.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	y := tensor.SumTo(1, 3)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	x.Cleargrad()

	y = tensor.SumTo(2, 1)(x)
	y.Backward()
	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[1 3]([5 7 9])
	// variable[2 3]([[1 1 1] [1 1 1]])
	// variable[2 1]([[6] [15]])
	// variable[2 3]([[1 1 1] [1 1 1]])
}

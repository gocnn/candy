package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleMax() {
	A := tensor.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 10, 6},
	)

	y := tensor.Max(A)
	y.Backward()

	fmt.Println(y)
	fmt.Println(A.Grad)

	// Output:
	// variable(10)
	// variable[2 3]([[0 0 0] [0 1 0]])
}

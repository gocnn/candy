package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleTranspose() {
	// p286
	x := tensor.NewOf([]float64{1, 2, 3}, []float64{4, 5, 6})
	y := tensor.Transpose(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3 2]([[1 4] [2 5] [3 6]])
	// variable[2 3]([[1 1 1] [1 1 1]])
}

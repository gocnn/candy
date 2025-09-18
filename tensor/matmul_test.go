package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/internal/mat"
	"github.com/qntx/spark/tensor"
)

func ExampleMatMul() {
	x := tensor.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	w := tensor.NewOf(
		[]float64{1, 2, 3, 4},
		[]float64{5, 6, 7, 8},
		[]float64{9, 10, 11, 12},
	)

	y := tensor.MatMul(x, w)
	y.Backward()

	fmt.Println(mat.Shape(x.Grad.Data))
	fmt.Println(mat.Shape(w.Grad.Data))

	// Output:
	// [2 3]
	// [3 4]
}

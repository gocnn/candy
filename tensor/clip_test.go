package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleClip() {
	x := tensor.New(-2, -1, 0, 1, 2, 4, 6, 8)
	y := tensor.Clip(0, 5)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[1 8]([0 0 0 1 2 4 5 5])
	// variable[1 8]([0 0 1 1 1 1 0 0])
}

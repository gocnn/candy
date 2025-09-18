package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleBroadcastTo() {
	x := tensor.New(2)
	y := tensor.BroadcastTo(1, 3)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[1 3]([2 2 2])
	// variable(3)
}

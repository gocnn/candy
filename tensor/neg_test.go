package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleNeg() {
	// p139
	x := tensor.New(3.0)
	y := tensor.Neg(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(-3)
	// variable(-1)
}

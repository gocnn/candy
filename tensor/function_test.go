package tensor_test

import (
	"fmt"

	"github.com/qntx/spark/tensor"
)

func ExampleFunction() {
	f := &tensor.Function{
		Forwarder: &tensor.SinT{},
	}

	y := f.Forward(tensor.New(1.0))
	fmt.Println(f)
	fmt.Println(y)

	// Output:
	// *tensor.SinT[variable(1)]
	// [variable(0.8414709848078965)]
}

package tensor_test

import (
	"fmt"
	"math"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

// ExampleTensor_String_empty demonstrates formatting an empty tensor.
func ExampleTensor_String_empty() {
	t := tensor.Zeros[float32](spark.NewShape(0), spark.CPU)
	fmt.Println("\n", t)
	// Output:
	// tensor([], shape=[0], dtype=float32, device=cpu)
}

// ExampleTensor_String_scalar demonstrates formatting a scalar tensor.
func ExampleTensor_String_scalar() {
	t := tensor.New([]float32{3.14159}, spark.NewShape(), spark.CPU)
	fmt.Println("\n", t)
	// Output:
	// tensor(3.14159, shape=[], dtype=float32, device=cpu)
}

// ExampleTensor_String_vector demonstrates formatting a 1D tensor (vector).
func ExampleTensor_String_vector() {
	t := tensor.New([]float32{1.0, 2.5, 3.0, 4.25, 5.0}, spark.NewShape(5), spark.CPU)
	fmt.Println("\n", t)
	// Output:
	// tensor([  1.,  2.5,   3., 4.25,   5.], shape=[5], dtype=float32, device=cpu)
}

// ExampleTensor_String_matrix demonstrates formatting a 2D tensor (matrix).
func ExampleTensor_String_matrix() {
	t := tensor.New([]float32{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	}, spark.NewShape(2, 3), spark.CPU)
	fmt.Println("\n", t)
	// Output:
	// tensor([[1., 2., 3.],
	//         [4., 5., 6.]], shape=[2 3], dtype=float32, device=cpu)
}

// ExampleTensor_String_special demonstrates formatting with NaN, infinity, and extreme values.
func ExampleTensor_String_special() {
	data := []float64{
		math.NaN(), math.Inf(1), math.Inf(-1), math.MaxFloat64, // NaN, +Inf, -Inf, MaxFloat64
		-math.MaxFloat64, 2.2250738585072014e-308, 4.9406564584124654e-324, 0.0, // -MaxFloat64, MinNormalFloat64, SmallestNonzeroFloat64, Zero
		0.0, 42.0, -999999999999.999, 0.000000000000001, // Negative zero, normal, large negative, tiny positive
	}
	t := tensor.New(data, spark.NewShape(3, 4), spark.CPU)
	fmt.Println("\n", t)
	// Output:
	// tensor([[         nan,          inf,         -inf,  1.7977e+308],
	//         [-1.7977e+308,  2.2251e-308,  4.9407e-324,           0.],
	//         [          0.,          42.,       -1e+12,        1e-15]], shape=[3 4], dtype=float64, device=cpu)
}

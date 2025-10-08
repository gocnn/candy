package main

import (
	"fmt"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

// Computes f(x, y) = sqrt((x + y) * (x - y)) + x / y and its gradients
func main() {
	// Initialize input tensors
	x := tensor.MustNew([]float64{3.0}, spark.NewShape(1), spark.CPU).RequiresGrad()
	y := tensor.MustNew([]float64{2.0}, spark.NewShape(1), spark.CPU).RequiresGrad()

	// Forward pass
	result := x.MustAdd(y).MustMul(x.MustSub(y)).MustSqrt().MustAdd(x.MustDiv(y))

	// Backward pass
	store, err := result.Backward()
	if err != nil {
		panic(err)
	}

	// Print results
	fmt.Printf("f(%.1f, %.1f) = %.4f\n", 3.0, 2.0, result.Data()[0])
	fmt.Printf("∂f/∂x = %.4f (Expected ≈ %.4f)\n", store.Get(x).Data()[0], 1.8416)
	fmt.Printf("∂f/∂y = %.4f (Expected ≈ %.4f)\n", store.Get(y).Data()[0], -1.6444)
	fmt.Printf("x grad.IsVar() = %v\n", store.Get(x).IsVar())
	fmt.Printf("y grad.IsVar() = %v\n", store.Get(y).IsVar())
}

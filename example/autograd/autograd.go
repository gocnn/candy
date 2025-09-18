package main

import (
	"fmt"
	"math"

	"github.com/qntx/spark"
	"github.com/qntx/spark/dot"
	"github.com/qntx/spark/nn"
)

// Three-layer neural network with automatic differentiation
// Structure: Input(2) -> Hidden(2, Sigmoid) -> Output(1, Sigmoid)
func main() {
	// Initialize data
	x := spark.NewOf([]float64{0.35, 0.9})
	x.Name = "X"
	target := 0.5

	// Initialize weights
	w0 := spark.NewOf(
		[]float64{0.1, 0.8},
		[]float64{0.4, 0.6},
	)
	w0.Name = "W0"
	w1 := spark.NewOf(
		[]float64{0.3},
		[]float64{0.9},
	)
	w1.Name = "W1"

	// Forward propagation
	z1 := spark.MatMul(x, w0)
	z1.Name = "Z1"
	y1 := nn.Sigmoid(z1)
	y1.Name = "Y1"
	z2 := spark.MatMul(y1, w1)
	z2.Name = "Z2"
	y2 := nn.Sigmoid(z2)
	y2.Name = "Y2"

	// Loss calculation
	diff := spark.SubC(target, y2)
	loss := spark.Mul(spark.New(0.5), spark.Mul(diff, diff))
	loss.Name = "Loss"
	lossValue := 0.5 * math.Pow(y2.At(0, 0)-target, 2)

	// Backward propagation
	w0.Cleargrad()
	w1.Cleargrad()
	loss.Backward()

	// Weight update
	lr := 0.1
	w0_new := spark.Sub(w0, spark.MulC(lr, w0.Grad))
	w1_new := spark.Sub(w1, spark.MulC(lr, w1.Grad))

	// Validation
	z1_new := spark.MatMul(x, w0_new)
	y1_new := nn.Sigmoid(z1_new)
	z2_new := spark.MatMul(y1_new, w1_new)
	y2_new := nn.Sigmoid(z2_new)
	newLoss := 0.5 * math.Pow(y2_new.At(0, 0)-target, 2)

	// Output results
	fmt.Printf("Input %s: %s\n", x.Name, x)
	fmt.Printf("Target: %.1f\n\n", target)
	fmt.Printf("Initial Weights %s: %s\n", w0.Name, w0)
	fmt.Printf("Initial Weights %s: %s\n\n", w1.Name, w1)
	fmt.Printf("Output %s: %.6f\n", y2.Name, y2.At(0, 0))
	fmt.Printf("Loss: %.6f\n\n", lossValue)
	fmt.Printf("Gradients %s: %s\n", w0.Name, w0.Grad)
	fmt.Printf("Gradients %s: %s\n\n", w1.Name, w1.Grad)
	fmt.Printf("Updated Weights %s: %s\n", w0.Name, w0_new)
	fmt.Printf("Updated Weights %s: %s\n\n", w1.Name, w1_new)
	fmt.Printf("New Output %s: %.6f\n", y2_new.Name, y2_new.At(0, 0))
	fmt.Printf("New Loss: %.6f\n\n", newLoss)

	dot.SaveGraph(loss, "graph.dot", dot.Opts{Verbose: true})
}

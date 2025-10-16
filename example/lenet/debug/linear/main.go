package main

import (
	"fmt"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/nn"
	"github.com/gocnn/spark/tensor"
)

func runCase(xPath, wPath, bPath, yPath string, inDim, outDim int) {
	x := tensor.MustReadNPY[float32](xPath)
	w := tensor.MustReadNPY[float32](wPath)
	b := tensor.MustReadNPY[float32](bPath)
	y := tensor.MustReadNPY[float32](yPath)

	fmt.Printf("Input: %v, Weight: %v, Bias: %v, Expected: %v\n", x.Dims(), w.Dims(), b.Dims(), y.Dims())

	// Manual: x @ W^T + b
	wt := w.MustTranspose(-1, -2)
	manual := x.MustMatMul(wt).MustBroadcastAdd(b)
	fmt.Printf("manual first_5: %v\n", manual.Data()[:5])

	// nn.Linear
	lin := nn.NewLinearLayer[float32](inDim, outDim, true, spark.CPU)
	copy(lin.Weight().Data(), w.Data())
	copy(lin.Bias().Data(), b.Data())
	nnOut := lin.MustForward(x)
	fmt.Printf("nn first_5: %v\n", nnOut.Data()[:5])

	// diffs
	mf, nf, ef := manual.Data(), nnOut.Data(), y.Data()
	var dManExp, dNNExp, dManNN float32
	for i := range ef {
		if t := abs32(mf[i] - ef[i]); t > dManExp {
			dManExp = t
		}
		if t := abs32(nf[i] - ef[i]); t > dNNExp {
			dNNExp = t
		}
		if t := abs32(mf[i] - nf[i]); t > dManNN {
			dManNN = t
		}
	}
	fmt.Printf("max_abs_diff manual vs expected: %g\n", dManExp)
	fmt.Printf("max_abs_diff nn vs expected: %g\n", dNNExp)
	fmt.Printf("max_abs_diff manual vs nn: %g\n", dManNN)
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func main() {
	fmt.Println("=== FC1 ===")
	runCase("real_input_fc1.npy", "real_weight_fc1.npy", "real_bias_fc1.npy", "real_output_fc1.npy", 9216, 128)

	fmt.Println("\n=== FC2 ===")
	runCase("real_input_fc2.npy", "real_weight_fc2.npy", "real_bias_fc2.npy", "real_output_fc2.npy", 128, 10)
}

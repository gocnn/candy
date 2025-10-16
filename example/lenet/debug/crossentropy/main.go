package main

import (
	"fmt"

	"github.com/gocnn/spark/nn/loss"
	"github.com/gocnn/spark/tensor"
)

func main() {
	x := tensor.MustReadNPY[float32]("x.npy")         // [B,C]
	y := tensor.MustReadNPY[float32]("y.npy")         // [B]
	lr := tensor.MustReadNPY[float32]("loss_ref.npy") // [1]

	ls := loss.MustCrossEntropy(x, y)

	got := ls.Data()[0]
	ref := lr.Data()[0]
	diff := got - ref
	if diff < 0 {
		diff = -diff
	}

	fmt.Printf("x shape: %v, y shape: %v\n", x.Shape().Dims(), y.Shape().Dims())
	fmt.Printf("loss: %.9g, ref: %.9g, abs_diff: %.6g\n", got, ref, diff)
}

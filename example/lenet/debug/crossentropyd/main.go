package main

import (
	"fmt"

	"github.com/gocnn/spark/nn/loss"
	"github.com/gocnn/spark/tensor"
)

func maxAbsDiff(a, b []float32) float32 {
	m := float32(0)
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

func main() {
	x := tensor.MustReadNPY[float32]("x.npy") // [B,C]
	x.SetIsVar(true)
	y := tensor.MustReadNPY[float32]("y.npy") // [B]

	ls := loss.MustCrossEntropy(x, y)
	gs := ls.MustBackward()
	dx := gs.Get(x)

	dxRef := tensor.MustReadNPY[float32]("dx_ref.npy")

	fmt.Printf("dx shape: %v, dxRef shape: %v\n", dx.Shape().Dims(), dxRef.Shape().Dims())
	fmt.Printf("max_abs_diff dx: %.6g\n", maxAbsDiff(dx.Data(), dxRef.Data()))
}

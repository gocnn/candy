package main

import (
	"fmt"
	"math"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/nn/optim"
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

func testManualStep() {
	x := tensor.MustFull[float32](2.0, spark.NewShape(2), spark.CPU).RequiresGrad()
	g := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)
	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{x}, 0.1)
	s.MustStep(gs)
	expected := []float32{1.9, 1.9}
	got := x.Data()
	fmt.Printf("manual step: got=%v expected=%v diff=%.6g\n", got, expected, maxAbsDiff(got, expected))
}

func testOptimizeMSE() {
	x := tensor.MustFull[float32](3.0, spark.NewShape(2), spark.CPU).RequiresGrad()
	y := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	loss := x.MustSub(y).MustPowf(2).MustSum([]int{0})
	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{x}, 0.1)
	s.MustOptimize(loss)
	expected := []float32{2.6, 2.6}
	got := x.Data()
	fmt.Printf("optimize MSE: got=%v expected=%v diff=%.6g\n", got, expected, maxAbsDiff(got, expected))
}

func main() {
	fmt.Println("SGD accuracy tests")
	testManualStep()
	testOptimizeMSE()
	_ = math.SmallestNonzeroFloat32
}

package main

import (
	"fmt"

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
	x := tensor.MustReadNPY[float32]("x_step.npy").RequiresGrad()
	g := tensor.MustReadNPY[float32]("g_step.npy")
	xRef := tensor.MustReadNPY[float32]("x_step_ref.npy")

	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)

	p := optim.DefaultAdamWParams()
	p.LearningRate = 0.01
	p.Beta1 = 0.9
	p.Beta2 = 0.999
	p.Epsilon = 1e-8
	p.WeightDecay = 0.01
	opt, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)
	opt.MustStep(gs)

	got := x.Data()
	exp := xRef.Data()
	fmt.Printf("manual step: got=%v expected=%v diff=%.6g\n", got, exp, maxAbsDiff(got, exp))
}

func testOptimizeMSE() {
	x := tensor.MustReadNPY[float32]("x_opt.npy").RequiresGrad()
	y := tensor.MustReadNPY[float32]("y_opt.npy")
	xRef := tensor.MustReadNPY[float32]("x_opt_ref.npy")

	p := optim.DefaultAdamWParams()
	p.LearningRate = 0.01
	p.Beta1 = 0.9
	p.Beta2 = 0.999
	p.Epsilon = 1e-8
	p.WeightDecay = 0.01
	opt, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)

	loss := x.MustSub(y).MustPowf(2).MustSumAll()
	opt.MustOptimize(loss)

	got := x.Data()
	exp := xRef.Data()
	fmt.Printf("optimize MSE: got=%v expected=%v diff=%.6g\n", got, exp, maxAbsDiff(got, exp))
}

func main() {
	fmt.Println("AdamW accuracy tests")
	testManualStep()
	testOptimizeMSE()
}

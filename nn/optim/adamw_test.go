package optim_test

import (
	"math"
	"testing"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/nn/optim"
	"github.com/gocnn/spark/tensor"
)

func TestDefaultAdamWParams(t *testing.T) {
	p := optim.DefaultAdamWParams()
	if p.LearningRate != 0.001 {
		t.Errorf("got lr %f, want 0.001", p.LearningRate)
	}
	if p.Beta1 != 0.9 {
		t.Errorf("got beta1 %f, want 0.9", p.Beta1)
	}
	if p.Beta2 != 0.999 {
		t.Errorf("got beta2 %f, want 0.999", p.Beta2)
	}
	if p.Epsilon != 1e-8 {
		t.Errorf("got eps %e, want 1e-8", p.Epsilon)
	}
	if p.WeightDecay != 0.01 {
		t.Errorf("got wd %f, want 0.01", p.WeightDecay)
	}
}

func TestNewAdamW(t *testing.T) {
	x := tensor.MustOnes[float32](spark.NewShape(2, 3), spark.CPU).RequiresGrad()
	y := tensor.MustOnes[float32](spark.NewShape(2, 3), spark.CPU)

	p := optim.DefaultAdamWParams()
	a, err := optim.NewAdamW([]*tensor.Tensor[float32]{x, y}, p)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Vars()) != 1 {
		t.Errorf("got %d vars, want 1", len(a.Vars()))
	}
	if a.LearningRate() != 0.001 {
		t.Errorf("got lr %f, want 0.001", a.LearningRate())
	}
}

func TestNewAdamWWithLR(t *testing.T) {
	x := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU).RequiresGrad()

	a, err := optim.NewAdamWWithLR([]*tensor.Tensor[float32]{x}, 0.01)
	if err != nil {
		t.Fatal(err)
	}
	if a.LearningRate() != 0.01 {
		t.Errorf("got lr %f, want 0.01", a.LearningRate())
	}
}

func TestAdamWParams(t *testing.T) {
	p1 := optim.AdamWParams{LearningRate: 0.002, Beta1: 0.8, Beta2: 0.99, Epsilon: 1e-7, WeightDecay: 0.02}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{}, p1)

	p2 := a.Params()
	if p2.LearningRate != 0.002 {
		t.Errorf("got lr %f, want 0.002", p2.LearningRate)
	}

	p3 := optim.AdamWParams{LearningRate: 0.005}
	a.SetParams(p3)
	if a.LearningRate() != 0.005 {
		t.Errorf("got lr %f, want 0.005", a.LearningRate())
	}
}

func TestAdamWLearningRate(t *testing.T) {
	a, _ := optim.NewAdamWWithLR([]*tensor.Tensor[float32]{}, 0.001)

	if a.LearningRate() != 0.001 {
		t.Errorf("got %f, want 0.001", a.LearningRate())
	}

	a.SetLearningRate(0.01)
	if a.LearningRate() != 0.01 {
		t.Errorf("got %f, want 0.01", a.LearningRate())
	}
}

func TestAdamWAdd(t *testing.T) {
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{}, optim.DefaultAdamWParams())

	x := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	err := a.Add(x)
	if err == nil {
		t.Error("expected error for non-var")
	}

	x.SetIsVar(true)
	err = a.Add(x)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Vars()) != 1 {
		t.Errorf("got %d vars, want 1", len(a.Vars()))
	}
}

func TestAdamWStep(t *testing.T) {
	x := tensor.MustFull[float32](2.0, spark.NewShape(2), spark.CPU)
	x.SetIsVar(true)

	p := optim.AdamWParams{LearningRate: 0.1, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.0}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)

	g := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)

	err := a.Step(gs)
	if err != nil {
		t.Fatal(err)
	}

	d := x.Data()
	expected := float32(2.0 - 0.1*1.0/math.Sqrt(1.0+1e-8))
	for _, v := range d {
		if math.Abs(float64(v-expected)) > 1e-6 {
			t.Errorf("got %f, want %f", v, expected)
		}
	}
}

func TestAdamWStepNoGrad(t *testing.T) {
	x := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	x.SetIsVar(true)

	a, _ := optim.NewAdamWWithLR([]*tensor.Tensor[float32]{x}, 0.1)
	gs := tensor.NewGradStore[float32]()

	err := a.Step(gs)
	if err != nil {
		t.Fatal(err)
	}

	d := x.Data()
	for _, v := range d {
		if v != 1.0 {
			t.Errorf("got %f, want 1.0", v)
		}
	}
}

func TestAdamWOptimize(t *testing.T) {
	x := tensor.MustFull[float32](3.0, spark.NewShape(2), spark.CPU)
	x.SetIsVar(true)

	y := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	loss := x.MustSub(y).MustSum([]int{0})

	p := optim.AdamWParams{LearningRate: 0.1, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.0}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)

	err := a.Optimize(loss)
	if err != nil {
		t.Fatal(err)
	}

	d := x.Data()
	expected := float32(3.0 - 0.1*1.0/math.Sqrt(1.0+1e-8))
	for _, v := range d {
		if math.Abs(float64(v-expected)) > 1e-6 {
			t.Errorf("got %f, want %f", v, expected)
		}
	}
}

func TestAdamWMustOptimize(t *testing.T) {
	x := tensor.MustFull[float32](5.0, spark.NewShape(2), spark.CPU)
	x.SetIsVar(true)

	y := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	loss := x.MustSub(y).MustSum([]int{0})

	p := optim.AdamWParams{LearningRate: 0.05, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.0}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)

	a.MustOptimize(loss)

	d := x.Data()
	expected := float32(5.0 - 0.05*1.0/math.Sqrt(1.0+1e-8))
	for _, v := range d {
		if math.Abs(float64(v-expected)) > 1e-6 {
			t.Errorf("got %f, want %f", v, expected)
		}
	}
}

func TestAdamWMustStep(t *testing.T) {
	x := tensor.MustFull[float32](4.0, spark.NewShape(3), spark.CPU)
	x.SetIsVar(true)

	p := optim.AdamWParams{LearningRate: 0.2, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.0}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)

	g := tensor.MustOnes[float32](spark.NewShape(3), spark.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)

	a.MustStep(gs)

	d := x.Data()
	expected := float32(4.0 - 0.2*1.0/math.Sqrt(1.0+1e-8))
	for _, v := range d {
		if math.Abs(float64(v-expected)) > 1e-6 {
			t.Errorf("got %f, want %f", v, expected)
		}
	}
}

func TestAdamWFloat64(t *testing.T) {
	x := tensor.MustFull[float64](1.5, spark.NewShape(2), spark.CPU)
	x.SetIsVar(true)

	p := optim.AdamWParams{LearningRate: 0.1, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.0}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float64]{x}, p)

	g := tensor.MustOnes[float64](spark.NewShape(2), spark.CPU)
	gs := tensor.NewGradStore[float64]()
	gs.Set(x, g)

	err := a.Step(gs)
	if err != nil {
		t.Fatal(err)
	}

	d := x.Data()
	expected := 1.5 - 0.1*1.0/math.Sqrt(1.0+1e-8)
	for _, v := range d {
		if math.Abs(v-expected) > 1e-6 {
			t.Errorf("got %f, want %f", v, expected)
		}
	}
}

func TestAdamWWeightDecay(t *testing.T) {
	x := tensor.MustFull[float32](2.0, spark.NewShape(2), spark.CPU)
	x.SetIsVar(true)

	p := optim.AdamWParams{LearningRate: 0.1, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.1}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)

	g := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)

	err := a.Step(gs)
	if err != nil {
		t.Fatal(err)
	}

	d := x.Data()
	adamUpdate := float32(0.1 * 1.0 / math.Sqrt(1.0+1e-8))
	weightDecayUpdate := float32(0.1 * 0.1 * 2.0)
	expected := float32(2.0) - adamUpdate - weightDecayUpdate
	for _, v := range d {
		if math.Abs(float64(v-expected)) > 1e-6 {
			t.Errorf("got %f, want %f", v, expected)
		}
	}
}

package optim_test

import (
	"math"
	"slices"
	"testing"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/nn/optim"
	"github.com/gocnn/candy/tensor"
)

func TestNewSGD(t *testing.T) {
	x := tensor.MustOnes[float32](candy.NewShape(2, 3), candy.CPU).RequiresGrad()
	y := tensor.MustOnes[float32](candy.NewShape(2, 3), candy.CPU)
	s, err := optim.NewSGD([]*tensor.Tensor[float32]{x, y}, 0.01)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(s.Vars()), 1; got != want {
		t.Errorf("got %d vars, want %d", got, want)
	}
	if got, want := s.LearningRate(), 0.01; got != want {
		t.Errorf("got lr %f, want %f", got, want)
	}
}

func TestSGDLearningRate(t *testing.T) {
	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{}, 0.01)
	if got, want := s.LearningRate(), 0.01; got != want {
		t.Errorf("got %f, want %f", got, want)
	}
	s.SetLearningRate(0.001)
	if got, want := s.LearningRate(), 0.001; got != want {
		t.Errorf("got %f, want %f", got, want)
	}
}

func TestSGDAdd(t *testing.T) {
	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{}, 0.01)
	x := tensor.MustOnes[float32](candy.NewShape(2), candy.CPU)
	if err := s.Add(x); err == nil {
		t.Error("expected error for non-var")
	}
	x.SetIsVar(true)
	if err := s.Add(x); err != nil {
		t.Fatal(err)
	}
	if got, want := len(s.Vars()), 1; got != want {
		t.Errorf("got %d vars, want %d", got, want)
	}
}

func TestSGDStep(t *testing.T) {
	x := tensor.MustFull[float32](2.0, candy.NewShape(2), candy.CPU).RequiresGrad()
	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{x}, 0.1)
	g := tensor.MustOnes[float32](candy.NewShape(2), candy.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)
	if err := s.Step(gs); err != nil {
		t.Fatal(err)
	}
	got := x.Data()
	want := []float32{1.9, 1.9}
	if !slices.EqualFunc(got, want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestSGDStepNoGrad(t *testing.T) {
	x := tensor.MustOnes[float32](candy.NewShape(2, 3), candy.CPU).RequiresGrad()
	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{x}, 0.1)
	gs := tensor.NewGradStore[float32]()
	if err := s.Step(gs); err != nil {
		t.Fatal(err)
	}
	got := x.Data()
	want := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	if !slices.EqualFunc(got, want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestSGDOptimize(t *testing.T) {
	x := tensor.MustFull[float32](3.0, candy.NewShape(2), candy.CPU).RequiresGrad()
	y := tensor.MustOnes[float32](candy.NewShape(2), candy.CPU)
	loss := x.MustSub(y).MustPowf(2).MustSum([]int{0})
	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{x}, 0.1)
	if err := s.Optimize(loss); err != nil {
		t.Fatal(err)
	}
	got := x.Data()
	want := []float32{2.6, 2.6}
	if !slices.EqualFunc(got, want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestSGDMustStep(t *testing.T) {
	x := tensor.MustFull[float32](4.0, candy.NewShape(3), candy.CPU).RequiresGrad()
	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{x}, 0.2)
	g := tensor.MustOnes[float32](candy.NewShape(3), candy.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)
	s.MustStep(gs)
	got := x.Data()
	want := []float32{3.8, 3.8, 3.8}
	if !slices.EqualFunc(got, want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestSGDFloat64(t *testing.T) {
	x := tensor.MustFull[float64](1.5, candy.NewShape(2), candy.CPU).RequiresGrad()
	s, _ := optim.NewSGD([]*tensor.Tensor[float64]{x}, 0.1)
	g := tensor.MustOnes[float64](candy.NewShape(2), candy.CPU)
	gs := tensor.NewGradStore[float64]()
	gs.Set(x, g)
	if err := s.Step(gs); err != nil {
		t.Fatal(err)
	}
	got := x.Data()
	want := []float64{1.4, 1.4}
	if !slices.EqualFunc(got, want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
		t.Errorf("got %v, want %v", got, want)
	}
}

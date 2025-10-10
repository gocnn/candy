package optim_test

import (
	"testing"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/nn/optim"
	"github.com/gocnn/spark/tensor"
)

func TestNewSGD(t *testing.T) {
	x := tensor.MustOnes[float32](spark.NewShape(2, 3), spark.CPU)
	x.SetIsVar(true)
	y := tensor.MustOnes[float32](spark.NewShape(2, 3), spark.CPU)

	s, err := optim.NewSGD([]*tensor.Tensor[float32]{x, y}, 0.01)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Vars()) != 1 {
		t.Errorf("got %d vars, want 1", len(s.Vars()))
	}
	if s.LearningRate() != 0.01 {
		t.Errorf("got lr %f, want 0.01", s.LearningRate())
	}
}

func TestSGDLearningRate(t *testing.T) {
	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{}, 0.01)

	if s.LearningRate() != 0.01 {
		t.Errorf("got %f, want 0.01", s.LearningRate())
	}

	s.SetLearningRate(0.001)
	if s.LearningRate() != 0.001 {
		t.Errorf("got %f, want 0.001", s.LearningRate())
	}
}

func TestSGDAdd(t *testing.T) {
	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{}, 0.01)

	x := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	err := s.Add(x)
	if err == nil {
		t.Error("expected error for non-var")
	}

	x.SetIsVar(true)
	err = s.Add(x)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Vars()) != 1 {
		t.Errorf("got %d vars, want 1", len(s.Vars()))
	}
}

func TestSGDStep(t *testing.T) {
	x := tensor.MustFull[float32](2.0, spark.NewShape(2), spark.CPU)
	x.SetIsVar(true)

	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{x}, 0.1)

	g := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)

	err := s.Step(gs)
	if err != nil {
		t.Fatal(err)
	}

	d := x.Data()
	for _, v := range d {
		if v != 1.9 {
			t.Errorf("got %f, want 1.9", v)
		}
	}
}

func TestSGDStepNoGrad(t *testing.T) {
	x := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	x.SetIsVar(true)

	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{x}, 0.1)
	gs := tensor.NewGradStore[float32]()

	err := s.Step(gs)
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

func TestSGDOptimize(t *testing.T) {
	x := tensor.MustFull[float32](3.0, spark.NewShape(2), spark.CPU)
	x.SetIsVar(true)

	y := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	loss := x.MustSub(y).MustSum([]int{0})

	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{x}, 0.1)

	err := s.Optimize(loss)
	if err != nil {
		t.Fatal(err)
	}

	d := x.Data()
	for _, v := range d {
		if v != 2.9 {
			t.Errorf("got %f, want 2.9", v)
		}
	}
}

func TestSGDMustOptimize(t *testing.T) {
	x := tensor.MustFull[float32](5.0, spark.NewShape(2), spark.CPU)
	x.SetIsVar(true)

	y := tensor.MustOnes[float32](spark.NewShape(2), spark.CPU)
	loss := x.MustSub(y).MustSum([]int{0})

	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{x}, 0.05)

	s.MustOptimize(loss)

	d := x.Data()
	for _, v := range d {
		if v != 4.95 {
			t.Errorf("got %f, want 4.95", v)
		}
	}
}

func TestSGDMustStep(t *testing.T) {
	x := tensor.MustFull[float32](4.0, spark.NewShape(3), spark.CPU)
	x.SetIsVar(true)

	s, _ := optim.NewSGD([]*tensor.Tensor[float32]{x}, 0.2)

	g := tensor.MustOnes[float32](spark.NewShape(3), spark.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)

	s.MustStep(gs)

	d := x.Data()
	for _, v := range d {
		if v != 3.8 {
			t.Errorf("got %f, want 3.8", v)
		}
	}
}

func TestSGDFloat64(t *testing.T) {
	x := tensor.MustFull[float64](1.5, spark.NewShape(2), spark.CPU)
	x.SetIsVar(true)

	s, _ := optim.NewSGD([]*tensor.Tensor[float64]{x}, 0.1)

	g := tensor.MustOnes[float64](spark.NewShape(2), spark.CPU)
	gs := tensor.NewGradStore[float64]()
	gs.Set(x, g)

	err := s.Step(gs)
	if err != nil {
		t.Fatal(err)
	}

	d := x.Data()
	for _, v := range d {
		if v != 1.4 {
			t.Errorf("got %f, want 1.4", v)
		}
	}
}

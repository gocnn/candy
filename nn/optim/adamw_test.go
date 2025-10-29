package optim_test

import (
	"math"
	"slices"
	"testing"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/nn/optim"
	"github.com/gocnn/candy/tensor"
)

func TestNewAdamW(t *testing.T) {
	x := tensor.MustFull[float32](2.5, candy.NewShape(3, 4), candy.CPU).RequiresGrad()
	y := tensor.MustFull[float32](-1.0, candy.NewShape(3, 4), candy.CPU).RequiresGrad()
	p := optim.DefaultAdamWParams()
	a, err := optim.NewAdamW([]*tensor.Tensor[float32]{x, y}, p)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(a.Vars()), 2; got != want {
		t.Errorf("got %d vars, want %d", got, want)
	}
	if got, want := a.LearningRate(), 0.001; got != want {
		t.Errorf("got lr %f, want %f", got, want)
	}
}

func TestNewAdamWWithLR(t *testing.T) {
	x := tensor.MustFull[float32](0.5, candy.NewShape(2, 3), candy.CPU).RequiresGrad()
	a, err := optim.NewAdamWWithLR([]*tensor.Tensor[float32]{x}, 0.02)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := a.LearningRate(), 0.02; got != want {
		t.Errorf("got lr %f, want %f", got, want)
	}
}

func TestAdamWParams(t *testing.T) {
	p1 := optim.AdamWParams{LearningRate: 0.003, Beta1: 0.85, Beta2: 0.995, Epsilon: 1e-7, WeightDecay: 0.05}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{}, p1)
	p2 := a.Params()
	if got, want := p2.LearningRate, 0.003; got != want {
		t.Errorf("got lr %f, want %f", got, want)
	}
	p3 := optim.AdamWParams{LearningRate: 0.007, WeightDecay: 0.02}
	a.SetParams(p3)
	if got, want := a.LearningRate(), 0.007; got != want {
		t.Errorf("got lr %f, want %f", got, want)
	}
}

func TestAdamWLearningRate(t *testing.T) {
	a, _ := optim.NewAdamWWithLR([]*tensor.Tensor[float32]{}, 0.002)
	if got, want := a.LearningRate(), 0.002; got != want {
		t.Errorf("got %f, want %f", got, want)
	}
	a.SetLearningRate(0.015)
	if got, want := a.LearningRate(), 0.015; got != want {
		t.Errorf("got %f, want %f", got, want)
	}
}

func TestAdamWAdd(t *testing.T) {
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{}, optim.DefaultAdamWParams())
	x := tensor.MustFull[float32](1.5, candy.NewShape(2, 2), candy.CPU)
	if err := a.Add(x); err == nil {
		t.Error("expected error for non-var")
	}
	x.SetIsVar(true)
	if err := a.Add(x); err != nil {
		t.Fatal(err)
	}
	if got, want := len(a.Vars()), 1; got != want {
		t.Errorf("got %d vars, want %d", got, want)
	}
}

func TestAdamWStep(t *testing.T) {
	x := tensor.MustFull[float32](2.0, candy.NewShape(2), candy.CPU).RequiresGrad()
	p := optim.AdamWParams{LearningRate: 0.1, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.0}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)
	g := tensor.MustOnes[float32](candy.NewShape(2), candy.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)
	if err := a.Step(gs); err != nil {
		t.Fatal(err)
	}
	got := x.Data()
	want := []float32{1.899999976158142, 1.899999976158142}
	if !slices.EqualFunc(got, want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAdamWStepNoGrad(t *testing.T) {
	x := tensor.MustFull[float32](2.0, candy.NewShape(3, 2), candy.CPU).RequiresGrad()
	a, _ := optim.NewAdamWWithLR([]*tensor.Tensor[float32]{x}, 0.05)
	gs := tensor.NewGradStore[float32]()
	if err := a.Step(gs); err != nil {
		t.Fatal(err)
	}
	got := x.Data()
	for _, v := range got {
		if v != 2.0 {
			t.Errorf("got %f, want 2.0", v)
		}
	}
}

func TestAdamWStepMultipleTensors(t *testing.T) {
	x := tensor.MustFull[float32](4.0, candy.NewShape(2, 3), candy.CPU).RequiresGrad()
	y := tensor.MustFull[float32](-2.0, candy.NewShape(3, 2), candy.CPU).RequiresGrad()
	p := optim.AdamWParams{LearningRate: 0.05, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.01}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x, y}, p)
	gx := tensor.MustFull[float32](1.5, candy.NewShape(2, 3), candy.CPU)
	gy := tensor.MustFull[float32](-0.5, candy.NewShape(3, 2), candy.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, gx)
	gs.Set(y, gy)
	if err := a.Step(gs); err != nil {
		t.Fatal(err)
	}
	gotX := x.Data()
	gotY := y.Data()
	wantX := []float32{3.947999954223633, 3.947999954223633, 3.947999954223633, 3.947999954223633, 3.947999954223633, 3.947999954223633}
	wantY := []float32{-1.9490000009536743, -1.9490000009536743, -1.9490000009536743, -1.9490000009536743, -1.9490000009536743, -1.9490000009536743}
	if !slices.EqualFunc(gotX, wantX, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
		t.Errorf("x: got %v, want %v", gotX, wantX)
	}
	if !slices.EqualFunc(gotY, wantY, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
		t.Errorf("y: got %v, want %v", gotY, wantY)
	}
}

func TestAdamWOptimize(t *testing.T) {
	x := tensor.MustFull[float32](2.5, candy.NewShape(2, 2), candy.CPU).RequiresGrad()
	y := tensor.MustFull[float32](0.5, candy.NewShape(2, 2), candy.CPU)
	loss := x.MustSub(y).MustPowf(2).MustSum([]int{0})
	p := optim.AdamWParams{LearningRate: 0.05, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.0}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)
	if err := a.Optimize(loss); err != nil {
		t.Fatal(err)
	}
	got := x.Data()
	want := []float32{2.450000047683716, 2.450000047683716, 2.450000047683716, 2.450000047683716}
	if !slices.EqualFunc(got, want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAdamWMustStep(t *testing.T) {
	x := tensor.MustFull[float32](5.0, candy.NewShape(3, 2), candy.CPU).RequiresGrad()
	p := optim.AdamWParams{LearningRate: 0.15, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.0}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)
	g := tensor.MustFull[float32](2.5, candy.NewShape(3, 2), candy.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)
	a.MustStep(gs)
	got := x.Data()
	want := []float32{4.849999904632568, 4.849999904632568, 4.849999904632568, 4.849999904632568, 4.849999904632568, 4.849999904632568}
	if !slices.EqualFunc(got, want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAdamWFloat64(t *testing.T) {
	x := tensor.MustFull[float64](2.75, candy.NewShape(2, 3), candy.CPU).RequiresGrad()
	p := optim.AdamWParams{LearningRate: 0.05, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.0}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float64]{x}, p)
	g := tensor.MustFull[float64](1.25, candy.NewShape(2, 3), candy.CPU)
	gs := tensor.NewGradStore[float64]()
	gs.Set(x, g)
	if err := a.Step(gs); err != nil {
		t.Fatal(err)
	}
	got := x.Data()
	want := []float64{2.7000000004, 2.7000000004, 2.7000000004, 2.7000000004, 2.7000000004, 2.7000000004}
	if !slices.EqualFunc(got, want, func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAdamWWeightDecay(t *testing.T) {
	x := tensor.MustFull[float32](2.0, candy.NewShape(2), candy.CPU).RequiresGrad()
	p := optim.AdamWParams{LearningRate: 0.1, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.1}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)
	g := tensor.MustOnes[float32](candy.NewShape(2), candy.CPU)
	gs := tensor.NewGradStore[float32]()
	gs.Set(x, g)
	if err := a.Step(gs); err != nil {
		t.Fatal(err)
	}
	got := x.Data()
	want := []float32{1.8799999952316284, 1.8799999952316284}
	if !slices.EqualFunc(got, want, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestAdamWMultipleStepsComplex(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0}
	x := tensor.MustNew(data, candy.NewShape(2, 2), candy.CPU).RequiresGrad()
	p := optim.AdamWParams{LearningRate: 0.01, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: 0.01}
	a, _ := optim.NewAdamW([]*tensor.Tensor[float32]{x}, p)

	// Step 1
	g1 := []float32{0.5, 1.0, 1.5, 2.0}
	gs1 := tensor.NewGradStore[float32]()
	gs1.Set(x, tensor.MustNew(g1, candy.NewShape(2, 2), candy.CPU))
	if err := a.Step(gs1); err != nil {
		t.Fatal(err)
	}
	got1 := x.Data()
	want1 := []float32{0.9898999929428101, 1.989799976348877, 2.9897000789642334, 3.9895999431610107}
	if !slices.EqualFunc(got1, want1, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-5 }) {
		t.Errorf("step1: got %v, want %v", got1, want1)
	}

	// Step 2
	g2 := []float32{-0.5, 0.0, 0.5, 1.0}
	gs2 := tensor.NewGradStore[float32]()
	gs2.Set(x, tensor.MustNew(g2, candy.NewShape(2, 2), candy.CPU))
	if err := a.Step(gs2); err != nil {
		t.Fatal(err)
	}
	got2 := x.Data()
	want2 := []float32{0.9903272986412048, 1.9829003810882568, 2.9806904792785645, 3.979879140853882}
	if !slices.EqualFunc(got2, want2, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-5 }) {
		t.Errorf("step2: got %v, want %v", got2, want2)
	}

	// Step 3
	g3 := tensor.MustOnes[float32](candy.NewShape(2, 2), candy.CPU)
	gs3 := tensor.NewGradStore[float32]()
	gs3.Set(x, g3)
	if err := a.Step(gs3); err != nil {
		t.Fatal(err)
	}
	got3 := x.Data()
	want3 := []float32{0.9852458238601685, 1.9745219945907593, 2.9712862968444824, 3.970294237136841}
	if !slices.EqualFunc(got3, want3, func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-5 }) {
		t.Errorf("step3: got %v, want %v", got3, want3)
	}
}

package loss_test

import (
	"math"
	"testing"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/nn/loss"
	"github.com/gocnn/spark/tensor"
)

func TestNLL(t *testing.T) {
	x := tensor.MustNew([]float32{0.1, 0.2, 0.7, 0.3, 0.4, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.0}, spark.NewShape(3, 5), spark.CPU)
	y := tensor.MustNew([]float32{2, 3, 1}, spark.NewShape(3), spark.CPU)
	got, _ := loss.NLL(x, y)
	if got, want := got.Data()[0], -0.4333333; math.Abs(float64(got)-want) > 1e-6 {
		t.Errorf("got %f, want %f", got, want)
	}
}

func TestCrossEntropy(t *testing.T) {
	x := tensor.MustNew([]float32{1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 3.5, 4.5, 1.2, 2.2, 3.2, 4.2, 5.2}, spark.NewShape(3, 5), spark.CPU)
	y := tensor.MustNew([]float32{2, 3, 0}, spark.NewShape(3), spark.CPU)
	got, _ := loss.CrossEntropy(x, y)
	if got, want := got.Data()[0], 2.785248; math.Abs(float64(got)-want) > 1e-6 {
		t.Errorf("got %f, want %f", got, want)
	}
}

func TestMSE(t *testing.T) {
	x := tensor.MustNew([]float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, spark.NewShape(3, 3), spark.CPU)
	y := tensor.MustNew([]float32{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5}, spark.NewShape(3, 3), spark.CPU)
	got, _ := loss.MSE(x, y)
	if got, want := got.Data()[0], 0.2500; math.Abs(float64(got)-want) > 1e-6 {
		t.Errorf("got %f, want %f", got, want)
	}
}

func TestBCE(t *testing.T) {
	x := tensor.MustNew([]float32{1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5, -1.5, 0.0}, spark.NewShape(3, 3), spark.CPU)
	y := tensor.MustNew([]float32{1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5}, spark.NewShape(3, 3), spark.CPU)
	got, _ := loss.BCE(x, y)
	if got, want := got.Data()[0], 0.324945; math.Abs(float64(got)-want) > 1e-6 {
		t.Errorf("got %f, want %f", got, want)
	}
}

func TestL1Loss(t *testing.T) {
	x := tensor.MustNew([]float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, spark.NewShape(3, 3), spark.CPU)
	y := tensor.MustNew([]float32{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5}, spark.NewShape(3, 3), spark.CPU)
	got, _ := loss.L1Loss(x, y)
	if got, want := got.Data()[0], 0.5000; math.Abs(float64(got)-want) > 1e-6 {
		t.Errorf("got %f, want %f", got, want)
	}
}

func TestSmoothL1Loss(t *testing.T) {
	x := tensor.MustNew([]float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, spark.NewShape(3, 3), spark.CPU)
	y := tensor.MustNew([]float32{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5}, spark.NewShape(3, 3), spark.CPU)
	got, _ := loss.SmoothL1Loss(x, y, 1.0)
	if got, want := got.Data()[0], 0.1250; math.Abs(float64(got)-want) > 1e-6 {
		t.Errorf("got %f, want %f", got, want)
	}

	// Linear region
	x = tensor.MustNew([]float32{3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0}, spark.NewShape(3, 3), spark.CPU)
	y = tensor.MustNew([]float32{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5}, spark.NewShape(3, 3), spark.CPU)
	got, _ = loss.SmoothL1Loss(x, y, 1.0)
	if got, want := got.Data()[0], 2.0000; math.Abs(float64(got)-want) > 1e-6 {
		t.Errorf("got %f, want %f", got, want)
	}
}

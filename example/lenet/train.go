package main

import (
	"fmt"
	"time"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/dataset/mnist"
	"github.com/gocnn/candy/nn/loss"
	"github.com/gocnn/candy/nn/optim"
	"github.com/gocnn/candy/tensor"
)

const (
	batch  = 64
	epochs = 5
	lr     = 0.001
	gamma  = 0.7
)

type stepLR struct{ gamma float64 }

func (s stepLR) Step(o optim.Optimizer[float32]) { o.SetLearningRate(o.LearningRate() * s.gamma) }

func acc(z, y *tensor.Tensor[float32]) float64 {
	zd, yd := z.Data(), y.Data()
	n, c := z.Dim(0), z.Dim(1)
	var cr int
	for i := range n {
		mx, p := zd[i*c], 0
		for j := 1; j < c; j++ {
			if v := zd[i*c+j]; v > mx {
				mx, p = v, j
			}
		}
		if p == int(yd[i]) {
			cr++
		}
	}
	return float64(cr) / float64(n)
}

func train(m *LeNet[float32], o optim.Optimizer[float32], l *mnist.DataLoader[float32], epoch int) {
	es := time.Now()
	var tl, a float64
	n := 0
	for x, y := range l.AllTensors() {
		z := m.MustForward(x)
		ls := loss.MustCrossEntropy(z, y)
		o.MustOptimize(ls)
		tl += float64(ls.Data()[0])
		a += acc(z, y)
		n++
		if n%10 == 0 {
			fmt.Printf("  Batch %d/%d - Loss: %.4f, Acc: %.2f%%\n", n, l.Len(), tl/float64(n), 100*a/float64(n))
		}
	}
	fmt.Printf("Epoch %d/%d - Loss: %.4f, Acc: %.2f%%, Time: %v\n", epoch, epochs, tl/float64(n), 100*a/float64(n), time.Since(es))
}

func eval(m *LeNet[float32], ds *mnist.Dataset[float32], b int, d candy.Device) float64 {
	fmt.Println("Evaluating on test set...")
	l := ds.NewDataLoader(b, false, d)
	var a float64
	n := 0
	for x, y := range l.AllTensors() {
		z := m.MustForward(x)
		a += acc(z, y)
		n++
	}
	return a / float64(n)
}

func RunTrain(dir, out string) error {
	fmt.Println("Starting LeNet-5 Training on MNIST")

	d := candy.CPU
	tr, err := mnist.New[float32](dir, true, true)
	if err != nil {
		return err
	}
	te, err := mnist.New[float32](dir, false, true)
	if err != nil {
		return err
	}

	m := NewLeNet[float32](d)
	o, err := optim.NewAdamWWithLR(m.Parameters(), lr)
	if err != nil {
		return err
	}

	l := tr.NewDataLoader(batch, true, d)
	s := time.Now()
	sch := stepLR{gamma: gamma}
	for i := range epochs {
		train(m, o, l, i+1)
		fmt.Printf("Test Acc: %.2f%%\n", 100*eval(m, te, batch, d))
		sch.Step(o)
		l.Reset()
	}
	fmt.Printf("\nTraining completed in %v\n", time.Since(s))

	if err := m.Save(out); err != nil {
		return err
	}
	fmt.Println("Saved weights to", out)
	return nil
}

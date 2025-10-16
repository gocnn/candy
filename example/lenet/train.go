package main

import (
	"fmt"
	"time"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/dataset/mnist"
	"github.com/gocnn/spark/nn/loss"
	"github.com/gocnn/spark/nn/optim"
	"github.com/gocnn/spark/tensor"
)

const (
	batch  = 64
	epochs = 5
	lr     = 0.001
)

// func main() {
// 	fmt.Println("Starting LeNet-5 Training on MNIST")

// 	d := spark.CPU
// 	traindata, err := mnist.New[float32]("./data", true, true)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	testdata, err := mnist.New[float32]("./data", false, true)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	m := NewLeNet[float32](d)
// 	o, err := optim.NewAdamWWithLR(m.Parameters(), lr)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	train(m, o, traindata, epochs, batch, d)
// 	fmt.Printf("Final Test Accuracy: %.2f%%\n", 100*eval(m, testdata, batch, d))
// 	if err := m.Save("lenet.npz"); err != nil {
// 		log.Fatal(err)
// 	}
// 	fmt.Println("Saved weights to lenet.npz")
// }

func train(m *LeNet[float32], o optim.Optimizer[float32], t *mnist.Dataset[float32], e, b int, d spark.Device) {
	l := t.NewDataLoader(b, true, d)
	s := time.Now()
	for i := range e {
		es := time.Now()
		var tl, a float64
		n := 0
		for x, y := range l.AllTensors() {
			z := m.MustForward(x)
			ls := loss.MustCrossEntropy(z, y)
			gs := ls.MustBackward()
			o.MustStep(gs)

			tl += float64(ls.Data()[0])
			a += acc(z, y)
			n++
			if n%10 == 0 {
				fmt.Printf("  Batch %d/%d - Loss: %.4f, Acc: %.2f%%\n", n, l.Len(), tl/float64(n), 100*a/float64(n))
			}
		}
		fmt.Printf("Epoch %d/%d - Loss: %.4f, Acc: %.2f%%, Time: %v\n", i+1, e, tl/float64(n), 100*a/float64(n), time.Since(es))
		l.Reset()
	}
	fmt.Printf("\nTraining completed in %v\n", time.Since(s))
}

func eval(m *LeNet[float32], t *mnist.Dataset[float32], b int, d spark.Device) float64 {
	fmt.Println("Evaluating on test set...")
	l := t.NewDataLoader(b, false, d)
	var a float64
	n := 0
	for x, y := range l.AllTensors() {
		z := m.MustForward(x)
		a += acc(z, y)
		n++
	}
	return a / float64(n)
}

func acc[T spark.D](z, y *tensor.Tensor[T]) float64 {
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

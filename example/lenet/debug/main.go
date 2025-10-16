package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/nn"
	"github.com/gocnn/spark/tensor"
)

func mustDims(t *tensor.Tensor[float32], want ...int) {
	d := t.Dims()
	if len(d) != len(want) {
		log.Fatalf("shape mismatch %v vs %v", d, want)
	}
	for i := range d {
		if d[i] != want[i] {
			log.Fatalf("shape mismatch %v vs %v", d, want)
		}
	}
}

func mustKey(m map[string]*tensor.Tensor[float32], k string) *tensor.Tensor[float32] {
	t, ok := m[k]
	if !ok || t == nil {
		log.Fatalf("missing key %s in weights", k)
	}
	return t
}

func write(outDir, name string, t *tensor.Tensor[float32]) {
	path := filepath.Join(outDir, name)
	t.MustWriteNPY(path)
}

func main() {
	var weightsPath, inputPath, outDir string
	flag.StringVar(&weightsPath, "weights", "artifacts/weights.npz", "path to weights.npz")
	flag.StringVar(&inputPath, "input", "artifacts/py_out/00_input.npy", "path to input.npy")
	flag.StringVar(&outDir, "out", "artifacts/go_out", "output directory for activations")
	flag.Parse()

	if err := os.MkdirAll(outDir, 0755); err != nil {
		log.Fatal(err)
	}

	// Load weights
	wmap := tensor.MustReadNPZ(weightsPath)

	// Build same network as Python:
	// conv1: in=1, out=32, k=3, stride=1, pad=0
	// conv2: in=32, out=64, k=3, stride=1, pad=0
	// maxpool: k=2,s=2
	// flatten -> fc1: 9216->128 -> relu -> fc2: 128->10 -> log_softmax
	c1 := nn.NewConv2d[float32](1, 32, 3, 1, 0, spark.CPU)
	c2 := nn.NewConv2d[float32](32, 64, 3, 1, 0, spark.CPU)
	f1 := nn.NewLinearLayer[float32](9216, 128, true, spark.CPU)
	f2 := nn.NewLinearLayer[float32](128, 10, true, spark.CPU)

	c1w, c1b := c1.Parameters()[0], c1.Parameters()[1]
	wc1 := mustKey(wmap, "conv1.weight")
	bc1 := mustKey(wmap, "conv1.bias")
	mustDims(wc1, 32, 1, 3, 3)
	mustDims(bc1, 32)
	copy(c1w.Data(), wc1.Data())
	copy(c1b.Data(), bc1.Data())

	c2w, c2b := c2.Parameters()[0], c2.Parameters()[1]
	wc2 := mustKey(wmap, "conv2.weight")
	bc2 := mustKey(wmap, "conv2.bias")
	mustDims(wc2, 64, 32, 3, 3)
	mustDims(bc2, 64)
	copy(c2w.Data(), wc2.Data())
	copy(c2b.Data(), bc2.Data())

	f1w, f1b := f1.Weight(), f1.Bias()
	wf1 := mustKey(wmap, "fc1.weight")
	bf1 := mustKey(wmap, "fc1.bias")
	mustDims(wf1, 128, 9216)
	mustDims(bf1, 128)
	copy(f1w.Data(), wf1.Data())
	copy(f1b.Data(), bf1.Data())

	f2w, f2b := f2.Weight(), f2.Bias()
	wf2 := mustKey(wmap, "fc2.weight")
	bf2 := mustKey(wmap, "fc2.bias")
	mustDims(wf2, 10, 128)
	mustDims(bf2, 10)
	copy(f2w.Data(), wf2.Data())
	copy(f2b.Data(), bf2.Data())

	// Debug: Print weight statistics
	fmt.Printf("=== Go Weight Analysis ===\n")
	fmt.Printf("conv1.weight: shape=%v, first_5=%v\n", wc1.Dims(), wc1.Data()[:5])
	fmt.Printf("conv1.bias: shape=%v, first_5=%v\n", bc1.Dims(), bc1.Data()[:5])
	fmt.Printf("conv2.weight: shape=%v, first_5=%v\n", wc2.Dims(), wc2.Data()[:5])
	fmt.Printf("conv2.bias: shape=%v, first_5=%v\n", bc2.Dims(), bc2.Data()[:5])

	// Load input
	x := tensor.MustReadNPY[float32](inputPath)
	mustDims(x, 1, 1, 28, 28)

	write(outDir, "00_input.npy", x)

	x1 := c1.MustForward(x)
	mustDims(x1, 1, 32, 26, 26)
	write(outDir, "10_conv1_out.npy", x1)

	x2 := x1.MustRelu()
	write(outDir, "11_relu1_out.npy", x2)

	x3 := c2.MustForward(x2)
	mustDims(x3, 1, 64, 24, 24)
	write(outDir, "20_conv2_out.npy", x3)

	x4 := x3.MustRelu()
	write(outDir, "21_relu2_out.npy", x4)

	x5 := x4.MustMaxPool2d(2, 2, 2, 2)
	mustDims(x5, 1, 64, 12, 12)
	write(outDir, "30_maxpool_out.npy", x5)

	// Flatten
	b := x5.Dim(0)
	x7 := x5.MustReshape(b, -1)
	mustDims(x7, b, 9216)
	write(outDir, "40_flatten_out.npy", x7)

	x8 := f1.MustForward(x7)
	mustDims(x8, b, 128)
	write(outDir, "50_fc1_out.npy", x8)

	x9 := x8.MustRelu()
	write(outDir, "51_relu3_out.npy", x9)

	x11 := f2.MustForward(x9)
	mustDims(x11, b, 10)
	write(outDir, "60_fc2_out.npy", x11)

	x12 := x11.MustLogSoftmax(1)
	mustDims(x12, b, 10)
	write(outDir, "70_logsoftmax_out.npy", x12)

	fmt.Println("Done. Go activations written to:", outDir)
}

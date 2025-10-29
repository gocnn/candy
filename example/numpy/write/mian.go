package main

import (
	"github.com/gocnn/candy"
	"github.com/gocnn/candy/tensor"
)

func main() {
	// Write f32.npy
	tA := tensor.MustNew([]float32{-1.5, 0.0, 3.14, 2.7, -0.8, 1.2}, candy.NewShape(2, 3), candy.CPU)
	tA.MustWriteNPY("f32.npy")

	// Write f64.npy
	tB := tensor.MustNew([]float64{100.5, -50.25, 0.333}, candy.NewShape(3), candy.CPU)
	tB.MustWriteNPY("f64.npy")

	// Write u8.npy
	tC := tensor.MustNew([]uint8{255, 128, 0}, candy.NewShape(3), candy.CPU)
	tC.MustWriteNPY("u8.npy")

	// Write u32.npy
	tD := tensor.MustNew([]uint32{1000000, 500, 42}, candy.NewShape(3), candy.CPU)
	tD.MustWriteNPY("u32.npy")

	// Write i64.npy
	tE := tensor.MustNew([]int64{-9876543210, 0, 1234567890}, candy.NewShape(3), candy.CPU)
	tE.MustWriteNPY("i64.npy")

	// Write pack.npz with keys
	tF := tensor.MustNew([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, candy.NewShape(2, 3), candy.CPU)
	tG := tensor.MustNew([]float32{10.0, 20.0, 30.0, 40.0, 50.0, 60.0}, candy.NewShape(2, 3), candy.CPU)
	tensor.MustWriteNPZ("pack.npz", map[string]*tensor.Tensor[float32]{
		"matrix_A": tA,
		"matrix_F": tF,
		"matrix_G": tG,
	})
}

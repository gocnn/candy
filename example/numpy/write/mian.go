package main

import (
	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

func main() {
	// Write f32.npy
	tA := tensor.MustNew([]float32{-1.5, 0.0, 3.14, 2.7, -0.8, 1.2}, spark.NewShape(2, 3), spark.CPU)
	tA.MustWriteNPY("f32.npy")

	// Write f64.npy
	tB := tensor.MustNew([]float64{100.5, -50.25, 0.333}, spark.NewShape(3), spark.CPU)
	tB.MustWriteNPY("f64.npy")

	// Write u8.npy
	tC := tensor.MustNew([]uint8{255, 128, 0}, spark.NewShape(3), spark.CPU)
	tC.MustWriteNPY("u8.npy")

	// Write u32.npy
	tD := tensor.MustNew([]uint32{1000000, 500, 42}, spark.NewShape(3), spark.CPU)
	tD.MustWriteNPY("u32.npy")

	// Write i64.npy
	tE := tensor.MustNew([]int64{-9876543210, 0, 1234567890}, spark.NewShape(3), spark.CPU)
	tE.MustWriteNPY("i64.npy")

	// Write pack.npz with keys
	tF := tensor.MustNew([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, spark.NewShape(2, 3), spark.CPU)
	tG := tensor.MustNew([]float32{10.0, 20.0, 30.0, 40.0, 50.0, 60.0}, spark.NewShape(2, 3), spark.CPU)
	tensor.MustWriteNPZ("pack.npz", map[string]*tensor.Tensor[float32]{
		"matrix_A": tA,
		"matrix_F": tF,
		"matrix_G": tG,
	})
}

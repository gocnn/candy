package main

import (
	"fmt"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

func maxAbsDiff(a, b []float32) float32 {
	m := float32(0)
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

func main() {
	// Load data produced by convb/main.py
	x := tensor.MustReadNPY[float32]("x.npy")
	w := tensor.MustReadNPY[float32]("w.npy")
	b := tensor.MustReadNPY[float32]("b.npy")
	// Mark as variables to enable gradient tracking
	x.SetIsVar(true)
	w.SetIsVar(true)
	b.SetIsVar(true)

	B, Cin, H, W := x.Dim(0), x.Dim(1), x.Dim(2), x.Dim(3)
	Cout, KH, KW := w.Dim(0), w.Dim(2), w.Dim(3)

	// Build conv2d params
	p := &spark.Conv2DParams{Batch: B, InH: H, InW: W, KH: KH, KW: KW, OutCh: Cout, InCh: Cin, Pad: 1, Stride: 1, Dilate: 1}

	// Forward: y = conv2d(x, w) + b
	y, err := x.Conv2d(w, p)
	if err != nil {
		panic(err)
	}
	br, err := b.Reshape(1, Cout, 1, 1)
	if err != nil {
		panic(err)
	}
	y, err = y.BroadcastAdd(br)
	if err != nil {
		panic(err)
	}

	// Loss = sum(y)
	loss := y.MustSumAll()

	// Backward
	gs := loss.MustBackward()
	dx := gs.Get(x)
	dw := gs.Get(w)
	db := gs.Get(b)

	// Load references
	dxRef := tensor.MustReadNPY[float32]("dx_ref.npy")
	dwRef := tensor.MustReadNPY[float32]("dw_ref.npy")
	dbRef := tensor.MustReadNPY[float32]("db_ref.npy")

	if dx == nil {
		fmt.Println("dx is nil (x not marked as variable?)")
	} else {
		fmt.Printf("dx shape: %v, dxRef shape: %v\n", dx.Shape().Dims(), dxRef.Shape().Dims())
		fmt.Printf("max_abs_diff dx: %.6g\n", maxAbsDiff(dx.Data(), dxRef.Data()))
	}
	if dw == nil {
		fmt.Println("dw is nil (w not marked as variable?)")
	} else {
		fmt.Printf("dw shape: %v, dwRef shape: %v\n", dw.Shape().Dims(), dwRef.Shape().Dims())
		fmt.Printf("max_abs_diff dw: %.6g\n", maxAbsDiff(dw.Data(), dwRef.Data()))
	}
	if db == nil {
		fmt.Println("db is nil (b not marked as variable?)")
	} else {
		fmt.Printf("db shape: %v, dbRef shape: %v\n", db.Shape().Dims(), dbRef.Shape().Dims())
		fmt.Printf("max_abs_diff db: %.6g\n", maxAbsDiff(db.Data(), dbRef.Data()))
	}
}

package tensor_test

import (
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/tensor"
)

func TestNPYRoundTripF32(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "f32.npy")
	data := []float32{0, 1.5, -2.25, 3.75}
	shape := candy.NewShapeFrom([]int{2, 2})
	tx, err := tensor.New(data, shape, candy.CPU)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.WriteNPY(path); err != nil {
		t.Fatalf("WriteNPY: %v", err)
	}
	tr, err := tensor.ReadNPY[float32](path)
	if err != nil {
		t.Fatalf("ReadNPY: %v", err)
	}
	if !tr.Shape().Equal(shape) {
		t.Fatalf("shape mismatch: got %v want %v", tr.Shape(), shape)
	}
	got := tr.Data()
	if len(got) != len(data) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(data))
	}

	if !slices.Equal(got, data) {
		t.Fatalf("data mismatch: got %v want %v", got, data)
	}
}

func TestNPYRoundTripF64(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "f64.npy")
	data := []float64{math.Pi, -1e-3, 7.0}
	shape := candy.NewShapeFrom([]int{3})
	tx, err := tensor.New(data, shape, candy.CPU)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.WriteNPY(path); err != nil {
		t.Fatalf("WriteNPY: %v", err)
	}
	tr, err := tensor.ReadNPY[float64](path)
	if err != nil {
		t.Fatalf("ReadNPY: %v", err)
	}
	if !tr.Shape().Equal(shape) {
		t.Fatalf("shape mismatch: got %v want %v", tr.Shape(), shape)
	}
	got := tr.Data()
	if len(got) != len(data) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(data))
	}
	if !slices.Equal(got, data) {
		t.Fatalf("data mismatch: got %v want %v", got, data)
	}
}

func TestNPYRoundTripU8(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "u8.npy")
	data := []uint8{0, 1, 255, 42}
	shape := candy.NewShapeFrom([]int{2, 2})
	tx, err := tensor.New(data, shape, candy.CPU)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.WriteNPY(path); err != nil {
		t.Fatalf("WriteNPY: %v", err)
	}
	tr, err := tensor.ReadNPY[uint8](path)
	if err != nil {
		t.Fatalf("ReadNPY: %v", err)
	}
	if !tr.Shape().Equal(shape) {
		t.Fatalf("shape mismatch: got %v want %v", tr.Shape(), shape)
	}
	got := tr.Data()
	if len(got) != len(data) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(data))
	}
	if !slices.Equal(got, data) {
		t.Fatalf("data mismatch: got %v want %v", got, data)
	}
}

func TestNPYRoundTripU32(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "u32.npy")
	data := []uint32{0, 1, 4000000000 - 1}
	shape := candy.NewShapeFrom([]int{3})
	tx, err := tensor.New(data, shape, candy.CPU)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.WriteNPY(path); err != nil {
		t.Fatalf("WriteNPY u32: %v", err)
	}
	tr, err := tensor.ReadNPY[uint32](path)
	if err != nil {
		t.Fatalf("ReadNPY u32: %v", err)
	}
	if !tr.Shape().Equal(shape) {
		t.Fatalf("u32 shape mismatch")
	}
	if !slices.Equal(tr.Data(), data) {
		t.Fatalf("u32 data mismatch")
	}
}

func TestNPYRoundTripI64(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "i64.npy")
	data := []int64{-9, 0, 7, math.MaxInt32 + 1}
	shape := candy.NewShapeFrom([]int{2, 2})
	tx, err := tensor.New(data, shape, candy.CPU)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.WriteNPY(path); err != nil {
		t.Fatalf("WriteNPY i64: %v", err)
	}
	tr, err := tensor.ReadNPY[int64](path)
	if err != nil {
		t.Fatalf("ReadNPY i64: %v", err)
	}
	if !tr.Shape().Equal(shape) {
		t.Fatalf("i64 shape mismatch")
	}
	if !slices.Equal(tr.Data(), data) {
		t.Fatalf("i64 data mismatch")
	}
}

func TestNPYScalar(t *testing.T) {
	t.Parallel()
	p1 := filepath.Join(t.TempDir(), "scalar.npy")
	t1, err := tensor.New([]float32{42}, candy.NewShape(), candy.CPU)
	if err != nil {
		t.Fatal(err)
	}
	if err := t1.WriteNPY(p1); err != nil {
		t.Fatalf("WriteNPY scalar: %v", err)
	}
	r1, err := tensor.ReadNPY[float32](p1)
	if err != nil {
		t.Fatalf("ReadNPY scalar: %v", err)
	}
	if r1.Shape().Rank() != 0 {
		t.Fatalf("scalar rank: got %d", r1.Shape().Rank())
	}
	if got := r1.Data(); len(got) != 1 || got[0] != 42 {
		t.Fatalf("scalar data: %v", got)
	}
}

func TestNPYEmpty(t *testing.T) {
	t.Parallel()
	p2 := filepath.Join(t.TempDir(), "empty.npy")
	t2, err := tensor.New([]uint8{}, candy.NewShapeFrom([]int{0}), candy.CPU)
	if err != nil {
		t.Fatal(err)
	}
	if err := t2.WriteNPY(p2); err != nil {
		t.Fatalf("WriteNPY empty: %v", err)
	}
	r2, err := tensor.ReadNPY[uint8](p2)
	if err != nil {
		t.Fatalf("ReadNPY empty: %v", err)
	}
	if !r2.Shape().Equal(candy.NewShapeFrom([]int{0})) {
		t.Fatalf("empty shape: %v", r2.Shape())
	}
	if len(r2.Data()) != 0 {
		t.Fatalf("empty data len: %d", len(r2.Data()))
	}
}

func TestNPZRoundTrip(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "a.npz")
	a := tensor.MustNew([]float32{1, 2, 3, 4}, candy.NewShapeFrom([]int{2, 2}), candy.CPU)
	b := tensor.MustNew([]float32{5, 6, 7}, candy.NewShapeFrom([]int{3}), candy.CPU)
	if err := tensor.WriteNPZ(path, map[string]*tensor.Tensor[float32]{"A": a, "B": b}); err != nil {
		t.Fatalf("WriteNPZ: %v", err)
	}
	m, err := tensor.ReadNPZ[float32](path)
	if err != nil {
		t.Fatalf("ReadNPZ: %v", err)
	}
	if _, ok := m["A"]; !ok {
		t.Fatalf("missing A")
	}
	if _, ok := m["B"]; !ok {
		t.Fatalf("missing B")
	}
	if !m["A"].Shape().Equal(a.Shape()) || !slices.Equal(m["A"].Data(), a.Data()) {
		t.Fatalf("A mismatch")
	}
	if !m["B"].Shape().Equal(b.Shape()) || !slices.Equal(m["B"].Data(), b.Data()) {
		t.Fatalf("B mismatch")
	}
}

func TestNPZReadByNameOrder(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "b.npz")
	a := tensor.MustNew([]float32{1, 2}, candy.NewShapeFrom([]int{2}), candy.CPU)
	b := tensor.MustNew([]float32{3, 4, 5}, candy.NewShapeFrom([]int{3}), candy.CPU)
	if err := tensor.WriteNPZ(path, map[string]*tensor.Tensor[float32]{"first": a, "second": b}); err != nil {
		t.Fatalf("WriteNPZ: %v", err)
	}
	res, err := tensor.ReadNPZByName[float32](path, []string{"second", "first"})
	if err != nil {
		t.Fatalf("ReadNPZByName: %v", err)
	}
	if len(res) != 2 {
		t.Fatalf("len=%d", len(res))
	}
	if !slices.Equal(res[0].Data(), b.Data()) {
		t.Fatalf("order[0] mismatch")
	}
	if !slices.Equal(res[1].Data(), a.Data()) {
		t.Fatalf("order[1] mismatch")
	}
}

package spark_test

import (
	"slices"
	"testing"

	"github.com/gocnn/spark"
)

func TestLayoutNewLayout(t *testing.T) {
	shape := spark.NewShape(2, 3)
	stride := []int{3, 1}
	l := spark.NewLayout(shape, stride, 5)

	if !l.Shape().Equal(shape) {
		t.Errorf("NewLayout shape = %v; want %v", l.Shape(), shape)
	}
	if !slices.Equal(l.Stride(), stride) {
		t.Errorf("NewLayout stride = %v; want %v", l.Stride(), stride)
	}
	if l.StartOffset() != 5 {
		t.Errorf("NewLayout offset = %d; want 5", l.StartOffset())
	}
}

func TestLayoutNewLayoutPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("NewLayout did not panic on stride mismatch")
		}
	}()
	spark.NewLayout(spark.NewShape(2, 3), []int{1}, 0)
}

func TestLayoutContiguousWithOffset(t *testing.T) {
	shape := spark.NewShape(2, 3)
	l := spark.ContiguousWithOffset(shape, 5)

	expectedStride := []int{3, 1}
	if !slices.Equal(l.Stride(), expectedStride) {
		t.Errorf("ContiguousWithOffset stride = %v; want %v", l.Stride(), expectedStride)
	}
	if l.StartOffset() != 5 {
		t.Errorf("ContiguousWithOffset offset = %d; want 5", l.StartOffset())
	}
}

func TestLayoutContiguous(t *testing.T) {
	shape := spark.NewShape(2, 3)
	l := spark.Contiguous(shape)

	expectedStride := []int{3, 1}
	if !slices.Equal(l.Stride(), expectedStride) {
		t.Errorf("Contiguous stride = %v; want %v", l.Stride(), expectedStride)
	}
	if l.StartOffset() != 0 {
		t.Errorf("Contiguous offset = %d; want 0", l.StartOffset())
	}
}

func TestLayoutShape(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3))
	s := l.Shape()
	expected := spark.NewShape(2, 3)
	if !s.Equal(expected) {
		t.Errorf("Shape() = %v; want %v", s, expected)
	}
}

func TestLayoutStride(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3))
	stride := l.Stride()
	expected := []int{3, 1}
	if !slices.Equal(stride, expected) {
		t.Errorf("Stride() = %v; want %v", stride, expected)
	}

	stride[0] = 99
	newStride := l.Stride()
	if newStride[0] == 99 {
		t.Errorf("Stride() not cloned; modification affected original")
	}
}

func TestLayoutStartOffset(t *testing.T) {
	l := spark.ContiguousWithOffset(spark.NewShape(2, 3), 5)
	if l.StartOffset() != 5 {
		t.Errorf("StartOffset() = %d; want 5", l.StartOffset())
	}
}

func TestLayoutString(t *testing.T) {
	l1 := spark.Contiguous(spark.NewShape(2, 3))
	l2 := spark.ContiguousWithOffset(spark.NewShape(2, 3), 5)

	expected1 := "Layout{shape=[2 3], stride=[3 1]}"
	if l1.String() != expected1 {
		t.Errorf("String() = %q; want %q", l1.String(), expected1)
	}

	expected2 := "Layout{shape=[2 3], stride=[3 1], offset=5}"
	if l2.String() != expected2 {
		t.Errorf("String() = %q; want %q", l2.String(), expected2)
	}
}

func TestLayoutClone(t *testing.T) {
	original := spark.Contiguous(spark.NewShape(2, 3))
	clone := original.Clone()

	if !clone.Shape().Equal(original.Shape()) {
		t.Errorf("Clone shape = %v; want %v", clone.Shape(), original.Shape())
	}
	if !slices.Equal(clone.Stride(), original.Stride()) {
		t.Errorf("Clone stride = %v; want %v", clone.Stride(), original.Stride())
	}
	if clone.StartOffset() != original.StartOffset() {
		t.Errorf("Clone offset = %d; want %d", clone.StartOffset(), original.StartOffset())
	}

	cloneStride := clone.Stride()
	cloneStride[0] = 99
	originalStride := original.Stride()
	if originalStride[0] == 99 {
		t.Errorf("Clone() not deep; modifying clone affected original")
	}
}

func TestLayoutDims(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3))
	dims := l.Dims()
	expected := []int{2, 3}
	if !slices.Equal(dims, expected) {
		t.Errorf("Dims() = %v; want %v", dims, expected)
	}
}

func TestLayoutDim(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3, 4))

	if l.Dim(1) != 3 {
		t.Errorf("Dim(1) = %d; want 3", l.Dim(1))
	}
	if l.Dim(-1) != 4 {
		t.Errorf("Dim(-1) = %d; want 4", l.Dim(-1))
	}
}

func TestLayoutContiguousOffsets(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3))
	start, end, ok := l.ContiguousOffsets()
	if !ok || start != 0 || end != 6 {
		t.Errorf("ContiguousOffsets() = %d, %d, %v; want 0, 6, true", start, end, ok)
	}

	nonContiguous := spark.NewLayout(spark.NewShape(2, 3), []int{1, 2}, 0)
	_, _, ok = nonContiguous.ContiguousOffsets()
	if ok {
		t.Errorf("ContiguousOffsets() ok=true for non-contiguous layout")
	}
}

func TestLayoutIsContiguous(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3))
	if !l.IsContiguous() {
		t.Errorf("IsContiguous() = false; want true")
	}

	nonContiguous := spark.NewLayout(spark.NewShape(2, 3), []int{1, 2}, 0)
	if nonContiguous.IsContiguous() {
		t.Errorf("IsContiguous() = true for non-contiguous layout")
	}
}

func TestLayoutIsFortranContiguous(t *testing.T) {
	shape := spark.NewShape(2, 3)
	stride := []int{1, 2}
	l := spark.NewLayout(shape, stride, 0)

	if !l.IsFortranContiguous() {
		t.Errorf("IsFortranContiguous() = false; want true")
	}

	cContiguous := spark.Contiguous(shape)
	if cContiguous.IsFortranContiguous() {
		t.Errorf("IsFortranContiguous() = true for C-contiguous layout")
	}
}

func TestLayoutNarrow(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3, 4))
	nl, err := l.Narrow(1, 1, 2)

	if err != nil {
		t.Errorf("Narrow failed: %v", err)
	}

	expectedShape := spark.NewShape(2, 2, 4)
	if !nl.Shape().Equal(expectedShape) {
		t.Errorf("Narrow shape = %v; want %v", nl.Shape(), expectedShape)
	}

	expectedStride := []int{12, 4, 1}
	if !slices.Equal(nl.Stride(), expectedStride) {
		t.Errorf("Narrow stride = %v; want %v", nl.Stride(), expectedStride)
	}

	if nl.StartOffset() != 4 {
		t.Errorf("Narrow offset = %d; want 4", nl.StartOffset())
	}
}

func TestLayoutNarrowError(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3))

	_, err := l.Narrow(0, 0, 3)
	if err == nil {
		t.Errorf("Narrow did not error on invalid len")
	}

	_, err = l.Narrow(2, 0, 1)
	if err == nil {
		t.Errorf("Narrow did not error on out of range dim")
	}

	_, err = l.Narrow(0, -1, 1)
	if err == nil {
		t.Errorf("Narrow did not error on negative start")
	}

	_, err = l.Narrow(0, 0, -1)
	if err == nil {
		t.Errorf("Narrow did not error on negative len")
	}
}

func TestLayoutTranspose(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3, 4))
	tl, err := l.Transpose(0, 2)

	if err != nil {
		t.Errorf("Transpose failed: %v", err)
	}

	expectedShape := spark.NewShape(4, 3, 2)
	if !tl.Shape().Equal(expectedShape) {
		t.Errorf("Transpose shape = %v; want %v", tl.Shape(), expectedShape)
	}

	expectedStride := []int{1, 4, 12}
	if !slices.Equal(tl.Stride(), expectedStride) {
		t.Errorf("Transpose stride = %v; want %v", tl.Stride(), expectedStride)
	}
}

func TestLayoutTransposeError(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3))

	_, err := l.Transpose(0, 2)
	if err == nil {
		t.Errorf("Transpose did not error on out of range")
	}

	_, err = l.Transpose(-3, 0)
	if err == nil {
		t.Errorf("Transpose did not error on negative out of range")
	}
}

func TestLayoutPermute(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3, 4))
	pl, err := l.Permute([]int{2, 0, 1})

	if err != nil {
		t.Errorf("Permute failed: %v", err)
	}

	expectedShape := spark.NewShape(4, 2, 3)
	if !pl.Shape().Equal(expectedShape) {
		t.Errorf("Permute shape = %v; want %v", pl.Shape(), expectedShape)
	}

	expectedStride := []int{1, 12, 4}
	if !slices.Equal(pl.Stride(), expectedStride) {
		t.Errorf("Permute stride = %v; want %v", pl.Stride(), expectedStride)
	}
}

func TestLayoutPermuteError(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3))

	_, err := l.Permute([]int{0})
	if err == nil {
		t.Errorf("Permute did not error on len mismatch")
	}

	_, err = l.Permute([]int{0, 0})
	if err == nil {
		t.Errorf("Permute did not error on duplicate index")
	}

	_, err = l.Permute([]int{0, 2})
	if err == nil {
		t.Errorf("Permute did not error on out of range index")
	}
}

func TestLayoutBroadcastAs(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(1, 3))
	tgt := spark.NewShape(2, 1, 3)
	bl, err := l.BroadcastAs(tgt)

	if err != nil {
		t.Errorf("BroadcastAs failed: %v", err)
	}

	if !bl.Shape().Equal(tgt) {
		t.Errorf("BroadcastAs shape = %v; want %v", bl.Shape(), tgt)
	}

	expectedStride := []int{0, 3, 1}
	if !slices.Equal(bl.Stride(), expectedStride) {
		t.Errorf("BroadcastAs stride = %v; want %v", bl.Stride(), expectedStride)
	}
}

func TestLayoutBroadcastAsError(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3))

	tgt := spark.NewShape(3)
	_, err := l.BroadcastAs(tgt)
	if err == nil {
		t.Errorf("BroadcastAs did not error on lower rank target")
	}

	tgt = spark.NewShape(2, 1, 4)
	_, err = l.BroadcastAs(tgt)
	if err == nil {
		t.Errorf("BroadcastAs did not error on incompatible dims")
	}
}

func TestLayoutOffsetsB(t *testing.T) {
	shape := spark.NewShape(2, 1, 3, 1)
	stride := []int{0, 0, 1, 0}
	l := spark.NewLayout(shape, stride, 5)

	offs, ok := l.OffsetsB()
	if !ok {
		t.Errorf("OffsetsB failed for valid broadcast layout")
	}

	if offs.Start != 5 {
		t.Errorf("OffsetsB Start = %d; want 5", offs.Start)
	}
	if offs.Len != 3 {
		t.Errorf("OffsetsB Len = %d; want 3", offs.Len)
	}
	if offs.LeftBroadcast != 2 {
		t.Errorf("OffsetsB LeftBroadcast = %d; want 2", offs.LeftBroadcast)
	}
	if offs.RightBroadcast != 1 {
		t.Errorf("OffsetsB RightBroadcast = %d; want 1", offs.RightBroadcast)
	}
}

package candy_test

import (
	"slices"
	"testing"

	"github.com/gocnn/candy"
)

func TestShapeNewShape(t *testing.T) {
	s := candy.NewShape(2, 3, 4)
	if !slices.Equal(s.Dims(), []int{2, 3, 4}) {
		t.Errorf("NewShape(2, 3, 4) = %v; want [2 3 4]", s.Dims())
	}
}

func TestShapeNewShapeFrom(t *testing.T) {
	s := candy.NewShapeFrom([]int{1, 2})
	if !slices.Equal(s.Dims(), []int{1, 2}) {
		t.Errorf("NewShapeFrom([1, 2]) = %v; want [1 2]", s.Dims())
	}
}

func TestShapeClone(t *testing.T) {
	s := candy.NewShape(2, 3)
	clone := s.Clone()
	if !clone.Equal(s) {
		t.Errorf("Clone() = %v; want %v", clone, s)
	}
}

func TestShapeEqual(t *testing.T) {
	s1 := candy.NewShape(2, 3)
	s2 := candy.NewShape(2, 3)
	s3 := candy.NewShape(2, 4)
	if !s1.Equal(s2) {
		t.Errorf("Equal([2 3], [2 3]) = false; want true")
	}
	if s1.Equal(s3) {
		t.Errorf("Equal([2 3], [2 4]) = true; want false")
	}
}

func TestShapeIsScalar(t *testing.T) {
	s1 := candy.NewShape()
	s2 := candy.NewShape(1)
	if !s1.IsScalar() {
		t.Errorf("IsScalar([]) = false; want true")
	}
	if s2.IsScalar() {
		t.Errorf("IsScalar([1]) = true; want false")
	}
}

func TestShapeIsVector(t *testing.T) {
	s1 := candy.NewShape(5)
	s2 := candy.NewShape(2, 3)
	if !s1.IsVector() {
		t.Errorf("IsVector([5]) = false; want true")
	}
	if s2.IsVector() {
		t.Errorf("IsVector([2 3]) = true; want false")
	}
}

func TestShapeIsMatrix(t *testing.T) {
	s1 := candy.NewShape(2, 3)
	s2 := candy.NewShape(4)
	if !s1.IsMatrix() {
		t.Errorf("IsMatrix([2 3]) = false; want true")
	}
	if s2.IsMatrix() {
		t.Errorf("IsMatrix([4]) = true; want false")
	}
}

func TestShapeString(t *testing.T) {
	s1 := candy.NewShape()
	s2 := candy.NewShape(2, 3, 4)
	if s1.String() != "[]" {
		t.Errorf("String([]) = %q; want []", s1.String())
	}
	if s2.String() != "[2 3 4]" {
		t.Errorf("String([2 3 4]) = %q; want [2 3 4]", s2.String())
	}
}

func TestShapeRank(t *testing.T) {
	s := candy.NewShape(1, 2, 3)
	if s.Rank() != 3 {
		t.Errorf("Rank([1 2 3]) = %d; want 3", s.Rank())
	}
}

func TestShapeDims(t *testing.T) {
	s := candy.NewShape(2, 3, 4)
	dims := s.Dims()
	if !slices.Equal(dims, []int{2, 3, 4}) {
		t.Errorf("Dims() = %v; want [2 3 4]", dims)
	}
	dims[0] = 5 // Modify returned slice
	if slices.Equal(s.Dims(), dims) {
		t.Errorf("Dims() not deep; modifying returned slice affected original")
	}
}

func TestShapeDim(t *testing.T) {
	s := candy.NewShape(2, 3, 4)
	if s.Dim(1) != 3 {
		t.Errorf("Dim(1) = %d; want 3", s.Dim(1))
	}
	if s.Dim(-1) != 4 {
		t.Errorf("Dim(-1) = %d; want 4", s.Dim(-1))
	}
}

func TestShapeDimPanic(t *testing.T) {
	s := candy.NewShape(2, 3)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Dim(2) did not panic")
		}
	}()
	s.Dim(2)
}

func TestShapeNumel(t *testing.T) {
	s1 := candy.NewShape()
	s2 := candy.NewShape(2, 3, 4)
	s3 := candy.NewShape(2, 0, 4)
	if s1.Numel() != 1 {
		t.Errorf("Numel([]) = %d; want 1", s1.Numel())
	}
	if s2.Numel() != 24 {
		t.Errorf("Numel([2 3 4]) = %d; want 24", s2.Numel())
	}
	if s3.Numel() != 0 {
		t.Errorf("Numel([2 0 4]) = %d; want 0", s3.Numel())
	}
}

func TestShapeStrideContiguous(t *testing.T) {
	s := candy.NewShape(2, 3, 4)
	strides := s.StrideContiguous()
	if !slices.Equal(strides, []int{12, 4, 1}) {
		t.Errorf("StrideContiguous([2 3 4]) = %v; want [12 4 1]", strides)
	}
}

func TestShapeIsContiguous(t *testing.T) {
	s := candy.NewShape(2, 3, 4)
	if !s.IsContiguous([]int{12, 4, 1}) {
		t.Errorf("IsContiguous([12 4 1]) = false; want true")
	}
	if s.IsContiguous([]int{4, 12, 1}) {
		t.Errorf("IsContiguous([4 12 1]) = true; want false")
	}
}

func TestShapeIsFortranContiguous(t *testing.T) {
	s := candy.NewShape(2, 3, 4)
	if !s.IsFortranContiguous([]int{1, 2, 6}) {
		t.Errorf("IsFortranContiguous([1 2 6]) = false; want true")
	}
	if s.IsFortranContiguous([]int{4, 12, 1}) {
		t.Errorf("IsFortranContiguous([4 12 1]) = true; want false")
	}
}

func TestShapeExtend(t *testing.T) {
	s := candy.NewShape(2, 3)
	newShape := s.Extend(4, 5)
	if !newShape.Equal(candy.NewShape(2, 3, 4, 5)) {
		t.Errorf("Extend(4, 5) = %v; want [2 3 4 5]", newShape)
	}
}

func TestShapeBroadcastShapeBinaryOp(t *testing.T) {
	s1 := candy.NewShape(3, 1, 4)
	s2 := candy.NewShape(2, 4)
	result, err := s1.BroadcastShapeBinaryOp(s2)
	if err != nil {
		t.Errorf("BroadcastShapeBinaryOp failed: %v", err)
	}
	if !result.Equal(candy.NewShape(3, 2, 4)) {
		t.Errorf("BroadcastShapeBinaryOp([3 1 4], [2 4]) = %v; want [3 2 4]", result)
	}
}

func TestShapeBroadcastShapeBinaryOpError(t *testing.T) {
	s1 := candy.NewShape(3, 4)
	s2 := candy.NewShape(2, 5)
	_, err := s1.BroadcastShapeBinaryOp(s2)
	if err == nil {
		t.Errorf("BroadcastShapeBinaryOp([3 4], [2 5]) did not error")
	}
}

func TestShapeBroadcastShapeMatmul(t *testing.T) {
	s1 := candy.NewShape(2, 3, 4)
	s2 := candy.NewShape(4, 5)
	lhs, rhs, err := s1.BroadcastShapeMatmul(s2)
	if err != nil {
		t.Errorf("BroadcastShapeMatmul failed: %v", err)
	}
	if !lhs.Equal(candy.NewShape(2, 3, 4)) || !rhs.Equal(candy.NewShape(2, 4, 5)) {
		t.Errorf("BroadcastShapeMatmul([2 3 4], [4 5]) = %v, %v; want [2 3 4], [2 4 5]", lhs, rhs)
	}
}

func TestShapeBroadcastShapeMatmulError(t *testing.T) {
	s1 := candy.NewShape(2, 3, 5)
	s2 := candy.NewShape(4, 5)
	_, _, err := s1.BroadcastShapeMatmul(s2)
	if err == nil {
		t.Errorf("BroadcastShapeMatmul([2 3 5], [4 5]) did not error")
	}
}

func TestShapeResolveAxes(t *testing.T) {
	s := candy.NewShape(2, 3, 4)
	axes, err := candy.ResolveAxes([]int{0, -1}, s)
	if err != nil {
		t.Errorf("ResolveAxes([0, -1]) failed: %v", err)
	}
	if !slices.Equal(axes, []int{0, 2}) {
		t.Errorf("ResolveAxes([0, -1]) = %v; want [0 2]", axes)
	}
}

func TestShapeResolveAxesError(t *testing.T) {
	s := candy.NewShape(2, 3)
	_, err := candy.ResolveAxes([]int{0, 0}, s)
	if err == nil {
		t.Errorf("ResolveAxes([0, 0]) did not error")
	}
}

func TestShapeDims0(t *testing.T) {
	s := candy.NewShape()
	if err := s.Dims0(); err != nil {
		t.Errorf("Dims0() failed: %v", err)
	}
	s2 := candy.NewShape(1)
	if err := s2.Dims0(); err == nil {
		t.Errorf("Dims0([1]) did not error")
	}
}

func TestShapeDims1(t *testing.T) {
	s := candy.NewShape(5)
	d, err := s.Dims1()
	if err != nil || d != 5 {
		t.Errorf("Dims1([5]) = %d, %v; want 5, nil", d, err)
	}
}

func TestShapeDims2(t *testing.T) {
	s := candy.NewShape(2, 3)
	d1, d2, err := s.Dims2()
	if err != nil || d1 != 2 || d2 != 3 {
		t.Errorf("Dims2([2 3]) = %d, %d, %v; want 2, 3, nil", d1, d2, err)
	}
}

func TestShapeDims3(t *testing.T) {
	s := candy.NewShape(2, 3, 4)
	d1, d2, d3, err := s.Dims3()
	if err != nil || d1 != 2 || d2 != 3 || d3 != 4 {
		t.Errorf("Dims3([2 3 4]) = %d, %d, %d, %v; want 2, 3, 4, nil", d1, d2, d3, err)
	}
}

func TestShapeDims4(t *testing.T) {
	s := candy.NewShape(2, 3, 4, 5)
	d1, d2, d3, d4, err := s.Dims4()
	if err != nil || d1 != 2 || d2 != 3 || d3 != 4 || d4 != 5 {
		t.Errorf("Dims4([2 3 4 5]) = %d, %d, %d, %d, %v; want 2, 3, 4, 5, nil", d1, d2, d3, d4, err)
	}
}

func TestShapeDims5(t *testing.T) {
	s := candy.NewShape(2, 3, 4, 5, 6)
	d1, d2, d3, d4, d5, err := s.Dims5()
	if err != nil || d1 != 2 || d2 != 3 || d3 != 4 || d4 != 5 || d5 != 6 {
		t.Errorf("Dims5([2 3 4 5 6]) = %d, %d, %d, %d, %d, %v; want 2, 3, 4, 5, 6, nil", d1, d2, d3, d4, d5, err)
	}
}

func TestShapeReshape(t *testing.T) {
	s := candy.NewShape(2, 3, 4)
	newShape, err := s.Reshape(4, -1, 3)
	if err != nil {
		t.Errorf("Reshape(4, -1, 3) failed: %v", err)
	}
	if !newShape.Equal(candy.NewShape(4, 2, 3)) {
		t.Errorf("Reshape(4, -1, 3) = %v; want [4 2 3]", newShape)
	}
}

func TestShapeReshapeError(t *testing.T) {
	s := candy.NewShape(2, 3, 4)
	_, err := s.Reshape(4, 5, 6)
	if err == nil {
		t.Errorf("Reshape(4, 5, 6) did not error")
	}
}

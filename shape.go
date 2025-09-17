package goml

import (
	"fmt"
	"strings"
)

// Shape represents the dimensions of a tensor.
type Shape struct {
	dims []int
}

// NewShape creates a new Shape from the given dimensions.
func NewShape(dims ...int) Shape {
	// Make a copy to avoid external modifications
	result := make([]int, len(dims))
	copy(result, dims)
	return Shape{dims: result}
}

// NewShapeFromSlice creates a new Shape from a slice of dimensions.
func NewShapeFromSlice(dims []int) Shape {
	result := make([]int, len(dims))
	copy(result, dims)
	return Shape{dims: result}
}

// Dims returns a copy of the dimensions slice.
func (s Shape) Dims() []int {
	result := make([]int, len(s.dims))
	copy(result, s.dims)
	return result
}

// Ndim returns the number of dimensions.
func (s Shape) Ndim() int {
	return len(s.dims)
}

// Size returns the total number of elements (product of all dimensions).
func (s Shape) Size() int {
	if len(s.dims) == 0 {
		return 0
	}
	size := 1
	for _, dim := range s.dims {
		size *= dim
	}
	return size
}

// At returns the dimension at the given index.
func (s Shape) At(i int) int {
	if i < 0 || i >= len(s.dims) {
		panic(fmt.Sprintf("shape: index %d out of range [0, %d)", i, len(s.dims)))
	}
	return s.dims[i]
}

// Equals checks if two shapes are equal.
func (s Shape) Equals(other Shape) bool {
	if len(s.dims) != len(other.dims) {
		return false
	}
	for i, dim := range s.dims {
		if dim != other.dims[i] {
			return false
		}
	}
	return true
}

// IsScalar returns true if the shape represents a scalar (0 dimensions).
func (s Shape) IsScalar() bool {
	return len(s.dims) == 0
}

// IsVector returns true if the shape represents a vector (1 dimension).
func (s Shape) IsVector() bool {
	return len(s.dims) == 1
}

// IsMatrix returns true if the shape represents a matrix (2 dimensions).
func (s Shape) IsMatrix() bool {
	return len(s.dims) == 2
}

// Clone creates a deep copy of the shape.
func (s Shape) Clone() Shape {
	return NewShapeFromSlice(s.dims)
}

// String returns a string representation of the shape.
func (s Shape) String() string {
	if len(s.dims) == 0 {
		return "[]"
	}
	dimStrs := make([]string, len(s.dims))
	for i, dim := range s.dims {
		dimStrs[i] = fmt.Sprintf("%d", dim)
	}
	return "[" + strings.Join(dimStrs, ", ") + "]"
}

// Reshape returns a new shape with the given dimensions.
// Use -1 for one dimension to auto-infer its size.
// The total size must remain the same.
func (s Shape) Reshape(dims ...int) Shape {
	totalSize := s.Size()
	inferIndex := -1
	knownSize := 1

	// Find the -1 dimension and calculate known size
	for i, dim := range dims {
		switch {
		case dim == -1:
			if inferIndex != -1 {
				panic("shape: only one dimension can be -1")
			}
			inferIndex = i
		case dim <= 0:
			panic(fmt.Sprintf("shape: dimension %d must be positive, got %d", i, dim))
		default:
			knownSize *= dim
		}
	}

	// Handle dimension inference
	if inferIndex != -1 {
		if totalSize%knownSize != 0 {
			panic(fmt.Sprintf("shape: cannot infer dimension, total size %d not divisible by known size %d",
				totalSize, knownSize))
		}
		dims[inferIndex] = totalSize / knownSize
	}

	// Validate total size
	newShape := NewShape(dims...)
	if totalSize != newShape.Size() {
		panic(fmt.Sprintf("shape: cannot reshape from %v to %v: size mismatch (%d vs %d)",
			s, newShape, totalSize, newShape.Size()))
	}

	return newShape
}

// CanBroadcastWith checks if this shape can be broadcast with another shape.
func (s Shape) CanBroadcastWith(other Shape) bool {
	// Broadcasting rules: dimensions are compatible if they are equal,
	// or one of them is 1, starting from the trailing dimensions.
	maxNdim := max(len(other.dims), len(s.dims))

	for i := range maxNdim {
		sDim := 1
		otherDim := 1

		if i < len(s.dims) {
			sDim = s.dims[len(s.dims)-1-i]
		}
		if i < len(other.dims) {
			otherDim = other.dims[len(other.dims)-1-i]
		}

		if sDim != otherDim && sDim != 1 && otherDim != 1 {
			return false
		}
	}
	return true
}

// BroadcastWith returns the resulting shape after broadcasting with another shape.
func (s Shape) BroadcastWith(other Shape) Shape {
	if !s.CanBroadcastWith(other) {
		panic(fmt.Sprintf("shape: cannot broadcast %v with %v", s, other))
	}

	maxNdim := max(len(other.dims), len(s.dims))

	result := make([]int, maxNdim)
	for i := range maxNdim {
		sDim := 1
		otherDim := 1

		if i < len(s.dims) {
			sDim = s.dims[len(s.dims)-1-i]
		}
		if i < len(other.dims) {
			otherDim = other.dims[len(other.dims)-1-i]
		}

		if sDim > otherDim {
			result[maxNdim-1-i] = sDim
		} else {
			result[maxNdim-1-i] = otherDim
		}
	}

	return NewShapeFromSlice(result)
}

// StrideContiguous returns the stride for contiguous (row-major) layout.
func (s Shape) StrideContiguous() []int {
	if len(s.dims) == 0 {
		return []int{}
	}

	stride := make([]int, len(s.dims))
	stride[len(s.dims)-1] = 1

	for i := len(s.dims) - 2; i >= 0; i-- {
		stride[i] = stride[i+1] * s.dims[i+1]
	}

	return stride
}

// StrideFortran returns the stride for Fortran contiguous (column-major) layout.
func (s Shape) StrideFortran() []int {
	if len(s.dims) == 0 {
		return []int{}
	}

	stride := make([]int, len(s.dims))
	stride[0] = 1

	for i := 1; i < len(s.dims); i++ {
		stride[i] = stride[i-1] * s.dims[i-1]
	}

	return stride
}

// IsContiguous checks if the given stride represents contiguous (row-major) layout.
func (s Shape) IsContiguous(stride []int) bool {
	if len(stride) != len(s.dims) {
		return false
	}
	if len(s.dims) == 0 {
		return true
	}

	expectedStride := s.StrideContiguous()
	for i, st := range stride {
		if st != expectedStride[i] {
			return false
		}
	}
	return true
}

// IsFortranContiguous checks if the given stride represents Fortran contiguous (column-major) layout.
func (s Shape) IsFortranContiguous(stride []int) bool {
	if len(stride) != len(s.dims) {
		return false
	}
	if len(s.dims) == 0 {
		return true
	}

	expectedStride := s.StrideFortran()
	for i, st := range stride {
		if st != expectedStride[i] {
			return false
		}
	}
	return true
}

// Rank returns the number of dimensions (same as Ndim, for compatibility).
func (s Shape) Rank() int {
	return len(s.dims)
}

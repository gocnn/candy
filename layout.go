package spark

import (
	"fmt"
	"strings"
)

// Layout represents tensor memory layout including shape, stride, and offset.
// This design is inspired by Candle's Layout for efficient tensor operations.
type Layout struct {
	shape       Shape
	stride      []int // Stride in number of elements (not bytes)
	startOffset int   // Starting offset in the underlying storage
}

// NewLayout creates a new Layout with the given shape, stride, and start offset.
func NewLayout(shape Shape, stride []int, startOffset int) Layout {
	if len(stride) != shape.Ndim() {
		panic(fmt.Sprintf("layout: stride length %d must match shape dimensions %d", len(stride), shape.Ndim()))
	}

	// Make copies to avoid external modifications
	strideCopy := make([]int, len(stride))
	copy(strideCopy, stride)

	return Layout{
		shape:       shape.Clone(),
		stride:      strideCopy,
		startOffset: startOffset,
	}
}

// Contiguous creates a contiguous layout with the given shape and zero offset.
func Contiguous(shape Shape) Layout {
	return ContiguousWithOffset(shape, 0)
}

// ContiguousWithOffset creates a contiguous layout with the given shape and start offset.
func ContiguousWithOffset(shape Shape, startOffset int) Layout {
	stride := shape.StrideContiguous()
	return Layout{
		shape:       shape.Clone(),
		stride:      stride,
		startOffset: startOffset,
	}
}

// Shape returns the shape of the layout.
func (l Layout) Shape() Shape {
	return l.shape.Clone()
}

// Dims returns a copy of the shape dimensions.
func (l Layout) Dims() []int {
	return l.shape.Dims()
}

// Dim returns the size of the specified dimension.
func (l Layout) Dim(dim int) int {
	return l.shape.At(dim)
}

// Stride returns a copy of the stride slice.
func (l Layout) Stride() []int {
	result := make([]int, len(l.stride))
	copy(result, l.stride)
	return result
}

// StartOffset returns the starting offset in the underlying storage.
func (l Layout) StartOffset() int {
	return l.startOffset
}

// IsContiguous returns true if the layout represents contiguous memory (row-major).
func (l Layout) IsContiguous() bool {
	return l.shape.IsContiguous(l.stride)
}

// IsFortranContiguous returns true if the layout represents Fortran contiguous memory (column-major).
func (l Layout) IsFortranContiguous() bool {
	return l.shape.IsFortranContiguous(l.stride)
}

// ContiguousOffsets returns the start and end offsets if the data is contiguous.
// Returns (start, end, true) if contiguous, (0, 0, false) otherwise.
func (l Layout) ContiguousOffsets() (int, int, bool) {
	if l.IsContiguous() {
		start := l.startOffset
		end := start + l.shape.Size()
		return start, end, true
	}
	return 0, 0, false
}

// Narrow creates a new layout by narrowing the specified dimension.
// This is equivalent to slicing: tensor[..., start:start+length, ...]
func (l Layout) Narrow(dim, start, length int) Layout {
	dims := l.shape.Dims()
	if dim < 0 || dim >= len(dims) {
		panic(fmt.Sprintf("layout: dimension %d out of range [0, %d)", dim, len(dims)))
	}
	if start < 0 || start+length > dims[dim] {
		panic(fmt.Sprintf("layout: narrow range [%d:%d] out of bounds for dimension size %d",
			start, start+length, dims[dim]))
	}

	// Update the shape
	newDims := make([]int, len(dims))
	copy(newDims, dims)
	newDims[dim] = length
	newShape := NewShapeFromSlice(newDims)

	// Calculate new start offset
	newStartOffset := l.startOffset + l.stride[dim]*start

	return Layout{
		shape:       newShape,
		stride:      l.Stride(), // Copy stride
		startOffset: newStartOffset,
	}
}

// Transpose swaps two dimensions.
func (l Layout) Transpose(dim1, dim2 int) Layout {
	rank := l.shape.Ndim()
	if dim1 < 0 || dim1 >= rank || dim2 < 0 || dim2 >= rank {
		panic(fmt.Sprintf("layout: transpose dimensions [%d, %d] out of range [0, %d)", dim1, dim2, rank))
	}

	dims := l.shape.Dims()
	stride := l.Stride()

	// Swap dimensions and strides
	dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
	stride[dim1], stride[dim2] = stride[dim2], stride[dim1]

	return Layout{
		shape:       NewShapeFromSlice(dims),
		stride:      stride,
		startOffset: l.startOffset,
	}
}

// Permute reorders dimensions according to the given indices.
func (l Layout) Permute(indices []int) Layout {
	rank := l.shape.Ndim()
	if len(indices) != rank {
		panic(fmt.Sprintf("layout: permute indices length %d must match rank %d", len(indices), rank))
	}

	// Validate that indices form a valid permutation
	used := make([]bool, rank)
	for _, idx := range indices {
		if idx < 0 || idx >= rank {
			panic(fmt.Sprintf("layout: permute index %d out of range [0, %d)", idx, rank))
		}
		if used[idx] {
			panic(fmt.Sprintf("layout: duplicate index %d in permutation", idx))
		}
		used[idx] = true
	}

	dims := l.shape.Dims()
	stride := l.Stride()

	newDims := make([]int, rank)
	newStride := make([]int, rank)

	for i, idx := range indices {
		newDims[i] = dims[idx]
		newStride[i] = stride[idx]
	}

	return Layout{
		shape:       NewShapeFromSlice(newDims),
		stride:      newStride,
		startOffset: l.startOffset,
	}
}

// BroadcastAs creates a new layout by broadcasting to the target shape.
func (l Layout) BroadcastAs(targetShape Shape) Layout {
	srcRank := l.shape.Ndim()
	dstRank := targetShape.Ndim()

	if dstRank < srcRank {
		panic(fmt.Sprintf("layout: cannot broadcast shape %v to smaller shape %v", l.shape, targetShape))
	}

	srcDims := l.shape.Dims()
	dstDims := targetShape.Dims()
	addedDims := dstRank - srcRank

	// Create new stride with zeros for added dimensions
	newStride := make([]int, dstRank)

	// Set stride for added dimensions to 0 (broadcasting)
	for i := 0; i < addedDims; i++ {
		newStride[i] = 0
	}

	// Check compatibility and set stride for existing dimensions
	for i := 0; i < srcRank; i++ {
		srcDim := srcDims[i]
		dstDim := dstDims[addedDims+i]

		if srcDim == dstDim {
			newStride[addedDims+i] = l.stride[i]
		} else if srcDim == 1 {
			newStride[addedDims+i] = 0 // Broadcasting
		} else {
			panic(fmt.Sprintf("layout: cannot broadcast dimension %d from %d to %d", i, srcDim, dstDim))
		}
	}

	return Layout{
		shape:       targetShape.Clone(),
		stride:      newStride,
		startOffset: l.startOffset,
	}
}

// Clone creates a deep copy of the layout.
func (l Layout) Clone() Layout {
	return Layout{
		shape:       l.shape.Clone(),
		stride:      l.Stride(),
		startOffset: l.startOffset,
	}
}

// String returns a string representation of the layout.
func (l Layout) String() string {
	var parts []string
	parts = append(parts, fmt.Sprintf("shape=%v", l.shape))
	parts = append(parts, fmt.Sprintf("stride=%v", l.stride))
	if l.startOffset != 0 {
		parts = append(parts, fmt.Sprintf("offset=%d", l.startOffset))
	}
	return fmt.Sprintf("Layout{%s}", strings.Join(parts, ", "))
}

// Equals checks if two layouts are equal.
func (l Layout) Equals(other Layout) bool {
	if !l.shape.Equals(other.shape) || l.startOffset != other.startOffset {
		return false
	}
	if len(l.stride) != len(other.stride) {
		return false
	}
	for i, s := range l.stride {
		if s != other.stride[i] {
			return false
		}
	}
	return true
}

// FlatIndex calculates the flat index in the underlying storage for given multi-dimensional indices.
func (l Layout) FlatIndex(indices ...int) int {
	if len(indices) != l.shape.Ndim() {
		panic(fmt.Sprintf("layout: expected %d indices, got %d", l.shape.Ndim(), len(indices)))
	}

	dims := l.shape.Dims()
	for i, idx := range indices {
		if idx < 0 || idx >= dims[i] {
			panic(fmt.Sprintf("layout: index %d out of range [0, %d) for dimension %d", idx, dims[i], i))
		}
	}

	flatIdx := l.startOffset
	for i, idx := range indices {
		flatIdx += idx * l.stride[i]
	}
	return flatIdx
}

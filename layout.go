package candy

import (
	"fmt"
	"slices"
)

// Layout represents the layout of a tensor, including shape, strides, and starting offset.
// Strides are in number of elements, not bytes.
type Layout struct {
	shape       *Shape
	stride      []int
	startOffset int
}

// NewLayout creates a new Layout with the given shape, stride, and start offset.
// It clones the inputs to ensure immutability.
func NewLayout(shape *Shape, stride []int, startOffset int) *Layout {
	if len(stride) != shape.Rank() {
		panic(fmt.Sprintf("stride len %d != shape rank %d", len(stride), shape.Rank()))
	}
	return &Layout{
		shape:       shape,
		stride:      slices.Clone(stride),
		startOffset: startOffset,
	}
}

// Contiguous creates a contiguous (row-major) layout starting at offset 0.
func Contiguous(shape *Shape) *Layout {
	return ContiguousWithOffset(shape, 0)
}

// ContiguousWithOffset creates a contiguous (row-major) layout with the given start offset.
func ContiguousWithOffset(shape *Shape, startOffset int) *Layout {
	stride := shape.StrideContiguous()
	return NewLayout(shape, stride, startOffset)
}

// Shape returns a copy of the shape.
func (l *Layout) Shape() *Shape {
	return l.shape
}

// Stride returns a copy of the stride slice.
func (l *Layout) Stride() []int {
	return slices.Clone(l.stride)
}

// StartOffset returns the starting offset.
func (l *Layout) StartOffset() int {
	return l.startOffset
}

// String returns a string representation of the layout.
func (l *Layout) String() string {
	if l.startOffset == 0 {
		return fmt.Sprintf("Layout{shape=%v, stride=%v}", l.shape, l.stride)
	}
	return fmt.Sprintf("Layout{shape=%v, stride=%v, offset=%d}", l.shape, l.stride, l.startOffset)
}

// Clone returns a deep copy of the layout.
func (l *Layout) Clone() *Layout {
	return &Layout{
		shape:       l.shape.Clone(),
		stride:      slices.Clone(l.stride),
		startOffset: l.startOffset,
	}
}

// Rank returns the number of dimensions (rank) of the layout.
func (l *Layout) Rank() int {
	return l.shape.Rank()
}

// Dims returns the dimensions of the shape.
func (l *Layout) Dims() []int {
	return l.shape.Dims()
}

// Dim returns the size of the specified dimension, supporting negative indices.
func (l *Layout) Dim(dim int) int {
	return l.shape.Dim(dim)
}

// Dims0 checks if the shape has 0 dimensions (scalar).
func (s *Layout) Dims0() error {
	return s.shape.Dims0()
}

// Dims1 extracts the single dimension from a 1D shape.
func (s *Layout) Dims1() (int, error) {
	return s.shape.Dims1()
}

// Dims2 extracts the two dimensions from a 2D shape.
func (s *Layout) Dims2() (int, int, error) {
	return s.shape.Dims2()
}

// Dims3 extracts the three dimensions from a 3D shape.
func (s *Layout) Dims3() (int, int, int, error) {
	return s.shape.Dims3()
}

// Dims4 extracts the four dimensions from a 4D shape.
func (s *Layout) Dims4() (int, int, int, int, error) {
	return s.shape.Dims4()
}

// Dims5 extracts the five dimensions from a 5D shape.
func (s *Layout) Dims5() (int, int, int, int, int, error) {
	return s.shape.Dims5()
}

// Numel returns the total number of elements in the layout.
func (l *Layout) Numel() int {
	return l.shape.Numel()
}

// ContiguousOffsets returns the start and end offsets if the layout is contiguous,
// along with a boolean indicating if it is contiguous.
func (l *Layout) ContiguousOffsets() (start, end int, ok bool) {
	if !l.IsContiguous() {
		return 0, 0, false
	}
	start = l.StartOffset()
	end = start + l.shape.Numel()
	return start, end, true
}

// IsContiguous returns true if the strides represent a C-contiguous (row-major) layout.
func (l *Layout) IsContiguous() bool {
	return l.shape.IsContiguous(l.stride)
}

// IsFortranContiguous returns true if the strides represent a Fortran-contiguous (column-major) layout.
func (l *Layout) IsFortranContiguous() bool {
	return l.shape.IsFortranContiguous(l.stride)
}

// Narrow returns a new layout narrowed along the specified dimension from start to start+len.
func (l *Layout) Narrow(dim, start, len int) (*Layout, error) {
	rank := l.shape.Rank()
	resolvedDim, err := ResolveAxis(dim, rank)
	if err != nil {
		return nil, err
	}
	if start < 0 || len < 0 || start+len > l.Dims()[resolvedDim] {
		return nil, fmt.Errorf("invalid narrow args: dim %d, start %d, len %d, shape %v", resolvedDim, start, len, l.shape)
	}
	newDims := slices.Clone(l.Dims())
	newDims[resolvedDim] = len
	newOffset := l.startOffset + l.stride[resolvedDim]*start
	return NewLayout(NewShapeFrom(newDims), l.stride, newOffset), nil
}

// Transpose returns a new layout with the two specified dimensions swapped.
func (l *Layout) Transpose(dim1, dim2 int) (*Layout, error) {
	rank := l.shape.Rank()
	resolvedDim1, err := ResolveAxis(dim1, rank)
	if err != nil {
		return nil, err
	}
	resolvedDim2, err := ResolveAxis(dim2, rank)
	if err != nil {
		return nil, err
	}
	newDims := slices.Clone(l.Dims())
	newStride := slices.Clone(l.stride)
	newDims[resolvedDim1], newDims[resolvedDim2] = newDims[resolvedDim2], newDims[resolvedDim1]
	newStride[resolvedDim1], newStride[resolvedDim2] = newStride[resolvedDim2], newStride[resolvedDim1]
	return NewLayout(NewShapeFrom(newDims), newStride, l.startOffset), nil
}

// Permute returns a new layout with dimensions reordered according to the permutation indices.
func (l *Layout) Permute(idxs []int) (*Layout, error) {
	rank := l.shape.Rank()
	if len(idxs) != rank {
		return nil, fmt.Errorf("permute idxs len %d != rank %d", len(idxs), rank)
	}
	seen := make(map[int]struct{}, rank)
	resolvedIdxs := make([]int, len(idxs))
	for i, idx := range idxs {
		resolved, err := ResolveAxis(idx, rank)
		if err != nil {
			return nil, err
		}
		if _, exists := seen[resolved]; exists {
			return nil, fmt.Errorf("duplicate index in permute: %v", idxs)
		}
		seen[resolved] = struct{}{}
		resolvedIdxs[i] = resolved
	}
	newDims := make([]int, rank)
	newStride := make([]int, rank)
	for i, idx := range resolvedIdxs {
		newDims[i] = l.Dims()[idx]
		newStride[i] = l.stride[idx]
	}
	return NewLayout(NewShapeFrom(newDims), newStride, l.startOffset), nil
}

// BroadcastAs returns a new layout broadcasted to the target shape.
func (l *Layout) BroadcastAs(target *Shape) (*Layout, error) {
	srcRank := l.shape.Rank()
	tgtRank := target.Rank()
	if tgtRank < srcRank {
		return nil, fmt.Errorf("cannot broadcast to lower rank: src %d, tgt %d", srcRank, tgtRank)
	}
	addedDims := tgtRank - srcRank
	newStride := make([]int, tgtRank)
	for i := range addedDims {
		newStride[i] = 0
	}
	for i := range srcRank {
		srcDim := l.Dims()[i]
		tgtDim := target.Dims()[addedDims+i]
		switch srcDim {
		case tgtDim:
			newStride[addedDims+i] = l.stride[i]
		case 1:
			newStride[addedDims+i] = 0
		default:
			return nil, fmt.Errorf("broadcast incompatible shapes: src %v, tgt %v", l.shape, target)
		}
	}
	return NewLayout(target, newStride, l.startOffset), nil
}

// ContiguousOffsetsWithBroadcast represents contiguous storage with broadcasted dimensions.
type ContiguousOffsetsWithBroadcast struct {
	Start          int
	Len            int
	LeftBroadcast  int
	RightBroadcast int
}

// OffsetsB returns contiguous offsets with broadcast dimensions if applicable,
// along with a boolean indicating success.
func (l *Layout) OffsetsB() (ContiguousOffsetsWithBroadcast, bool) {
	dims := l.Dims()
	strides := l.stride
	leftBroadcast := 1
	rightBroadcast := 1
	startCont := 0
	endCont := len(dims)

	for i := range dims {
		if strides[i] != 0 {
			break
		}
		leftBroadcast *= dims[i]
		startCont++
	}

	if startCont == len(dims) {
		return ContiguousOffsetsWithBroadcast{
			Start:          l.startOffset,
			Len:            1,
			LeftBroadcast:  leftBroadcast,
			RightBroadcast: 1,
		}, true
	}

	for i := len(dims) - 1; i >= 0; i-- {
		if strides[i] != 0 {
			break
		}
		rightBroadcast *= dims[i]
		endCont--
	}

	mStrides := strides[startCont:endCont]
	mDims := dims[startCont:endCont]
	contLen := 1
	for i := len(mDims) - 1; i >= 0; i-- {
		if mStrides[i] != contLen {
			return ContiguousOffsetsWithBroadcast{}, false
		}
		contLen *= mDims[i]
	}

	return ContiguousOffsetsWithBroadcast{
		Start:          l.startOffset,
		Len:            contLen,
		LeftBroadcast:  leftBroadcast,
		RightBroadcast: rightBroadcast,
	}, true
}

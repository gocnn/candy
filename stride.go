package spark

import (
	"iter"
	"slices"
)

// StridedIndex represents an iterator over storage offsets for elements in a multi-dimensional
// tensor stored in a flat buffer with strides.
type StridedIndex struct {
	nextStorageIndex *int  // Nil when iteration is complete.
	multiIndex       []int // Current multi-dimensional index.
	dims             []int // Dimensions (cloned for ownership).
	stride           []int // Strides (cloned for ownership).
}

// NewStridedIndex creates a new StridedIndex iterator.
func NewStridedIndex(dims, stride []int, startOffset int) *StridedIndex {
	if len(dims) != len(stride) {
		panic("dims and stride must have the same length")
	}
	elemCount := 1
	for _, d := range dims {
		if d < 0 {
			panic("negative dimensions not allowed")
		}
		elemCount *= d
	}
	var next *int
	if elemCount > 0 {
		n := startOffset
		next = &n
	}
	return &StridedIndex{
		nextStorageIndex: next,
		multiIndex:       make([]int, len(dims)),
		dims:             slices.Clone(dims),
		stride:           slices.Clone(stride),
	}
}

// NewStridedIndexFromLayout creates a StridedIndex from a Layout.
func NewStridedIndexFromLayout(l Layout) *StridedIndex {
	return NewStridedIndex(l.Dims(), l.stride, l.startOffset)
}

func (si *StridedIndex) NextStorageIndex() int {
	if si.nextStorageIndex == nil {
		return -1
	}
	return *si.nextStorageIndex
}

func (si *StridedIndex) IsComplete() bool {
	return si.nextStorageIndex == nil
}

func (si *StridedIndex) MultiIndex() []int {
	return slices.Clone(si.multiIndex)
}

// Dims returns a copy of the dimensions of the tensor.
func (si *StridedIndex) Dims() []int {
	return slices.Clone(si.dims)
}

// Stride returns a copy of the strides of the tensor.
func (si *StridedIndex) Stride() []int {
	return slices.Clone(si.stride)
}

// All returns an iterator over storage offsets, compatible with Go 1.23 range syntax.
func (si *StridedIndex) All() iter.Seq[int] {
	return func(yield func(int) bool) {
		for si.nextStorageIndex != nil {
			storageIndex := *si.nextStorageIndex
			updated := false
			nextStorageIndex := storageIndex
			// Increment the multi-index like an odometer, starting from the least significant dimension.
			for i := len(si.multiIndex) - 1; i >= 0; i-- {
				nextI := si.multiIndex[i] + 1
				if nextI < si.dims[i] {
					si.multiIndex[i] = nextI
					updated = true
					nextStorageIndex += si.stride[i]
					break
				} else {
					nextStorageIndex -= si.multiIndex[i] * si.stride[i]
					si.multiIndex[i] = 0
				}
			}
			if updated {
				si.nextStorageIndex = &nextStorageIndex
			} else {
				si.nextStorageIndex = nil
			}
			if !yield(storageIndex) {
				return
			}
		}
	}
}

// StridedBlockType distinguishes between single and multiple block variants.
type StridedBlockType int

const (
	SingleBlock StridedBlockType = iota
	MultipleBlocks
)

// StridedBlocks represents either a single contiguous block or multiple contiguous blocks
// with strided starting positions.
type StridedBlocks struct {
	Type            StridedBlockType
	StartOffset     int           // Used for SingleBlock.
	Len             int           // Length for SingleBlock or block length for MultipleBlocks.
	BlockStartIndex *StridedIndex // Used for MultipleBlocks: iterator over block start offsets.
}

// In the Layout struct, add the following method:

// StridedBlocks computes the strided blocks representation for the layout.
// It identifies contiguous inner dimensions and represents the layout as either
// a single block or multiple blocks with strided starts.
func (l Layout) StridedBlocks() StridedBlocks {
	blockLen := 1
	contiguousDims := 0
	dims := l.Dims()
	strides := l.stride
	// Count contiguous dimensions from the right (innermost).
	for i := len(dims) - 1; i >= 0; i-- {
		if strides[i] != blockLen {
			break
		}
		blockLen *= dims[i]
		contiguousDims++
	}
	indexDims := len(dims) - contiguousDims
	if indexDims == 0 {
		return StridedBlocks{
			Type:        SingleBlock,
			StartOffset: l.startOffset,
			Len:         blockLen,
		}
	}
	blockStartIndex := NewStridedIndex(
		dims[:indexDims],
		strides[:indexDims],
		l.startOffset,
	)
	return StridedBlocks{
		Type:            MultipleBlocks,
		Len:             blockLen,
		BlockStartIndex: blockStartIndex,
	}
}

package spark

import (
	"fmt"
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
// Panics if dims and stride lengths mismatch or if any dimension is negative.
// Sets nextStorageIndex to nil for empty dimensions.
func NewStridedIndex(dims, stride []int, startOffset int) *StridedIndex {
	if len(dims) != len(stride) {
		panic(fmt.Sprintf("dims and stride length mismatch: %d != %d", len(dims), len(stride)))
	}
	for _, d := range dims {
		if d < 0 {
			panic(fmt.Sprintf("negative dimension not allowed: %d", d))
		}
	}
	var next *int
	if len(dims) > 0 {
		elemCount := 1
		for _, d := range dims {
			elemCount *= d
		}
		if elemCount > 0 {
			n := startOffset
			next = &n
		}
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
	return NewStridedIndex(l.Dims(), l.Stride(), l.StartOffset())
}

// NextStorageIndex returns the next storage index or nil if iteration is complete.
func (si *StridedIndex) NextStorageIndex() *int {
	return si.nextStorageIndex
}

// IsComplete returns true if the iterator has no more elements.
func (si *StridedIndex) IsComplete() bool {
	return si.nextStorageIndex == nil
}

// MultiIndex returns a copy of the current multi-dimensional index.
func (si *StridedIndex) MultiIndex() []int {
	return slices.Clone(si.multiIndex)
}

// Dims returns a copy of the tensor dimensions.
func (si *StridedIndex) Dims() []int {
	return slices.Clone(si.dims)
}

// Stride returns a copy of the tensor strides.
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
			// Increment the multi-index like an odometer, from right to left.
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

// StridedBlocks computes the strided blocks representation for the layout.
// It identifies contiguous inner dimensions and represents the layout as either
// a single block or multiple blocks with strided starts.
func (l Layout) StridedBlocks() StridedBlocks {
	dims := l.Dims()
	strides := l.Stride()

	// Handle scalar case
	if len(dims) == 0 {
		return StridedBlocks{
			Type:        SingleBlock,
			StartOffset: l.StartOffset(),
			Len:         1,
		}
	}

	// Compute block length for contiguous dimensions from the right
	blockLen := 1
	contiguousDims := 0
	currentStride := 1
	for i := len(dims) - 1; i >= 0; i-- {
		if strides[i] != currentStride {
			break
		}
		blockLen *= dims[i]
		contiguousDims++
		currentStride *= dims[i]
	}

	indexDims := len(dims) - contiguousDims
	if indexDims == 0 {
		return StridedBlocks{
			Type:        SingleBlock,
			StartOffset: l.StartOffset(),
			Len:         blockLen,
		}
	}

	// Create iterator for non-contiguous outer dimensions
	blockStartIndex := NewStridedIndex(
		dims[:indexDims],
		strides[:indexDims],
		l.StartOffset(),
	)
	return StridedBlocks{
		Type:            MultipleBlocks,
		Len:             blockLen,
		BlockStartIndex: blockStartIndex,
	}
}

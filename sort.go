package spark

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// Comparable defines types that support partial comparison for sorting.
// Implementations must handle NaN and other edge cases appropriately.
type Comparable interface {
	// Compare returns -1 if x < y, 0 if x == y, 1 if x > y, or handles special cases (e.g., NaN).
	Compare(other Comparable) int
}

// Float32 implements Comparable for float32.
type Float32 float32

func (x Float32) Compare(other Comparable) int {
	y, ok := other.(Float32)
	if !ok {
		panic("incompatible types for comparison")
	}
	if float32(x) < float32(y) {
		return -1
	} else if float32(x) > float32(y) {
		return 1
	} else if float32(x) == float32(y) {
		return 0
	}
	// Handle NaN: treat as greater than all other values (as in Rust).
	if math.IsNaN(float64(x)) {
		return 1
	}
	return -1
}

// Float64 implements Comparable for float64.
type Float64 float64

func (x Float64) Compare(other Comparable) int {
	y, ok := other.(Float64)
	if !ok {
		panic("incompatible types for comparison")
	}
	if float64(x) < float64(y) {
		return -1
	} else if float64(x) > float64(y) {
		return 1
	} else if float64(x) == float64(y) {
		return 0
	}
	// Handle NaN: treat as greater than all other values.
	if math.IsNaN(float64(x)) {
		return 1
	}
	return -1
}

// ArgSort defines parameters for an argsort operation on a tensor.
type ArgSort struct {
	Asc     bool // True for ascending, false for descending.
	LastDim int  // Size of the last dimension to sort.
}

// NewArgSort creates a new ArgSort with the given parameters.
func NewArgSort(asc bool, lastDim int) ArgSort {
	return ArgSort{Asc: asc, LastDim: lastDim}
}

// ASort performs an argsort on the input slice along the last dimension.
// It returns a slice of indices representing the sorted order of elements.
// The input slice must be a multiple of LastDim, and values must implement Comparable.
func (a ArgSort) ASort(vs []Comparable, layout *Layout) ([]uint32, error) {
	elCount := layout.Shape().ElemCount()
	if elCount == 0 || a.LastDim <= 0 || len(vs) != elCount {
		return nil, fmt.Errorf("invalid argsort parameters: el_count=%d, last_dim=%d, len(vs)=%d", elCount, a.LastDim, len(vs))
	}
	if len(vs)%a.LastDim != 0 {
		return nil, fmt.Errorf("input length %d not divisible by last_dim %d", len(vs), a.LastDim)
	}

	// Initialize index slice.
	indexes := make([]uint32, elCount)
	chunkSize := a.LastDim
	numChunks := len(vs) / chunkSize

	// Parallelize sorting across chunks using goroutines.
	var wg sync.WaitGroup
	wg.Add(numChunks)
	for i := range numChunks {
		go func(chunkIdx int) {
			defer wg.Done()
			start := chunkIdx * chunkSize
			end := start + chunkSize
			chunkIndexes := indexes[start:end]
			chunkVs := vs[start:end]

			// Initialize indices: 0, 1, 2, ..., last_dim-1.
			for j := range chunkIndexes {
				chunkIndexes[j] = uint32(j)
			}

			// Sort indices based on values.
			if a.Asc {
				sort.Slice(chunkIndexes, func(i, j int) bool {
					return chunkVs[chunkIndexes[i]].Compare(chunkVs[chunkIndexes[j]]) < 0
				})
			} else {
				sort.Slice(chunkIndexes, func(i, j int) bool {
					return chunkVs[chunkIndexes[j]].Compare(chunkVs[chunkIndexes[i]]) < 0
				})
			}
		}(i)
	}
	wg.Wait()

	return indexes, nil
}

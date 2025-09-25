package mnist

import (
	"iter"
	"math/rand"

	"github.com/gocnn/spark"
)

// DataLoader manages batch iteration over a generic dataset.
type DataLoader[T spark.D] struct {
	dataset   *Dataset[T]
	batchSize int
	shuffle   bool
	indices   []int
}

// NewDataLoader creates a DataLoader for batch iteration.
func (ds *Dataset[T]) NewDataLoader(batchSize int, shuffle bool) *DataLoader[T] {
	indices := make([]int, ds.Len())
	for i := range indices {
		indices[i] = i
	}
	dl := &DataLoader[T]{
		dataset:   ds,
		batchSize: batchSize,
		shuffle:   shuffle,
		indices:   indices,
	}
	if shuffle {
		rand.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
	}
	return dl
}

// All returns an iterator over batch pairs using iter.Seq2.
func (dl *DataLoader[T]) All() iter.Seq2[[][]T, []uint8] {
	return func(yield func([][]T, []uint8) bool) {
		for start := 0; start < len(dl.indices); start += dl.batchSize {
			end := min(start+dl.batchSize, len(dl.indices))
			batchIndices := dl.indices[start:end]
			images := make([][]T, len(batchIndices))
			labels := make([]uint8, len(batchIndices))
			for i, idx := range batchIndices {
				images[i], labels[i] = dl.dataset.Get(idx)
			}
			if !yield(images, labels) {
				return
			}
		}
	}
}

// Reset reshuffles the indices if shuffle is enabled.
func (dl *DataLoader[T]) Reset() {
	if dl.shuffle {
		rand.Shuffle(len(dl.indices), func(i, j int) { dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i] })
	}
}

// Len returns the number of batches in the DataLoader.
func (dl *DataLoader[T]) Len() int {
	return (len(dl.indices) + dl.batchSize - 1) / dl.batchSize
}

// BatchSize returns the batch size.
func (dl *DataLoader[T]) BatchSize() int {
	return dl.batchSize
}

// IsShuffled returns whether the DataLoader shuffles data.
func (dl *DataLoader[T]) IsShuffled() bool {
	return dl.shuffle
}

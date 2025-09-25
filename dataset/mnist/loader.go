package mnist

import (
	"iter"
	"math/rand"
)

// DataLoader manages batch iteration over a dataset.
type DataLoader struct {
	dataset   *Dataset
	batchSize int
	shuffle   bool
	indices   []int
}

// NewDataLoader creates a DataLoader for batch iteration.
func (ds *Dataset) NewDataLoader(batchSize int, shuffle bool) *DataLoader {
	indices := make([]int, ds.Len())
	for i := range indices {
		indices[i] = i
	}
	dl := &DataLoader{
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
// This approach provides direct access to images and labels without wrapper struct.
// Usage: for images, labels := range dl.All() { ... }
func (dl *DataLoader) All() iter.Seq2[[][]float32, []uint8] {
	return func(yield func([][]float32, []uint8) bool) {
		for start := 0; start < len(dl.indices); start += dl.batchSize {
			end := min(start+dl.batchSize, len(dl.indices))
			batchIndices := dl.indices[start:end]
			images := make([][]float32, len(batchIndices))
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
func (dl *DataLoader) Reset() {
	if dl.shuffle {
		rand.Shuffle(len(dl.indices), func(i, j int) { dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i] })
	}
}

// Len returns the number of batches in the DataLoader.
func (dl *DataLoader) Len() int {
	return (len(dl.indices) + dl.batchSize - 1) / dl.batchSize
}

// BatchSize returns the batch size.
func (dl *DataLoader) BatchSize() int {
	return dl.batchSize
}

// IsShuffled returns whether the DataLoader shuffles data.
func (dl *DataLoader) IsShuffled() bool {
	return dl.shuffle
}

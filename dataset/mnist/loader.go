package mnist

import (
	"iter"
	"math/rand"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

// DataLoader manages batch iteration over a generic dataset.
type DataLoader[T spark.D] struct {
	dataset   *Dataset[T]
	batchSize int
	shuffle   bool
	indices   []int
	device    spark.Device
}

// NewDataLoader creates a DataLoader for batch iteration.
func (ds *Dataset[T]) NewDataLoader(batchSize int, shuffle bool, device spark.Device) *DataLoader[T] {
	indices := make([]int, ds.Len())
	for i := range indices {
		indices[i] = i
	}
	dl := &DataLoader[T]{
		dataset:   ds,
		batchSize: batchSize,
		shuffle:   shuffle,
		indices:   indices,
		device:    device,
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

// AllTensors returns an iterator over tensor batches.
func (dl *DataLoader[T]) AllTensors() iter.Seq2[*tensor.Tensor[T], *tensor.Tensor[T]] {
	return func(yield func(*tensor.Tensor[T], *tensor.Tensor[T]) bool) {
		for start := 0; start < len(dl.indices); start += dl.batchSize {
			end := min(start+dl.batchSize, len(dl.indices))
			batchIndices := dl.indices[start:end]
			batchSize := len(batchIndices)

			// Prepare image data
			imageData := make([]T, batchSize*28*28)
			labelData := make([]T, batchSize)

			for i, idx := range batchIndices {
				img, lbl := dl.dataset.Get(idx)
				copy(imageData[i*28*28:(i+1)*28*28], img)
				labelData[i] = T(lbl)
			}

			// Create tensors
			imageTensor, err := tensor.New(imageData, spark.NewShape(batchSize, 1, 28, 28), dl.device)
			if err != nil {
				panic(err) // In production, consider better error handling
			}

			labelTensor, err := tensor.New(labelData, spark.NewShape(batchSize), dl.device)
			if err != nil {
				panic(err) // In production, consider better error handling
			}

			if !yield(imageTensor, labelTensor) {
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

// Device returns the device for tensor creation.
func (dl *DataLoader[T]) Device() spark.Device {
	return dl.device
}

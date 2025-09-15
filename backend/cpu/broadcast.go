package cpu

import (
	"errors"

	"github.com/qntx/goml"
)

// Broadcasting and repetition methods for Tensor tensors

// BroadcastAs broadcasts this tensor to the specified shape.
func (d *Tensor[T]) BroadcastAs(targetShape goml.Shape) (*Tensor[T], error) {
	if !d.shape.CanBroadcastWith(targetShape) {
		return nil, NewShapeError("BroadcastAs", ErrIncompatibleShape, targetShape, d.shape)
	}

	if d.shape.Equals(targetShape) {
		// Already the target shape
		return &Tensor[T]{
			data:         d.data,
			shape:        d.shape,
			requiresGrad: d.requiresGrad,
		}, nil
	}

	// Create new data with broadcasting
	newData := make([]T, targetShape.Size())
	d.broadcastData(newData, targetShape)

	return &Tensor[T]{
		data:         newData,
		shape:        targetShape,
		requiresGrad: d.requiresGrad,
	}, nil
}

// Expand is an alias for BroadcastAs.
func (d *Tensor[T]) Expand(targetShape goml.Shape) (*Tensor[T], error) {
	return d.BroadcastAs(targetShape)
}

// BroadcastLeft broadcasts by adding dimensions on the left.
func (d *Tensor[T]) BroadcastLeft(leftShape goml.Shape) (*Tensor[T], error) {
	// Combine left shape with current shape
	leftDims := leftShape.Dims()
	currentDims := d.shape.Dims()
	newDims := make([]int, len(leftDims)+len(currentDims))
	copy(newDims, leftDims)
	copy(newDims[len(leftDims):], currentDims)

	targetShape := goml.NewShapeFromSlice(newDims)
	return d.BroadcastAs(targetShape)
}

// Repeat repeats the tensor along each dimension according to the given counts.
func (d *Tensor[T]) Repeat(repeats ...int) (*Tensor[T], error) {
	if len(repeats) != d.shape.Ndim() {
		return nil, ErrInvalidRepeatCounts
	}

	// Calculate new shape
	currentDims := d.shape.Dims()
	newDims := make([]int, len(currentDims))
	for i, repeat := range repeats {
		if repeat <= 0 {
			return nil, ErrInvalidRepeatCount
		}
		newDims[i] = currentDims[i] * repeat
	}

	newShape := goml.NewShapeFromSlice(newDims)
	newData := make([]T, newShape.Size())
	d.repeatData(newData, repeats, newShape)

	return &Tensor[T]{
		data:         newData,
		shape:        newShape,
		requiresGrad: d.requiresGrad,
	}, nil
}

// broadcastData performs the actual broadcasting operation
func (d *Tensor[T]) broadcastData(newData []T, targetShape goml.Shape) {
	sourceDims := d.shape.Dims()
	targetDims := targetShape.Dims()

	sourceStrides := calculateStrides(sourceDims)
	targetStrides := calculateStrides(targetDims)

	// Align dimensions from the right
	dimOffset := len(targetDims) - len(sourceDims)

	for i := 0; i < len(newData); i++ {
		targetIndices := flatToMultiIndex(i, targetDims, targetStrides)
		sourceIndices := make([]int, len(sourceDims))

		// Map target indices to source indices
		for j := 0; j < len(sourceDims); j++ {
			targetIdx := targetIndices[dimOffset+j]
			if sourceDims[j] == 1 {
				sourceIndices[j] = 0 // Broadcast dimension
			} else {
				sourceIndices[j] = targetIdx
			}
		}

		sourceIndex := multiToFlatIndex(sourceIndices, sourceStrides)
		newData[i] = d.data[sourceIndex]
	}
}

// repeatData performs the actual repetition operation
func (d *Tensor[T]) repeatData(newData []T, repeats []int, newShape goml.Shape) {
	sourceDims := d.shape.Dims()
	targetDims := newShape.Dims()

	sourceStrides := calculateStrides(sourceDims)
	targetStrides := calculateStrides(targetDims)

	for i := 0; i < len(newData); i++ {
		targetIndices := flatToMultiIndex(i, targetDims, targetStrides)
		sourceIndices := make([]int, len(sourceDims))

		// Map target indices to source indices
		for j := 0; j < len(sourceDims); j++ {
			sourceIndices[j] = targetIndices[j] % sourceDims[j]
		}

		sourceIndex := multiToFlatIndex(sourceIndices, sourceStrides)
		newData[i] = d.data[sourceIndex]
	}
}

// Contiguous returns a contiguous copy of the tensor if it's not already contiguous.
// For now, all Tensor tensors are contiguous, so this just returns a reference.
func (d *Tensor[T]) Contiguous() *Tensor[T] {
	return &Tensor[T]{
		data:         d.data,
		shape:        d.shape,
		requiresGrad: d.requiresGrad,
	}
}

// ForceContiguous always creates a new contiguous copy of the tensor.
func (d *Tensor[T]) ForceContiguous() *Tensor[T] {
	newData := make([]T, len(d.data))
	copy(newData, d.data)

	return &Tensor[T]{
		data:         newData,
		shape:        d.shape,
		requiresGrad: d.requiresGrad,
	}
}

var (
	ErrInvalidRepeatCounts = errors.New("invalid repeat counts")
	ErrInvalidRepeatCount  = errors.New("invalid repeat count")
)

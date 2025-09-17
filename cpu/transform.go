package cpu

import (
	"fmt"

	"github.com/qntx/goml"
)

// Reshape returns a new tensor with the specified shape.
// Use -1 for one dimension to auto-infer its size.
func (d *Tensor[T]) Reshape(dims ...int) goml.Tensor[T] {
	newShape := d.Shape().Reshape(dims...)

	return &Tensor[T]{
		data:         d.data, // Share the same underlying data
		layout:       goml.Contiguous(newShape),
		requiresGrad: d.requiresGrad,
	}
}

// Squeeze removes dimensions of size 1 at the specified positions.
func (d *Tensor[T]) Squeeze(dims ...int) (*Tensor[T], error) {
	if len(dims) == 0 {
		// Remove all dimensions of size 1
		return d.squeezeAll()
	}

	newDims := make([]int, 0, d.Shape().Ndim())
	currentDims := d.Shape().Dims()

	// Create a set of dimensions to squeeze
	squeezeSet := make(map[int]bool)
	for _, dim := range dims {
		if dim < 0 {
			dim = d.Shape().Ndim() + dim
		}
		if dim < 0 || dim >= d.Shape().Ndim() {
			return nil, NewDimensionError("Squeeze", ErrDimensionOutOfRange, dim, d.Shape().Ndim())
		}
		if currentDims[dim] != 1 {
			return nil, NewOperationError("Squeeze", fmt.Sprintf("dimension %d has size %d, not 1", dim, currentDims[dim]))
		}
		squeezeSet[dim] = true
	}

	// Build new dimensions
	for i, size := range currentDims {
		if !squeezeSet[i] {
			newDims = append(newDims, size)
		}
	}

	return &Tensor[T]{
		data:         d.data,
		layout:       goml.Contiguous(goml.NewShapeFromSlice(newDims)),
		requiresGrad: d.requiresGrad,
	}, nil
}

// squeezeAll removes all dimensions of size 1
func (d *Tensor[T]) squeezeAll() (*Tensor[T], error) {
	newDims := make([]int, 0, d.Shape().Ndim())
	for _, size := range d.Shape().Dims() {
		if size != 1 {
			newDims = append(newDims, size)
		}
	}

	// If all dimensions were 1, keep one dimension
	if len(newDims) == 0 {
		newDims = []int{1}
	}

	return &Tensor[T]{
		data:         d.data,
		layout:       goml.Contiguous(goml.NewShapeFromSlice(newDims)),
		requiresGrad: d.requiresGrad,
	}, nil
}

// MustSqueeze is like Squeeze but panics on error.
func (d *Tensor[T]) MustSqueeze(dims ...int) *Tensor[T] {
	result, err := d.Squeeze(dims...)
	if err != nil {
		panic(err)
	}
	return result
}

// Unsqueeze adds a dimension of size 1 at the specified position.
func (d *Tensor[T]) Unsqueeze(dim int) (*Tensor[T], error) {
	ndim := d.Shape().Ndim()
	if dim < 0 {
		dim = ndim + 1 + dim
	}
	if dim < 0 || dim > ndim {
		return nil, NewDimensionError("Unsqueeze", ErrDimensionOutOfRange, dim, ndim+1)
	}

	currentDims := d.Shape().Dims()
	newDims := make([]int, 0, ndim+1)

	// Insert the new dimension
	for i := 0; i < dim; i++ {
		newDims = append(newDims, currentDims[i])
	}
	newDims = append(newDims, 1)
	for i := dim; i < ndim; i++ {
		newDims = append(newDims, currentDims[i])
	}

	return &Tensor[T]{
		data:         d.data,
		layout:       goml.Contiguous(goml.NewShapeFromSlice(newDims)),
		requiresGrad: d.requiresGrad,
	}, nil
}

// MustUnsqueeze is like Unsqueeze but panics on error.
func (d *Tensor[T]) MustUnsqueeze(dim int) *Tensor[T] {
	result, err := d.Unsqueeze(dim)
	if err != nil {
		panic(err)
	}
	return result
}

// Flatten flattens dimensions from startDim to endDim (inclusive).
func (d *Tensor[T]) Flatten(startDim, endDim int) (*Tensor[T], error) {
	ndim := d.Shape().Ndim()
	if startDim < 0 {
		startDim = ndim + startDim
	}
	if endDim < 0 {
		endDim = ndim + endDim
	}

	if startDim < 0 || startDim >= ndim {
		return nil, NewDimensionError("Flatten", ErrDimensionOutOfRange, startDim, ndim)
	}
	if endDim < 0 || endDim >= ndim {
		return nil, NewDimensionError("Flatten", ErrDimensionOutOfRange, endDim, ndim)
	}
	if startDim > endDim {
		return nil, NewOperationError("Flatten", fmt.Sprintf("startDim %d > endDim %d", startDim, endDim))
	}

	currentDims := d.Shape().Dims()
	newDims := make([]int, 0, ndim-(endDim-startDim))

	// Add dimensions before startDim
	for i := 0; i < startDim; i++ {
		newDims = append(newDims, currentDims[i])
	}

	// Calculate flattened dimension size
	flattenedSize := 1
	for i := startDim; i <= endDim; i++ {
		flattenedSize *= currentDims[i]
	}
	newDims = append(newDims, flattenedSize)

	// Add dimensions after endDim
	for i := endDim + 1; i < ndim; i++ {
		newDims = append(newDims, currentDims[i])
	}

	return &Tensor[T]{
		data:         d.data,
		layout:       goml.Contiguous(goml.NewShapeFromSlice(newDims)),
		requiresGrad: d.requiresGrad,
	}, nil
}

// FlattenTo flattens from dimension 0 to endDim (inclusive).
func (d *Tensor[T]) FlattenTo(endDim int) (*Tensor[T], error) {
	return d.Flatten(0, endDim)
}

// FlattenFrom flattens from startDim to the last dimension.
func (d *Tensor[T]) FlattenFrom(startDim int) (*Tensor[T], error) {
	return d.Flatten(startDim, d.Shape().Ndim()-1)
}

// FlattenAll flattens the tensor to 1D.
func (d *Tensor[T]) FlattenAll() *Tensor[T] {
	return &Tensor[T]{
		data:         d.data,
		layout:       goml.Contiguous(goml.NewShape(d.Shape().Size())),
		requiresGrad: d.requiresGrad,
	}
}

// MustFlatten is like Flatten but panics on error.
func (d *Tensor[T]) MustFlatten(startDim, endDim int) *Tensor[T] {
	result, err := d.Flatten(startDim, endDim)
	if err != nil {
		panic(err)
	}
	return result
}

// Transpose swaps two dimensions.
func (d *Tensor[T]) Transpose(dim1, dim2 int) (*Tensor[T], error) {
	ndim := d.Shape().Ndim()
	if dim1 < 0 {
		dim1 = ndim + dim1
	}
	if dim2 < 0 {
		dim2 = ndim + dim2
	}

	if dim1 < 0 || dim1 >= ndim {
		return nil, NewDimensionError("Transpose", ErrDimensionOutOfRange, dim1, ndim)
	}
	if dim2 < 0 || dim2 >= ndim {
		return nil, NewDimensionError("Transpose", ErrDimensionOutOfRange, dim2, ndim)
	}

	if dim1 == dim2 {
		// No change needed
		return &Tensor[T]{
			data:         d.data,
			layout:       d.layout,
			requiresGrad: d.requiresGrad,
		}, nil
	}

	// Create permutation array
	perm := make([]int, ndim)
	for i := range perm {
		perm[i] = i
	}
	perm[dim1], perm[dim2] = perm[dim2], perm[dim1]

	return d.Permute(perm...)
}

// T transposes the last two dimensions (for 2D tensors, this is matrix transpose).
func (d *Tensor[T]) T() (*Tensor[T], error) {
	ndim := d.Shape().Ndim()
	if ndim < 2 {
		return nil, NewOperationError("T", fmt.Sprintf("tensor must have at least 2 dimensions, got %d", ndim))
	}
	return d.Transpose(ndim-2, ndim-1)
}

// MustT is like T but panics on error.
func (d *Tensor[T]) MustT() *Tensor[T] {
	result, err := d.T()
	if err != nil {
		panic(err)
	}
	return result
}

// Permute rearranges the dimensions according to the given permutation.
func (d *Tensor[T]) Permute(dims ...int) (*Tensor[T], error) {
	ndim := d.Shape().Ndim()
	if len(dims) != ndim {
		return nil, NewOperationError("Permute", fmt.Sprintf("expected %d dimensions, got %d", ndim, len(dims)))
	}

	// Validate permutation
	used := make([]bool, ndim)
	newDims := make([]int, ndim)
	currentDims := d.Shape().Dims()

	for i, dim := range dims {
		if dim < 0 {
			dim = ndim + dim
		}
		if dim < 0 || dim >= ndim {
			return nil, NewDimensionError("Permute", ErrDimensionOutOfRange, dim, ndim)
		}
		if used[dim] {
			return nil, NewOperationError("Permute", fmt.Sprintf("dimension %d used multiple times", dim))
		}
		used[dim] = true
		newDims[i] = currentDims[dim]
	}

	// For now, we'll create a new tensor with copied data
	// TODO: Implement efficient strided tensor operations
	newData := make([]T, len(d.data))
	d.permuteData(newData, dims)

	return &Tensor[T]{
		data:         newData,
		layout:       goml.Contiguous(goml.NewShapeFromSlice(newDims)),
		requiresGrad: d.requiresGrad,
	}, nil
}

// permuteData performs the actual data permutation
func (d *Tensor[T]) permuteData(newData []T, dims []int) {
	// This is a simplified implementation
	// For production, you'd want to use more efficient algorithms
	currentDims := d.Shape().Dims()
	newDims := make([]int, len(dims))
	for i, dim := range dims {
		newDims[i] = currentDims[dim]
	}

	// Calculate strides for both old and new layouts
	oldStrides := calculateStrides(currentDims)
	newStrides := calculateStrides(newDims)

	// Copy data with permutation
	for i := 0; i < len(d.data); i++ {
		oldIndices := flatToMultiIndex(i, currentDims, oldStrides)
		newIndices := make([]int, len(oldIndices))
		for j, dim := range dims {
			newIndices[j] = oldIndices[dim]
		}
		newIndex := multiToFlatIndex(newIndices, newStrides)
		newData[newIndex] = d.data[i]
	}
}

// Helper functions for index calculations
func calculateStrides(dims []int) []int {
	strides := make([]int, len(dims))
	stride := 1
	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= dims[i]
	}
	return strides
}

func flatToMultiIndex(flatIndex int, dims, strides []int) []int {
	indices := make([]int, len(dims))
	for i := range dims {
		indices[i] = flatIndex / strides[i]
		flatIndex %= strides[i]
	}
	return indices
}

func multiToFlatIndex(indices, strides []int) int {
	flatIndex := 0
	for i, idx := range indices {
		flatIndex += idx * strides[i]
	}
	return flatIndex
}

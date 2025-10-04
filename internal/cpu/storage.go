package cpu

import (
	"errors"
	"slices"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/internal/cpu/kernels"
)

var _ spark.BackendStorage[float32] = (*CpuStorage[float32])(nil)
var _ spark.BackendStorage[float64] = (*CpuStorage[float64])(nil)

type CpuStorage[T kernels.D] struct {
	data   []T
	device *CpuDevice[T]
	dtype  spark.DType
}

func New[T kernels.D](data []T) *CpuStorage[T] {
	return &CpuStorage[T]{data: data, device: &CpuDevice[T]{}, dtype: spark.DTypeOf[T]()}
}

func (s *CpuStorage[T]) TryClone() (spark.BackendStorage[T], error) {
	return &CpuStorage[T]{data: slices.Clone(s.data), device: s.device, dtype: s.dtype}, nil
}

func (s *CpuStorage[T]) Data() []T {
	return slices.Clone(s.data)
}

func (s *CpuStorage[T]) Device() spark.BackendDevice[T] {
	return s.device
}

func (s *CpuStorage[T]) DType() spark.DType {
	return s.dtype
}

// Affine performs an affine transformation on the storage.
func (s *CpuStorage[T]) Affine(layout *spark.Layout, scale, bias T) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))
	kernels.AffineStrided(
		numel,           // numel
		layout.Rank(),   // ndims
		layout.Dims(),   // dims
		layout.Stride(), // strides
		scale,           // scale
		bias,            // bias
		s.data,          // x
		result.data,     // y
	)
	return result, nil
}

// Add performs element-wise addition of two tensors.
func (s *CpuStorage[T]) Add(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[T], error) {
	rhsC, ok := rhs.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("rhs storage must be CpuStorage")
	}
	if lhsLayout == nil {
		return nil, errors.New("lhsLayout cannot be nil")
	}
	if rhsLayout == nil {
		return nil, errors.New("rhsLayout cannot be nil")
	}
	if resLayout == nil {
		return nil, errors.New("resLayout cannot be nil")
	}
	if lhsLayout.ElemCount() != rhsLayout.ElemCount() || lhsLayout.ElemCount() != resLayout.ElemCount() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.ElemCount()))
	kernels.BAddStrided(
		lhsLayout.ElemCount(), // numel
		lhsLayout.Rank(),      // ndims
		lhsLayout.Dims(),      // dims
		lhsLayout.Stride(),    // stridesX1
		rhsLayout.Stride(),    // stridesX2
		resLayout.Stride(),    // stridesY
		s.data,                // x1
		rhsC.data,             // x2
		result.data,           // y
	)

	return result, nil
}

// Sub performs element-wise subtraction of two tensors.
func (s *CpuStorage[T]) Sub(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[T], error) {
	rhsC, ok := rhs.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("rhs storage must be CpuStorage")
	}
	if lhsLayout == nil {
		return nil, errors.New("lhsLayout cannot be nil")
	}
	if rhsLayout == nil {
		return nil, errors.New("rhsLayout cannot be nil")
	}
	if resLayout == nil {
		return nil, errors.New("resLayout cannot be nil")
	}
	if lhsLayout.ElemCount() != rhsLayout.ElemCount() || lhsLayout.ElemCount() != resLayout.ElemCount() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.ElemCount()))
	kernels.BSubStrided(
		lhsLayout.ElemCount(), // numel
		lhsLayout.Rank(),      // ndims
		lhsLayout.Dims(),      // dims
		lhsLayout.Stride(),    // stridesX1
		rhsLayout.Stride(),    // stridesX2
		resLayout.Stride(),    // stridesY
		s.data,                // x1
		rhsC.data,             // x2
		result.data,           // y
	)

	return result, nil
}

// Mul performs element-wise multiplication of two tensors.
func (s *CpuStorage[T]) Mul(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[T], error) {
	rhsC, ok := rhs.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("rhs storage must be CpuStorage")
	}
	if lhsLayout == nil {
		return nil, errors.New("lhsLayout cannot be nil")
	}
	if rhsLayout == nil {
		return nil, errors.New("rhsLayout cannot be nil")
	}
	if resLayout == nil {
		return nil, errors.New("resLayout cannot be nil")
	}
	if lhsLayout.ElemCount() != rhsLayout.ElemCount() || lhsLayout.ElemCount() != resLayout.ElemCount() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.ElemCount()))
	kernels.BMulStrided(
		lhsLayout.ElemCount(), // numel
		lhsLayout.Rank(),      // ndims
		lhsLayout.Dims(),      // dims
		lhsLayout.Stride(),    // stridesX1
		rhsLayout.Stride(),    // stridesX2
		resLayout.Stride(),    // stridesY
		s.data,                // x1
		rhsC.data,             // x2
		result.data,           // y
	)

	return result, nil
}

// Div performs element-wise division of two tensors.
func (s *CpuStorage[T]) Div(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[T], error) {
	rhsC, ok := rhs.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("rhs storage must be CpuStorage")
	}
	if lhsLayout == nil {
		return nil, errors.New("lhsLayout cannot be nil")
	}
	if rhsLayout == nil {
		return nil, errors.New("rhsLayout cannot be nil")
	}
	if resLayout == nil {
		return nil, errors.New("resLayout cannot be nil")
	}
	if lhsLayout.ElemCount() != rhsLayout.ElemCount() || lhsLayout.ElemCount() != resLayout.ElemCount() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.ElemCount()))
	kernels.BDivStrided(
		lhsLayout.ElemCount(), // numel
		lhsLayout.Rank(),      // ndims
		lhsLayout.Dims(),      // dims
		lhsLayout.Stride(),    // stridesX1
		rhsLayout.Stride(),    // stridesX2
		resLayout.Stride(),    // stridesY
		s.data,                // x1
		rhsC.data,             // x2
		result.data,           // y
	)

	return result, nil
}

// Sqrt performs element-wise square root operation.
func (s *CpuStorage[T]) Sqrt(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))
	kernels.USqrtStrided(
		numel,           // numel
		layout.Rank(),   // ndims
		layout.Dims(),   // dims
		layout.Stride(), // strides
		s.data,          // inp
		result.data,     // out
	)
	return result, nil
}

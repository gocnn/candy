package cpu

import (
	"errors"
	"slices"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/internal/cpu/kernels"
)

type CpuStorage[T kernels.D] struct {
	data   []T
	device spark.BackendDevice[T]
	dtype  spark.DType
}

func New[T kernels.D](data []T) *CpuStorage[T] {
	return &CpuStorage[T]{data: data, device: &CpuDevice[T]{}, dtype: spark.DTypeOf[T]()}
}

func (s *CpuStorage[T]) TryClone(layout *spark.Layout) (spark.BackendStorage[T], error) {
	return &CpuStorage[T]{data: slices.Clone(s.data), device: s.device, dtype: s.dtype}, nil
}

func (s *CpuStorage[T]) DType() spark.DType {
	return s.dtype
}

func (s *CpuStorage[T]) Device() spark.BackendDevice[T] {
	return s.device
}

func (s *CpuStorage[T]) Affine(layout *spark.Layout, scale, bias T) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}
	result := New(make([]T, numel))
	kernels.AffineStrided(numel, layout.Rank(), layout.Dims(), layout.Stride(), scale, bias, s.data, result.data)
	return result, nil
}

func (s *CpuStorage[T]) Add(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resultLayout *spark.Layout) (spark.BackendStorage[T], error) {
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
	if lhsLayout.ElemCount() != rhsLayout.ElemCount() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resultLayout.ElemCount()))
	kernels.BAddStrided(
		lhsLayout.ElemCount(), // numel
		lhsLayout.Rank(),      // numDims
		lhsLayout.Dims(),      // dims
		lhsLayout.Stride(),    // stridesX1
		rhsLayout.Stride(),    // stridesX2
		resultLayout.Stride(), // stridesY
		s.data,                // x1
		rhsC.data,             // x2
		result.data,           // y
	)

	return result, nil
}

func (s *CpuStorage[T]) Mul(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resultLayout *spark.Layout) (spark.BackendStorage[T], error) {
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
	if lhsLayout.ElemCount() != rhsLayout.ElemCount() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resultLayout.ElemCount()))
	kernels.BMulStrided(
		lhsLayout.ElemCount(), // numel
		lhsLayout.Rank(),      // numDims
		lhsLayout.Dims(),      // dims
		lhsLayout.Stride(),    // stridesX1
		rhsLayout.Stride(),    // stridesX2
		resultLayout.Stride(), // stridesY
		s.data,                // x1
		rhsC.data,             // x2
		result.data,           // y
	)

	return result, nil
}

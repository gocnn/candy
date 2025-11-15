package cpu

import (
	"errors"
	"fmt"
	"slices"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/tensor/internal/cpu/kernels"
)

var _ candy.BackendStorage[float32] = (*CpuStorage[float32])(nil)
var _ candy.BackendStorage[float64] = (*CpuStorage[float64])(nil)
var _ candy.BackendStorage[uint8] = (*CpuStorage[uint8])(nil)
var _ candy.BackendStorage[uint32] = (*CpuStorage[uint32])(nil)
var _ candy.BackendStorage[int64] = (*CpuStorage[int64])(nil)

type CpuStorage[T kernels.D] struct {
	data   []T
	device *CpuDevice[T]
	dtype  candy.DType
}

func New[T kernels.D](data []T) *CpuStorage[T] {
	return &CpuStorage[T]{data: data, device: &CpuDevice[T]{}, dtype: candy.DTypeOf[T]()}
}

func (s *CpuStorage[T]) Clone() (candy.BackendStorage[T], error) {
	return &CpuStorage[T]{data: slices.Clone(s.data), device: s.device, dtype: s.dtype}, nil
}

func (s *CpuStorage[T]) Data() []T {
	return s.data
}

func (s *CpuStorage[T]) Device() candy.BackendDevice[T] {
	return s.device
}

func (s *CpuStorage[T]) DType() candy.DType {
	return s.dtype
}

// Affine performs an affine transformation on the storage.
func (s *CpuStorage[T]) Affine(layout *candy.Layout, scale, bias T) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	numel := layout.Numel()
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
func (s *CpuStorage[T]) Add(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.BAddStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// Sub performs element-wise subtraction of two tensors.
func (s *CpuStorage[T]) Sub(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.BSubStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// Mul performs element-wise multiplication of two tensors.
func (s *CpuStorage[T]) Mul(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.BMulStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// Div performs element-wise division of two tensors.
func (s *CpuStorage[T]) Div(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.BDivStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// Max performs element-wise maximum of two tensors.
func (s *CpuStorage[T]) Maximum(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.BMaximumStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// Min performs element-wise minimum of two tensors.
func (s *CpuStorage[T]) Minimum(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.BMinimumStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// Eq performs element-wise equality comparison of two tensors.
func (s *CpuStorage[T]) Eq(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.EqStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// Ne performs element-wise not-equal comparison of two tensors.
func (s *CpuStorage[T]) Ne(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.NeStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// Lt performs element-wise less-than comparison of two tensors.
func (s *CpuStorage[T]) Lt(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.LtStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// Le performs element-wise less-than-or-equal comparison of two tensors.
func (s *CpuStorage[T]) Le(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.LeStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// Gt performs element-wise greater-than comparison of two tensors.
func (s *CpuStorage[T]) Gt(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.GtStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// Ge performs element-wise greater-than-or-equal comparison of two tensors.
func (s *CpuStorage[T]) Ge(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[T], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]T, resLayout.Numel()))
	kernels.GeStrided(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// EqU8 performs element-wise equality comparison of two tensors.
func (s *CpuStorage[T]) EqU8(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[uint8], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]uint8, resLayout.Numel()))
	kernels.EqStridedU8(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// NeU8 performs element-wise not-equal comparison of two tensors.
func (s *CpuStorage[T]) NeU8(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[uint8], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]uint8, resLayout.Numel()))
	kernels.NeStridedU8(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// LtU8 performs element-wise less-than comparison of two tensors.
func (s *CpuStorage[T]) LtU8(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[uint8], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]uint8, resLayout.Numel()))
	kernels.LtStridedU8(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// LeU8 performs element-wise less-than-or-equal comparison of two tensors.
func (s *CpuStorage[T]) LeU8(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[uint8], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]uint8, resLayout.Numel()))
	kernels.LeStridedU8(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// GtU8 performs element-wise greater-than comparison of two tensors.
func (s *CpuStorage[T]) GtU8(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[uint8], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]uint8, resLayout.Numel()))
	kernels.GtStridedU8(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// GeU8 performs element-wise greater-than-or-equal comparison of two tensors.
func (s *CpuStorage[T]) GeU8(rhs candy.BackendStorage[T], lhsLayout *candy.Layout, rhsLayout *candy.Layout, resLayout *candy.Layout) (candy.BackendStorage[uint8], error) {
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
	if lhsLayout.Numel() != rhsLayout.Numel() || lhsLayout.Numel() != resLayout.Numel() {
		return nil, errors.New("lhsLayout element count does not match rhsLayout element count")
	}

	result := New(make([]uint8, resLayout.Numel()))
	kernels.GeStridedU8(
		lhsLayout.Numel(),  // numel
		lhsLayout.Rank(),   // ndims
		lhsLayout.Dims(),   // dims
		lhsLayout.Stride(), // stridesX1
		rhsLayout.Stride(), // stridesX2
		resLayout.Stride(), // stridesY
		s.data,             // x1
		rhsC.data,          // x2
		result.data,        // y
	)

	return result, nil
}

// ToDtype performs type conversion to the specified target type.
func (s *CpuStorage[T]) ToDtype(layout *candy.Layout, dtype candy.DType) (any, error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	srcDtype := s.dtype
	if srcDtype == dtype {
		cloned, err := s.Clone()
		return cloned, err
	}

	// Handle type conversions based on source type
	switch srcDtype {
	case candy.F32:
		return s.CastFromF32(numel, layout, dtype)
	case candy.F64:
		return s.CastFromF64(numel, layout, dtype)
	case candy.U8:
		return s.CastFromU8(numel, layout, dtype)
	case candy.U32:
		return s.CastFromU32(numel, layout, dtype)
	case candy.I64:
		return s.CastFromI64(numel, layout, dtype)
	default:
		return nil, errors.New("unsupported source type: " + srcDtype.String())
	}
}

func (s *CpuStorage[T]) CastFromF32(numel int, layout *candy.Layout, dtype candy.DType) (any, error) {
	srcData := any(s.data).([]float32)
	stride := layout.Stride()
	dims := layout.Dims()
	ndims := layout.Rank()

	switch dtype {
	case candy.F64:
		result := New(make([]float64, numel))
		kernels.CastStridedF32F64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.U8:
		result := New(make([]uint8, numel))
		kernels.CastStridedF32U8(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.U32:
		result := New(make([]uint32, numel))
		kernels.CastStridedF32U32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.I64:
		result := New(make([]int64, numel))
		kernels.CastStridedF32I64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	}
	return nil, errors.New("unsupported target type: " + dtype.String())
}

func (s *CpuStorage[T]) CastFromF64(numel int, layout *candy.Layout, dtype candy.DType) (any, error) {
	srcData := any(s.data).([]float64)
	stride := layout.Stride()
	dims := layout.Dims()
	ndims := layout.Rank()

	switch dtype {
	case candy.F32:
		result := New(make([]float32, numel))
		kernels.CastStridedF64F32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.U8:
		result := New(make([]uint8, numel))
		kernels.CastStridedF64U8(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.U32:
		result := New(make([]uint32, numel))
		kernels.CastStridedF64U32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.I64:
		result := New(make([]int64, numel))
		kernels.CastStridedF64I64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	}
	return nil, errors.New("unsupported target type: " + dtype.String())
}

func (s *CpuStorage[T]) CastFromU8(numel int, layout *candy.Layout, dtype candy.DType) (any, error) {
	srcData := any(s.data).([]uint8)
	stride := layout.Stride()
	dims := layout.Dims()
	ndims := layout.Rank()

	switch dtype {
	case candy.F32:
		result := New(make([]float32, numel))
		kernels.CastStridedU8F32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.F64:
		result := New(make([]float64, numel))
		kernels.CastStridedU8F64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.U32:
		result := New(make([]uint32, numel))
		kernels.CastStridedU8U32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.I64:
		result := New(make([]int64, numel))
		kernels.CastStridedU8I64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	}
	return nil, errors.New("unsupported target type: " + dtype.String())
}

func (s *CpuStorage[T]) CastFromU32(numel int, layout *candy.Layout, dtype candy.DType) (any, error) {
	srcData := any(s.data).([]uint32)
	stride := layout.Stride()
	dims := layout.Dims()
	ndims := layout.Rank()

	switch dtype {
	case candy.F32:
		result := New(make([]float32, numel))
		kernels.CastStridedU32F32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.F64:
		result := New(make([]float64, numel))
		kernels.CastStridedU32F64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.U8:
		result := New(make([]uint8, numel))
		kernels.CastStridedU32U8(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.I64:
		result := New(make([]int64, numel))
		kernels.CastStridedU32I64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	}
	return nil, errors.New("unsupported target type: " + dtype.String())
}

func (s *CpuStorage[T]) CastFromI64(numel int, layout *candy.Layout, dtype candy.DType) (any, error) {
	srcData := any(s.data).([]int64)
	stride := layout.Stride()
	dims := layout.Dims()
	ndims := layout.Rank()

	switch dtype {
	case candy.F32:
		result := New(make([]float32, numel))
		kernels.CastStridedI64F32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.F64:
		result := New(make([]float64, numel))
		kernels.CastStridedI64F64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.U8:
		result := New(make([]uint8, numel))
		kernels.CastStridedI64U8(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case candy.U32:
		result := New(make([]uint32, numel))
		kernels.CastStridedI64U32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	}
	return nil, errors.New("unsupported target type: " + dtype.String())
}

// MatMul performs matrix multiplication: C = A * B
func (s *CpuStorage[T]) MatMul(lhsLayout *candy.Layout, rhs candy.BackendStorage[T], rhsLayout *candy.Layout, b, m, n, k int) (candy.BackendStorage[T], error) {
	rhsC, ok := rhs.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("rhs storage must be CpuStorage")
	}

	if lhsLayout == nil || rhsLayout == nil {
		return nil, errors.New("layouts cannot be nil")
	}

	if m <= 0 || n <= 0 || k <= 0 {
		return nil, errors.New("invalid matrix dimensions")
	}

	result := New(make([]T, b*m*n))

	switch any(s.data).(type) {
	case []float32:
		lhsData := any(s.data).([]float32)
		rhsData := any(rhsC.data).([]float32)
		resultData := any(result.data).([]float32)

		if lhsLayout.IsContiguous() && rhsLayout.IsContiguous() {
			kernels.NaiveBatchedMatMulF32(b, m, n, k, lhsData, rhsData, resultData)
		} else {
			if b == 1 {
				ls := lhsLayout.Stride()
				rs := rhsLayout.Stride()
				lhsStrides := []int{0, ls[len(ls)-2], ls[len(ls)-1]} // [batch,row,col]
				rhsStrides := []int{0, rs[len(rs)-2], rs[len(rs)-1]}
				cStrides := []int{0, n, 1}
				kernels.NaiveBatchedMatMulStridedF32(
					b,
					m,
					n,
					k,
					lhsData,
					rhsData,
					resultData,
					lhsStrides,
					rhsStrides,
					cStrides,
				)
			} else {
				lhsStrides := []int{m * k, k, 1} // [batch, row, col]
				rhsStrides := []int{k * n, n, 1}
				cStrides := []int{m * n, n, 1}
				kernels.NaiveBatchedMatMulStridedF32(
					b,
					m,
					n,
					k,
					lhsData,
					rhsData,
					resultData,
					lhsStrides,
					rhsStrides,
					cStrides,
				)
			}
		}
	case []float64:
		lhsData := any(s.data).([]float64)
		rhsData := any(rhsC.data).([]float64)
		resultData := any(result.data).([]float64)

		if lhsLayout.IsContiguous() && rhsLayout.IsContiguous() {
			kernels.NaiveBatchedMatMulF64(b, m, n, k, lhsData, rhsData, resultData)
		} else {
			if b == 1 {
				ls := lhsLayout.Stride()
				rs := rhsLayout.Stride()
				lhsStrides := []int{0, ls[len(ls)-2], ls[len(ls)-1]}
				rhsStrides := []int{0, rs[len(rs)-2], rs[len(rs)-1]}
				cStrides := []int{0, n, 1}
				kernels.NaiveBatchedMatMulStridedF64(
					b,
					m,
					n,
					k,
					lhsData,
					rhsData,
					resultData,
					lhsStrides,
					rhsStrides,
					cStrides,
				)
			} else {
				lhsStrides := []int{m * k, k, 1}
				rhsStrides := []int{k * n, n, 1}
				cStrides := []int{m * n, n, 1}
				kernels.NaiveBatchedMatMulStridedF64(
					b,
					m,
					n,
					k,
					lhsData,
					rhsData,
					resultData,
					lhsStrides,
					rhsStrides,
					cStrides,
				)
			}
		}
	case []uint8, []uint32, []int64:
		if lhsLayout.IsContiguous() && rhsLayout.IsContiguous() {
			kernels.NaiveBatchedMatMul(b, m, n, k, s.data, rhsC.data, result.data)
		} else {
			if b == 1 {
				ls := lhsLayout.Stride()
				rs := rhsLayout.Stride()
				lhsStrides := []int{0, ls[len(ls)-2], ls[len(ls)-1]}
				rhsStrides := []int{0, rs[len(rs)-2], rs[len(rs)-1]}
				cStrides := []int{0, n, 1}
				kernels.NaiveBatchedMatMulStrided(
					b,
					m,
					n,
					k,
					s.data,
					rhsC.data,
					result.data,
					lhsStrides,
					rhsStrides,
					cStrides,
				)
			} else {
				lhsStrides := []int{m * k, k, 1}
				rhsStrides := []int{k * n, n, 1}
				cStrides := []int{m * n, n, 1}
				kernels.NaiveBatchedMatMulStrided(
					b,
					m,
					n,
					k,
					s.data,
					rhsC.data,
					result.data,
					lhsStrides,
					rhsStrides,
					cStrides,
				)
			}
		}
	default:
		return nil, errors.New("unsupported type for matmul")
	}

	return result, nil
}

// Conv1d performs 1D convolution using im2col + BLAS for supported types.
func (s *CpuStorage[T]) Conv1d(layout *candy.Layout, kernel candy.BackendStorage[T], kernelLayout *candy.Layout, params *candy.Conv1DParams) (candy.BackendStorage[T], error) {
	kernelC, ok := kernel.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("kernel storage must be CpuStorage")
	}

	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if kernel == nil {
		return nil, errors.New("kernel cannot be nil")
	}
	if kernelLayout == nil {
		return nil, errors.New("kernelLayout cannot be nil")
	}
	if params == nil {
		return nil, errors.New("params cannot be nil")
	}
	lOut := params.OutLen()
	if lOut <= 0 {
		return nil, errors.New("invalid convolution parameters: output length <= 0")
	}

	result := New(make([]T, params.Batch*params.OutCh*lOut))
	switch any(s.data).(type) {
	case []float32:
		if layout.IsContiguous() && kernelLayout.IsContiguous() {
			kernels.Im2colConv1dF32(
				params.Batch,
				params.InCh,
				params.InLen,
				params.OutCh,
				params.KSize,
				params.Stride,
				params.Pad,
				params.Dilate,
				any(s.data).([]float32),
				any(kernelC.data).([]float32),
				any(result.data).([]float32),
			)
		} else {
			kernels.NaiveConv1dF32(
				params.Batch,
				params.InCh,
				params.InLen,
				params.OutCh,
				params.KSize,
				params.Stride,
				params.Pad,
				params.Dilate,
				any(s.data).([]float32),
				any(kernelC.data).([]float32),
				any(result.data).([]float32),
			)
		}
	case []float64:
		if layout.IsContiguous() && kernelLayout.IsContiguous() {
			kernels.Im2colConv1dF64(
				params.Batch,
				params.InCh,
				params.InLen,
				params.OutCh,
				params.KSize,
				params.Stride,
				params.Pad,
				params.Dilate,
				any(s.data).([]float64),
				any(kernelC.data).([]float64),
				any(result.data).([]float64),
			)
		} else {
			kernels.NaiveConv1dF64(
				params.Batch,
				params.InCh,
				params.InLen,
				params.OutCh,
				params.KSize,
				params.Stride,
				params.Pad,
				params.Dilate,
				any(s.data).([]float64),
				any(kernelC.data).([]float64),
				any(result.data).([]float64),
			)
		}
	case []uint8, []uint32, []int64:
		kernels.NaiveConv1d(
			params.Batch,
			params.InCh,
			params.InLen,
			params.OutCh,
			params.KSize,
			params.Stride,
			params.Pad,
			params.Dilate,
			s.data,
			kernelC.data,
			result.data,
		)
	default:
		return nil, errors.New("unsupported data type for conv1d")
	}

	return result, nil
}

// ConvTranspose1d performs 1D transposed convolution (deconvolution) for supported types.
func (s *CpuStorage[T]) ConvTranspose1d(layout *candy.Layout, kernel candy.BackendStorage[T], kernelLayout *candy.Layout, params *candy.ConvT1DParams) (candy.BackendStorage[T], error) {
	kernelC, ok := kernel.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("kernel storage must be CpuStorage")
	}
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if kernel == nil {
		return nil, errors.New("kernel cannot be nil")
	}
	if kernelLayout == nil {
		return nil, errors.New("kernelLayout cannot be nil")
	}
	if params == nil {
		return nil, errors.New("params cannot be nil")
	}
	lOut := params.OutLen()
	if lOut <= 0 {
		return nil, errors.New("invalid convolution parameters: output length <= 0")
	}
	result := New(make([]T, params.Batch*params.OutCh*lOut))

	switch any(s.data).(type) {
	case []float32:
		kernels.NaiveConvTranspose1dF32(
			params.Batch,
			params.InCh,
			params.InLen,
			params.OutCh,
			params.KSize,
			params.Stride,
			params.Pad,
			params.OutPad,
			params.Dilate,
			any(s.data).([]float32),
			any(kernelC.data).([]float32),
			any(result.data).([]float32),
		)
	case []float64:
		kernels.NaiveConvTranspose1dF64(
			params.Batch,
			params.InCh,
			params.InLen,
			params.OutCh,
			params.KSize,
			params.Stride,
			params.Pad,
			params.OutPad,
			params.Dilate,
			any(s.data).([]float64),
			any(kernelC.data).([]float64),
			any(result.data).([]float64),
		)
	case []uint8, []uint32, []int64:
		kernels.NaiveConvTranspose1d(
			params.Batch,
			params.InCh,
			params.InLen,
			params.OutCh,
			params.KSize,
			params.Stride,
			params.Pad,
			params.OutPad,
			params.Dilate,
			s.data,
			kernelC.data,
			result.data,
		)
	default:
		return nil, errors.New("unsupported data type for conv_transpose1d")
	}

	return result, nil
}

// Conv2d performs 2D convolution using im2col + BLAS for supported types.
func (s *CpuStorage[T]) Conv2d(layout *candy.Layout, kernel candy.BackendStorage[T], kernelLayout *candy.Layout, params *candy.Conv2DParams) (candy.BackendStorage[T], error) {
	kernelC, ok := kernel.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("kernel storage must be CpuStorage")
	}
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if kernel == nil {
		return nil, errors.New("kernel cannot be nil")
	}
	if kernelLayout == nil {
		return nil, errors.New("kernelLayout cannot be nil")
	}
	if params == nil {
		return nil, errors.New("params cannot be nil")
	}

	hOut := params.OutH()
	wOut := params.OutW()
	if hOut <= 0 || wOut <= 0 {
		return nil, errors.New("invalid convolution parameters: output dimensions <= 0")
	}

	result := New(make([]T, params.Batch*params.OutCh*hOut*wOut))

	switch any(s.data).(type) {
	case []float32:
		if layout.IsContiguous() && kernelLayout.IsContiguous() {
			kernels.Im2colConv2dF32(
				params.Batch,
				params.InCh,
				params.InH,
				params.InW,
				params.OutCh,
				params.KH,
				params.KW,
				params.Stride,
				params.Pad,
				params.Dilate,
				any(s.data).([]float32),
				any(kernelC.data).([]float32),
				any(result.data).([]float32),
			)
		} else {
			kernels.NaiveConv2dF32(
				params.Batch,
				params.InCh,
				params.InH,
				params.InW,
				params.OutCh,
				params.KH,
				params.KW,
				params.Stride,
				params.Pad,
				params.Dilate,
				any(s.data).([]float32),
				any(kernelC.data).([]float32),
				any(result.data).([]float32),
			)
		}
	case []float64:
		if layout.IsContiguous() && kernelLayout.IsContiguous() {
			kernels.Im2colConv2dF64(
				params.Batch,
				params.InCh,
				params.InH,
				params.InW,
				params.OutCh,
				params.KH,
				params.KW,
				params.Stride,
				params.Pad,
				params.Dilate,
				any(s.data).([]float64),
				any(kernelC.data).([]float64),
				any(result.data).([]float64),
			)
		} else {
			kernels.NaiveConv2dF64(
				params.Batch,
				params.InCh,
				params.InH,
				params.InW,
				params.OutCh,
				params.KH,
				params.KW,
				params.Stride,
				params.Pad,
				params.Dilate,
				any(s.data).([]float64),
				any(kernelC.data).([]float64),
				any(result.data).([]float64),
			)
		}
	case []uint8, []uint32, []int64:
		kernels.NaiveConv2d(
			params.Batch,
			params.InCh,
			params.InH,
			params.InW,
			params.OutCh,
			params.KH,
			params.KW,
			params.Stride,
			params.Pad,
			params.Dilate,
			s.data,
			kernelC.data,
			result.data,
		)
	default:
		return nil, errors.New("unsupported data type for conv2d")
	}

	return result, nil
}

// ConvTranspose2d performs 2D transposed convolution (deconvolution) for supported types.
func (s *CpuStorage[T]) ConvTranspose2d(layout *candy.Layout, kernel candy.BackendStorage[T], kernelLayout *candy.Layout, params *candy.ConvT2DParams) (candy.BackendStorage[T], error) {
	kernelC, ok := kernel.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("kernel storage must be CpuStorage")
	}
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if kernel == nil {
		return nil, errors.New("kernel cannot be nil")
	}
	if kernelLayout == nil {
		return nil, errors.New("kernelLayout cannot be nil")
	}
	if params == nil {
		return nil, errors.New("params cannot be nil")
	}

	hOut := params.OutH()
	wOut := params.OutW()
	if hOut <= 0 || wOut <= 0 {
		return nil, errors.New("invalid convolution parameters: output dimensions <= 0")
	}

	result := New(make([]T, params.Batch*params.OutCh*hOut*wOut))

	switch any(s.data).(type) {
	case []float32:
		kernels.NaiveConvTranspose2dF32(
			params.Batch,
			params.InCh,
			params.InH,
			params.InW,
			params.OutCh,
			params.KH,
			params.KW,
			params.Stride,
			params.Pad,
			params.OutPad,
			params.Dilate,
			any(s.data).([]float32),
			any(kernelC.data).([]float32),
			any(result.data).([]float32),
		)
	case []float64:
		kernels.NaiveConvTranspose2dF64(
			params.Batch,
			params.InCh,
			params.InH,
			params.InW,
			params.OutCh,
			params.KH,
			params.KW,
			params.Stride,
			params.Pad,
			params.OutPad,
			params.Dilate,
			any(s.data).([]float64),
			any(kernelC.data).([]float64),
			any(result.data).([]float64),
		)
	case []uint8, []uint32, []int64:
		kernels.NaiveConvTranspose2d(
			params.Batch,
			params.InCh,
			params.InH,
			params.InW,
			params.OutCh,
			params.KH,
			params.KW,
			params.Stride,
			params.Pad,
			params.OutPad,
			params.Dilate,
			s.data,
			kernelC.data,
			result.data,
		)
	default:
		return nil, errors.New("unsupported data type for conv_transpose2d")
	}

	return result, nil
}

// AvgPool2d performs 2D average pooling for supported types.
func (s *CpuStorage[T]) AvgPool2d(layout *candy.Layout, kH, kW, sH, sW int) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	n, c, h, w, err := layout.Dims4()
	if err != nil {
		return nil, fmt.Errorf("expected 4D tensor for avg_pool2d, got: %w", err)
	}
	if h < kH || w < kW {
		return nil, fmt.Errorf("kernel size (%d,%d) is larger than input size (%d,%d)", kH, kW, h, w)
	}
	hOut := (h-kH)/sH + 1
	wOut := (w-kW)/sW + 1

	if hOut <= 0 || wOut <= 0 {
		return nil, fmt.Errorf("invalid pooling parameters: output dimensions (%d,%d) <= 0", hOut, wOut)
	}
	outputSize := n * c * hOut * wOut
	result := New(make([]T, outputSize))
	dstStrides := []int{c * hOut * wOut, hOut * wOut, wOut, 1}

	switch any(s.data).(type) {
	case []float32:
		if layout.IsContiguous() {
			kernels.AvgPool2dF32(
				n,  // batch
				c,  // channels
				h,  // input height
				w,  // input width
				kH, // kernel height
				kW, // kernel width
				sH, // stride height
				sW, // stride width
				any(s.data).([]float32),
				any(result.data).([]float32),
			)
		} else {
			kernels.AvgPool2dStridedF32(
				n, c, h, w,
				kH, kW, sH, sW,
				any(s.data).([]float32),
				any(result.data).([]float32),
				layout.Stride(),
				dstStrides,
			)
		}
	case []float64:
		if layout.IsContiguous() {
			kernels.AvgPool2dF64(
				n, c, h, w,
				kH, kW, sH, sW,
				any(s.data).([]float64),
				any(result.data).([]float64),
			)
		} else {
			kernels.AvgPool2dStridedF64(
				n, c, h, w,
				kH, kW, sH, sW,
				any(s.data).([]float64),
				any(result.data).([]float64),
				layout.Stride(),
				dstStrides,
			)
		}
	case []uint8, []uint32, []int64:
		if layout.IsContiguous() {
			kernels.AvgPool2d(
				n, c, h, w,
				kH, kW, sH, sW,
				s.data,
				result.data,
			)
		} else {
			kernels.AvgPool2dStrided(
				n, c, h, w,
				kH, kW, sH, sW,
				s.data,
				result.data,
				layout.Stride(),
				dstStrides,
			)
		}
	default:
		return nil, errors.New("unsupported data type for avg_pool2d")
	}

	return result, nil
}

// MaxPool2d performs 2D max pooling for supported types.
func (s *CpuStorage[T]) MaxPool2d(layout *candy.Layout, kH, kW, sH, sW int) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	n, c, h, w, err := layout.Dims4()
	if err != nil {
		return nil, fmt.Errorf("expected 4D tensor for max_pool2d, got: %w", err)
	}
	if h < kH || w < kW {
		return nil, fmt.Errorf("kernel size (%d,%d) is larger than input size (%d,%d)", kH, kW, h, w)
	}
	if kH <= 0 || kW <= 0 || sH <= 0 || sW <= 0 {
		return nil, fmt.Errorf("kernel and stride must be positive")
	}
	hOut := (h-kH)/sH + 1
	wOut := (w-kW)/sW + 1

	if hOut <= 0 || wOut <= 0 {
		return nil, fmt.Errorf("invalid pooling parameters: output dimensions (%d,%d) <= 0", hOut, wOut)
	}

	outputSize := n * c * hOut * wOut
	result := New(make([]T, outputSize))
	dstStrides := []int{c * hOut * wOut, hOut * wOut, wOut, 1}

	switch any(s.data).(type) {
	case []float32:
		if layout.IsContiguous() {
			kernels.MaxPool2dF32(
				n,  // batch
				c,  // channels
				h,  // input height
				w,  // input width
				kH, // kernel height
				kW, // kernel width
				sH, // stride height
				sW, // stride width
				any(s.data).([]float32),
				any(result.data).([]float32),
			)
		} else {
			kernels.MaxPool2dStridedF32(
				n, c, h, w,
				kH, kW, sH, sW,
				any(s.data).([]float32),
				any(result.data).([]float32),
				layout.Stride(),
				dstStrides,
			)
		}
	case []float64:
		if layout.IsContiguous() {
			kernels.MaxPool2dF64(
				n, c, h, w,
				kH, kW, sH, sW,
				any(s.data).([]float64),
				any(result.data).([]float64),
			)
		} else {
			kernels.MaxPool2dStridedF64(
				n, c, h, w,
				kH, kW, sH, sW,
				any(s.data).([]float64),
				any(result.data).([]float64),
				layout.Stride(),
				dstStrides,
			)
		}
	case []uint8, []uint32, []int64:
		if layout.IsContiguous() {
			kernels.MaxPool2d(
				n, c, h, w,
				kH, kW, sH, sW,
				s.data,
				result.data,
			)
		} else {
			kernels.MaxPool2dStrided(
				n, c, h, w,
				kH, kW, sH, sW,
				s.data,
				result.data,
				layout.Stride(),
				dstStrides,
			)
		}
	default:
		return nil, errors.New("unsupported data type for max_pool2d")
	}

	return result, nil
}

// UpsampleNearest2d performs 2D nearest neighbor upsampling for supported types.
func (s *CpuStorage[T]) UpsampleNearest2d(layout *candy.Layout, targetH, targetW int) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	b, c, srcH, srcW, err := layout.Dims4()
	if err != nil {
		return nil, fmt.Errorf("expected 4D tensor for upsample_nearest2d, got: %w", err)
	}
	if targetH <= 0 || targetW <= 0 {
		return nil, fmt.Errorf("target dimensions must be positive, got (%d,%d)", targetH, targetW)
	}
	scaleH := float64(srcH) / float64(targetH)
	scaleW := float64(srcW) / float64(targetW)
	outputSize := b * c * targetH * targetW
	result := New(make([]T, outputSize))
	dstStrides := []int{c * targetH * targetW, targetH * targetW, targetW, 1}

	switch any(s.data).(type) {
	case []float32:
		if layout.IsContiguous() {
			kernels.UpsampleNearest2dF32(
				b,       // batch
				c,       // channels
				srcH,    // source height
				srcW,    // source width
				targetH, // target height
				targetW, // target width
				scaleH,  // height scale
				scaleW,  // width scale
				any(s.data).([]float32),
				any(result.data).([]float32),
			)
		} else {
			kernels.UpsampleNearest2dStridedF32(
				b, c, srcH, srcW,
				targetH, targetW, scaleH, scaleW,
				any(s.data).([]float32),
				any(result.data).([]float32),
				layout.Stride(),
				dstStrides,
			)
		}
	case []float64:
		if layout.IsContiguous() {
			kernels.UpsampleNearest2dF64(
				b, c, srcH, srcW,
				targetH, targetW, scaleH, scaleW,
				any(s.data).([]float64),
				any(result.data).([]float64),
			)
		} else {
			kernels.UpsampleNearest2dStridedF64(
				b, c, srcH, srcW,
				targetH, targetW, scaleH, scaleW,
				any(s.data).([]float64),
				any(result.data).([]float64),
				layout.Stride(),
				dstStrides,
			)
		}
	case []uint8, []uint32, []int64:
		if layout.IsContiguous() {
			kernels.UpsampleNearest2d(
				b, c, srcH, srcW,
				targetH, targetW, scaleH, scaleW,
				s.data,
				result.data,
			)
		} else {
			kernels.UpsampleNearest2dStrided(
				b, c, srcH, srcW,
				targetH, targetW, scaleH, scaleW,
				s.data,
				result.data,
				layout.Stride(),
				dstStrides,
			)
		}
	default:
		return nil, errors.New("unsupported data type for upsample_nearest2d")
	}

	return result, nil
}

// ConstSet sets all elements to a constant value for supported types.
func (s *CpuStorage[T]) ConstSet(layout *candy.Layout, val T) error {
	if layout == nil {
		return errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel <= 0 {
		return errors.New("invalid layout: number of elements <= 0")
	}

	kernels.FillStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		val,
		s.data,
	)

	return nil
}

// Gather performs gather operation along a specified dimension with same-type indices
func (s *CpuStorage[T]) Gather(layout *candy.Layout, ids candy.BackendStorage[T], idsLayout *candy.Layout, dim int) (candy.BackendStorage[T], error) {
	if layout == nil || idsLayout == nil {
		return nil, errors.New("layout and idsLayout cannot be nil")
	}
	if ids == nil {
		return nil, errors.New("ids cannot be nil")
	}

	idsC, ok := ids.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("ids storage must be CpuStorage")
	}

	// Calculate dimensions
	srcDims := layout.Dims()
	idsDims := idsLayout.Dims()

	if dim < 0 || dim >= len(srcDims) {
		return nil, fmt.Errorf("dimension %d out of range", dim)
	}

	// Result shape matches the index tensor shape
	resultDims := make([]int, len(idsDims))
	copy(resultDims, idsDims)

	numel := 1
	for _, d := range resultDims {
		numel *= d
	}
	result := New(make([]T, numel))

	// Calculate parameters
	leftSize := 1
	for i := range dim {
		leftSize *= srcDims[i]
	}

	srcDimSize := srcDims[dim]

	rightSize := 1
	for i := dim + 1; i < len(srcDims); i++ {
		rightSize *= srcDims[i]
	}

	idsDimSize := idsLayout.Numel() / leftSize / rightSize

	// Call kernel based on type
	switch any(s.data).(type) {
	case []float32:
		kernels.GatherF32F32(numel, any(idsC.data).([]float32), any(s.data).([]float32), any(result.data).([]float32), leftSize, srcDimSize, idsDimSize, rightSize)
	case []float64:
		kernels.GatherF64F64(numel, any(idsC.data).([]float64), any(s.data).([]float64), any(result.data).([]float64), leftSize, srcDimSize, idsDimSize, rightSize)
	case []uint8:
		kernels.GatherU8U8(numel, any(idsC.data).([]uint8), any(s.data).([]uint8), any(result.data).([]uint8), leftSize, srcDimSize, idsDimSize, rightSize)
	case []uint32:
		kernels.GatherU32U32(numel, any(idsC.data).([]uint32), any(s.data).([]uint32), any(result.data).([]uint32), leftSize, srcDimSize, idsDimSize, rightSize)
	case []int64:
		kernels.GatherI64I64(numel, any(idsC.data).([]int64), any(s.data).([]int64), any(result.data).([]int64), leftSize, srcDimSize, idsDimSize, rightSize)
	default:
		return nil, errors.New("unsupported data type for Gather")
	}

	return result, nil
}

// Scatter performs scatter operation along a specified dimension with same-type indices
func (s *CpuStorage[T]) Scatter(layout *candy.Layout, ids candy.BackendStorage[T], idsLayout *candy.Layout, src candy.BackendStorage[T], srcLayout *candy.Layout, dim int) (candy.BackendStorage[T], error) {
	if layout == nil || idsLayout == nil || srcLayout == nil {
		return nil, errors.New("layouts cannot be nil")
	}
	if ids == nil || src == nil {
		return nil, errors.New("ids and src cannot be nil")
	}

	idsC, ok := ids.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("ids storage must be CpuStorage")
	}

	srcC, ok := src.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("src storage must be CpuStorage")
	}

	// Calculate dimensions
	dstDims := layout.Dims()
	srcDims := srcLayout.Dims()

	if dim < 0 || dim >= len(dstDims) {
		return nil, fmt.Errorf("dimension %d out of range", dim)
	}

	// Create result storage with same shape as destination
	numel := layout.Numel()
	result := New(make([]T, numel))

	// Copy destination data to result first
	copy(result.data, s.data)

	// Calculate parameters
	leftSize := 1
	for i := 0; i < dim; i++ {
		leftSize *= dstDims[i]
	}

	srcDimSize := srcDims[dim]
	dstDimSize := dstDims[dim]

	rightSize := 1
	for i := dim + 1; i < len(dstDims); i++ {
		rightSize *= dstDims[i]
	}

	// Call kernel based on type
	switch any(s.data).(type) {
	case []float32:
		kernels.ScatterF32F32(leftSize, srcDimSize, dstDimSize, rightSize, any(idsC.data).([]float32), any(srcC.data).([]float32), any(result.data).([]float32))
	case []float64:
		kernels.ScatterF64F64(leftSize, srcDimSize, dstDimSize, rightSize, any(idsC.data).([]float64), any(srcC.data).([]float64), any(result.data).([]float64))
	case []uint8:
		kernels.ScatterU8U8(leftSize, srcDimSize, dstDimSize, rightSize, any(idsC.data).([]uint8), any(srcC.data).([]uint8), any(result.data).([]uint8))
	case []uint32:
		kernels.ScatterU32U32(leftSize, srcDimSize, dstDimSize, rightSize, any(idsC.data).([]uint32), any(srcC.data).([]uint32), any(result.data).([]uint32))
	case []int64:
		kernels.ScatterI64I64(leftSize, srcDimSize, dstDimSize, rightSize, any(idsC.data).([]int64), any(srcC.data).([]int64), any(result.data).([]int64))
	default:
		return nil, errors.New("unsupported data type for Scatter")
	}

	return result, nil
}

// ScatterAdd performs scatter-add operation along a specified dimension with same-type indices
func (s *CpuStorage[T]) ScatterAdd(layout *candy.Layout, ids candy.BackendStorage[T], idsLayout *candy.Layout, src candy.BackendStorage[T], srcLayout *candy.Layout, dim int) (candy.BackendStorage[T], error) {
	if layout == nil || idsLayout == nil || srcLayout == nil {
		return nil, errors.New("layouts cannot be nil")
	}
	if ids == nil || src == nil {
		return nil, errors.New("ids and src cannot be nil")
	}

	idsC, ok := ids.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("ids storage must be CpuStorage")
	}

	srcC, ok := src.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("src storage must be CpuStorage")
	}

	// Calculate dimensions
	dstDims := layout.Dims()
	srcDims := srcLayout.Dims()

	if dim < 0 || dim >= len(dstDims) {
		return nil, fmt.Errorf("dimension %d out of range", dim)
	}

	// Create result storage with same shape as destination
	numel := layout.Numel()
	result := New(make([]T, numel))

	// Copy destination data to result first
	copy(result.data, s.data)

	// Calculate parameters
	leftSize := 1
	for i := range dim {
		leftSize *= dstDims[i]
	}

	srcDimSize := srcDims[dim]
	dstDimSize := dstDims[dim]

	rightSize := 1
	for i := dim + 1; i < len(dstDims); i++ {
		rightSize *= dstDims[i]
	}

	// Call kernel based on type
	switch any(s.data).(type) {
	case []float32:
		kernels.ScatterAddF32F32(leftSize, srcDimSize, dstDimSize, rightSize, any(idsC.data).([]float32), any(srcC.data).([]float32), any(result.data).([]float32))
	case []float64:
		kernels.ScatterAddF64F64(leftSize, srcDimSize, dstDimSize, rightSize, any(idsC.data).([]float64), any(srcC.data).([]float64), any(result.data).([]float64))
	case []uint8:
		kernels.ScatterAddU8U8(leftSize, srcDimSize, dstDimSize, rightSize, any(idsC.data).([]uint8), any(srcC.data).([]uint8), any(result.data).([]uint8))
	case []uint32:
		kernels.ScatterAddU32U32(leftSize, srcDimSize, dstDimSize, rightSize, any(idsC.data).([]uint32), any(srcC.data).([]uint32), any(result.data).([]uint32))
	case []int64:
		kernels.ScatterAddI64I64(leftSize, srcDimSize, dstDimSize, rightSize, any(idsC.data).([]int64), any(srcC.data).([]int64), any(result.data).([]int64))
	default:
		return nil, errors.New("unsupported data type for ScatterAdd")
	}

	return result, nil
}

// Copy2d copies a 2D region from source to destination for supported types.
func (s *CpuStorage[T]) Copy2d(dst candy.BackendStorage[T], d1, d2 int, srcStride1, dstStride1, srcOffset, dstOffset int) error {
	dstC, ok := dst.(*CpuStorage[T])
	if !ok {
		return errors.New("dst storage must be CpuStorage")
	}
	if dst == nil {
		return errors.New("dst cannot be nil")
	}
	if d1 <= 0 || d2 <= 0 {
		return errors.New("invalid copy parameters: d1 or d2 <= 0")
	}
	if srcStride1 <= 0 || dstStride1 <= 0 {
		return errors.New("invalid stride parameters: stride <= 0")
	}

	srcRequired := srcOffset + (d1-1)*srcStride1 + d2
	dstRequired := dstOffset + (d1-1)*dstStride1 + d2
	if len(s.data) < srcRequired {
		return errors.New("source storage too small")
	}
	if len(dstC.data) < dstRequired {
		return errors.New("destination storage too small")
	}

	kernels.Copy2d(
		d1,         // rows
		d2,         // cols
		srcStride1, // lda
		dstStride1, // ldc
		srcOffset,  // srcOffset
		dstOffset,  // dstOffset
		s.data,     // src
		dstC.data,  // dst
	)

	return nil
}

// FastSum computes the sum over the last dimension
func (s *CpuStorage[T]) FastSum(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	dims := layout.Dims()
	if len(dims) == 0 {
		return nil, errors.New("cannot reduce scalar tensor")
	}
	dstSize := numel / dims[len(dims)-1]

	result := New(make([]T, dstSize))
	kernels.FastSumStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// FastMin computes the minimum over the last dimension
func (s *CpuStorage[T]) FastMin(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	dims := layout.Dims()
	if len(dims) == 0 {
		return nil, errors.New("cannot reduce scalar tensor")
	}
	dstSize := numel / dims[len(dims)-1]

	result := New(make([]T, dstSize))
	kernels.FastMinStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// FastMax computes the maximum over the last dimension
func (s *CpuStorage[T]) FastMax(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	dims := layout.Dims()
	if len(dims) == 0 {
		return nil, errors.New("cannot reduce scalar tensor")
	}
	dstSize := numel / dims[len(dims)-1]

	result := New(make([]T, dstSize))
	kernels.FastMaxStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// FastArgmin computes the indices of minimum values over the last dimension
func (s *CpuStorage[T]) FastArgmin(layout *candy.Layout) (candy.BackendStorage[uint32], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	dims := layout.Dims()
	if len(dims) == 0 {
		return nil, errors.New("cannot reduce scalar tensor")
	}
	dstSize := numel / dims[len(dims)-1]

	result := New(make([]uint32, dstSize))
	kernels.FastArgminStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// FastArgmax computes the indices of maximum values over the last dimension
func (s *CpuStorage[T]) FastArgmax(layout *candy.Layout) (candy.BackendStorage[uint32], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	dims := layout.Dims()
	if len(dims) == 0 {
		return nil, errors.New("cannot reduce scalar tensor")
	}
	dstSize := numel / dims[len(dims)-1]

	result := New(make([]uint32, dstSize))
	kernels.FastArgmaxStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

func (s *CpuStorage[T]) Sum(layout *candy.Layout, dims []int) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if len(dims) == 0 {
		return nil, errors.New("sumDims cannot be empty")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	outputDims := make([]int, len(layout.Dims()))
	copy(outputDims, layout.Dims())
	for _, dim := range dims {
		if dim < 0 || dim >= len(outputDims) {
			return nil, errors.New("invalid sum dimension")
		}
		outputDims[dim] = 1
	}

	outputSize := 1
	for _, d := range outputDims {
		outputSize *= d
	}

	result := New(make([]T, outputSize))
	kernels.SumStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		dims,
		s.data,
		result.data,
	)

	return result, nil
}

// Min computes the minimum over the specified dimension
func (s *CpuStorage[T]) Min(layout *candy.Layout, dim int) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	dims := layout.Dims()
	if dim < 0 || dim >= len(dims) {
		return nil, errors.New("invalid dimension")
	}

	// Calculate output dimensions (remove the reduced dimension)
	outputDims := make([]int, 0, len(dims)-1)
	for i, d := range dims {
		if i != dim {
			outputDims = append(outputDims, d)
		}
	}

	outputSize := 1
	for _, d := range outputDims {
		outputSize *= d
	}

	result := New(make([]T, outputSize))
	kernels.MinStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		dim,
		s.data,
		result.data,
	)

	return result, nil
}

// Max computes the maximum over the specified dimension
func (s *CpuStorage[T]) Max(layout *candy.Layout, dim int) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	dims := layout.Dims()
	if dim < 0 || dim >= len(dims) {
		return nil, errors.New("invalid dimension")
	}

	// Calculate output dimensions (remove the reduced dimension)
	outputDims := make([]int, 0, len(dims)-1)
	for i, d := range dims {
		if i != dim {
			outputDims = append(outputDims, d)
		}
	}

	outputSize := 1
	for _, d := range outputDims {
		outputSize *= d
	}

	result := New(make([]T, outputSize))
	kernels.MaxStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		dim,
		s.data,
		result.data,
	)

	return result, nil
}

// Argmin computes the index of minimum over the specified dimension
func (s *CpuStorage[T]) Argmin(layout *candy.Layout, dim int) (candy.BackendStorage[uint32], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	dims := layout.Dims()
	if dim < 0 || dim >= len(dims) {
		return nil, errors.New("invalid dimension")
	}

	// Calculate output dimensions (remove the reduced dimension)
	outputDims := make([]int, 0, len(dims)-1)
	for i, d := range dims {
		if i != dim {
			outputDims = append(outputDims, d)
		}
	}

	outputSize := 1
	for _, d := range outputDims {
		outputSize *= d
	}

	result := New(make([]uint32, outputSize))
	kernels.ArgminStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		dim,
		s.data,
		result.data,
	)

	return result, nil
}

// Argmax computes the index of maximum over the specified dimension
func (s *CpuStorage[T]) Argmax(layout *candy.Layout, dim int) (candy.BackendStorage[uint32], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	dims := layout.Dims()
	if dim < 0 || dim >= len(dims) {
		return nil, errors.New("invalid dimension")
	}

	// Calculate output dimensions (remove the reduced dimension)
	outputDims := make([]int, 0, len(dims)-1)
	for i, d := range dims {
		if i != dim {
			outputDims = append(outputDims, d)
		}
	}

	outputSize := 1
	for _, d := range outputDims {
		outputSize *= d
	}

	result := New(make([]uint32, outputSize))
	kernels.ArgmaxStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		dim,
		s.data,
		result.data,
	)

	return result, nil
}

// FastSoftmax performs softmax along the last dimension
func (s *CpuStorage[T]) FastSoftmax(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))
	kernels.FastSoftmaxStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// FastRmsNorm performs RMS normalization along the last dimension
func (s *CpuStorage[T]) FastRmsNorm(layout *candy.Layout, alpha candy.BackendStorage[T], alphaLayout *candy.Layout, eps T) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	alphaC, ok := alpha.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("alpha must be CPU storage")
	}

	if !alphaLayout.IsContiguous() {
		return nil, errors.New("alpha must be contiguous")
	}

	dims := layout.Dims()
	if len(dims) == 0 {
		return nil, errors.New("cannot normalize scalar tensor")
	}
	lastDim := dims[len(dims)-1]
	if len(alphaC.data) != lastDim {
		return nil, fmt.Errorf("alpha size %d must match last dimension %d", len(alphaC.data), lastDim)
	}

	result := New(make([]T, numel))
	kernels.FastRmsNormStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		eps,
		alphaC.data,
		s.data,
		result.data,
	)

	return result, nil
}

// FastLayerNorm performs Layer normalization along the last dimension
func (s *CpuStorage[T]) FastLayerNorm(layout *candy.Layout, alpha candy.BackendStorage[T], alphaLayout *candy.Layout, beta candy.BackendStorage[T], betaLayout *candy.Layout, eps T) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	alphaC, ok := alpha.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("alpha must be CPU storage")
	}

	betaC, ok := beta.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("beta must be CPU storage")
	}

	if !alphaLayout.IsContiguous() {
		return nil, errors.New("alpha must be contiguous")
	}
	if !betaLayout.IsContiguous() {
		return nil, errors.New("beta must be contiguous")
	}

	dims := layout.Dims()
	if len(dims) == 0 {
		return nil, errors.New("cannot normalize scalar tensor")
	}
	lastDim := dims[len(dims)-1]
	if len(alphaC.data) != lastDim {
		return nil, fmt.Errorf("alpha size %d must match last dimension %d", len(alphaC.data), lastDim)
	}
	if len(betaC.data) != lastDim {
		return nil, fmt.Errorf("beta size %d must match last dimension %d", len(betaC.data), lastDim)
	}

	result := New(make([]T, numel))
	kernels.FastLayerNormStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		eps,
		alphaC.data,
		betaC.data,
		s.data,
		result.data,
	)

	return result, nil
}

// RopeI performs rotary position embedding (rope_i variant)
func (s *CpuStorage[T]) RopeI(layout *candy.Layout, cos candy.BackendStorage[T], cosLayout *candy.Layout, sin candy.BackendStorage[T], sinLayout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if cosLayout == nil {
		return nil, errors.New("cos layout cannot be nil")
	}
	if sinLayout == nil {
		return nil, errors.New("sin layout cannot be nil")
	}
	if len(s.data) == 0 {
		return nil, errors.New("source storage is empty")
	}

	if !cosLayout.IsContiguous() {
		return nil, errors.New("input cos must be contiguous")
	}
	if !sinLayout.IsContiguous() {
		return nil, errors.New("input sin must be contiguous")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	cosC, ok := cos.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("cos must be CPU storage")
	}
	sinC, ok := sin.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("sin must be CPU storage")
	}

	if layout.Rank() != 4 {
		return nil, errors.New("input must be 4D for RoPE")
	}

	b, h, t, d, err := layout.Dims4()
	if err != nil {
		return nil, err
	}

	expectedSize := b * t * d / 2
	if cosLayout.Numel() != expectedSize {
		return nil, fmt.Errorf("cos size mismatch: expected %d, got %d", expectedSize, cosLayout.Numel())
	}
	if sinLayout.Numel() != expectedSize {
		return nil, fmt.Errorf("sin size mismatch: expected %d, got %d", expectedSize, sinLayout.Numel())
	}

	bh := b * h
	td := t * d
	strideB := td

	result := New(make([]T, numel))
	kernels.RopeIStrided(
		layout.Rank(),   // rank
		layout.Dims(),   // dims
		layout.Stride(), // strides
		bh,              // bh
		td,              // td
		strideB,         // strideB
		s.data,          // src
		cosC.data,       // cos
		sinC.data,       // sin
		result.data,     // dst
	)

	return result, nil
}

// Rope performs rotary position embedding (rope variant)
func (s *CpuStorage[T]) Rope(layout *candy.Layout, cos candy.BackendStorage[T], cosLayout *candy.Layout, sin candy.BackendStorage[T], sinLayout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if cosLayout == nil {
		return nil, errors.New("cos layout cannot be nil")
	}
	if sinLayout == nil {
		return nil, errors.New("sin layout cannot be nil")
	}
	if len(s.data) == 0 {
		return nil, errors.New("source storage is empty")
	}

	if !cosLayout.IsContiguous() {
		return nil, errors.New("input cos must be contiguous")
	}
	if !sinLayout.IsContiguous() {
		return nil, errors.New("input sin must be contiguous")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	cosC, ok := cos.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("cos must be CPU storage")
	}
	sinC, ok := sin.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("sin must be CPU storage")
	}

	if layout.Rank() != 4 {
		return nil, errors.New("input must be 4D for RoPE")
	}

	b, h, t, d, err := layout.Dims4()
	if err != nil {
		return nil, err
	}

	expectedSize := b * t * d / 2
	if cosLayout.Numel() != expectedSize {
		return nil, fmt.Errorf("cos size mismatch: expected %d, got %d", expectedSize, cosLayout.Numel())
	}
	if sinLayout.Numel() != expectedSize {
		return nil, fmt.Errorf("sin size mismatch: expected %d, got %d", expectedSize, sinLayout.Numel())
	}

	bh := b * h
	td := t * d
	strideB := td

	result := New(make([]T, numel))
	kernels.RopeStrided(
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		bh,
		td,
		d,
		strideB,
		s.data,
		cosC.data,
		sinC.data,
		result.data,
	)

	return result, nil
}

// RopeThd performs rotary position embedding (rope_thd variant)
func (s *CpuStorage[T]) RopeThd(layout *candy.Layout, cos candy.BackendStorage[T], cosLayout *candy.Layout, sin candy.BackendStorage[T], sinLayout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if cosLayout == nil {
		return nil, errors.New("cos layout cannot be nil")
	}
	if sinLayout == nil {
		return nil, errors.New("sin layout cannot be nil")
	}
	if len(s.data) == 0 {
		return nil, errors.New("source storage is empty")
	}

	if !cosLayout.IsContiguous() {
		return nil, errors.New("input cos must be contiguous")
	}
	if !sinLayout.IsContiguous() {
		return nil, errors.New("input sin must be contiguous")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	cosC, ok := cos.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("cos must be CPU storage")
	}
	sinC, ok := sin.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("sin must be CPU storage")
	}

	if layout.Rank() != 4 {
		return nil, errors.New("input must be 4D for RoPE")
	}

	b, t, h, d, err := layout.Dims4()
	if err != nil {
		return nil, err
	}

	expectedSize := b * t * h * d / 2
	if cosLayout.Numel() != expectedSize {
		return nil, fmt.Errorf("cos size mismatch: expected %d, got %d", expectedSize, cosLayout.Numel())
	}
	if sinLayout.Numel() != expectedSize {
		return nil, fmt.Errorf("sin size mismatch: expected %d, got %d", expectedSize, sinLayout.Numel())
	}

	strideB := t * h * d

	result := New(make([]T, numel))
	kernels.RopeThdStrided(
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		b,
		t,
		h,
		d,
		strideB,
		s.data,
		cosC.data,
		sinC.data,
		result.data,
	)

	return result, nil
}

// WhereCond performs element-wise selection based on condition.
// If s[i] != 0, result[i] = t[i], otherwise result[i] = f[i].
// Note: s can be uint8, uint32, or int64 type (condition mask).
func (s *CpuStorage[T]) WhereCond(condLayout *candy.Layout, t candy.BackendStorage[T], tLayout *candy.Layout, f candy.BackendStorage[T], fLayout *candy.Layout) (candy.BackendStorage[T], error) {
	tC, ok := t.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("true storage must be CpuStorage")
	}

	fC, ok := f.(*CpuStorage[T])
	if !ok {
		return nil, errors.New("false storage must be CpuStorage")
	}

	if condLayout == nil || tLayout == nil || fLayout == nil {
		return nil, errors.New("layouts cannot be nil")
	}

	elemCount := condLayout.Numel()
	if tLayout.Numel() != elemCount || fLayout.Numel() != elemCount {
		return nil, errors.New("layout element counts must match")
	}

	result := New(make([]T, elemCount))
	switch cond := any(s.data).(type) {
	case []float32:
		kernels.WhereStridedF32(
			elemCount, condLayout.Rank(), condLayout.Dims(),
			condLayout.Stride(), tLayout.Stride(), fLayout.Stride(),
			cond,
			tC.data,
			fC.data,
			result.data,
		)
	case []float64:
		kernels.WhereStridedF64(
			elemCount, condLayout.Rank(), condLayout.Dims(),
			condLayout.Stride(), tLayout.Stride(), fLayout.Stride(),
			cond,
			tC.data,
			fC.data,
			result.data,
		)
	case []uint8:
		kernels.WhereStridedU8(
			elemCount, condLayout.Rank(), condLayout.Dims(),
			condLayout.Stride(), tLayout.Stride(), fLayout.Stride(),
			cond,
			tC.data,
			fC.data,
			result.data,
		)
	case []uint32:
		kernels.WhereStridedU32(
			elemCount, condLayout.Rank(), condLayout.Dims(),
			condLayout.Stride(), tLayout.Stride(), fLayout.Stride(),
			cond,
			tC.data,
			fC.data,
			result.data,
		)
	case []int64:
		kernels.WhereStridedI64(
			elemCount, condLayout.Rank(), condLayout.Dims(),
			condLayout.Stride(), tLayout.Stride(), fLayout.Stride(),
			cond,
			tC.data,
			fC.data,
			result.data,
		)
	default:
		return nil, errors.New("condition must be uint8, uint32, or int64 type")
	}

	return result, nil
}

// Copy performs element-wise copy operation
func (s *CpuStorage[T]) Copy(layout *candy.Layout, src candy.BackendStorage[T]) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	srcData := src.(*CpuStorage[T]).data
	result := New(make([]T, numel))

	kernels.UCopyStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		srcData,
		result.data,
	)

	return result, nil
}

// Neg performs element-wise negation operation
func (s *CpuStorage[T]) Neg(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))

	kernels.UNegStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Recip performs element-wise reciprocal operation
func (s *CpuStorage[T]) Recip(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))

	kernels.URecipStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Exp performs element-wise exponential operation
func (s *CpuStorage[T]) Exp(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UExpStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Log performs element-wise logarithm operation
func (s *CpuStorage[T]) Log(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.ULogStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Sin performs element-wise sine operation
func (s *CpuStorage[T]) Sin(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.USinStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Cos performs element-wise cosine operation
func (s *CpuStorage[T]) Cos(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UCosStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Tanh performs element-wise hyperbolic tangent operation
func (s *CpuStorage[T]) Tanh(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UTanhStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Erf performs element-wise error function operation
func (s *CpuStorage[T]) Erf(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UErfStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Ceil performs element-wise ceiling operation
func (s *CpuStorage[T]) Ceil(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UCeilStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Floor performs element-wise floor operation
func (s *CpuStorage[T]) Floor(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UFloorStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Round performs element-wise round operation
func (s *CpuStorage[T]) Round(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.URoundStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Normcdf performs element-wise normal CDF operation
func (s *CpuStorage[T]) Normcdf(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UNormcdfStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Abs performs element-wise absolute value operation
func (s *CpuStorage[T]) Abs(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UAbsStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Sqr performs element-wise square operation
func (s *CpuStorage[T]) Sqr(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.USqrStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Sqrt performs element-wise square root operation
func (s *CpuStorage[T]) Sqrt(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))

	kernels.USqrtStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Gelu performs element-wise GELU activation operation
func (s *CpuStorage[T]) Gelu(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UGeluStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// GeluErf performs element-wise GELU (ERF-based) activation operation
func (s *CpuStorage[T]) GeluErf(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UGeluErfStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Relu performs element-wise ReLU activation operation
func (s *CpuStorage[T]) Relu(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UReluStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Elu performs element-wise ELU activation operation with parameter alpha
func (s *CpuStorage[T]) Elu(layout *candy.Layout, alpha T) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UEluStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		alpha,
		s.data,
		result.data,
	)

	return result, nil
}

// Silu performs element-wise SiLU (Swish) activation operation
func (s *CpuStorage[T]) Silu(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.USiluStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Powf performs element-wise power operation with parameter param
func (s *CpuStorage[T]) Powf(layout *candy.Layout, param T) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.UPowfStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		param,
		s.data,
		result.data,
	)

	return result, nil
}

// Sigmoid performs element-wise sigmoid activation operation
func (s *CpuStorage[T]) Sigmoid(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	kernels.USigmoidStrided(
		numel,
		layout.Rank(),
		layout.Dims(),
		layout.Stride(),
		s.data,
		result.data,
	)

	return result, nil
}

// Sign performs element-wise sign operation
func (s *CpuStorage[T]) Sign(layout *candy.Layout) (candy.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.Numel()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.USignStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.USignStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.USignStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.USignStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.USignStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Sign")
	}

	return result, nil
}

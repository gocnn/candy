package cpu

import (
	"errors"
	"slices"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/internal/cpu/kernels"
)

var _ spark.BackendStorage[float32] = (*CpuStorage[float32])(nil)
var _ spark.BackendStorage[float64] = (*CpuStorage[float64])(nil)
var _ spark.BackendStorage[uint8] = (*CpuStorage[uint8])(nil)
var _ spark.BackendStorage[uint32] = (*CpuStorage[uint32])(nil)
var _ spark.BackendStorage[int64] = (*CpuStorage[int64])(nil)

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

// Max performs element-wise maximum of two tensors.
func (s *CpuStorage[T]) Max(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[T], error) {
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
	kernels.BMaxStrided(
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

// Min performs element-wise minimum of two tensors.
func (s *CpuStorage[T]) Min(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[T], error) {
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
	kernels.BMinStrided(
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

// Eq performs element-wise equality comparison of two tensors.
func (s *CpuStorage[T]) Eq(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[uint8], error) {
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

	result := New(make([]uint8, resLayout.ElemCount()))
	kernels.EqStrided(
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

// Ne performs element-wise not-equal comparison of two tensors.
func (s *CpuStorage[T]) Ne(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[uint8], error) {
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

	result := New(make([]uint8, resLayout.ElemCount()))
	kernels.NeStrided(
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

// Lt performs element-wise less-than comparison of two tensors.
func (s *CpuStorage[T]) Lt(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[uint8], error) {
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

	result := New(make([]uint8, resLayout.ElemCount()))
	kernels.LtStrided(
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

// Le performs element-wise less-than-or-equal comparison of two tensors.
func (s *CpuStorage[T]) Le(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[uint8], error) {
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

	result := New(make([]uint8, resLayout.ElemCount()))
	kernels.LeStrided(
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

// Gt performs element-wise greater-than comparison of two tensors.
func (s *CpuStorage[T]) Gt(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[uint8], error) {
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

	result := New(make([]uint8, resLayout.ElemCount()))
	kernels.GtStrided(
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

// Ge performs element-wise greater-than-or-equal comparison of two tensors.
func (s *CpuStorage[T]) Ge(rhs spark.BackendStorage[T], lhsLayout *spark.Layout, rhsLayout *spark.Layout, resLayout *spark.Layout) (spark.BackendStorage[uint8], error) {
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

	result := New(make([]uint8, resLayout.ElemCount()))
	kernels.GeStrided(
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

// ToDtype performs type conversion to the specified target type.
func (s *CpuStorage[T]) ToDtype(layout *spark.Layout, dtype spark.DType) (any, error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	srcDtype := s.dtype
	if srcDtype == dtype {
		cloned, err := s.TryClone()
		return cloned, err
	}

	// Handle type conversions based on source type
	switch srcDtype {
	case spark.F32:
		return s.CastFromF32(numel, layout, dtype)
	case spark.F64:
		return s.CastFromF64(numel, layout, dtype)
	case spark.U8:
		return s.CastFromU8(numel, layout, dtype)
	case spark.U32:
		return s.CastFromU32(numel, layout, dtype)
	case spark.I64:
		return s.CastFromI64(numel, layout, dtype)
	default:
		return nil, errors.New("unsupported source type: " + srcDtype.String())
	}
}

func (s *CpuStorage[T]) CastFromF32(numel int, layout *spark.Layout, dtype spark.DType) (any, error) {
	srcData := any(s.data).([]float32)
	stride := layout.Stride()
	dims := layout.Dims()
	ndims := layout.Rank()

	switch dtype {
	case spark.F64:
		result := New(make([]float64, numel))
		kernels.CastStridedF32F64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.U8:
		result := New(make([]uint8, numel))
		kernels.CastStridedF32U8(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.U32:
		result := New(make([]uint32, numel))
		kernels.CastStridedF32U32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.I64:
		result := New(make([]int64, numel))
		kernels.CastStridedF32I64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	}
	return nil, errors.New("unsupported target type: " + dtype.String())
}

func (s *CpuStorage[T]) CastFromF64(numel int, layout *spark.Layout, dtype spark.DType) (any, error) {
	srcData := any(s.data).([]float64)
	stride := layout.Stride()
	dims := layout.Dims()
	ndims := layout.Rank()

	switch dtype {
	case spark.F32:
		result := New(make([]float32, numel))
		kernels.CastStridedF64F32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.U8:
		result := New(make([]uint8, numel))
		kernels.CastStridedF64U8(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.U32:
		result := New(make([]uint32, numel))
		kernels.CastStridedF64U32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.I64:
		result := New(make([]int64, numel))
		kernels.CastStridedF64I64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	}
	return nil, errors.New("unsupported target type: " + dtype.String())
}

func (s *CpuStorage[T]) CastFromU8(numel int, layout *spark.Layout, dtype spark.DType) (any, error) {
	srcData := any(s.data).([]uint8)
	stride := layout.Stride()
	dims := layout.Dims()
	ndims := layout.Rank()

	switch dtype {
	case spark.F32:
		result := New(make([]float32, numel))
		kernels.CastStridedU8F32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.F64:
		result := New(make([]float64, numel))
		kernels.CastStridedU8F64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.U32:
		result := New(make([]uint32, numel))
		kernels.CastStridedU8U32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.I64:
		result := New(make([]int64, numel))
		kernels.CastStridedU8I64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	}
	return nil, errors.New("unsupported target type: " + dtype.String())
}

func (s *CpuStorage[T]) CastFromU32(numel int, layout *spark.Layout, dtype spark.DType) (any, error) {
	srcData := any(s.data).([]uint32)
	stride := layout.Stride()
	dims := layout.Dims()
	ndims := layout.Rank()

	switch dtype {
	case spark.F32:
		result := New(make([]float32, numel))
		kernels.CastStridedU32F32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.F64:
		result := New(make([]float64, numel))
		kernels.CastStridedU32F64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.U8:
		result := New(make([]uint8, numel))
		kernels.CastStridedU32U8(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.I64:
		result := New(make([]int64, numel))
		kernels.CastStridedU32I64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	}
	return nil, errors.New("unsupported target type: " + dtype.String())
}

func (s *CpuStorage[T]) CastFromI64(numel int, layout *spark.Layout, dtype spark.DType) (any, error) {
	srcData := any(s.data).([]int64)
	stride := layout.Stride()
	dims := layout.Dims()
	ndims := layout.Rank()

	switch dtype {
	case spark.F32:
		result := New(make([]float32, numel))
		kernels.CastStridedI64F32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.F64:
		result := New(make([]float64, numel))
		kernels.CastStridedI64F64(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.U8:
		result := New(make([]uint8, numel))
		kernels.CastStridedI64U8(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	case spark.U32:
		result := New(make([]uint32, numel))
		kernels.CastStridedI64U32(numel, ndims, dims, stride, stride, srcData, result.data)
		return result, nil
	}
	return nil, errors.New("unsupported target type: " + dtype.String())
}

// Conv1d performs 1D convolution using im2col + BLAS for supported types.
func (s *CpuStorage[T]) Conv1d(layout *spark.Layout, kernel spark.BackendStorage[T], kernelLayout *spark.Layout, params *spark.Conv1DParams) (spark.BackendStorage[T], error) {
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
		srcData := any(s.data).([]float32)
		kernelData := any(kernelC.data).([]float32)
		dstData := any(result.data).([]float32)
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
				srcData,
				kernelData,
				dstData,
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
				srcData,
				kernelData,
				dstData,
			)
		}
	case []float64:
		srcData := any(s.data).([]float64)
		kernelData := any(kernelC.data).([]float64)
		dstData := any(result.data).([]float64)
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
				srcData,
				kernelData,
				dstData,
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
				srcData,
				kernelData,
				dstData,
			)
		}
	case []uint8:
		srcData := any(s.data).([]uint8)
		kernelData := any(kernelC.data).([]uint8)
		dstData := any(result.data).([]uint8)
		kernels.NaiveConv1dU8(
			params.Batch,
			params.InCh,
			params.InLen,
			params.OutCh,
			params.KSize,
			params.Stride,
			params.Pad,
			params.Dilate,
			srcData,
			kernelData,
			dstData,
		)
	case []uint32:
		srcData := any(s.data).([]uint32)
		kernelData := any(kernelC.data).([]uint32)
		dstData := any(result.data).([]uint32)
		kernels.NaiveConv1dU32(
			params.Batch,
			params.InCh,
			params.InLen,
			params.OutCh,
			params.KSize,
			params.Stride,
			params.Pad,
			params.Dilate,
			srcData,
			kernelData,
			dstData,
		)
	case []int64:
		srcData := any(s.data).([]int64)
		kernelData := any(kernelC.data).([]int64)
		dstData := any(result.data).([]int64)
		kernels.NaiveConv1dI64(
			params.Batch,
			params.InCh,
			params.InLen,
			params.OutCh,
			params.KSize,
			params.Stride,
			params.Pad,
			params.Dilate,
			srcData,
			kernelData,
			dstData,
		)
	default:
		return nil, errors.New("unsupported data type for conv1d")
	}

	return result, nil
}

// ConvTranspose1d performs 1D transposed convolution (deconvolution) for supported types.
func (s *CpuStorage[T]) ConvTranspose1d(layout *spark.Layout, kernel spark.BackendStorage[T], kernelLayout *spark.Layout, params *spark.ConvT1DParams) (spark.BackendStorage[T], error) {
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
		srcData := any(s.data).([]float32)
		kernelData := any(kernelC.data).([]float32)
		dstData := any(result.data).([]float32)
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
			srcData,
			kernelData,
			dstData,
		)
	case []float64:
		srcData := any(s.data).([]float64)
		kernelData := any(kernelC.data).([]float64)
		dstData := any(result.data).([]float64)
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
			srcData,
			kernelData,
			dstData,
		)
	case []uint8:
		srcData := any(s.data).([]uint8)
		kernelData := any(kernelC.data).([]uint8)
		dstData := any(result.data).([]uint8)
		kernels.NaiveConvTranspose1dU8(
			params.Batch,
			params.InCh,
			params.InLen,
			params.OutCh,
			params.KSize,
			params.Stride,
			params.Pad,
			params.OutPad,
			params.Dilate,
			srcData,
			kernelData,
			dstData,
		)
	case []uint32:
		srcData := any(s.data).([]uint32)
		kernelData := any(kernelC.data).([]uint32)
		dstData := any(result.data).([]uint32)
		kernels.NaiveConvTranspose1dU32(
			params.Batch,
			params.InCh,
			params.InLen,
			params.OutCh,
			params.KSize,
			params.Stride,
			params.Pad,
			params.OutPad,
			params.Dilate,
			srcData,
			kernelData,
			dstData,
		)
	case []int64:
		srcData := any(s.data).([]int64)
		kernelData := any(kernelC.data).([]int64)
		dstData := any(result.data).([]int64)
		kernels.NaiveConvTranspose1dI64(
			params.Batch,
			params.InCh,
			params.InLen,
			params.OutCh,
			params.KSize,
			params.Stride,
			params.Pad,
			params.OutPad,
			params.Dilate,
			srcData,
			kernelData,
			dstData,
		)
	default:
		return nil, errors.New("unsupported data type for conv_transpose1d")
	}

	return result, nil
}

// Conv2d performs 2D convolution using im2col + BLAS for supported types.
func (s *CpuStorage[T]) Conv2d(layout *spark.Layout, kernel spark.BackendStorage[T], kernelLayout *spark.Layout, params *spark.Conv2DParams) (spark.BackendStorage[T], error) {
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
		srcData := any(s.data).([]float32)
		kernelData := any(kernelC.data).([]float32)
		dstData := any(result.data).([]float32)
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
				srcData,
				kernelData,
				dstData,
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
				srcData,
				kernelData,
				dstData,
			)
		}
	case []float64:
		srcData := any(s.data).([]float64)
		kernelData := any(kernelC.data).([]float64)
		dstData := any(result.data).([]float64)
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
				srcData,
				kernelData,
				dstData,
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
				srcData,
				kernelData,
				dstData,
			)
		}
	case []uint8:
		srcData := any(s.data).([]uint8)
		kernelData := any(kernelC.data).([]uint8)
		dstData := any(result.data).([]uint8)
		kernels.NaiveConv2dU8(
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
			srcData,
			kernelData,
			dstData,
		)
	case []uint32:
		srcData := any(s.data).([]uint32)
		kernelData := any(kernelC.data).([]uint32)
		dstData := any(result.data).([]uint32)
		kernels.NaiveConv2dU32(
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
			srcData,
			kernelData,
			dstData,
		)
	case []int64:
		srcData := any(s.data).([]int64)
		kernelData := any(kernelC.data).([]int64)
		dstData := any(result.data).([]int64)
		kernels.NaiveConv2dI64(
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
			srcData,
			kernelData,
			dstData,
		)
	default:
		return nil, errors.New("unsupported data type for conv2d")
	}

	return result, nil
}

// ConvTranspose2d performs 2D transposed convolution (deconvolution) for supported types.
func (s *CpuStorage[T]) ConvTranspose2d(layout *spark.Layout, kernel spark.BackendStorage[T], kernelLayout *spark.Layout, params *spark.ConvT2DParams) (spark.BackendStorage[T], error) {
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
		srcData := any(s.data).([]float32)
		kernelData := any(kernelC.data).([]float32)
		dstData := any(result.data).([]float32)
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
			srcData,
			kernelData,
			dstData,
		)
	case []float64:
		srcData := any(s.data).([]float64)
		kernelData := any(kernelC.data).([]float64)
		dstData := any(result.data).([]float64)
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
			srcData,
			kernelData,
			dstData,
		)
	case []uint8:
		srcData := any(s.data).([]uint8)
		kernelData := any(kernelC.data).([]uint8)
		dstData := any(result.data).([]uint8)
		kernels.NaiveConvTranspose2dU8(
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
			srcData,
			kernelData,
			dstData,
		)
	case []uint32:
		srcData := any(s.data).([]uint32)
		kernelData := any(kernelC.data).([]uint32)
		dstData := any(result.data).([]uint32)
		kernels.NaiveConvTranspose2dU32(
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
			srcData,
			kernelData,
			dstData,
		)
	case []int64:
		srcData := any(s.data).([]int64)
		kernelData := any(kernelC.data).([]int64)
		dstData := any(result.data).([]int64)
		kernels.NaiveConvTranspose2dI64(
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
			srcData,
			kernelData,
			dstData,
		)
	default:
		return nil, errors.New("unsupported data type for conv_transpose2d")
	}

	return result, nil
}

// AvgPool2d performs 2D average pooling for supported types.
func (s *CpuStorage[T]) AvgPool2d(layout *spark.Layout, params *spark.Pool2DParams) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if params == nil {
		return nil, errors.New("params cannot be nil")
	}

	hOut := params.OutH()
	wOut := params.OutW()
	if hOut <= 0 || wOut <= 0 {
		return nil, errors.New("invalid pooling parameters: output dimensions <= 0")
	}

	result := New(make([]T, params.Batch*params.Ch*hOut*wOut))

	switch any(s.data).(type) {
	case []float32:
		srcData := any(s.data).([]float32)
		dstData := any(result.data).([]float32)
		if layout.IsContiguous() {
			kernels.AvgPool2dF32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
			)
		} else {
			kernels.AvgPool2dStridedF32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []float64:
		srcData := any(s.data).([]float64)
		dstData := any(result.data).([]float64)
		if layout.IsContiguous() {
			kernels.AvgPool2dF64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
			)
		} else {
			kernels.AvgPool2dStridedF64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []uint8:
		srcData := any(s.data).([]uint8)
		dstData := any(result.data).([]uint8)
		if layout.IsContiguous() {
			kernels.AvgPool2dU8(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
			)
		} else {
			kernels.AvgPool2dStridedU8(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []uint32:
		srcData := any(s.data).([]uint32)
		dstData := any(result.data).([]uint32)
		if layout.IsContiguous() {
			kernels.AvgPool2dU32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
			)
		} else {
			kernels.AvgPool2dStridedU32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []int64:
		srcData := any(s.data).([]int64)
		dstData := any(result.data).([]int64)
		if layout.IsContiguous() {
			kernels.AvgPool2dI64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
			)
		} else {
			kernels.AvgPool2dStridedI64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	default:
		return nil, errors.New("unsupported data type for avg_pool2d")
	}

	return result, nil
}

// MaxPool2d performs 2D max pooling for supported types.
func (s *CpuStorage[T]) MaxPool2d(layout *spark.Layout, params *spark.MaxPool2DParams) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if params == nil {
		return nil, errors.New("params cannot be nil")
	}

	hOut := params.OutH()
	wOut := params.OutW()
	if hOut <= 0 || wOut <= 0 {
		return nil, errors.New("invalid pooling parameters: output dimensions <= 0")
	}

	result := New(make([]T, params.Batch*params.Ch*hOut*wOut))

	switch any(s.data).(type) {
	case []float32:
		srcData := any(s.data).([]float32)
		dstData := any(result.data).([]float32)
		if layout.IsContiguous() {
			kernels.MaxPool2dF32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
			)
		} else {
			kernels.MaxPool2dStridedF32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []float64:
		srcData := any(s.data).([]float64)
		dstData := any(result.data).([]float64)
		if layout.IsContiguous() {
			kernels.MaxPool2dF64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
			)
		} else {
			kernels.MaxPool2dStridedF64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []uint8:
		srcData := any(s.data).([]uint8)
		dstData := any(result.data).([]uint8)
		if layout.IsContiguous() {
			kernels.MaxPool2dU8(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
			)
		} else {
			kernels.MaxPool2dStridedU8(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []uint32:
		srcData := any(s.data).([]uint32)
		dstData := any(result.data).([]uint32)
		if layout.IsContiguous() {
			kernels.MaxPool2dU32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
			)
		} else {
			kernels.MaxPool2dStridedU32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []int64:
		srcData := any(s.data).([]int64)
		dstData := any(result.data).([]int64)
		if layout.IsContiguous() {
			kernels.MaxPool2dI64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
			)
		} else {
			kernels.MaxPool2dStridedI64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.KH,
				params.KW,
				params.HStride,
				params.WStride,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	default:
		return nil, errors.New("unsupported data type for max_pool2d")
	}

	return result, nil
}

// UpsampleNearest2d performs 2D nearest neighbor upsampling for supported types.
func (s *CpuStorage[T]) UpsampleNearest2d(layout *spark.Layout, params *spark.UpsampleParams) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if params == nil {
		return nil, errors.New("params cannot be nil")
	}

	hOut := params.HOut
	wOut := params.WOut
	if hOut <= 0 || wOut <= 0 {
		return nil, errors.New("invalid upsampling parameters: output dimensions <= 0")
	}

	result := New(make([]T, params.Batch*params.Ch*hOut*wOut))

	switch any(s.data).(type) {
	case []float32:
		srcData := any(s.data).([]float32)
		dstData := any(result.data).([]float32)
		if layout.IsContiguous() {
			kernels.UpsampleNearest2dF32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.HOut,
				params.WOut,
				params.HScale,
				params.WScale,
				srcData,
				dstData,
			)
		} else {
			kernels.UpsampleNearest2dStridedF32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.HOut,
				params.WOut,
				params.HScale,
				params.WScale,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []float64:
		srcData := any(s.data).([]float64)
		dstData := any(result.data).([]float64)
		if layout.IsContiguous() {
			kernels.UpsampleNearest2dF64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.HOut,
				params.WOut,
				params.HScale,
				params.WScale,
				srcData,
				dstData,
			)
		} else {
			kernels.UpsampleNearest2dStridedF64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.HOut,
				params.WOut,
				params.HScale,
				params.WScale,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []uint8:
		srcData := any(s.data).([]uint8)
		dstData := any(result.data).([]uint8)
		if layout.IsContiguous() {
			kernels.UpsampleNearest2dU8(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.HOut,
				params.WOut,
				params.HScale,
				params.WScale,
				srcData,
				dstData,
			)
		} else {
			kernels.UpsampleNearest2dStridedU8(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.HOut,
				params.WOut,
				params.HScale,
				params.WScale,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []uint32:
		srcData := any(s.data).([]uint32)
		dstData := any(result.data).([]uint32)
		if layout.IsContiguous() {
			kernels.UpsampleNearest2dU32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.HOut,
				params.WOut,
				params.HScale,
				params.WScale,
				srcData,
				dstData,
			)
		} else {
			kernels.UpsampleNearest2dStridedU32(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.HOut,
				params.WOut,
				params.HScale,
				params.WScale,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	case []int64:
		srcData := any(s.data).([]int64)
		dstData := any(result.data).([]int64)
		if layout.IsContiguous() {
			kernels.UpsampleNearest2dI64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.HOut,
				params.WOut,
				params.HScale,
				params.WScale,
				srcData,
				dstData,
			)
		} else {
			kernels.UpsampleNearest2dStridedI64(
				params.Batch,
				params.Ch,
				params.InH,
				params.InW,
				params.HOut,
				params.WOut,
				params.HScale,
				params.WScale,
				srcData,
				dstData,
				layout.Stride(),
				[]int{params.Ch * hOut * wOut, hOut * wOut, wOut, 1},
			)
		}
	default:
		return nil, errors.New("unsupported data type for upsample_nearest2d")
	}

	return result, nil
}

// ConstSet sets all elements to a constant value for supported types.
func (s *CpuStorage[T]) ConstSet(layout *spark.Layout, val T) error {
	if layout == nil {
		return errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
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

// Copy2d copies a 2D region from source to destination for supported types.
func (s *CpuStorage[T]) Copy2d(dst spark.BackendStorage[T], d1, d2 int, srcStride1, dstStride1, srcOffset, dstOffset int) error {
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
func (s *CpuStorage[T]) FastSum(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
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
func (s *CpuStorage[T]) FastMin(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
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
func (s *CpuStorage[T]) FastMax(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
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
func (s *CpuStorage[T]) FastArgmin(layout *spark.Layout) (spark.BackendStorage[uint32], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
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
func (s *CpuStorage[T]) FastArgmax(layout *spark.Layout) (spark.BackendStorage[uint32], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
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

func (s *CpuStorage[T]) Sum(layout *spark.Layout, sumDims []int) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if len(sumDims) == 0 {
		return nil, errors.New("sumDims cannot be empty")
	}

	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	outputDims := make([]int, len(layout.Dims()))
	copy(outputDims, layout.Dims())
	for _, dim := range sumDims {
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
		sumDims,
		s.data,
		result.data,
	)

	return result, nil
}

// Softmax performs softmax along the last dimension
func (s *CpuStorage[T]) Softmax(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))
	switch any(s.data).(type) {
	case []float32:
		kernels.SoftmaxStridedF32(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(s.data).([]float32),
			any(result.data).([]float32),
		)
	case []float64:
		kernels.SoftmaxStridedF64(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(s.data).([]float64),
			any(result.data).([]float64),
		)
	case []uint8:
		kernels.SoftmaxStridedU8(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(s.data).([]uint8),
			any(result.data).([]uint8),
		)
	case []uint32:
		kernels.SoftmaxStridedU32(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(s.data).([]uint32),
			any(result.data).([]uint32),
		)
	case []int64:
		kernels.SoftmaxStridedI64(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(s.data).([]int64),
			any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for softmax")
	}

	return result, nil
}

// RmsNorm performs RMS normalization along the last dimension
func (s *CpuStorage[T]) RmsNorm(layout *spark.Layout, eps T, alpha []T) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))
	switch any(s.data).(type) {
	case []float32:
		kernels.RmsNormStridedF32(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(eps).(float32),
			any(alpha).([]float32),
			any(s.data).([]float32),
			any(result.data).([]float32),
		)
	case []float64:
		kernels.RmsNormStridedF64(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(eps).(float64),
			any(alpha).([]float64),
			any(s.data).([]float64),
			any(result.data).([]float64),
		)
	case []uint8:
		kernels.RmsNormStridedU8(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(eps).(float64),
			any(alpha).([]uint8),
			any(s.data).([]uint8),
			any(result.data).([]uint8),
		)
	case []uint32:
		kernels.RmsNormStridedU32(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(eps).(float64),
			any(alpha).([]uint32),
			any(s.data).([]uint32),
			any(result.data).([]uint32),
		)
	case []int64:
		kernels.RmsNormStridedI64(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(eps).(float64),
			any(alpha).([]int64),
			any(s.data).([]int64),
			any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for RmsNorm")
	}

	return result, nil
}

// LayerNorm performs Layer normalization along the last dimension
func (s *CpuStorage[T]) LayerNorm(layout *spark.Layout, eps T, alpha, beta []T) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))
	switch any(s.data).(type) {
	case []float32:
		kernels.LayerNormStridedF32(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(eps).(float32),
			any(alpha).([]float32),
			any(beta).([]float32),
			any(s.data).([]float32),
			any(result.data).([]float32),
		)
	case []float64:
		kernels.LayerNormStridedF64(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(eps).(float64),
			any(alpha).([]float64),
			any(beta).([]float64),
			any(s.data).([]float64),
			any(result.data).([]float64),
		)
	case []uint8:
		kernels.LayerNormStridedU8(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(eps).(float64),
			any(alpha).([]uint8),
			any(beta).([]uint8),
			any(s.data).([]uint8),
			any(result.data).([]uint8),
		)
	case []uint32:
		kernels.LayerNormStridedU32(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(eps).(float64),
			any(alpha).([]uint32),
			any(beta).([]uint32),
			any(s.data).([]uint32),
			any(result.data).([]uint32),
		)
	case []int64:
		kernels.LayerNormStridedI64(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(eps).(float64),
			any(alpha).([]int64),
			any(beta).([]int64),
			any(s.data).([]int64),
			any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for LayerNorm")
	}

	return result, nil
}

// RopeI performs rotary position embedding (rope_i variant)
func (s *CpuStorage[T]) RopeI(layout *spark.Layout, bh, td, strideB int, cos, sin []T) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if len(s.data) == 0 {
		return nil, errors.New("source storage is empty")
	}

	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))
	switch any(s.data).(type) {
	case []float32:
		kernels.RopeIStridedF32(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			bh,
			td,
			strideB,
			any(s.data).([]float32),
			any(cos).([]float32),
			any(sin).([]float32),
			any(result.data).([]float32),
		)
	case []float64:
		kernels.RopeIStridedF64(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			bh,
			td,
			strideB,
			any(s.data).([]float64),
			any(cos).([]float64),
			any(sin).([]float64),
			any(result.data).([]float64),
		)
	case []uint8:
		kernels.RopeIStridedU8(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			bh,
			td,
			strideB,
			any(s.data).([]uint8),
			any(cos).([]uint8),
			any(sin).([]uint8),
			any(result.data).([]uint8),
		)
	case []uint32:
		kernels.RopeIStridedU32(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			bh,
			td,
			strideB,
			any(s.data).([]uint32),
			any(cos).([]uint32),
			any(sin).([]uint32),
			any(result.data).([]uint32),
		)
	case []int64:
		kernels.RopeIStridedI64(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			bh,
			td,
			strideB,
			any(s.data).([]int64),
			any(cos).([]int64),
			any(sin).([]int64),
			any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for RopeI")
	}

	return result, nil
}

// Rope performs rotary position embedding (rope variant)
func (s *CpuStorage[T]) Rope(layout *spark.Layout, bh, td, d, strideB int, cos, sin []T) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if len(s.data) == 0 {
		return nil, errors.New("source storage is empty")
	}

	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))
	switch any(s.data).(type) {
	case []float32:
		kernels.RopeStridedF32(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			bh,
			td,
			d,
			strideB,
			any(s.data).([]float32),
			any(cos).([]float32),
			any(sin).([]float32),
			any(result.data).([]float32),
		)
	case []float64:
		kernels.RopeStridedF64(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			bh,
			td,
			d,
			strideB,
			any(s.data).([]float64),
			any(cos).([]float64),
			any(sin).([]float64),
			any(result.data).([]float64),
		)
	case []uint8:
		kernels.RopeStridedU8(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			bh,
			td,
			d,
			strideB,
			any(s.data).([]uint8),
			any(cos).([]uint8),
			any(sin).([]uint8),
			any(result.data).([]uint8),
		)
	case []uint32:
		kernels.RopeStridedU32(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			bh,
			td,
			d,
			strideB,
			any(s.data).([]uint32),
			any(cos).([]uint32),
			any(sin).([]uint32),
			any(result.data).([]uint32),
		)
	case []int64:
		kernels.RopeStridedI64(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			bh,
			td,
			d,
			strideB,
			any(s.data).([]int64),
			any(cos).([]int64),
			any(sin).([]int64),
			any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Rope")
	}

	return result, nil
}

// RopeThd performs rotary position embedding (rope_thd variant)
func (s *CpuStorage[T]) RopeThd(layout *spark.Layout, b, t, h, d, strideB int, cos, sin []T) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}
	if len(s.data) == 0 {
		return nil, errors.New("source storage is empty")
	}

	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))
	switch any(s.data).(type) {
	case []float32:
		kernels.RopeThdStridedF32(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			b,
			t,
			h,
			d,
			strideB,
			any(s.data).([]float32),
			any(cos).([]float32),
			any(sin).([]float32),
			any(result.data).([]float32),
		)
	case []float64:
		kernels.RopeThdStridedF64(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			b,
			t,
			h,
			d,
			strideB,
			any(s.data).([]float64),
			any(cos).([]float64),
			any(sin).([]float64),
			any(result.data).([]float64),
		)
	case []uint8:
		kernels.RopeThdStridedU8(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			b,
			t,
			h,
			d,
			strideB,
			any(s.data).([]uint8),
			any(cos).([]uint8),
			any(sin).([]uint8),
			any(result.data).([]uint8),
		)
	case []uint32:
		kernels.RopeThdStridedU32(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			b,
			t,
			h,
			d,
			strideB,
			any(s.data).([]uint32),
			any(cos).([]uint32),
			any(sin).([]uint32),
			any(result.data).([]uint32),
		)
	case []int64:
		kernels.RopeThdStridedI64(
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			b,
			t,
			h,
			d,
			strideB,
			any(s.data).([]int64),
			any(cos).([]int64),
			any(sin).([]int64),
			any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for RopeThd")
	}

	return result, nil
}

// Copy performs element-wise copy operation
func (s *CpuStorage[T]) Copy(layout *spark.Layout, src spark.BackendStorage[T]) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	srcData := src.(*CpuStorage[T]).data
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UCopyStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(srcData).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UCopyStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(srcData).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UCopyStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(srcData).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UCopyStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(srcData).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UCopyStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(srcData).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Copy")
	}

	return result, nil
}

// Neg performs element-wise negation operation
func (s *CpuStorage[T]) Neg(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UNegStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UNegStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UNegStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UNegStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UNegStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Neg")
	}

	return result, nil
}

// Recip performs element-wise reciprocal operation
func (s *CpuStorage[T]) Recip(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	if numel != len(s.data) {
		return nil, errors.New("layout element count does not match storage size")
	}

	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.URecipStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.URecipStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.URecipStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.URecipStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.URecipStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Recip")
	}

	return result, nil
}

// Exp performs element-wise exponential operation
func (s *CpuStorage[T]) Exp(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UExpStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UExpStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UExpStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UExpStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UExpStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Exp")
	}

	return result, nil
}

// Log performs element-wise logarithm operation
func (s *CpuStorage[T]) Log(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.ULogStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.ULogStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.ULogStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.ULogStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.ULogStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Log")
	}

	return result, nil
}

// Sin performs element-wise sine operation
func (s *CpuStorage[T]) Sin(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.USinStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.USinStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.USinStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.USinStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.USinStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Sin")
	}

	return result, nil
}

// Cos performs element-wise cosine operation
func (s *CpuStorage[T]) Cos(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UCosStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UCosStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UCosStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UCosStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UCosStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Cos")
	}

	return result, nil
}

// Tanh performs element-wise hyperbolic tangent operation
func (s *CpuStorage[T]) Tanh(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UTanhStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UTanhStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UTanhStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UTanhStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UTanhStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Tanh")
	}

	return result, nil
}

// Erf performs element-wise error function operation
func (s *CpuStorage[T]) Erf(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UErfStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UErfStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UErfStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UErfStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UErfStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Erf")
	}

	return result, nil
}

// Ceil performs element-wise ceiling operation
func (s *CpuStorage[T]) Ceil(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UCeilStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UCeilStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UCeilStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UCeilStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UCeilStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Ceil")
	}

	return result, nil
}

// Floor performs element-wise floor operation
func (s *CpuStorage[T]) Floor(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UFloorStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UFloorStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UFloorStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UFloorStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UFloorStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Floor")
	}

	return result, nil
}

// Round performs element-wise round operation
func (s *CpuStorage[T]) Round(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.URoundStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.URoundStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.URoundStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.URoundStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.URoundStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Round")
	}

	return result, nil
}

// Normcdf performs element-wise normal CDF operation
func (s *CpuStorage[T]) Normcdf(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UNormcdfStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UNormcdfStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UNormcdfStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UNormcdfStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UNormcdfStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Normcdf")
	}

	return result, nil
}

// Abs performs element-wise absolute value operation
func (s *CpuStorage[T]) Abs(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UAbsStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UAbsStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UAbsStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UAbsStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UAbsStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Abs")
	}

	return result, nil
}

// Sqr performs element-wise square operation
func (s *CpuStorage[T]) Sqr(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.USqrStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.USqrStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.USqrStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.USqrStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.USqrStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Sqr")
	}

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
	switch any(s.data).(type) {
	case []float32:
		kernels.USqrtStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.USqrtStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.USqrtStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.USqrtStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.USqrtStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Sqrt")
	}

	return result, nil
}

// Gelu performs element-wise GELU activation operation
func (s *CpuStorage[T]) Gelu(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UGeluStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UGeluStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UGeluStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UGeluStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UGeluStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Gelu")
	}

	return result, nil
}

// GeluErf performs element-wise GELU (ERF-based) activation operation
func (s *CpuStorage[T]) GeluErf(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UGeluErfStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UGeluErfStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UGeluErfStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UGeluErfStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UGeluErfStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for GeluErf")
	}

	return result, nil
}

// Relu performs element-wise ReLU activation operation
func (s *CpuStorage[T]) Relu(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UReluStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.UReluStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.UReluStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UReluStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.UReluStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Relu")
	}

	return result, nil
}

// Elu performs element-wise ELU activation operation with parameter alpha
func (s *CpuStorage[T]) Elu(layout *spark.Layout, alpha T) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UEluStridedF32(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(alpha).(float32),
			any(s.data).([]float32),
			any(result.data).([]float32),
		)
	case []float64:
		kernels.UEluStridedF64(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(alpha).(float64),
			any(s.data).([]float64),
			any(result.data).([]float64),
		)
	case []uint8:
		kernels.UEluStridedU8(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(alpha).(uint8),
			any(s.data).([]uint8),
			any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UEluStridedU32(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(alpha).(uint32),
			any(s.data).([]uint32),
			any(result.data).([]uint32),
		)
	case []int64:
		kernels.UEluStridedI64(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(alpha).(int64),
			any(s.data).([]int64),
			any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Elu")
	}

	return result, nil
}

// Silu performs element-wise SiLU (Swish) activation operation
func (s *CpuStorage[T]) Silu(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.USiluStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.USiluStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.USiluStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.USiluStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.USiluStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Silu")
	}

	return result, nil
}

// Powf performs element-wise power operation with parameter param
func (s *CpuStorage[T]) Powf(layout *spark.Layout, param T) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.UPowfStridedF32(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(param).(float32),
			any(s.data).([]float32),
			any(result.data).([]float32),
		)
	case []float64:
		kernels.UPowfStridedF64(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(param).(float64),
			any(s.data).([]float64),
			any(result.data).([]float64),
		)
	case []uint8:
		kernels.UPowfStridedU8(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(param).(uint8),
			any(s.data).([]uint8),
			any(result.data).([]uint8),
		)
	case []uint32:
		kernels.UPowfStridedU32(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(param).(uint32),
			any(s.data).([]uint32),
			any(result.data).([]uint32),
		)
	case []int64:
		kernels.UPowfStridedI64(
			numel,
			layout.Rank(),
			layout.Dims(),
			layout.Stride(),
			any(param).(int64),
			any(s.data).([]int64),
			any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Powf")
	}

	return result, nil
}

// Sign performs element-wise sign operation
func (s *CpuStorage[T]) Sign(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
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

// Sigmoid performs element-wise sigmoid activation operation
func (s *CpuStorage[T]) Sigmoid(layout *spark.Layout) (spark.BackendStorage[T], error) {
	if layout == nil {
		return nil, errors.New("layout cannot be nil")
	}

	numel := layout.ElemCount()
	result := New(make([]T, numel))

	switch any(s.data).(type) {
	case []float32:
		kernels.USigmoidStridedF32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float32), any(result.data).([]float32),
		)
	case []float64:
		kernels.USigmoidStridedF64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]float64), any(result.data).([]float64),
		)
	case []uint8:
		kernels.USigmoidStridedU8(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint8), any(result.data).([]uint8),
		)
	case []uint32:
		kernels.USigmoidStridedU32(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]uint32), any(result.data).([]uint32),
		)
	case []int64:
		kernels.USigmoidStridedI64(
			numel, layout.Rank(), layout.Dims(), layout.Stride(),
			any(s.data).([]int64), any(result.data).([]int64),
		)
	default:
		return nil, errors.New("unsupported data type for Sigmoid")
	}

	return result, nil
}

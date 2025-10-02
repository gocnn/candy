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

func (s *CpuStorage[T]) BinaryImpl(op spark.BinaryOp, other spark.BackendStorage[T], layout *spark.Layout, otherLayout *spark.Layout) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) Powf(layout *spark.Layout, power T) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) Elu(layout *spark.Layout, alpha T) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) ReduceOp(op spark.ReduceOp, layout *spark.Layout, dims []int) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) Cmp(op spark.CmpOp, other spark.BackendStorage[T], layout *spark.Layout, otherLayout *spark.Layout) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) ToDType(layout *spark.Layout, dtype spark.DType) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) UnaryImpl(op spark.UnaryOp, layout *spark.Layout) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) WhereCond(condLayout *spark.Layout, trueValue spark.BackendStorage[T], trueLayout *spark.Layout, falseValue spark.BackendStorage[T], falseLayout *spark.Layout) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) Conv1D(layout *spark.Layout, kernel spark.BackendStorage[T], kernelLayout *spark.Layout, params *spark.ParamsConv1D) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) ConvTranspose1D(layout *spark.Layout, kernel spark.BackendStorage[T], kernelLayout *spark.Layout, params *spark.ParamsConvTranspose1D) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) Conv2D(layout *spark.Layout, kernel spark.BackendStorage[T], kernelLayout *spark.Layout, params *spark.ParamsConv2D) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) ConvTranspose2D(layout *spark.Layout, kernel spark.BackendStorage[T], kernelLayout *spark.Layout, params *spark.ParamsConvTranspose2D) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) AvgPool2D(layout *spark.Layout, kernelSize, strides [2]int) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) MaxPool2D(layout *spark.Layout, kernelSize, strides [2]int) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) UpsampleNearest1D(layout *spark.Layout, outSize int) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) UpsampleNearest2D(layout *spark.Layout, outH, outW int) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) Gather(layout *spark.Layout, indices spark.BackendStorage[T], indicesLayout *spark.Layout, dim int) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) ScatterSet(layout *spark.Layout, indices spark.BackendStorage[T], indicesLayout *spark.Layout, values spark.BackendStorage[T], valuesLayout *spark.Layout, dim int) error {
	return nil
}

func (s *CpuStorage[T]) ScatterAddSet(layout *spark.Layout, indices spark.BackendStorage[T], indicesLayout *spark.Layout, values spark.BackendStorage[T], valuesLayout *spark.Layout, dim int) error {
	return nil
}

func (s *CpuStorage[T]) IndexSelect(indices spark.BackendStorage[T], indicesLayout, layout *spark.Layout, dim int) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) IndexAdd(layout *spark.Layout, indices spark.BackendStorage[T], indicesLayout *spark.Layout, values spark.BackendStorage[T], valuesLayout *spark.Layout, dim int) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) Matmul(rhs spark.BackendStorage[T], dims [4]int, lhsLayout, rhsLayout *spark.Layout) (spark.BackendStorage[T], error) {
	return nil, nil
}

func (s *CpuStorage[T]) CopyStridedSrc(dst spark.BackendStorage[T], dstOffset int, srcLayout *spark.Layout) error {
	return nil
}

func (s *CpuStorage[T]) Copy2D(dst spark.BackendStorage[T], d1, d2, srcStride1, dstStride1, srcOffset, dstOffset int) error {
	return nil
}

func (s *CpuStorage[T]) ConstSet(scalar interface{}, layout *spark.Layout) error {
	return nil
}

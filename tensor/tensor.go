package tensor

import (
	"fmt"
	"sync/atomic"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/internal/cpu"
)

// counter is an atomic counter for unique TensorId values.
var counter uint64

// TensorId is a unique identifier for a tensor.
type TensorId uint64

// NewId generates a unique TensorId thread-safely.
func NewId() TensorId {
	return TensorId(atomic.AddUint64(&counter, 1))
}

// Tensor represents a multi-dimensional array with automatic differentiation support.
type Tensor[T spark.D] struct {
	id      TensorId
	storage spark.BackendStorage[T]
	layout  *spark.Layout
	op      *Op[T]
	isVar   bool
	dtype   spark.DType
	device  spark.Device
}

// NewFrom creates a tensor from existing storage and layout.
func NewFrom[T spark.D](storage spark.BackendStorage[T], layout *spark.Layout, dtype spark.DType, dev spark.Device) *Tensor[T] {
	return &Tensor[T]{
		id:      NewId(),
		storage: storage,
		layout:  layout,
		dtype:   dtype,
		device:  dev,
	}
}

// New creates a tensor from an array and shape on the specified device.
func New[T spark.D](data []T, shape *spark.Shape, dev spark.Device) (*Tensor[T], error) {
	var storage spark.BackendStorage[T]
	switch dev {
	case spark.CPU:
		storage = cpu.New(data)
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, spark.Contiguous(shape), spark.DTypeOf[T](), dev), nil
}

// Full creates a tensor filled with the specified value.
func Full[T spark.D](value float64, shape *spark.Shape, dev spark.Device) (*Tensor[T], error) {
	var storage spark.BackendStorage[T]
	switch dev {
	case spark.CPU:
		var err error
		storage, err = cpu.NewCpuDevice[T]().Full(shape, spark.DTypeOf[T](), value)
		if err != nil {
			return nil, fmt.Errorf("failed to create full: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, spark.Contiguous(shape), spark.DTypeOf[T](), dev), nil
}

// Ones creates a tensor filled with ones.
func Ones[T spark.D](shape *spark.Shape, dev spark.Device) (*Tensor[T], error) {
	var storage spark.BackendStorage[T]
	switch dev {
	case spark.CPU:
		var err error
		storage, err = cpu.NewCpuDevice[T]().Ones(shape, spark.DTypeOf[T]())
		if err != nil {
			return nil, fmt.Errorf("failed to create ones: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, spark.Contiguous(shape), spark.DTypeOf[T](), dev), nil
}

// Zeros creates a tensor filled with zeros.
func Zeros[T spark.D](shape *spark.Shape, dev spark.Device) (*Tensor[T], error) {
	var storage spark.BackendStorage[T]
	switch dev {
	case spark.CPU:
		var err error
		storage, err = cpu.NewCpuDevice[T]().Zeros(shape, spark.DTypeOf[T]())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, spark.Contiguous(shape), spark.DTypeOf[T](), dev), nil
}

// Rand creates a tensor with values uniformly sampled between lo and up.
func Rand[T spark.D](lo, up float64, shape *spark.Shape, dev spark.Device) (*Tensor[T], error) {
	var storage spark.BackendStorage[T]
	switch dev {
	case spark.CPU:
		var err error
		storage, err = cpu.NewCpuDevice[T]().RandUniform(shape, spark.DTypeOf[T](), lo, up)
		if err != nil {
			return nil, fmt.Errorf("failed to create rand: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, spark.Contiguous(shape), spark.DTypeOf[T](), dev), nil
}

// RandN creates a tensor with values sampled from a normal distribution.
func RandN[T spark.D](mean, std float64, shape *spark.Shape, dev spark.Device) (*Tensor[T], error) {
	var storage spark.BackendStorage[T]
	switch dev {
	case spark.CPU:
		var err error
		storage, err = cpu.NewCpuDevice[T]().RandNormal(shape, spark.DTypeOf[T](), mean, std)
		if err != nil {
			return nil, fmt.Errorf("failed to create randn: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, spark.Contiguous(shape), spark.DTypeOf[T](), dev), nil
}

// MustNew creates a tensor from an array and shape on the specified device, panicking on error.
func MustNew[T spark.D](data []T, shape *spark.Shape, dev spark.Device) *Tensor[T] {
	t, err := New[T](data, shape, dev)
	if err != nil {
		panic(err)
	}
	return t
}

// MustFull creates a tensor filled with the specified value, panicking on error.
func MustFull[T spark.D](value float64, shape *spark.Shape, dev spark.Device) *Tensor[T] {
	t, err := Full[T](value, shape, dev)
	if err != nil {
		panic(err)
	}
	return t
}

// MustOnes creates a tensor filled with ones, panicking on error.
func MustOnes[T spark.D](shape *spark.Shape, dev spark.Device) *Tensor[T] {
	t, err := Ones[T](shape, dev)
	if err != nil {
		panic(err)
	}
	return t
}

// MustZeros creates a tensor filled with zeros, panicking on error.
func MustZeros[T spark.D](shape *spark.Shape, dev spark.Device) *Tensor[T] {
	t, err := Zeros[T](shape, dev)
	if err != nil {
		panic(err)
	}
	return t
}

// MustRand creates a tensor with values uniformly sampled between lo and up, panicking on error.
func MustRand[T spark.D](lo, up float64, shape *spark.Shape, dev spark.Device) *Tensor[T] {
	t, err := Rand[T](lo, up, shape, dev)
	if err != nil {
		panic(err)
	}
	return t
}

// MustRandN creates a tensor with values sampled from a normal distribution, panicking on error.
func MustRandN[T spark.D](mean, std float64, shape *spark.Shape, dev spark.Device) *Tensor[T] {
	t, err := RandN[T](mean, std, shape, dev)
	if err != nil {
		panic(err)
	}
	return t
}

// FullLike creates a tensor with the same shape and device as t, filled with value.
func (t *Tensor[T]) FullLike(value float64) (*Tensor[T], error) {
	return Full[T](value, t.Shape(), t.device)
}

// OnesLike creates a tensor with the same shape and device as t, filled with ones.
func (t *Tensor[T]) OnesLike() (*Tensor[T], error) {
	return Ones[T](t.Shape(), t.device)
}

// ZerosLike creates a tensor with the same shape and device as t, filled with zeros.
func (t *Tensor[T]) ZerosLike() (*Tensor[T], error) {
	return Zeros[T](t.Shape(), t.device)
}

// RandLike creates a tensor with the same shape and device as t, with random values.
func (t *Tensor[T]) RandLike(lo, up float64) (*Tensor[T], error) {
	return Rand[T](lo, up, t.Shape(), t.device)
}

// RandNLike creates a tensor with the same shape and device as t, with normal-distributed values.
func (t *Tensor[T]) RandNLike(mean, std float64) (*Tensor[T], error) {
	return RandN[T](mean, std, t.Shape(), t.device)
}

// MustFullLike creates a tensor with the same shape and device as t, filled with value, panicking on error.
func (t *Tensor[T]) MustFullLike(value float64) *Tensor[T] {
	t, err := Full[T](value, t.Shape(), t.device)
	if err != nil {
		panic(err)
	}
	return t
}

// MustOnesLike creates a tensor with the same shape and device as t, filled with ones, panicking on error.
func (t *Tensor[T]) MustOnesLike() *Tensor[T] {
	t, err := Ones[T](t.Shape(), t.device)
	if err != nil {
		panic(err)
	}
	return t
}

// MustZerosLike creates a tensor with the same shape and device as t, filled with zeros, panicking on error.
func (t *Tensor[T]) MustZerosLike() *Tensor[T] {
	t, err := Zeros[T](t.Shape(), t.device)
	if err != nil {
		panic(err)
	}
	return t
}

// MustRandLike creates a tensor with the same shape and device as t, with random values, panicking on error.
func (t *Tensor[T]) MustRandLike(lo, up float64) *Tensor[T] {
	t, err := Rand[T](lo, up, t.Shape(), t.device)
	if err != nil {
		panic(err)
	}
	return t
}

// MustRandNLike creates a tensor with the same shape and device as t, with normal-distributed values, panicking on error.
func (t *Tensor[T]) MustRandNLike(mean, std float64) *Tensor[T] {
	t, err := RandN[T](mean, std, t.Shape(), t.device)
	if err != nil {
		panic(err)
	}
	return t
}

// ID returns the tensor's unique identifier.
func (t *Tensor[T]) ID() TensorId {
	return t.id
}

// Storage returns the tensor's backend storage.
func (t *Tensor[T]) Storage() spark.BackendStorage[T] {
	return t.storage
}

// Layout returns a clone of the tensor's layout.
func (t *Tensor[T]) Layout() *spark.Layout {
	return t.layout.Clone()
}

// Op returns the operation that created the tensor.
func (t *Tensor[T]) Op() *Op[T] {
	return t.op
}

// IsVar reports whether the tensor requires gradient computation.
func (t *Tensor[T]) IsVar() bool {
	return t.isVar
}

// DType returns the tensor's data type.
func (t *Tensor[T]) DType() spark.DType {
	return t.dtype
}

// Device returns the tensor's device.
func (t *Tensor[T]) Device() spark.Device {
	return t.device
}

// Data returns the tensor's data slice; panics if not CPU-based.
func (t *Tensor[T]) Data() []T {
	return t.storage.Data()
}

// Stride returns the tensor's strides.
func (t *Tensor[T]) Stride() []int {
	return t.layout.Stride()
}

// Shape returns the tensor's shape.
func (t *Tensor[T]) Shape() *spark.Shape {
	return t.layout.Shape()
}

// Dims returns the tensor's dimensions.
func (t *Tensor[T]) Dims() []int {
	return t.layout.Dims()
}

// Dim returns the size of the specified dimension.
func (t *Tensor[T]) Dim(dim int) int {
	return t.layout.Dim(dim)
}

// Dims0 checks if the shape has 0 dimensions (scalar).
func (s *Tensor[T]) Dims0() error {
	return s.layout.Dims0()
}

// Dims1 extracts the single dimension from a 1D shape.
func (s *Tensor[T]) Dims1() (int, error) {
	return s.layout.Dims1()
}

// Dims2 extracts the two dimensions from a 2D shape.
func (s *Tensor[T]) Dims2() (int, int, error) {
	return s.layout.Dims2()
}

// Dims3 extracts the three dimensions from a 3D shape.
func (s *Tensor[T]) Dims3() (int, int, int, error) {
	return s.layout.Dims3()
}

// Dims4 extracts the four dimensions from a 4D shape.
func (s *Tensor[T]) Dims4() (int, int, int, int, error) {
	return s.layout.Dims4()
}

// Dims5 extracts the five dimensions from a 5D shape.
func (s *Tensor[T]) Dims5() (int, int, int, int, int, error) {
	return s.layout.Dims5()
}

// Rank returns the tensor's rank.
func (t *Tensor[T]) Rank() int {
	return t.layout.Rank()
}

// ElemCount returns the number of elements in the tensor.
func (t *Tensor[T]) ElemCount() int {
	return t.layout.ElemCount()
}

// SetStorage sets the tensor's storage and returns the tensor.
func (t *Tensor[T]) SetStorage(storage spark.BackendStorage[T]) *Tensor[T] {
	t.storage = storage
	return t
}

// SetLayout sets the tensor's layout and returns the tensor.
func (t *Tensor[T]) SetLayout(layout *spark.Layout) *Tensor[T] {
	t.layout = layout.Clone()
	return t
}

// SetOp sets the tensor's operation and returns the tensor.
func (t *Tensor[T]) SetOp(op *Op[T]) *Tensor[T] {
	t.op = op
	return t
}

// SetIsVar sets whether the tensor requires gradients and returns the tensor.
func (t *Tensor[T]) SetIsVar(isVar bool) *Tensor[T] {
	t.isVar = isVar
	return t
}

// SetDType sets the tensor's data type and returns the tensor.
func (t *Tensor[T]) SetDType(dtype spark.DType) *Tensor[T] {
	t.dtype = dtype
	return t
}

// SetDevice sets the tensor's device and returns the tensor.
func (t *Tensor[T]) SetDevice(dev spark.Device) *Tensor[T] {
	t.device = dev
	return t
}

// RequiresGrad marks the tensor as requiring gradients and returns the tensor.
func (t *Tensor[T]) RequiresGrad() *Tensor[T] {
	t.isVar = true
	return t
}

// Detach creates a new tensor detached from the computation graph.
func (t *Tensor[T]) Detach() *Tensor[T] {
	return NewFrom(t.storage, t.layout, t.dtype, t.device)
}

// Clone creates a clone of the tensor.
func (t *Tensor[T]) Clone() *Tensor[T] {
	return &Tensor[T]{
		id:      NewId(),
		storage: t.storage,
		layout:  t.layout.Clone(),
		op:      nil,
		isVar:   false,
		dtype:   t.dtype,
		device:  t.device,
	}
}

// ToDtype converts the tensor to the specified dtype, returning a new tensor.
func ToDtype[T spark.D, U spark.D](t *Tensor[T], dtype spark.DType) (*Tensor[U], error) {
	storage, err := t.storage.ToDtype(t.layout, dtype)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to %v: %w", dtype, err)
	}
	return NewFrom(storage.(spark.BackendStorage[U]), t.layout.Clone(), dtype, t.device), nil
}

// ToFloat32 converts the tensor to float32.
func (t *Tensor[T]) ToFloat32() (*Tensor[float32], error) {
	return ToDtype[T, float32](t, spark.F32)
}

// ToFloat64 converts the tensor to float64.
func (t *Tensor[T]) ToFloat64() (*Tensor[float64], error) {
	return ToDtype[T, float64](t, spark.F64)
}

// ToUint8 converts the tensor to uint8.
func (t *Tensor[T]) ToUint8() (*Tensor[uint8], error) {
	return ToDtype[T, uint8](t, spark.U8)
}

// ToUint32 converts the tensor to uint32.
func (t *Tensor[T]) ToUint32() (*Tensor[uint32], error) {
	return ToDtype[T, uint32](t, spark.U32)
}

// ToInt64 converts the tensor to int64.
func (t *Tensor[T]) ToInt64() (*Tensor[int64], error) {
	return ToDtype[T, int64](t, spark.I64)
}

// MustToFloat32 converts the tensor to float32, panicking on error.
func (t *Tensor[T]) MustToFloat32() *Tensor[float32] {
	result, err := t.ToFloat32()
	if err != nil {
		panic(fmt.Sprintf("failed to convert to float32: %v", err))
	}
	return result
}

// MustToFloat64 converts the tensor to float64, panicking on error.
func (t *Tensor[T]) MustToFloat64() *Tensor[float64] {
	result, err := t.ToFloat64()
	if err != nil {
		panic(fmt.Sprintf("failed to convert to float64: %v", err))
	}
	return result
}

// MustToUint8 converts the tensor to uint8, panicking on error.
func (t *Tensor[T]) MustToUint8() *Tensor[uint8] {
	result, err := t.ToUint8()
	if err != nil {
		panic(fmt.Sprintf("failed to convert to uint8: %v", err))
	}
	return result
}

// MustToUint32 converts the tensor to uint32, panicking on error.
func (t *Tensor[T]) MustToUint32() *Tensor[uint32] {
	result, err := t.ToUint32()
	if err != nil {
		panic(fmt.Sprintf("failed to convert to uint32: %v", err))
	}
	return result
}

// MustToInt64 converts the tensor to int64, panicking on error.
func (t *Tensor[T]) MustToInt64() *Tensor[int64] {
	result, err := t.ToInt64()
	if err != nil {
		panic(fmt.Sprintf("failed to convert to int64: %v", err))
	}
	return result
}

// Affine performs affine transformation: y = scale * x + bias
func (t *Tensor[T]) Affine(scale, bias float64) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, AffineForward[T](scale, bias), AffineBackward[T](scale, bias))
}

// MustAffine performs affine transformation, panicking on error.
func (t *Tensor[T]) MustAffine(scale, bias float64) *Tensor[T] {
	result, err := t.Affine(scale, bias)
	if err != nil {
		panic(err)
	}
	return result
}

// Add performs element-wise addition of two tensors.
func (a *Tensor[T]) Add(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, AddForward[T](), AddBackward[T]())
}

// MustAdd performs element-wise addition of two tensors, panicking on error.
func (a *Tensor[T]) MustAdd(b *Tensor[T]) *Tensor[T] {
	t, err := a.Add(b)
	if err != nil {
		panic(err)
	}
	return t
}

// Sub performs element-wise subtraction of two tensors.
func (t *Tensor[T]) Sub(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, SubForward[T](), SubBackward[T]())
}

// MustSub performs element-wise subtraction of two tensors, panicking on error.
func (t *Tensor[T]) MustSub(other *Tensor[T]) *Tensor[T] {
	t, err := t.Sub(other)
	if err != nil {
		panic(err)
	}
	return t
}

// Mul performs element-wise multiplication of two tensors.
func (a *Tensor[T]) Mul(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, MulForward[T](), MulBackward[T]())
}

// MustMul performs element-wise multiplication of two tensors, panicking on error.
func (a *Tensor[T]) MustMul(b *Tensor[T]) *Tensor[T] {
	t, err := a.Mul(b)
	if err != nil {
		panic(err)
	}
	return t
}

// Div performs element-wise division of two tensors.
func (t *Tensor[T]) Div(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, DivForward[T](), DivBackward[T]())
}

// MustDiv performs element-wise division of two tensors, panicking on error.
func (t *Tensor[T]) MustDiv(other *Tensor[T]) *Tensor[T] {
	t, err := t.Div(other)
	if err != nil {
		panic(err)
	}
	return t
}

// Max performs element-wise maximum of two tensors.
func (a *Tensor[T]) Maximum(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, MaximumForward[T](), MaximumBackward[T]())
}

// MustMax performs element-wise maximum of two tensors, panicking on error.
func (a *Tensor[T]) MustMaximum(b *Tensor[T]) *Tensor[T] {
	result, err := a.Maximum(b)
	if err != nil {
		panic(err)
	}
	return result
}

// Min performs element-wise minimum of two tensors.
func (a *Tensor[T]) Minimum(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, MinimumForward[T](), MinimumBackward[T]())
}

// MustMin performs element-wise minimum of two tensors, panicking on error.
func (a *Tensor[T]) MustMinimum(b *Tensor[T]) *Tensor[T] {
	result, err := a.Minimum(b)
	if err != nil {
		panic(err)
	}
	return result
}

// Eq compares two tensors element-wise for equality: result = (a == b)
// Returns a uint8 tensor with 1 for equal elements and 0 for unequal elements.
func (a *Tensor[T]) Eq(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, EqForward[T](), EqBackward[T]())
}

// MustEq compares two tensors for equality, panicking on error.
func (a *Tensor[T]) MustEq(b *Tensor[T]) *Tensor[T] {
	result, err := a.Eq(b)
	if err != nil {
		panic(err)
	}
	return result
}

// Ne compares two tensors element-wise for inequality: result = (a != b)
// Returns a uint8 tensor with 1 for unequal elements and 0 for equal elements.
func (a *Tensor[T]) Ne(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, NeForward[T](), NeBackward[T]())
}

// MustNe compares two tensors for inequality, panicking on error.
func (a *Tensor[T]) MustNe(b *Tensor[T]) *Tensor[T] {
	result, err := a.Ne(b)
	if err != nil {
		panic(err)
	}
	return result
}

// Lt compares two tensors element-wise: result = (a < b)
// Returns a uint8 tensor with 1 where a < b and 0 otherwise.
func (a *Tensor[T]) Lt(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, LtForward[T](), LtBackward[T]())
}

// MustLt compares two tensors, panicking on error.
func (a *Tensor[T]) MustLt(b *Tensor[T]) *Tensor[T] {
	result, err := a.Lt(b)
	if err != nil {
		panic(err)
	}
	return result
}

// Le compares two tensors element-wise: result = (a <= b)
// Returns a uint8 tensor with 1 where a <= b and 0 otherwise.
func (a *Tensor[T]) Le(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, LeForward[T](), LeBackward[T]())
}

// MustLe compares two tensors, panicking on error.
func (a *Tensor[T]) MustLe(b *Tensor[T]) *Tensor[T] {
	result, err := a.Le(b)
	if err != nil {
		panic(err)
	}
	return result
}

// Gt compares two tensors element-wise: result = (a > b)
// Returns a uint8 tensor with 1 where a > b and 0 otherwise.
func (a *Tensor[T]) Gt(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, GtForward[T](), GtBackward[T]())
}

// MustGt compares two tensors, panicking on error.
func (a *Tensor[T]) MustGt(b *Tensor[T]) *Tensor[T] {
	result, err := a.Gt(b)
	if err != nil {
		panic(err)
	}
	return result
}

// Ge compares two tensors element-wise: result = (a >= b)
// Returns a uint8 tensor with 1 where a >= b and 0 otherwise.
func (a *Tensor[T]) Ge(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, GeForward[T](), GeBackward[T]())
}

// MustGe compares two tensors, panicking on error.
func (a *Tensor[T]) MustGe(b *Tensor[T]) *Tensor[T] {
	result, err := a.Ge(b)
	if err != nil {
		panic(err)
	}
	return result
}

// BroadcastAdd performs broadcasted addition: result = broadcast(a) + broadcast(b).
func (a *Tensor[T]) BroadcastAdd(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastAddForward[T](), BroadcastAddBackward[T]())
}

// MustBroadcastAdd performs broadcasted addition, panicking on error.
func (a *Tensor[T]) MustBroadcastAdd(b *Tensor[T]) *Tensor[T] {
	t, err := a.BroadcastAdd(b)
	if err != nil {
		panic(err)
	}
	return t
}

// BroadcastSub performs broadcasted subtraction: result = broadcast(a) - broadcast(b).
func (a *Tensor[T]) BroadcastSub(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastSubForward[T](), BroadcastSubBackward[T]())
}

// MustBroadcastSub performs broadcasted subtraction, panicking on error.
func (a *Tensor[T]) MustBroadcastSub(b *Tensor[T]) *Tensor[T] {
	t, err := a.BroadcastSub(b)
	if err != nil {
		panic(err)
	}
	return t
}

// BroadcastMul performs broadcasted multiplication: result = broadcast(a) * broadcast(b).
func (a *Tensor[T]) BroadcastMul(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastMulForward[T](), BroadcastMulBackward[T]())
}

// MustBroadcastMul performs broadcasted multiplication, panicking on error.
func (a *Tensor[T]) MustBroadcastMul(b *Tensor[T]) *Tensor[T] {
	t, err := a.BroadcastMul(b)
	if err != nil {
		panic(err)
	}
	return t
}

// BroadcastDiv performs broadcasted division: result = broadcast(a) / broadcast(b).
func (a *Tensor[T]) BroadcastDiv(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastDivForward[T](), BroadcastDivBackward[T]())
}

// MustBroadcastDiv performs broadcasted division, panicking on error.
func (a *Tensor[T]) MustBroadcastDiv(b *Tensor[T]) *Tensor[T] {
	t, err := a.BroadcastDiv(b)
	if err != nil {
		panic(err)
	}
	return t
}

// BroadcastMaximum performs element-wise maximum with broadcasting.
func (a *Tensor[T]) BroadcastMaximum(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastMaximumForward[T](), BroadcastMaximumBackward[T]())
}

// MustBroadcastMaximum performs element-wise maximum with broadcasting, panicking on error.
func (a *Tensor[T]) MustBroadcastMaximum(b *Tensor[T]) *Tensor[T] {
	result, err := a.BroadcastMaximum(b)
	if err != nil {
		panic(err)
	}
	return result
}

// BroadcastMinimum performs element-wise minimum with broadcasting.
func (a *Tensor[T]) BroadcastMinimum(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastMinimumForward[T](), BroadcastMinimumBackward[T]())
}

// MustBroadcastMinimum performs element-wise minimum with broadcasting, panicking on error.
func (a *Tensor[T]) MustBroadcastMinimum(b *Tensor[T]) *Tensor[T] {
	result, err := a.BroadcastMinimum(b)
	if err != nil {
		panic(err)
	}
	return result
}

// BroadcastEq performs broadcasted equality comparison: result = broadcast(a) == broadcast(b).
func (a *Tensor[T]) BroadcastEq(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastEqForward[T](), BroadcastEqBackward[T]())
}

// MustBroadcastEq performs broadcasted equality comparison, panicking on error.
func (a *Tensor[T]) MustBroadcastEq(b *Tensor[T]) *Tensor[T] {
	t, err := a.BroadcastEq(b)
	if err != nil {
		panic(err)
	}
	return t
}

// BroadcastNe performs broadcasted inequality comparison: result = broadcast(a) != broadcast(b).
func (a *Tensor[T]) BroadcastNe(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastNeForward[T](), BroadcastNeBackward[T]())
}

// MustBroadcastNe performs broadcasted inequality comparison, panicking on error.
func (a *Tensor[T]) MustBroadcastNe(b *Tensor[T]) *Tensor[T] {
	t, err := a.BroadcastNe(b)
	if err != nil {
		panic(err)
	}
	return t
}

// BroadcastLt performs broadcasted less-than comparison: result = broadcast(a) < broadcast(b).
func (a *Tensor[T]) BroadcastLt(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastLtForward[T](), BroadcastLtBackward[T]())
}

// MustBroadcastLt performs broadcasted less-than comparison, panicking on error.
func (a *Tensor[T]) MustBroadcastLt(b *Tensor[T]) *Tensor[T] {
	t, err := a.BroadcastLt(b)
	if err != nil {
		panic(err)
	}
	return t
}

// BroadcastLe performs broadcasted less-equal comparison: result = broadcast(a) <= broadcast(b).
func (a *Tensor[T]) BroadcastLe(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastLeForward[T](), BroadcastLeBackward[T]())
}

// MustBroadcastLe performs broadcasted less-equal comparison, panicking on error.
func (a *Tensor[T]) MustBroadcastLe(b *Tensor[T]) *Tensor[T] {
	t, err := a.BroadcastLe(b)
	if err != nil {
		panic(err)
	}
	return t
}

// BroadcastGt performs broadcasted greater-than comparison: result = broadcast(a) > broadcast(b).
func (a *Tensor[T]) BroadcastGt(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastGtForward[T](), BroadcastGtBackward[T]())
}

// MustBroadcastGt performs broadcasted greater-than comparison, panicking on error.
func (a *Tensor[T]) MustBroadcastGt(b *Tensor[T]) *Tensor[T] {
	t, err := a.BroadcastGt(b)
	if err != nil {
		panic(err)
	}
	return t
}

// BroadcastGe performs broadcasted greater-equal comparison: result = broadcast(a) >= broadcast(b).
func (a *Tensor[T]) BroadcastGe(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastGeForward[T](), BroadcastGeBackward[T]())
}

// MustBroadcastGe performs broadcasted greater-equal comparison, panicking on error.
func (a *Tensor[T]) MustBroadcastGe(b *Tensor[T]) *Tensor[T] {
	t, err := a.BroadcastGe(b)
	if err != nil {
		panic(err)
	}
	return t
}

// MatMul performs matrix multiplication: C = A × B
func (t *Tensor[T]) MatMul(rhs *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, rhs}, MatMulForward[T](), MatMulBackward[T]())
}

// MustMatMul performs matrix multiplication, panicking on error
func (t *Tensor[T]) MustMatMul(rhs *Tensor[T]) *Tensor[T] {
	result, err := t.MatMul(rhs)
	if err != nil {
		panic(err)
	}
	return result
}

// Conv1d performs 1D convolution.
func (t *Tensor[T]) Conv1d(kernel *Tensor[T], params *spark.Conv1DParams) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, kernel}, Conv1dForward[T](params), Conv1dBackward[T](params))
}

// MustConv1d performs 1D convolution, panicking on error.
func (t *Tensor[T]) MustConv1d(kernel *Tensor[T], params *spark.Conv1DParams) *Tensor[T] {
	result, err := t.Conv1d(kernel, params)
	if err != nil {
		panic(err)
	}
	return result
}

// ConvTranspose1d performs 1D transposed convolution (deconvolution).
func (t *Tensor[T]) ConvTranspose1d(kernel *Tensor[T], params *spark.ConvT1DParams) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, kernel}, ConvTranspose1dForward[T](params), ConvTranspose1dBackward[T](params))
}

// MustConvTranspose1d performs 1D transposed convolution, panicking on error.
func (t *Tensor[T]) MustConvTranspose1d(kernel *Tensor[T], params *spark.ConvT1DParams) *Tensor[T] {
	result, err := t.ConvTranspose1d(kernel, params)
	if err != nil {
		panic(err)
	}
	return result
}

// Conv2d performs 2D convolution.
func (t *Tensor[T]) Conv2d(kernel *Tensor[T], params *spark.Conv2DParams) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, kernel}, Conv2dForward[T](params), Conv2dBackward[T](params))
}

// MustConv2d performs 2D convolution, panicking on error.
func (t *Tensor[T]) MustConv2d(kernel *Tensor[T], params *spark.Conv2DParams) *Tensor[T] {
	result, err := t.Conv2d(kernel, params)
	if err != nil {
		panic(err)
	}
	return result
}

// ConvTranspose2d performs 2D transposed convolution (deconvolution).
func (t *Tensor[T]) ConvTranspose2d(kernel *Tensor[T], params *spark.ConvT2DParams) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, kernel}, ConvTranspose2dForward[T](params), ConvTranspose2dBackward[T](params))
}

// MustConvTranspose2d performs 2D transposed convolution, panicking on error.
func (t *Tensor[T]) MustConvTranspose2d(kernel *Tensor[T], params *spark.ConvT2DParams) *Tensor[T] {
	result, err := t.ConvTranspose2d(kernel, params)
	if err != nil {
		panic(err)
	}
	return result
}

// AvgPool2d performs 2D average pooling.
func (t *Tensor[T]) AvgPool2d(kH, kW, sH, sW int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, AvgPool2dForward[T](kH, kW, sH, sW), AvgPool2dBackward[T](kH, kW, sH, sW))
}

// MustAvgPool2d performs 2D average pooling, panicking on error.
func (t *Tensor[T]) MustAvgPool2d(kH, kW, sH, sW int) *Tensor[T] {
	result, err := t.AvgPool2d(kH, kW, sH, sW)
	if err != nil {
		panic(err)
	}
	return result
}

// MaxPool2d performs 2D max pooling.
func (t *Tensor[T]) MaxPool2d(kH, kW, sH, sW int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, MaxPool2dForward[T](kH, kW, sH, sW), MaxPool2dBackward[T](kH, kW, sH, sW))
}

// MustMaxPool2d performs 2D max pooling, panicking on error.
func (t *Tensor[T]) MustMaxPool2d(kH, kW, sH, sW int) *Tensor[T] {
	result, err := t.MaxPool2d(kH, kW, sH, sW)
	if err != nil {
		panic(err)
	}
	return result
}

// UpsampleNearest2d performs 2D nearest neighbor upsampling.
func (t *Tensor[T]) UpsampleNearest2d(targetH, targetW int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, UpsampleNearest2dForward[T](targetH, targetW), UpsampleNearest2dBackward[T](targetH, targetW))
}

// MustUpsampleNearest2d performs 2D nearest neighbor upsampling, panicking on error.
func (t *Tensor[T]) MustUpsampleNearest2d(targetH, targetW int) *Tensor[T] {
	result, err := t.UpsampleNearest2d(targetH, targetW)
	if err != nil {
		panic(err)
	}
	return result
}

// Gather performs gather along the specified dimension.
func (t *Tensor[T]) Gather(indexes *Tensor[T], dim int) (*Tensor[T], error) {
	return nil, nil
}

// MustGather performs gather along the specified dimension, panicking on error.
func (t *Tensor[T]) MustGather(indexes *Tensor[T], dim int) *Tensor[T] {
	result, err := t.Gather(indexes, dim)
	if err != nil {
		panic(err)
	}
	return result
}

// SumDim computes the sum along the specified dimensions.
// The dimensions to sum over are specified in dims.
// If keepdim is true, the summed dimensions are retained with size 1.
// If keepdim is false, the summed dimensions are removed.
func (t *Tensor[T]) SumDim(dims []int, keepdim bool) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SumDimForward[T](dims, keepdim), SumDimBackward[T](dims, keepdim))
}

// MustSumDim computes the sum along the specified dimensions, panicking on error.
func (t *Tensor[T]) MustSumDim(dims []int, keepdim bool) *Tensor[T] {
	t, err := t.SumDim(dims, keepdim)
	if err != nil {
		panic(err)
	}
	return t
}

// Sum computes the sum along the specified dimensions, removing the dimensions with size 1.
func (t *Tensor[T]) Sum(dims []int) (*Tensor[T], error) {
	return t.SumDim(dims, false)
}

// MustSum computes the sum along the specified dimensions, panicking on error.
func (t *Tensor[T]) MustSum(dims []int) *Tensor[T] {
	t, err := t.Sum(dims)
	if err != nil {
		panic(err)
	}
	return t
}

// SumKeepDim computes the sum along the specified dimensions, keeping the dimensions with size 1.
func (t *Tensor[T]) SumKeepDim(dims []int) (*Tensor[T], error) {
	return t.SumDim(dims, true)
}

// MustSumKeepDim computes the sum along the specified dimensions, panicking on error.
func (t *Tensor[T]) MustSumKeepDim(dims []int) *Tensor[T] {
	t, err := t.SumKeepDim(dims)
	if err != nil {
		panic(err)
	}
	return t
}

// SumAll computes the sum of all elements in the tensor.
func (t *Tensor[T]) SumAll() (*Tensor[T], error) {
	dims := make([]int, t.Rank())
	for i := range dims {
		dims[i] = i
	}
	return t.SumDim(dims, false)
}

// MustSumAll computes the sum of all elements in the tensor, panicking on error.
func (t *Tensor[T]) MustSumAll() *Tensor[T] {
	t, err := t.SumAll()
	if err != nil {
		panic(err)
	}
	return t
}

// MeanAll computes the mean of all elements in the tensor.
func (t *Tensor[T]) MeanAll() (*Tensor[T], error) {
	sum, err := t.SumAll()
	if err != nil {
		return nil, fmt.Errorf("failed to sum all elements: %w", err)
	}
	elemCount := float64(t.Shape().ElemCount())
	divisor, err := Full[T](elemCount, sum.Shape(), sum.Device())
	if err != nil {
		return nil, fmt.Errorf("failed to create divisor: %w", err)
	}
	mean, err := sum.Div(divisor)
	if err != nil {
		return nil, fmt.Errorf("failed to compute mean: %w", err)
	}
	return mean, nil
}

// MustMeanAll computes the mean of all elements in the tensor, panicking on error.
func (t *Tensor[T]) MustMeanAll() *Tensor[T] {
	result, err := t.MeanAll()
	if err != nil {
		panic(err)
	}
	return result
}

// MaxDim computes the max along the specified dimensions.
func (t *Tensor[T]) MaxDim(dims []int, keepdim bool) (*Tensor[T], error) {
	return nil, nil
}

// MustMaxDim computes the max along the specified dimensions, panicking on error.
func (t *Tensor[T]) MustMaxDim(dims []int, keepdim bool) *Tensor[T] {
	t, err := t.MaxDim(dims, keepdim)
	if err != nil {
		panic(err)
	}
	return t
}

// Max computes the max along the specified dimensions, removing the dimensions with size 1.
func (t *Tensor[T]) Max(dims []int) (*Tensor[T], error) {
	return t.MaxDim(dims, false)
}

// MustMax computes the max along the specified dimensions, panicking on error.
func (t *Tensor[T]) MustMax(dims []int) *Tensor[T] {
	t, err := t.Max(dims)
	if err != nil {
		panic(err)
	}
	return t
}

// MaxKeepDim computes the max along the specified dimensions, keeping the dimensions with size 1.
func (t *Tensor[T]) MaxKeepDim(dims []int) (*Tensor[T], error) {
	return t.MaxDim(dims, true)
}

// MustMaxKeepDim computes the max along the specified dimensions, panicking on error.
func (t *Tensor[T]) MustMaxKeepDim(dims []int) *Tensor[T] {
	t, err := t.MaxKeepDim(dims)
	if err != nil {
		panic(err)
	}
	return t
}

// MaxAll computes the max of all elements in the tensor.
func (t *Tensor[T]) MaxAll() (*Tensor[T], error) {
	dims := make([]int, t.Rank())
	for i := range dims {
		dims[i] = i
	}
	return t.MaxDim(dims, false)
}

// MustMaxAll computes the max of all elements in the tensor, panicking on error.
func (t *Tensor[T]) MaxSumAll() *Tensor[T] {
	t, err := t.MaxAll()
	if err != nil {
		panic(err)
	}
	return t
}

// FastMin computes the minimum over the last dimension.
func (t *Tensor[T]) FastMin() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, FastMinForward[T](), FastMinBackward[T]())
}

// MustFastMin computes the minimum over the last dimension, panicking on error.
func (t *Tensor[T]) MustFastMin() *Tensor[T] {
	result, err := t.FastMin()
	if err != nil {
		panic(err)
	}
	return result
}

// FastMax computes the maximum over the last dimension.
func (t *Tensor[T]) FastMax() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, FastMaxForward[T](), FastMaxBackward[T]())
}

// MustFastMax computes the maximum over the last dimension, panicking on error.
func (t *Tensor[T]) MustFastMax() *Tensor[T] {
	result, err := t.FastMax()
	if err != nil {
		panic(err)
	}
	return result
}

// FastSoftmax performs softmax activation along the last dimension.
func (t *Tensor[T]) FastSoftmax() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, FastSoftmaxForward[T](), FastSoftmaxBackward[T]())
}

// MustFastSoftmax performs softmax activation, panicking on error.
func (t *Tensor[T]) MustFastSoftmax() *Tensor[T] {
	result, err := t.FastSoftmax()
	if err != nil {
		panic(err)
	}
	return result
}

// LogSoftmax performs logsoftmax activation along the last dimension.
func (t *Tensor[T]) LogSoftmax(dim int) (*Tensor[T], error) {
	return nil, nil
}

// MustLogSoftmax performs logsoftmax activation, panicking on error.
func (t *Tensor[T]) MustLogSoftmax(dim int) *Tensor[T] {
	result, err := t.LogSoftmax(dim)
	if err != nil {
		panic(err)
	}
	return result
}

// WhereCond performs element-wise selection based on condition.
// result[i] = condition[i] != 0 ? trueVal[i] : falseVal[i]
// Supports automatic differentiation for trueVal and falseVal.
func (t *Tensor[T]) WhereCond(trueVal, falseVal *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{trueVal, falseVal}, WhereCondForward(t), WhereCondBackward(t))
}

// MustWhereCond performs conditional selection, panicking on error.
func (t *Tensor[T]) MustWhereCond(trueVal, falseVal *Tensor[T]) *Tensor[T] {
	result, err := t.WhereCond(trueVal, falseVal)
	if err != nil {
		panic(err)
	}
	return result
}

// Copy creates a copy of the tensor.
func (t *Tensor[T]) Copy() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, CopyForward[T](), CopyBackward[T]())
}

// MustCopy creates a copy of the tensor, panicking on error.
func (t *Tensor[T]) MustCopy() *Tensor[T] {
	result, err := t.Copy()
	if err != nil {
		panic(err)
	}
	return result
}

// Neg computes the negation of each element: neg(x) = -x
func (t *Tensor[T]) Neg() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, NegForward[T](), NegBackward[T]())
}

// MustNeg computes the negation of each element, panicking on error.
func (t *Tensor[T]) MustNeg() *Tensor[T] {
	result, err := t.Neg()
	if err != nil {
		panic(err)
	}
	return result
}

// Recip computes the reciprocal of each element: recip(x) = 1/x
func (t *Tensor[T]) Recip() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, RecipForward[T](), RecipBackward[T]())
}

// MustRecip computes the reciprocal of each element, panicking on error.
func (t *Tensor[T]) MustRecip() *Tensor[T] {
	result, err := t.Recip()
	if err != nil {
		panic(err)
	}
	return result
}

// Exp computes the exponential of each element: exp(x) = e^x
func (t *Tensor[T]) Exp() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, ExpForward[T](), ExpBackward[T]())
}

// MustExp computes the exponential of each element, panicking on error.
func (t *Tensor[T]) MustExp() *Tensor[T] {
	result, err := t.Exp()
	if err != nil {
		panic(err)
	}
	return result
}

// Log computes the natural logarithm of each element: log(x) = ln(x)
func (t *Tensor[T]) Log() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, LogForward[T](), LogBackward[T]())
}

// MustLog computes the natural logarithm of each element, panicking on error.
func (t *Tensor[T]) MustLog() *Tensor[T] {
	result, err := t.Log()
	if err != nil {
		panic(err)
	}
	return result
}

// Sin computes the sine of each element: sin(x)
func (t *Tensor[T]) Sin() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SinForward[T](), SinBackward[T]())
}

// MustSin computes the sine of each element, panicking on error.
func (t *Tensor[T]) MustSin() *Tensor[T] {
	result, err := t.Sin()
	if err != nil {
		panic(err)
	}
	return result
}

// Cos computes the cosine of each element: cos(x)
func (t *Tensor[T]) Cos() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, CosForward[T](), CosBackward[T]())
}

// MustCos computes the cosine of each element, panicking on error.
func (t *Tensor[T]) MustCos() *Tensor[T] {
	result, err := t.Cos()
	if err != nil {
		panic(err)
	}
	return result
}

// Tanh computes the hyperbolic tangent of each element: tanh(x)
func (t *Tensor[T]) Tanh() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, TanhForward[T](), TanhBackward[T]())
}

// MustTanh computes the hyperbolic tangent of each element, panicking on error.
func (t *Tensor[T]) MustTanh() *Tensor[T] {
	result, err := t.Tanh()
	if err != nil {
		panic(err)
	}
	return result
}

// Erf computes the error function of each element: erf(x)
func (t *Tensor[T]) Erf() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, ErfForward[T](), ErfBackward[T]())
}

// MustErf computes the error function of each element, panicking on error.
func (t *Tensor[T]) MustErf() *Tensor[T] {
	result, err := t.Erf()
	if err != nil {
		panic(err)
	}
	return result
}

// Ceil computes the ceiling of each element: ceil(x)
func (t *Tensor[T]) Ceil() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, CeilForward[T](), CeilBackward[T]())
}

// MustCeil computes the ceiling of each element, panicking on error.
func (t *Tensor[T]) MustCeil() *Tensor[T] {
	result, err := t.Ceil()
	if err != nil {
		panic(err)
	}
	return result
}

// Floor computes the floor of each element: floor(x)
func (t *Tensor[T]) Floor() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, FloorForward[T](), FloorBackward[T]())
}

// MustFloor computes the floor of each element, panicking on error.
func (t *Tensor[T]) MustFloor() *Tensor[T] {
	result, err := t.Floor()
	if err != nil {
		panic(err)
	}
	return result
}

// Round computes the round of each element: round(x)
func (t *Tensor[T]) Round() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, RoundForward[T](), RoundBackward[T]())
}

// MustRound computes the round of each element, panicking on error.
func (t *Tensor[T]) MustRound() *Tensor[T] {
	result, err := t.Round()
	if err != nil {
		panic(err)
	}
	return result
}

// Normcdf computes the normal CDF of each element: Φ(x)
func (t *Tensor[T]) Normcdf() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, NormcdfForward[T](), NormcdfBackward[T]())
}

// MustNormcdf computes the normal CDF of each element, panicking on error.
func (t *Tensor[T]) MustNormcdf() *Tensor[T] {
	result, err := t.Normcdf()
	if err != nil {
		panic(err)
	}
	return result
}

// Abs computes the absolute value of each element: abs(x) = |x|
func (t *Tensor[T]) Abs() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, AbsForward[T](), AbsBackward[T]())
}

// MustAbs computes the absolute value of each element, panicking on error.
func (t *Tensor[T]) MustAbs() *Tensor[T] {
	result, err := t.Abs()
	if err != nil {
		panic(err)
	}
	return result
}

// Sqr computes the square of each element: sqr(x) = x²
func (t *Tensor[T]) Sqr() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SqrForward[T](), SqrBackward[T]())
}

// MustSqr computes the square of each element, panicking on error.
func (t *Tensor[T]) MustSqr() *Tensor[T] {
	result, err := t.Sqr()
	if err != nil {
		panic(err)
	}
	return result
}

// Sqrt computes the square root of each element.
func (t *Tensor[T]) Sqrt() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SqrtForward[T](), SqrtBackward[T]())
}

// MustSqrt computes the square root of each element, panicking on error.
func (t *Tensor[T]) MustSqrt() *Tensor[T] {
	t, err := t.Sqrt()
	if err != nil {
		panic(err)
	}
	return t
}

// Gelu computes the GELU activation of each element: gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
func (t *Tensor[T]) Gelu() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, GeluForward[T](), GeluBackward[T]())
}

// MustGelu computes the GELU activation of each element, panicking on error.
func (t *Tensor[T]) MustGelu() *Tensor[T] {
	result, err := t.Gelu()
	if err != nil {
		panic(err)
	}
	return result
}

// GeluErf computes the ERF-based GELU activation of each element: gelu_erf(x) = 0.5 * x * (1 + erf(x/√2))
func (t *Tensor[T]) GeluErf() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, GeluErfForward[T](), GeluErfBackward[T]())
}

// MustGeluErf computes the ERF-based GELU activation of each element, panicking on error.
func (t *Tensor[T]) MustGeluErf() *Tensor[T] {
	result, err := t.GeluErf()
	if err != nil {
		panic(err)
	}
	return result
}

// Relu performs element-wise ReLU activation: relu(x) = max(0, x).
func (t *Tensor[T]) Relu() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, ReluForward[T](), ReluBackward[T]())
}

// MustRelu performs ReLU activation, panicking on error.
func (t *Tensor[T]) MustRelu() *Tensor[T] {
	result, err := t.Relu()
	if err != nil {
		panic(err)
	}
	return result
}

// Elu computes the ELU activation of each element: elu(x) = x if x >= 0, alpha * (exp(x) - 1) if x < 0
func (t *Tensor[T]) Elu(alpha float64) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, EluForward[T](alpha), EluBackward[T](alpha))
}

// MustElu computes the ELU activation of each element, panicking on error.
func (t *Tensor[T]) MustElu(alpha float64) *Tensor[T] {
	result, err := t.Elu(alpha)
	if err != nil {
		panic(err)
	}
	return result
}

// Silu computes the SiLU (Swish) activation of each element: silu(x) = x * sigmoid(x)
func (t *Tensor[T]) Silu() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SiluForward[T](), SiluBackward[T]())
}

// MustSilu computes the SiLU activation of each element, panicking on error.
func (t *Tensor[T]) MustSilu() *Tensor[T] {
	result, err := t.Silu()
	if err != nil {
		panic(err)
	}
	return result
}

// Powf computes the power of each element: powf(x) = x^param
func (t *Tensor[T]) Powf(param float64) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, PowfForward[T](param), PowfBackward[T](param))
}

// MustPowf computes the power of each element, panicking on error.
func (t *Tensor[T]) MustPowf(param float64) *Tensor[T] {
	result, err := t.Powf(param)
	if err != nil {
		panic(err)
	}
	return result
}

// Sigmoid computes the sigmoid activation: sigmoid(x) = 1/(1+e^(-x))
func (t *Tensor[T]) Sigmoid() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SigmoidForward[T](), SigmoidBackward[T]())
}

// MustSigmoid computes the sigmoid activation, panicking on error.
func (t *Tensor[T]) MustSigmoid() *Tensor[T] {
	result, err := t.Sigmoid()
	if err != nil {
		panic(err)
	}
	return result
}

// Sign computes the sign of each element: sign(x) = {1 if x>0, 0 if x=0, -1 if x<0}
func (t *Tensor[T]) Sign() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SignForward[T](), SignBackward[T]())
}

// MustSign computes the sign of each element, panicking on error.
func (t *Tensor[T]) MustSign() *Tensor[T] {
	result, err := t.Sign()
	if err != nil {
		panic(err)
	}
	return result
}

// Transpose returns a tensor that is a transposed version of the input.
func (t *Tensor[T]) Transpose(dim1, dim2 int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, TransposeForward[T](dim1, dim2), TransposeBackward[T](dim1, dim2))
}

// MustTranspose performs transpose, panicking on error.
func (t *Tensor[T]) MustTranspose(dim1, dim2 int) *Tensor[T] {
	result, err := t.Transpose(dim1, dim2)
	if err != nil {
		panic(err)
	}
	return result
}

// T is a convenient alias for transposing the last two dimensions.
// This is commonly used for matrix transpose operations.
func (t *Tensor[T]) T() (*Tensor[T], error) {
	rank := t.Rank()
	if rank < 2 {
		return nil, fmt.Errorf("tensor must have at least 2 dimensions for T(), got %d", rank)
	}
	return t.Transpose(rank-2, rank-1)
}

// MustT performs matrix transpose, panicking on error.
func (t *Tensor[T]) MustT() *Tensor[T] {
	result, err := t.T()
	if err != nil {
		panic(err)
	}
	return result
}

// BroadcastAs broadcasts the tensor to the target shape.
func (t *Tensor[T]) BroadcastAs(shape *spark.Shape) (*Tensor[T], error) {
	if t.layout.Shape().Equal(shape) {
		return t, nil
	}
	return ApplyOp([]*Tensor[T]{t}, BroadcastAsForward[T](shape), BroadcastAsBackward[T](t.layout.Shape()))
}

// MustBroadcastAs broadcasts the tensor to the target shape, panicking on error.
func (t *Tensor[T]) MustBroadcastAs(shape *spark.Shape) *Tensor[T] {
	result, err := t.BroadcastAs(shape)
	if err != nil {
		panic(err)
	}
	return result
}

// Expand broadcasts the tensor to the target shape.
func (t *Tensor[T]) Expand(shape *spark.Shape) (*Tensor[T], error) {
	return t.BroadcastAs(shape)
}

// MustExpand broadcasts the tensor to the target shape, panicking on error.
func (t *Tensor[T]) MustExpand(shape *spark.Shape) *Tensor[T] {
	t, err := t.Expand(shape)
	if err != nil {
		panic(err)
	}
	return t
}

// BroadcastLeft broadcasts the tensor by adding new dimensions on the left.
// For example, if tensor has shape [3, 4] and leftDims is [2, 5],
// the result will have shape [2, 5, 3, 4].
func (t *Tensor[T]) BroadcastLeft(leftDims ...int) (*Tensor[T], error) {
	currentDims := t.Dims()
	newDims := make([]int, 0, len(leftDims)+len(currentDims))
	newDims = append(newDims, leftDims...)
	newDims = append(newDims, currentDims...)

	targetShape := spark.NewShapeFrom(newDims)
	return t.BroadcastAs(targetShape)
}

// MustBroadcastLeft broadcasts the tensor by adding new dimensions on the left, panicking on error.
func (t *Tensor[T]) MustBroadcastLeft(leftDims ...int) *Tensor[T] {
	result, err := t.BroadcastLeft(leftDims...)
	if err != nil {
		panic(err)
	}
	return result
}

// Squeeze removes the specified dimension if its size is 1, returning a new tensor view.
// If the dimension size is not 1, returns a shallow copy of the tensor.
// Negative indices are supported (e.g., -1 for the last dimension).
func (t *Tensor[T]) Squeeze(dim int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SqueezeForward[T](dim), SqueezeBackward[T](dim))
}

// MustSqueeze removes the specified dimension if its size is 1, returning a new tensor view.
// If the dimension size is not 1, returns a shallow copy of the tensor.
// Negative indices are supported (e.g., -1 for the last dimension).
func (t *Tensor[T]) MustSqueeze(dim int) *Tensor[T] {
	t, err := t.Squeeze(dim)
	if err != nil {
		panic(err)
	}
	return t
}

// Unsqueeze inserts a dimension of size 1 at the specified position, returning a new tensor view.
// The dim can be in range [-rank-1, rank], where rank is the tensor's rank.
// Negative indices are supported (e.g., -1 to insert before the last dimension).
func (t *Tensor[T]) Unsqueeze(dim int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, UnsqueezeForward[T](dim), UnsqueezeBackward[T](dim))
}

// MustUnsqueeze inserts a dimension of size 1 at the specified position, returning a new tensor view.
// The dim can be in range [-rank-1, rank], where rank is the tensor's rank.
// Negative indices are supported (e.g., -1 to insert before the last dimension).
func (t *Tensor[T]) MustUnsqueeze(dim int) *Tensor[T] {
	t, err := t.Unsqueeze(dim)
	if err != nil {
		panic(err)
	}
	return t
}

func (t *Tensor[T]) SqueezeDims(dims []int) (*Tensor[T], error) {
	if len(dims) == 0 {
		return t, nil
	}

	if len(dims) == 1 {
		return t.Squeeze(dims[0])
	}

	dimSet := make(map[int]bool)
	for _, dim := range dims {
		resolved, err := spark.ResolveAxis(dim, t.Rank())
		if err != nil {
			return nil, err
		}
		dimSet[resolved] = true
	}

	currentDims := t.Dims()
	var newDims []int
	for i, size := range currentDims {
		if !dimSet[i] {
			newDims = append(newDims, size)
		}
	}

	return t.Reshape(newDims...)
}

// Reshape changes the tensor's shape while preserving the total number of elements.
func (t *Tensor[T]) Reshape(dims ...int) (*Tensor[T], error) {
	newShape := spark.NewShapeFrom(dims)
	return ApplyOp([]*Tensor[T]{t}, ReshapeForward[T](newShape), ReshapeBackward[T](t.layout.Shape()))
}

// MustReshape changes the tensor's shape, panicking on error.
func (t *Tensor[T]) MustReshape(dims ...int) *Tensor[T] {
	result, err := t.Reshape(dims...)
	if err != nil {
		panic(err)
	}
	return result
}

// FlattenAll flattens the tensor into a one-dimensional tensor.
func (t *Tensor[T]) FlattenAll() (*Tensor[T], error) {
	totalElements := t.layout.Shape().ElemCount()
	return t.Reshape(totalElements)
}

// MustFlattenAll flattens the tensor into 1D, panicking on error.
func (t *Tensor[T]) MustFlattenAll() *Tensor[T] {
	result, err := t.FlattenAll()
	if err != nil {
		panic(err)
	}
	return result
}

// Flatten flattens dimensions from startDim to endDim (inclusive).
func (t *Tensor[T]) Flatten(startDim, endDim int) (*Tensor[T], error) {
	shape := t.layout.Shape()
	rank := shape.Rank()

	resolvedStartDim, err := spark.ResolveAxis(startDim, rank)
	if err != nil {
		return nil, fmt.Errorf("flatten start_dim: %w", err)
	}
	resolvedEndDim, err := spark.ResolveAxis(endDim, rank)
	if err != nil {
		return nil, fmt.Errorf("flatten end_dim: %w", err)
	}

	if resolvedStartDim > resolvedEndDim {
		return nil, fmt.Errorf("flatten: start_dim %d must be <= end_dim %d", startDim, endDim)
	}
	if resolvedStartDim == resolvedEndDim {
		return NewFrom(t.storage, t.layout.Clone(), t.dtype, t.device), nil
	}

	dims := shape.Dims()
	var newDims []int

	// Add dimensions before startDim
	newDims = append(newDims, dims[:resolvedStartDim]...)

	// Calculate flattened dimension size
	flattenedSize := 1
	for i := resolvedStartDim; i <= resolvedEndDim; i++ {
		flattenedSize *= dims[i]
	}
	newDims = append(newDims, flattenedSize)

	// Add dimensions after endDim
	if resolvedEndDim+1 < len(dims) {
		newDims = append(newDims, dims[resolvedEndDim+1:]...)
	}

	return t.Reshape(newDims...)
}

// MustFlatten flattens dimensions, panicking on error.
func (t *Tensor[T]) MustFlatten(startDim, endDim int) *Tensor[T] {
	result, err := t.Flatten(startDim, endDim)
	if err != nil {
		panic(err)
	}
	return result
}

// FlattenFrom flattens from startDim to the last dimension.
func (t *Tensor[T]) FlattenFrom(startDim int) (*Tensor[T], error) {
	return t.Flatten(startDim, t.layout.Shape().Rank()-1)
}

// MustFlattenFrom flattens from startDim, panicking on error.
func (t *Tensor[T]) MustFlattenFrom(startDim int) *Tensor[T] {
	result, err := t.FlattenFrom(startDim)
	if err != nil {
		panic(err)
	}
	return result
}

// FlattenTo flattens from dimension 0 to endDim.
func (t *Tensor[T]) FlattenTo(endDim int) (*Tensor[T], error) {
	return t.Flatten(0, endDim)
}

// MustFlattenTo flattens to endDim, panicking on error.
func (t *Tensor[T]) MustFlattenTo(endDim int) *Tensor[T] {
	result, err := t.FlattenTo(endDim)
	if err != nil {
		panic(err)
	}
	return result
}

// Backward computes gradients for all variable tensors contributing to the root tensor.
// Returns the gradient store containing all computed gradients.
func (t *Tensor[T]) Backward() (*GradStore[T], error) {
	store := NewGradStore[T]()
	if err := Backward(t, store); err != nil {
		return nil, err
	}
	return store, nil
}

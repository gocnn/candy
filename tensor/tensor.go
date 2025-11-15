package tensor

import (
	"fmt"
	"sync/atomic"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/tensor/internal/cpu"
)

// counter provides atomic increment for unique IDs.
var counter uint64

// TensorID uniquely identifies a tensor.
type TensorID uint64

// NewID generates a unique TensorID safely.
func NewID() TensorID {
	return TensorID(atomic.AddUint64(&counter, 1))
}

// Tensor is a multi-dimensional array supporting autograd.
type Tensor[T candy.D] struct {
	id      TensorID
	storage candy.BackendStorage[T]
	layout  *candy.Layout
	op      *Op[T]
	isVar   bool
	dtype   candy.DType
	device  candy.Device
}

// NewFrom creates a tensor from storage and layout.
func NewFrom[T candy.D](storage candy.BackendStorage[T], layout *candy.Layout, dtype candy.DType, dev candy.Device) *Tensor[T] {
	return &Tensor[T]{
		id:      NewID(),
		storage: storage,
		layout:  layout.Clone(),
		dtype:   dtype,
		device:  dev,
	}
}

// New creates a tensor from data and shape on device.
func New[T candy.D](data []T, shape *candy.Shape, dev candy.Device) (*Tensor[T], error) {
	var storage candy.BackendStorage[T]
	switch dev {
	case candy.CPU:
		storage = cpu.New(data)
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, candy.Contiguous(shape), candy.DTypeOf[T](), dev), nil
}

// Full creates a tensor filled with value.
func Full[T candy.D](value float64, shape *candy.Shape, dev candy.Device) (*Tensor[T], error) {
	var storage candy.BackendStorage[T]
	switch dev {
	case candy.CPU:
		var err error
		storage, err = cpu.NewCpuDevice[T]().Full(shape, candy.DTypeOf[T](), value)
		if err != nil {
			return nil, fmt.Errorf("create full failed: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, candy.Contiguous(shape), candy.DTypeOf[T](), dev), nil
}

// Ones creates a tensor filled with ones.
func Ones[T candy.D](shape *candy.Shape, dev candy.Device) (*Tensor[T], error) {
	var storage candy.BackendStorage[T]
	switch dev {
	case candy.CPU:
		var err error
		storage, err = cpu.NewCpuDevice[T]().Ones(shape, candy.DTypeOf[T]())
		if err != nil {
			return nil, fmt.Errorf("create ones failed: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, candy.Contiguous(shape), candy.DTypeOf[T](), dev), nil
}

// Zeros creates a tensor filled with zeros.
func Zeros[T candy.D](shape *candy.Shape, dev candy.Device) (*Tensor[T], error) {
	var storage candy.BackendStorage[T]
	switch dev {
	case candy.CPU:
		var err error
		storage, err = cpu.NewCpuDevice[T]().Zeros(shape, candy.DTypeOf[T]())
		if err != nil {
			return nil, fmt.Errorf("create zeros failed: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, candy.Contiguous(shape), candy.DTypeOf[T](), dev), nil
}

// Rand creates a tensor with uniform samples in [lo, up).
func Rand[T candy.D](lo, up float64, shape *candy.Shape, dev candy.Device) (*Tensor[T], error) {
	var storage candy.BackendStorage[T]
	switch dev {
	case candy.CPU:
		var err error
		storage, err = cpu.NewCpuDevice[T]().RandUniform(shape, candy.DTypeOf[T](), lo, up)
		if err != nil {
			return nil, fmt.Errorf("create rand failed: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, candy.Contiguous(shape), candy.DTypeOf[T](), dev), nil
}

// RandN creates a tensor with normal distribution samples.
func RandN[T candy.D](mean, std float64, shape *candy.Shape, dev candy.Device) (*Tensor[T], error) {
	var storage candy.BackendStorage[T]
	switch dev {
	case candy.CPU:
		var err error
		storage, err = cpu.NewCpuDevice[T]().RandNormal(shape, candy.DTypeOf[T](), mean, std)
		if err != nil {
			return nil, fmt.Errorf("create randn failed: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported device: %v", dev)
	}
	return NewFrom(storage, candy.Contiguous(shape), candy.DTypeOf[T](), dev), nil
}

// MustNew creates tensor from data and shape, panics on error.
func MustNew[T candy.D](data []T, shape *candy.Shape, dev candy.Device) *Tensor[T] {
	res, err := New(data, shape, dev)
	if err != nil {
		panic(err)
	}
	return res
}

// MustFull creates tensor filled with value, panics on error.
func MustFull[T candy.D](value float64, shape *candy.Shape, dev candy.Device) *Tensor[T] {
	res, err := Full[T](value, shape, dev)
	if err != nil {
		panic(err)
	}
	return res
}

// MustOnes creates tensor filled with ones, panics on error.
func MustOnes[T candy.D](shape *candy.Shape, dev candy.Device) *Tensor[T] {
	res, err := Ones[T](shape, dev)
	if err != nil {
		panic(err)
	}
	return res
}

// MustZeros creates tensor filled with zeros, panics on error.
func MustZeros[T candy.D](shape *candy.Shape, dev candy.Device) *Tensor[T] {
	res, err := Zeros[T](shape, dev)
	if err != nil {
		panic(err)
	}
	return res
}

// MustRand creates tensor with uniform samples, panics on error.
func MustRand[T candy.D](lo, up float64, shape *candy.Shape, dev candy.Device) *Tensor[T] {
	res, err := Rand[T](lo, up, shape, dev)
	if err != nil {
		panic(err)
	}
	return res
}

// MustRandN creates tensor with normal samples, panics on error.
func MustRandN[T candy.D](mean, std float64, shape *candy.Shape, dev candy.Device) *Tensor[T] {
	res, err := RandN[T](mean, std, shape, dev)
	if err != nil {
		panic(err)
	}
	return res
}

// FullLike creates like t, filled with value.
func (t *Tensor[T]) FullLike(value float64) (*Tensor[T], error) {
	return Full[T](value, t.Shape(), t.device)
}

// OnesLike creates like t, filled with ones.
func (t *Tensor[T]) OnesLike() (*Tensor[T], error) {
	return Ones[T](t.Shape(), t.device)
}

// ZerosLike creates like t, filled with zeros.
func (t *Tensor[T]) ZerosLike() (*Tensor[T], error) {
	return Zeros[T](t.Shape(), t.device)
}

// RandLike creates like t, with uniform samples.
func (t *Tensor[T]) RandLike(lo, up float64) (*Tensor[T], error) {
	return Rand[T](lo, up, t.Shape(), t.device)
}

// RandNLike creates like t, with normal samples.
func (t *Tensor[T]) RandNLike(mean, std float64) (*Tensor[T], error) {
	return RandN[T](mean, std, t.Shape(), t.device)
}

// MustFullLike creates like t filled with value, panics on error.
func (t *Tensor[T]) MustFullLike(value float64) *Tensor[T] {
	res, err := t.FullLike(value)
	if err != nil {
		panic(err)
	}
	return res
}

// MustOnesLike creates like t filled with ones, panics on error.
func (t *Tensor[T]) MustOnesLike() *Tensor[T] {
	res, err := t.OnesLike()
	if err != nil {
		panic(err)
	}
	return res
}

// MustZerosLike creates like t filled with zeros, panics on error.
func (t *Tensor[T]) MustZerosLike() *Tensor[T] {
	res, err := t.ZerosLike()
	if err != nil {
		panic(err)
	}
	return res
}

// MustRandLike creates like t with uniform samples, panics on error.
func (t *Tensor[T]) MustRandLike(lo, up float64) *Tensor[T] {
	res, err := t.RandLike(lo, up)
	if err != nil {
		panic(err)
	}
	return res
}

// MustRandNLike creates like t with normal samples, panics on error.
func (t *Tensor[T]) MustRandNLike(mean, std float64) *Tensor[T] {
	res, err := t.RandNLike(mean, std)
	if err != nil {
		panic(err)
	}
	return res
}

// ID returns the unique identifier.
func (t *Tensor[T]) ID() TensorID {
	return t.id
}

// Storage returns the backend storage.
func (t *Tensor[T]) Storage() candy.BackendStorage[T] {
	return t.storage
}

// Layout returns a cloned layout.
func (t *Tensor[T]) Layout() *candy.Layout {
	return t.layout.Clone()
}

// Op returns the creating operation.
func (t *Tensor[T]) Op() *Op[T] {
	return t.op
}

// IsVar checks if gradient is required.
func (t *Tensor[T]) IsVar() bool {
	return t.isVar
}

// DType returns the data type.
func (t *Tensor[T]) DType() candy.DType {
	return t.dtype
}

// Device returns the device.
func (t *Tensor[T]) Device() candy.Device {
	return t.device
}

// Data returns CPU data slice, panics if not CPU.
func (t *Tensor[T]) Data() []T {
	return t.storage.Data()
}

// Stride returns strides.
func (t *Tensor[T]) Stride() []int {
	return t.layout.Stride()
}

// Shape returns shape.
func (t *Tensor[T]) Shape() *candy.Shape {
	return t.layout.Shape()
}

// Dims returns dimensions.
func (t *Tensor[T]) Dims() []int {
	return t.layout.Dims()
}

// Dim returns size of dimension.
func (t *Tensor[T]) Dim(dim int) int {
	return t.layout.Dim(dim)
}

// Dims0 validates scalar (0 dims).
func (t *Tensor[T]) Dims0() error {
	return t.layout.Dims0()
}

// Dims1 extracts 1D dimension.
func (t *Tensor[T]) Dims1() (int, error) {
	return t.layout.Dims1()
}

// Dims2 extracts 2D dimensions.
func (t *Tensor[T]) Dims2() (int, int, error) {
	return t.layout.Dims2()
}

// Dims3 extracts 3D dimensions.
func (t *Tensor[T]) Dims3() (int, int, int, error) {
	return t.layout.Dims3()
}

// Dims4 extracts 4D dimensions.
func (t *Tensor[T]) Dims4() (int, int, int, int, error) {
	return t.layout.Dims4()
}

// Dims5 extracts 5D dimensions.
func (t *Tensor[T]) Dims5() (int, int, int, int, int, error) {
	return t.layout.Dims5()
}

// Rank returns number of dimensions.
func (t *Tensor[T]) Rank() int {
	return t.layout.Rank()
}

// Numel returns element count.
func (t *Tensor[T]) Numel() int {
	return t.layout.Numel()
}

// SetStorage sets storage, returns self.
func (t *Tensor[T]) SetStorage(s candy.BackendStorage[T]) *Tensor[T] {
	t.storage = s
	return t
}

// SetLayout sets layout, returns self.
func (t *Tensor[T]) SetLayout(l *candy.Layout) *Tensor[T] {
	t.layout = l.Clone()
	return t
}

// SetOp sets operation, returns self.
func (t *Tensor[T]) SetOp(o *Op[T]) *Tensor[T] {
	t.op = o
	return t
}

// SetIsVar sets gradient flag, returns self.
func (t *Tensor[T]) SetIsVar(v bool) *Tensor[T] {
	t.isVar = v
	return t
}

// SetDType sets data type, returns self.
func (t *Tensor[T]) SetDType(d candy.DType) *Tensor[T] {
	t.dtype = d
	return t
}

// SetDevice sets device, returns self.
func (t *Tensor[T]) SetDevice(d candy.Device) *Tensor[T] {
	t.device = d
	return t
}

// RequiresGrad marks for gradient, returns self.
func (t *Tensor[T]) RequiresGrad() *Tensor[T] {
	t.isVar = true
	return t
}

// Detach detaches from graph.
func (t *Tensor[T]) Detach() *Tensor[T] {
	return NewFrom(t.storage, t.layout.Clone(), t.dtype, t.device)
}

// Clone clones the tensor.
func (t *Tensor[T]) Clone() (*Tensor[T], error) {
	storage, err := t.storage.Clone()
	if err != nil {
		return nil, err
	}
	return &Tensor[T]{
		id:      NewID(),
		storage: storage,
		layout:  t.layout.Clone(),
		op:      t.op,
		isVar:   t.isVar,
		dtype:   t.dtype,
		device:  t.device,
	}, nil
}

// MustClone clones, panics on error.
func (t *Tensor[T]) MustClone() *Tensor[T] {
	res, err := t.Clone()
	if err != nil {
		panic(err)
	}
	return res
}

// ToDtype converts to new dtype.
func ToDtype[T, U candy.D](t *Tensor[T], dtype candy.DType) (*Tensor[U], error) {
	s, err := t.storage.ToDtype(t.layout, dtype)
	if err != nil {
		return nil, fmt.Errorf("convert to %v failed: %w", dtype, err)
	}
	return NewFrom(s.(candy.BackendStorage[U]), t.layout.Clone(), dtype, t.device), nil
}

// ToFloat32 converts to float32.
func (t *Tensor[T]) ToFloat32() (*Tensor[float32], error) {
	return ToDtype[T, float32](t, candy.F32)
}

// ToFloat64 converts to float64.
func (t *Tensor[T]) ToFloat64() (*Tensor[float64], error) {
	return ToDtype[T, float64](t, candy.F64)
}

// ToUint8 converts to uint8.
func (t *Tensor[T]) ToUint8() (*Tensor[uint8], error) {
	return ToDtype[T, uint8](t, candy.U8)
}

// ToUint32 converts to uint32.
func (t *Tensor[T]) ToUint32() (*Tensor[uint32], error) {
	return ToDtype[T, uint32](t, candy.U32)
}

// ToInt64 converts to int64.
func (t *Tensor[T]) ToInt64() (*Tensor[int64], error) {
	return ToDtype[T, int64](t, candy.I64)
}

// MustToFloat32 converts to float32, panics on error.
func (t *Tensor[T]) MustToFloat32() *Tensor[float32] {
	res, err := t.ToFloat32()
	if err != nil {
		panic(fmt.Sprintf("to float32 failed: %v", err))
	}
	return res
}

// MustToFloat64 converts to float64, panics on error.
func (t *Tensor[T]) MustToFloat64() *Tensor[float64] {
	res, err := t.ToFloat64()
	if err != nil {
		panic(fmt.Sprintf("to float64 failed: %v", err))
	}
	return res
}

// MustToUint8 converts to uint8, panics on error.
func (t *Tensor[T]) MustToUint8() *Tensor[uint8] {
	res, err := t.ToUint8()
	if err != nil {
		panic(fmt.Sprintf("to uint8 failed: %v", err))
	}
	return res
}

// MustToUint32 converts to uint32, panics on error.
func (t *Tensor[T]) MustToUint32() *Tensor[uint32] {
	res, err := t.ToUint32()
	if err != nil {
		panic(fmt.Sprintf("to uint32 failed: %v", err))
	}
	return res
}

// MustToInt64 converts to int64, panics on error.
func (t *Tensor[T]) MustToInt64() *Tensor[int64] {
	res, err := t.ToInt64()
	if err != nil {
		panic(fmt.Sprintf("to int64 failed: %v", err))
	}
	return res
}

// Affine applies y = scale * x + bias.
func (t *Tensor[T]) Affine(scale, bias float64) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, AffineForward[T](scale, bias), AffineBackward[T](scale, bias))
}

// MustAffine applies affine, panics on error.
func (t *Tensor[T]) MustAffine(scale, bias float64) *Tensor[T] {
	res, err := t.Affine(scale, bias)
	if err != nil {
		panic(err)
	}
	return res
}

// AddScalar adds scalar using affine.
func (t *Tensor[T]) AddScalar(s float64) (*Tensor[T], error) {
	return t.Affine(1, s)
}

// MustAddScalar adds scalar, panics on error.
func (t *Tensor[T]) MustAddScalar(s float64) *Tensor[T] {
	res, err := t.AddScalar(s)
	if err != nil {
		panic(err)
	}
	return res
}

// SubScalar subtracts scalar using affine.
func (t *Tensor[T]) SubScalar(s float64) (*Tensor[T], error) {
	return t.Affine(1, -s)
}

// MustSubScalar subtracts scalar, panics on error.
func (t *Tensor[T]) MustSubScalar(s float64) *Tensor[T] {
	res, err := t.SubScalar(s)
	if err != nil {
		panic(err)
	}
	return res
}

// MulScalar multiplies by scalar using affine.
func (t *Tensor[T]) MulScalar(s float64) (*Tensor[T], error) {
	return t.Affine(s, 0)
}

// MustMulScalar multiplies by scalar, panics on error.
func (t *Tensor[T]) MustMulScalar(s float64) *Tensor[T] {
	res, err := t.MulScalar(s)
	if err != nil {
		panic(err)
	}
	return res
}

// DivScalar divides by scalar using affine.
func (t *Tensor[T]) DivScalar(s float64) (*Tensor[T], error) {
	if s == 0 {
		return nil, fmt.Errorf("division by zero")
	}
	return t.Affine(1/s, 0)
}

// MustDivScalar divides by scalar, panics on error.
func (t *Tensor[T]) MustDivScalar(s float64) *Tensor[T] {
	res, err := t.DivScalar(s)
	if err != nil {
		panic(err)
	}
	return res
}

// Add adds element-wise.
func (t *Tensor[T]) Add(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, AddForward[T](), AddBackward[T]())
}

// MustAdd adds element-wise, panics on error.
func (t *Tensor[T]) MustAdd(other *Tensor[T]) *Tensor[T] {
	res, err := t.Add(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Sub subtracts element-wise.
func (t *Tensor[T]) Sub(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, SubForward[T](), SubBackward[T]())
}

// MustSub subtracts element-wise, panics on error.
func (t *Tensor[T]) MustSub(other *Tensor[T]) *Tensor[T] {
	res, err := t.Sub(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Mul multiplies element-wise.
func (t *Tensor[T]) Mul(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, MulForward[T](), MulBackward[T]())
}

// MustMul multiplies element-wise, panics on error.
func (t *Tensor[T]) MustMul(other *Tensor[T]) *Tensor[T] {
	res, err := t.Mul(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Div divides element-wise.
func (t *Tensor[T]) Div(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, DivForward[T](), DivBackward[T]())
}

// MustDiv divides element-wise, panics on error.
func (t *Tensor[T]) MustDiv(other *Tensor[T]) *Tensor[T] {
	res, err := t.Div(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Maximum takes element-wise max.
func (t *Tensor[T]) Maximum(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, MaximumForward[T](), MaximumBackward[T]())
}

// MustMaximum takes element-wise max, panics on error.
func (t *Tensor[T]) MustMaximum(other *Tensor[T]) *Tensor[T] {
	res, err := t.Maximum(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Minimum takes element-wise min.
func (t *Tensor[T]) Minimum(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, MinimumForward[T](), MinimumBackward[T]())
}

// MustMinimum takes element-wise min, panics on error.
func (t *Tensor[T]) MustMinimum(other *Tensor[T]) *Tensor[T] {
	res, err := t.Minimum(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Eq compares equality element-wise.
func (t *Tensor[T]) Eq(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, EqForward[T](), EqBackward[T]())
}

// MustEq compares equality, panics on error.
func (t *Tensor[T]) MustEq(other *Tensor[T]) *Tensor[T] {
	res, err := t.Eq(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Ne compares inequality element-wise.
func (t *Tensor[T]) Ne(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, NeForward[T](), NeBackward[T]())
}

// MustNe compares inequality, panics on error.
func (t *Tensor[T]) MustNe(other *Tensor[T]) *Tensor[T] {
	res, err := t.Ne(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Lt compares less-than element-wise.
func (t *Tensor[T]) Lt(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, LtForward[T](), LtBackward[T]())
}

// MustLt compares less-than, panics on error.
func (t *Tensor[T]) MustLt(other *Tensor[T]) *Tensor[T] {
	res, err := t.Lt(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Le compares less-equal element-wise.
func (t *Tensor[T]) Le(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, LeForward[T](), LeBackward[T]())
}

// MustLe compares less-equal, panics on error.
func (t *Tensor[T]) MustLe(other *Tensor[T]) *Tensor[T] {
	res, err := t.Le(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Gt compares greater-than element-wise.
func (t *Tensor[T]) Gt(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, GtForward[T](), GtBackward[T]())
}

// MustGt compares greater-than, panics on error.
func (t *Tensor[T]) MustGt(other *Tensor[T]) *Tensor[T] {
	res, err := t.Gt(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Ge compares greater-equal element-wise.
func (t *Tensor[T]) Ge(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, GeForward[T](), GeBackward[T]())
}

// MustGe compares greater-equal, panics on error.
func (t *Tensor[T]) MustGe(other *Tensor[T]) *Tensor[T] {
	res, err := t.Ge(other)
	if err != nil {
		panic(err)
	}
	return res
}

// Clamp clamps values between tensor bounds.
func (t *Tensor[T]) Clamp(minT, maxT *Tensor[T]) (*Tensor[T], error) {
	r, err := t.Maximum(minT)
	if err != nil {
		return nil, err
	}
	return r.Minimum(maxT)
}

// MustClamp clamps between tensor bounds, panics on error.
func (t *Tensor[T]) MustClamp(minT, maxT *Tensor[T]) *Tensor[T] {
	res, err := t.Clamp(minT, maxT)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastAdd adds with broadcast.
func (t *Tensor[T]) BroadcastAdd(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastAddForward[T](), BroadcastAddBackward[T]())
}

// MustBroadcastAdd adds with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastAdd(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastAdd(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastSub subtracts with broadcast.
func (t *Tensor[T]) BroadcastSub(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastSubForward[T](), BroadcastSubBackward[T]())
}

// MustBroadcastSub subtracts with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastSub(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastSub(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastMul multiplies with broadcast.
func (t *Tensor[T]) BroadcastMul(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastMulForward[T](), BroadcastMulBackward[T]())
}

// MustBroadcastMul multiplies with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastMul(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastMul(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastDiv divides with broadcast.
func (t *Tensor[T]) BroadcastDiv(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastDivForward[T](), BroadcastDivBackward[T]())
}

// MustBroadcastDiv divides with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastDiv(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastDiv(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastMaximum takes max with broadcast.
func (t *Tensor[T]) BroadcastMaximum(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastMaximumForward[T](), BroadcastMaximumBackward[T]())
}

// MustBroadcastMaximum takes max with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastMaximum(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastMaximum(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastMinimum takes min with broadcast.
func (t *Tensor[T]) BroadcastMinimum(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastMinimumForward[T](), BroadcastMinimumBackward[T]())
}

// MustBroadcastMinimum takes min with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastMinimum(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastMinimum(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastEq equals with broadcast.
func (t *Tensor[T]) BroadcastEq(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastEqForward[T](), BroadcastEqBackward[T]())
}

// MustBroadcastEq equals with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastEq(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastEq(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastNe not equals with broadcast.
func (t *Tensor[T]) BroadcastNe(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastNeForward[T](), BroadcastNeBackward[T]())
}

// MustBroadcastNe not equals with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastNe(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastNe(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastLt less-than with broadcast.
func (t *Tensor[T]) BroadcastLt(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastLtForward[T](), BroadcastLtBackward[T]())
}

// MustBroadcastLt less-than with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastLt(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastLt(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastLe less-equal with broadcast.
func (t *Tensor[T]) BroadcastLe(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastLeForward[T](), BroadcastLeBackward[T]())
}

// MustBroadcastLe less-equal with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastLe(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastLe(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastGt greater-than with broadcast.
func (t *Tensor[T]) BroadcastGt(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastGtForward[T](), BroadcastGtBackward[T]())
}

// MustBroadcastGt greater-than with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastGt(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastGt(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastGe greater-equal with broadcast.
func (t *Tensor[T]) BroadcastGe(other *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, other}, BroadcastGeForward[T](), BroadcastGeBackward[T]())
}

// MustBroadcastGe greater-equal with broadcast, panics on error.
func (t *Tensor[T]) MustBroadcastGe(other *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastGe(other)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastClamp clamps values between broadcastable tensor bounds.
func (t *Tensor[T]) BroadcastClamp(minT, maxT *Tensor[T]) (*Tensor[T], error) {
	r, err := t.BroadcastMaximum(minT)
	if err != nil {
		return nil, err
	}
	return r.BroadcastMinimum(maxT)
}

// MustBroadcastClamp clamps between broadcastable bounds, panics on error.
func (t *Tensor[T]) MustBroadcastClamp(minT, maxT *Tensor[T]) *Tensor[T] {
	res, err := t.BroadcastClamp(minT, maxT)
	if err != nil {
		panic(err)
	}
	return res
}

// MatMul multiplies matrices.
func (t *Tensor[T]) MatMul(rhs *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, rhs}, MatMulForward[T](), MatMulBackward[T]())
}

// MustMatMul multiplies matrices, panics on error.
func (t *Tensor[T]) MustMatMul(rhs *Tensor[T]) *Tensor[T] {
	res, err := t.MatMul(rhs)
	if err != nil {
		panic(err)
	}
	return res
}

// Conv1d applies 1D convolution.
func (t *Tensor[T]) Conv1d(kernel *Tensor[T], params *candy.Conv1DParams) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, kernel}, Conv1dForward[T](params), Conv1dBackward[T](params))
}

// MustConv1d applies 1D convolution, panics on error.
func (t *Tensor[T]) MustConv1d(kernel *Tensor[T], params *candy.Conv1DParams) *Tensor[T] {
	res, err := t.Conv1d(kernel, params)
	if err != nil {
		panic(err)
	}
	return res
}

// ConvTranspose1d applies 1D transposed convolution.
func (t *Tensor[T]) ConvTranspose1d(kernel *Tensor[T], params *candy.ConvT1DParams) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, kernel}, ConvTranspose1dForward[T](params), ConvTranspose1dBackward[T](params))
}

// MustConvTranspose1d applies 1D transposed convolution, panics on error.
func (t *Tensor[T]) MustConvTranspose1d(kernel *Tensor[T], params *candy.ConvT1DParams) *Tensor[T] {
	res, err := t.ConvTranspose1d(kernel, params)
	if err != nil {
		panic(err)
	}
	return res
}

// Conv2d applies 2D convolution.
func (t *Tensor[T]) Conv2d(kernel *Tensor[T], params *candy.Conv2DParams) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, kernel}, Conv2dForward[T](params), Conv2dBackward[T](params))
}

// MustConv2d applies 2D convolution, panics on error.
func (t *Tensor[T]) MustConv2d(kernel *Tensor[T], params *candy.Conv2DParams) *Tensor[T] {
	res, err := t.Conv2d(kernel, params)
	if err != nil {
		panic(err)
	}
	return res
}

// ConvTranspose2d applies 2D transposed convolution.
func (t *Tensor[T]) ConvTranspose2d(kernel *Tensor[T], params *candy.ConvT2DParams) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, kernel}, ConvTranspose2dForward[T](params), ConvTranspose2dBackward[T](params))
}

// MustConvTranspose2d applies 2D transposed convolution, panics on error.
func (t *Tensor[T]) MustConvTranspose2d(kernel *Tensor[T], params *candy.ConvT2DParams) *Tensor[T] {
	res, err := t.ConvTranspose2d(kernel, params)
	if err != nil {
		panic(err)
	}
	return res
}

// AvgPool2d applies 2D average pooling.
func (t *Tensor[T]) AvgPool2d(kH, kW, sH, sW int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, AvgPool2dForward[T](kH, kW, sH, sW), AvgPool2dBackward[T](kH, kW, sH, sW))
}

// MustAvgPool2d applies 2D average pooling, panics on error.
func (t *Tensor[T]) MustAvgPool2d(kH, kW, sH, sW int) *Tensor[T] {
	res, err := t.AvgPool2d(kH, kW, sH, sW)
	if err != nil {
		panic(err)
	}
	return res
}

// AdaptiveAvgPool2d computes kernel and stride to produce exact outH×outW output, then calls AvgPool2d.
func (t *Tensor[T]) AdaptiveAvgPool2d(outH, outW int) (*Tensor[T], error) {
	if len(t.Dims()) != 4 {
		return nil, fmt.Errorf("adaptive avgpool2d: expected 4D input, got %dD", len(t.Dims()))
	}
	inH, inW := t.Dim(2), t.Dim(3)
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("adaptive avgpool2d: invalid output size %dx%d", outH, outW)
	}
	if inH < outH || inW < outW {
		return nil, fmt.Errorf("adaptive avgpool2d: input smaller than output: %dx%d -> %dx%d", inH, inW, outH, outW)
	}
	sH := inH / outH
	sW := inW / outW
	if sH <= 0 || sW <= 0 {
		return nil, fmt.Errorf("adaptive avgpool2d: computed stride <= 0")
	}
	kH := inH - (outH-1)*sH
	kW := inW - (outW-1)*sW
	if kH <= 0 || kW <= 0 {
		return nil, fmt.Errorf("adaptive avgpool2d: computed kernel <= 0")
	}
	return t.AvgPool2d(kH, kW, sH, sW)
}

// MustAdaptiveAvgPool2d applies AdaptiveAvgPool2d and panics on error.
func (t *Tensor[T]) MustAdaptiveAvgPool2d(outH, outW int) *Tensor[T] {
	res, err := t.AdaptiveAvgPool2d(outH, outW)
	if err != nil {
		panic(err)
	}
	return res
}

// MaxPool2d applies 2D max pooling.
func (t *Tensor[T]) MaxPool2d(kH, kW, sH, sW int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, MaxPool2dForward[T](kH, kW, sH, sW), MaxPool2dBackward[T](kH, kW, sH, sW))
}

// MustMaxPool2d applies 2D max pooling, panics on error.
func (t *Tensor[T]) MustMaxPool2d(kH, kW, sH, sW int) *Tensor[T] {
	res, err := t.MaxPool2d(kH, kW, sH, sW)
	if err != nil {
		panic(err)
	}
	return res
}

// UpsampleNearest2d upsamples 2D with nearest neighbor.
func (t *Tensor[T]) UpsampleNearest2d(h, w int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, UpsampleNearest2dForward[T](h, w), UpsampleNearest2dBackward[T](h, w))
}

// MustUpsampleNearest2d upsamples 2D, panics on error.
func (t *Tensor[T]) MustUpsampleNearest2d(h, w int) *Tensor[T] {
	res, err := t.UpsampleNearest2d(h, w)
	if err != nil {
		panic(err)
	}
	return res
}

// Gather gathers along dimension.
func (t *Tensor[T]) Gather(idx *Tensor[T], dim int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, idx}, GatherForward[T](dim), GatherBackward[T](dim))
}

// MustGather gathers along dimension, panics on error.
func (t *Tensor[T]) MustGather(idx *Tensor[T], dim int) *Tensor[T] {
	res, err := t.Gather(idx, dim)
	if err != nil {
		panic(err)
	}
	return res
}

// Scatter scatters src values along dimension using indices.
func (t *Tensor[T]) Scatter(idx *Tensor[T], src *Tensor[T], dim int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, idx, src}, ScatterForward[T](dim), ScatterBackward[T](dim))
}

// MustScatter scatters along dimension, panics on error.
func (t *Tensor[T]) MustScatter(idx *Tensor[T], src *Tensor[T], dim int) *Tensor[T] {
	res, err := t.Scatter(idx, src, dim)
	if err != nil {
		panic(err)
	}
	return res
}

// ScatterAdd performs scatter-add operation: adds src values to dst at indices along dimension.
func (t *Tensor[T]) ScatterAdd(idx *Tensor[T], src *Tensor[T], dim int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t, idx, src}, ScatterAddForward[T](dim), ScatterAddBackward[T](dim))
}

// MustScatterAdd performs scatter-add operation, panics on error.
func (t *Tensor[T]) MustScatterAdd(idx *Tensor[T], src *Tensor[T], dim int) *Tensor[T] {
	res, err := t.ScatterAdd(idx, src, dim)
	if err != nil {
		panic(err)
	}
	return res
}

// ReduceSum computes sum along dims, keepdim retains size 1.
func (t *Tensor[T]) ReduceSum(dims []int, keep bool) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, ReduceSumForward[T](dims, keep), ReduceSumBackward[T](dims, keep))
}

// MustReduceSum computes sum along dims, panics on error.
func (t *Tensor[T]) MustReduceSum(dims []int, keep bool) *Tensor[T] {
	res, err := t.ReduceSum(dims, keep)
	if err != nil {
		panic(err)
	}
	return res
}

// Sum computes sum along dims, removes dims.
func (t *Tensor[T]) Sum(dims []int) (*Tensor[T], error) {
	return t.ReduceSum(dims, false)
}

// MustSum computes sum along dims, panics on error.
func (t *Tensor[T]) MustSum(dims []int) *Tensor[T] {
	res, err := t.Sum(dims)
	if err != nil {
		panic(err)
	}
	return res
}

// SumKeep computes sum along dims, keeps size 1.
func (t *Tensor[T]) SumKeep(dims []int) (*Tensor[T], error) {
	return t.ReduceSum(dims, true)
}

// MustSumKeep computes sum along dims, panics on error.
func (t *Tensor[T]) MustSumKeep(dims []int) *Tensor[T] {
	res, err := t.SumKeep(dims)
	if err != nil {
		panic(err)
	}
	return res
}

// SumAll computes sum of all elements.
func (t *Tensor[T]) SumAll() (*Tensor[T], error) {
	dims := make([]int, t.Rank())
	for i := range dims {
		dims[i] = i
	}
	return t.ReduceSum(dims, false)
}

// MustSumAll computes sum of all elements, panics on error.
func (t *Tensor[T]) MustSumAll() *Tensor[T] {
	res, err := t.SumAll()
	if err != nil {
		panic(err)
	}
	return res
}

// ReduceMean computes mean along dims, keepdim retains size 1.
func (t *Tensor[T]) ReduceMean(dims []int, keep bool) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, ReduceMeanForward[T](dims, keep), ReduceMeanBackward[T](dims, keep))
}

// MustReduceMean computes mean along dims, panics on error.
func (t *Tensor[T]) MustReduceMean(dims []int, keep bool) *Tensor[T] {
	res, err := t.ReduceMean(dims, keep)
	if err != nil {
		panic(err)
	}
	return res
}

// Mean computes mean along dims, removes dims.
func (t *Tensor[T]) Mean(dims []int) (*Tensor[T], error) {
	return t.ReduceMean(dims, false)
}

// MustMean computes mean along dims, panics on error.
func (t *Tensor[T]) MustMean(dims []int) *Tensor[T] {
	res, err := t.Mean(dims)
	if err != nil {
		panic(err)
	}
	return res
}

// MeanKeep computes mean along dims, keeps size 1.
func (t *Tensor[T]) MeanKeep(dims []int) (*Tensor[T], error) {
	return t.ReduceMean(dims, true)
}

// MustMeanKeep computes mean along dims, panics on error.
func (t *Tensor[T]) MustMeanKeep(dims []int) *Tensor[T] {
	res, err := t.MeanKeep(dims)
	if err != nil {
		panic(err)
	}
	return res
}

// MeanAll computes mean of all elements.
func (t *Tensor[T]) MeanAll() (*Tensor[T], error) {
	dims := make([]int, t.Rank())
	for i := range dims {
		dims[i] = i
	}
	return t.ReduceMean(dims, false)
}

// MustMeanAll computes mean of all elements, panics on error.
func (t *Tensor[T]) MustMeanAll() *Tensor[T] {
	res, err := t.MeanAll()
	if err != nil {
		panic(err)
	}
	return res
}

// ReduceMin computes minimum along dims, keepdim retains size 1.
func (t *Tensor[T]) ReduceMin(dim int, keep bool) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, ReduceMinForward[T](dim, keep), ReduceMinBackward[T](dim, keep))
}

// MustReduceMin computes minimum along dims, panics on error.
func (t *Tensor[T]) MustReduceMin(dim int, keep bool) *Tensor[T] {
	res, err := t.ReduceMin(dim, keep)
	if err != nil {
		panic(err)
	}
	return res
}

// Min computes minimum along dims, removes dims.
func (t *Tensor[T]) Min(dim int) (*Tensor[T], error) {
	return t.ReduceMin(dim, false)
}

// MustMin computes minimum along dims, panics on error.
func (t *Tensor[T]) MustMin(dim int) *Tensor[T] {
	res, err := t.Min(dim)
	if err != nil {
		panic(err)
	}
	return res
}

// MinKeep computes minimum along dims, keeps size 1.
func (t *Tensor[T]) MinKeep(dim int) (*Tensor[T], error) {
	return t.ReduceMin(dim, true)
}

// MustMinKeep computes minimum along dims, panics on error.
func (t *Tensor[T]) MustMinKeep(dim int) *Tensor[T] {
	res, err := t.MinKeep(dim)
	if err != nil {
		panic(err)
	}
	return res
}

// ReduceMax computes maximum along dims, keepdim retains size 1.
func (t *Tensor[T]) ReduceMax(dim int, keep bool) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, ReduceMaxForward[T](dim, keep), ReduceMaxBackward[T](dim, keep))
}

// MustReduceMax computes maximum along dims, panics on error.
func (t *Tensor[T]) MustReduceMax(dim int, keep bool) *Tensor[T] {
	res, err := t.ReduceMax(dim, keep)
	if err != nil {
		panic(err)
	}
	return res
}

// Max computes maximum along dims, removes dims.
func (t *Tensor[T]) Max(dim int) (*Tensor[T], error) {
	return t.ReduceMax(dim, false)
}

// MustMax computes maximum along dims, panics on error.
func (t *Tensor[T]) MustMax(dim int) *Tensor[T] {
	res, err := t.Max(dim)
	if err != nil {
		panic(err)
	}
	return res
}

// MaxKeep computes maximum along dims, keeps size 1.
func (t *Tensor[T]) MaxKeep(dim int) (*Tensor[T], error) {
	return t.ReduceMax(dim, true)
}

// MustMaxKeep computes maximum along dims, panics on error.
func (t *Tensor[T]) MustMaxKeep(dim int) *Tensor[T] {
	res, err := t.MaxKeep(dim)
	if err != nil {
		panic(err)
	}
	return res
}

// FastMin mins over last dim.
func (t *Tensor[T]) FastMin() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, FastMinForward[T](), FastMinBackward[T]())
}

// MustFastMin mins over last dim, panics on error.
func (t *Tensor[T]) MustFastMin() *Tensor[T] {
	res, err := t.FastMin()
	if err != nil {
		panic(err)
	}
	return res
}

// FastMax maxes over last dim.
func (t *Tensor[T]) FastMax() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, FastMaxForward[T](), FastMaxBackward[T]())
}

// MustFastMax maxes over last dim, panics on error.
func (t *Tensor[T]) MustFastMax() *Tensor[T] {
	res, err := t.FastMax()
	if err != nil {
		panic(err)
	}
	return res
}

// FastSoftmax soft maxes over last dim.
func (t *Tensor[T]) FastSoftmax() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, FastSoftmaxForward[T](), FastSoftmaxBackward[T]())
}

// MustFastSoftmax soft maxes, panics on error.
func (t *Tensor[T]) MustFastSoftmax() *Tensor[T] {
	res, err := t.FastSoftmax()
	if err != nil {
		panic(err)
	}
	return res
}

// Softmax computes the softmax along the specified dimension.
func (x *Tensor[T]) Softmax(dim int) (*Tensor[T], error) {
	d, err := candy.ResolveAxis(dim, x.Rank())
	if err != nil {
		return nil, fmt.Errorf("failed to resolve dim: %w", err)
	}
	m, err := x.MaxKeep(d)
	if err != nil {
		return nil, fmt.Errorf("failed to compute max: %w", err)
	}
	s, err := x.BroadcastSub(m)
	if err != nil {
		return nil, fmt.Errorf("failed to subtract max: %w", err)
	}
	e, err := s.Exp()
	if err != nil {
		return nil, fmt.Errorf("failed to compute exp: %w", err)
	}
	se, err := e.SumKeep([]int{d})
	if err != nil {
		return nil, fmt.Errorf("failed to sum exp: %w", err)
	}
	r, err := e.BroadcastDiv(se)
	if err != nil {
		return nil, fmt.Errorf("failed to divide by sum: %w", err)
	}
	return r, nil
}

// MustSoftmax soft maxes, panics on error.
func (t *Tensor[T]) MustSoftmax(dim int) *Tensor[T] {
	res, err := t.Softmax(dim)
	if err != nil {
		panic(err)
	}
	return res
}

// LogSoftmax computes the log-softmax along the specified dimension.
func (x *Tensor[T]) LogSoftmax(dim int) (*Tensor[T], error) {
	d, err := candy.ResolveAxis(dim, x.Rank())
	if err != nil {
		return nil, fmt.Errorf("failed to resolve dim: %w", err)
	}
	m, err := x.MaxKeep(d)
	if err != nil {
		return nil, fmt.Errorf("failed to compute max: %w", err)
	}
	s, err := x.BroadcastSub(m)
	if err != nil {
		return nil, fmt.Errorf("failed to subtract max: %w", err)
	}
	e, err := s.Exp()
	if err != nil {
		return nil, fmt.Errorf("failed to compute exp: %w", err)
	}
	se, err := e.SumKeep([]int{d})
	if err != nil {
		return nil, fmt.Errorf("failed to sum exp: %w", err)
	}
	l, err := se.Log()
	if err != nil {
		return nil, fmt.Errorf("failed to compute log: %w", err)
	}
	r, err := s.BroadcastSub(l)
	if err != nil {
		return nil, fmt.Errorf("failed to subtract log-sum-exp: %w", err)
	}
	return r, nil
}

// MustLogSoftmax log soft maxes, panics on error.
func (t *Tensor[T]) MustLogSoftmax(dim int) *Tensor[T] {
	res, err := t.LogSoftmax(dim)
	if err != nil {
		panic(err)
	}
	return res
}

// Dropout randomly zeroes elements with probability dropProb and rescales the remainder.
func (x *Tensor[T]) Dropout(dropProb float64) (*Tensor[T], error) {
	var mask *Tensor[T]
	return ApplyOp([]*Tensor[T]{x}, DropoutForward(dropProb, &mask), DropoutBackward(&mask))
}

// MustDropout drops out values, panics on error.
func (t *Tensor[T]) MustDropout(dropProb float64) *Tensor[T] {
	res, err := t.Dropout(dropProb)
	if err != nil {
		panic(err)
	}
	return res
}

// WhereCond selects based on condition.
func (t *Tensor[T]) WhereCond(trueV, falseV *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{trueV, falseV}, WhereCondForward(t), WhereCondBackward(t))
}

// MustWhereCond selects, panics on error.
func (t *Tensor[T]) MustWhereCond(trueV, falseV *Tensor[T]) *Tensor[T] {
	res, err := t.WhereCond(trueV, falseV)
	if err != nil {
		panic(err)
	}
	return res
}

// Copy copies the tensor.
func (t *Tensor[T]) Copy() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, CopyForward[T](), CopyBackward[T]())
}

// MustCopy copies, panics on error.
func (t *Tensor[T]) MustCopy() *Tensor[T] {
	res, err := t.Copy()
	if err != nil {
		panic(err)
	}
	return res
}

// Neg negates element-wise.
func (t *Tensor[T]) Neg() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, NegForward[T](), NegBackward[T]())
}

// MustNeg negates, panics on error.
func (t *Tensor[T]) MustNeg() *Tensor[T] {
	res, err := t.Neg()
	if err != nil {
		panic(err)
	}
	return res
}

// Recip reciprocates element-wise.
func (t *Tensor[T]) Recip() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, RecipForward[T](), RecipBackward[T]())
}

// MustRecip reciprocates, panics on error.
func (t *Tensor[T]) MustRecip() *Tensor[T] {
	res, err := t.Recip()
	if err != nil {
		panic(err)
	}
	return res
}

// Exp exponentiates element-wise.
func (t *Tensor[T]) Exp() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, ExpForward[T](), ExpBackward[T]())
}

// MustExp exponentiates, panics on error.
func (t *Tensor[T]) MustExp() *Tensor[T] {
	res, err := t.Exp()
	if err != nil {
		panic(err)
	}
	return res
}

// Log logs element-wise (natural).
func (t *Tensor[T]) Log() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, LogForward[T](), LogBackward[T]())
}

// MustLog logs, panics on error.
func (t *Tensor[T]) MustLog() *Tensor[T] {
	res, err := t.Log()
	if err != nil {
		panic(err)
	}
	return res
}

// Sin sines element-wise.
func (t *Tensor[T]) Sin() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SinForward[T](), SinBackward[T]())
}

// MustSin sines, panics on error.
func (t *Tensor[T]) MustSin() *Tensor[T] {
	res, err := t.Sin()
	if err != nil {
		panic(err)
	}
	return res
}

// Cos cosines element-wise.
func (t *Tensor[T]) Cos() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, CosForward[T](), CosBackward[T]())
}

// MustCos cosines, panics on error.
func (t *Tensor[T]) MustCos() *Tensor[T] {
	res, err := t.Cos()
	if err != nil {
		panic(err)
	}
	return res
}

// Tanh hyperbolically tangents element-wise.
func (t *Tensor[T]) Tanh() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, TanhForward[T](), TanhBackward[T]())
}

// MustTanh tangents, panics on error.
func (t *Tensor[T]) MustTanh() *Tensor[T] {
	res, err := t.Tanh()
	if err != nil {
		panic(err)
	}
	return res
}

// Erf error functions element-wise.
func (t *Tensor[T]) Erf() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, ErfForward[T](), ErfBackward[T]())
}

// MustErf error functions, panics on error.
func (t *Tensor[T]) MustErf() *Tensor[T] {
	res, err := t.Erf()
	if err != nil {
		panic(err)
	}
	return res
}

// Ceil ceilings element-wise.
func (t *Tensor[T]) Ceil() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, CeilForward[T](), CeilBackward[T]())
}

// MustCeil ceilings, panics on error.
func (t *Tensor[T]) MustCeil() *Tensor[T] {
	res, err := t.Ceil()
	if err != nil {
		panic(err)
	}
	return res
}

// Floor floors element-wise.
func (t *Tensor[T]) Floor() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, FloorForward[T](), FloorBackward[T]())
}

// MustFloor floors, panics on error.
func (t *Tensor[T]) MustFloor() *Tensor[T] {
	res, err := t.Floor()
	if err != nil {
		panic(err)
	}
	return res
}

// Round rounds element-wise.
func (t *Tensor[T]) Round() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, RoundForward[T](), RoundBackward[T]())
}

// MustRound rounds, panics on error.
func (t *Tensor[T]) MustRound() *Tensor[T] {
	res, err := t.Round()
	if err != nil {
		panic(err)
	}
	return res
}

// Normcdf normal CDFs element-wise.
func (t *Tensor[T]) Normcdf() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, NormcdfForward[T](), NormcdfBackward[T]())
}

// MustNormcdf normal CDFs, panics on error.
func (t *Tensor[T]) MustNormcdf() *Tensor[T] {
	res, err := t.Normcdf()
	if err != nil {
		panic(err)
	}
	return res
}

// Abs absolutes element-wise.
func (t *Tensor[T]) Abs() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, AbsForward[T](), AbsBackward[T]())
}

// MustAbs absolutes, panics on error.
func (t *Tensor[T]) MustAbs() *Tensor[T] {
	res, err := t.Abs()
	if err != nil {
		panic(err)
	}
	return res
}

// Sqr squares element-wise.
func (t *Tensor[T]) Sqr() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SqrForward[T](), SqrBackward[T]())
}

// MustSqr squares, panics on error.
func (t *Tensor[T]) MustSqr() *Tensor[T] {
	res, err := t.Sqr()
	if err != nil {
		panic(err)
	}
	return res
}

// Sqrt square roots element-wise.
func (t *Tensor[T]) Sqrt() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SqrtForward[T](), SqrtBackward[T]())
}

// MustSqrt square roots, panics on error.
func (t *Tensor[T]) MustSqrt() *Tensor[T] {
	res, err := t.Sqrt()
	if err != nil {
		panic(err)
	}
	return res
}

// Gelu applies GELU activation.
func (t *Tensor[T]) Gelu() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, GeluForward[T](), GeluBackward[T]())
}

// MustGelu applies GELU, panics on error.
func (t *Tensor[T]) MustGelu() *Tensor[T] {
	res, err := t.Gelu()
	if err != nil {
		panic(err)
	}
	return res
}

// GeluErf applies ERF-based GELU.
func (t *Tensor[T]) GeluErf() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, GeluErfForward[T](), GeluErfBackward[T]())
}

// MustGeluErf applies ERF GELU, panics on error.
func (t *Tensor[T]) MustGeluErf() *Tensor[T] {
	res, err := t.GeluErf()
	if err != nil {
		panic(err)
	}
	return res
}

// Relu applies ReLU activation.
func (t *Tensor[T]) Relu() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, ReluForward[T](), ReluBackward[T]())
}

// MustRelu applies ReLU, panics on error.
func (t *Tensor[T]) MustRelu() *Tensor[T] {
	res, err := t.Relu()
	if err != nil {
		panic(err)
	}
	return res
}

// Elu applies ELU activation.
func (t *Tensor[T]) Elu(alpha float64) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, EluForward[T](alpha), EluBackward[T](alpha))
}

// MustElu applies ELU, panics on error.
func (t *Tensor[T]) MustElu(alpha float64) *Tensor[T] {
	res, err := t.Elu(alpha)
	if err != nil {
		panic(err)
	}
	return res
}

// Silu applies SiLU activation.
func (t *Tensor[T]) Silu() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SiluForward[T](), SiluBackward[T]())
}

// MustSilu applies SiLU, panics on error.
func (t *Tensor[T]) MustSilu() *Tensor[T] {
	res, err := t.Silu()
	if err != nil {
		panic(err)
	}
	return res
}

// Powf powers element-wise to param.
func (t *Tensor[T]) Powf(p float64) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, PowfForward[T](p), PowfBackward[T](p))
}

// MustPowf powers, panics on error.
func (t *Tensor[T]) MustPowf(p float64) *Tensor[T] {
	res, err := t.Powf(p)
	if err != nil {
		panic(err)
	}
	return res
}

// Sigmoid applies sigmoid activation.
func (t *Tensor[T]) Sigmoid() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SigmoidForward[T](), SigmoidBackward[T]())
}

// MustSigmoid applies sigmoid, panics on error.
func (t *Tensor[T]) MustSigmoid() *Tensor[T] {
	res, err := t.Sigmoid()
	if err != nil {
		panic(err)
	}
	return res
}

// Sign signs element-wise.
func (t *Tensor[T]) Sign() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SignForward[T](), SignBackward[T]())
}

// MustSign signs, panics on error.
func (t *Tensor[T]) MustSign() *Tensor[T] {
	res, err := t.Sign()
	if err != nil {
		panic(err)
	}
	return res
}

// Transpose transposes dims.
func (t *Tensor[T]) Transpose(d1, d2 int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, TransposeForward[T](d1, d2), TransposeBackward[T](d1, d2))
}

// MustTranspose transposes, panics on error.
func (t *Tensor[T]) MustTranspose(d1, d2 int) *Tensor[T] {
	res, err := t.Transpose(d1, d2)
	if err != nil {
		panic(err)
	}
	return res
}

// T transposes last two dims.
func (t *Tensor[T]) T() (*Tensor[T], error) {
	r := t.Rank()
	if r < 2 {
		return nil, fmt.Errorf("need >=2 dims for T, got %d", r)
	}
	return t.Transpose(r-2, r-1)
}

// MustT transposes last two, panics on error.
func (t *Tensor[T]) MustT() *Tensor[T] {
	res, err := t.T()
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastAs broadcasts to shape.
func (t *Tensor[T]) BroadcastAs(s *candy.Shape) (*Tensor[T], error) {
	if t.Shape().Equal(s) {
		return t, nil
	}
	return ApplyOp([]*Tensor[T]{t}, BroadcastAsForward[T](s), BroadcastAsBackward[T](t.Shape()))
}

// MustBroadcastAs broadcasts to shape, panics on error.
func (t *Tensor[T]) MustBroadcastAs(s *candy.Shape) *Tensor[T] {
	res, err := t.BroadcastAs(s)
	if err != nil {
		panic(err)
	}
	return res
}

// Expand expands to shape.
func (t *Tensor[T]) Expand(s *candy.Shape) (*Tensor[T], error) {
	return t.BroadcastAs(s)
}

// MustExpand expands to shape, panics on error.
func (t *Tensor[T]) MustExpand(s *candy.Shape) *Tensor[T] {
	res, err := t.Expand(s)
	if err != nil {
		panic(err)
	}
	return res
}

// BroadcastLeft adds left dims for broadcast.
func (t *Tensor[T]) BroadcastLeft(ld ...int) (*Tensor[T], error) {
	cd := t.Dims()
	nd := append(ld, cd...)
	return t.BroadcastAs(candy.NewShapeFrom(nd))
}

// MustBroadcastLeft adds left dims, panics on error.
func (t *Tensor[T]) MustBroadcastLeft(ld ...int) *Tensor[T] {
	res, err := t.BroadcastLeft(ld...)
	if err != nil {
		panic(err)
	}
	return res
}

// Squeeze squeezes dim if size 1.
func (t *Tensor[T]) Squeeze(dim int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SqueezeForward[T](dim), SqueezeBackward[T](dim))
}

// MustSqueeze squeezes dim, panics on error.
func (t *Tensor[T]) MustSqueeze(dim int) *Tensor[T] {
	res, err := t.Squeeze(dim)
	if err != nil {
		panic(err)
	}
	return res
}

// Unsqueeze inserts size 1 dim.
func (t *Tensor[T]) Unsqueeze(dim int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, UnsqueezeForward[T](dim), UnsqueezeBackward[T](dim))
}

// MustUnsqueeze inserts dim, panics on error.
func (t *Tensor[T]) MustUnsqueeze(dim int) *Tensor[T] {
	res, err := t.Unsqueeze(dim)
	if err != nil {
		panic(err)
	}
	return res
}

// SqueezeDims squeezes multiple dims if size 1.
func (t *Tensor[T]) SqueezeDims(dims []int) (*Tensor[T], error) {
	if len(dims) == 0 {
		return t, nil
	}
	if len(dims) == 1 {
		return t.Squeeze(dims[0])
	}
	ds := make(map[int]bool)
	for _, d := range dims {
		r, err := candy.ResolveAxis(d, t.Rank())
		if err != nil {
			return nil, err
		}
		ds[r] = true
	}
	cd := t.Dims()
	var nd []int
	for i, sz := range cd {
		if !ds[i] || sz != 1 {
			nd = append(nd, sz)
		}
	}
	return t.Reshape(nd...)
}

// Reshape reshapes preserving elements.
func (t *Tensor[T]) Reshape(d ...int) (*Tensor[T], error) {
	s, err := t.Shape().Reshape(d...)
	if err != nil {
		return nil, fmt.Errorf("reshape failed: %w", err)
	}
	return ApplyOp([]*Tensor[T]{t}, ReshapeForward[T](s), ReshapeBackward[T](t.Shape()))
}

// MustReshape reshapes, panics on error.
func (t *Tensor[T]) MustReshape(d ...int) *Tensor[T] {
	res, err := t.Reshape(d...)
	if err != nil {
		panic(err)
	}
	return res
}

// Flatten flattens from start to end dim.
func (t *Tensor[T]) Flatten(start, end int) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, FlattenForward[T](start, end), FlattenBackward[T](t.Shape()))
}

// MustFlatten flattens range, panics on error.
func (t *Tensor[T]) MustFlatten(start, end int) *Tensor[T] {
	res, err := t.Flatten(start, end)
	if err != nil {
		panic(err)
	}
	return res
}

// FlattenAll flattens to 1D.
func (t *Tensor[T]) FlattenAll() (*Tensor[T], error) {
	return t.Reshape(t.Numel())
}

// MustFlattenAll flattens to 1D, panics on error.
func (t *Tensor[T]) MustFlattenAll() *Tensor[T] {
	res, err := t.FlattenAll()
	if err != nil {
		panic(err)
	}
	return res
}

// Backward computes gradients.
func (t *Tensor[T]) Backward() (*GradStore[T], error) {
	s := NewGradStore[T]()
	if err := Backward(t, s); err != nil {
		return nil, err
	}
	return s, nil
}

// MustBackward computes gradients, panics on error.
func (t *Tensor[T]) MustBackward() *GradStore[T] {
	s, err := t.Backward()
	if err != nil {
		panic(err)
	}
	return s
}

package tensor

import (
	"sync/atomic"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/internal/cpu"
)

// counter is a package-level atomic counter for generating unique TensorId values.
var counter uint64

// TensorId is a unique identifier for a tensor.
type TensorId uint64

// NewId generates a new, unique TensorId in a thread-safe manner.
func NewId() TensorId {
	return TensorId(atomic.AddUint64(&counter, 1))
}

type Tensor[T spark.D] struct {
	id      TensorId
	storage spark.BackendStorage[T]
	layout  *spark.Layout
	op      *Op[T]
	isVar   bool
	dtype   spark.DType
	device  spark.Device
}

func New[T spark.D](array []T, shape *spark.Shape, device spark.Device) *Tensor[T] {
	var storage spark.BackendStorage[T]
	switch device {
	case spark.CPU:
		storage = cpu.New(array)
	default:
		panic("device not supported")
	}
	return &Tensor[T]{
		id:      NewId(),
		storage: storage,
		layout:  spark.Contiguous(shape),
		isVar:   false,
		dtype:   spark.DTypeOf[T](),
		device:  device,
	}
}

// NewFromStorage creates a new tensor from existing storage and layout.
// This is useful for creating tensors in backward passes without adding to computation graph.
func NewFromStorage[T spark.D](storage spark.BackendStorage[T], layout *spark.Layout, dtype spark.DType, device spark.Device) *Tensor[T] {
	return &Tensor[T]{
		id:      NewId(),
		storage: storage,
		layout:  layout.Clone(),
		isVar:   false, // Gradient tensors should not be variables by default
		dtype:   dtype,
		device:  device,
	}
}

func Ones[T spark.D](shape *spark.Shape, device spark.Device) *Tensor[T] {
	var storage spark.BackendStorage[T]
	var err error

	switch device {
	case spark.CPU:
		cpuDevice := cpu.NewCpuDevice[T]()
		storage, err = cpuDevice.Ones(shape, spark.DTypeOf[T]())
		if err != nil {
			panic(err)
		}
	default:
		panic("device not supported")
	}

	return &Tensor[T]{
		id:      NewId(),
		storage: storage,
		layout:  spark.Contiguous(shape),
		isVar:   false,
		dtype:   spark.DTypeOf[T](),
		device:  device,
	}
}

func Zeros[T spark.D](shape *spark.Shape, device spark.Device) *Tensor[T] {
	var storage spark.BackendStorage[T]
	var err error

	switch device {
	case spark.CPU:
		cpuDevice := cpu.NewCpuDevice[T]()
		storage, err = cpuDevice.Zeros(shape, spark.DTypeOf[T]())
		if err != nil {
			panic(err)
		}
	default:
		panic("device not supported")
	}

	return &Tensor[T]{
		id:      NewId(),
		storage: storage,
		layout:  spark.Contiguous(shape),
		isVar:   false,
		dtype:   spark.DTypeOf[T](),
		device:  device,
	}
}

// Rand creates a new tensor initialized with values sampled uniformly between lo and up.
func Rand[T spark.D](lo, up float64, shape *spark.Shape, device spark.Device) *Tensor[T] {
	var storage spark.BackendStorage[T]
	var err error

	switch device {
	case spark.CPU:
		cpuDevice := cpu.NewCpuDevice[T]()
		storage, err = cpuDevice.RandUniform(shape, spark.DTypeOf[T](), lo, up)
		if err != nil {
			panic(err)
		}
	default:
		panic("device not supported")
	}

	return &Tensor[T]{
		id:      NewId(),
		storage: storage,
		layout:  spark.Contiguous(shape),
		isVar:   false,
		dtype:   spark.DTypeOf[T](),
		device:  device,
	}
}

// RandN creates a new tensor initialized with values sampled from a normal distribution.
func RandN[T spark.D](mean, std float64, shape *spark.Shape, device spark.Device) *Tensor[T] {
	var storage spark.BackendStorage[T]
	var err error

	switch device {
	case spark.CPU:
		cpuDevice := cpu.NewCpuDevice[T]()
		storage, err = cpuDevice.RandNormal(shape, spark.DTypeOf[T](), mean, std)
		if err != nil {
			panic(err)
		}
	default:
		panic("device not supported")
	}

	return &Tensor[T]{
		id:      NewId(),
		storage: storage,
		layout:  spark.Contiguous(shape),
		isVar:   false,
		dtype:   spark.DTypeOf[T](),
		device:  device,
	}
}

func Full[T spark.D](value float64, shape *spark.Shape, device spark.Device) *Tensor[T] {
	var storage spark.BackendStorage[T]
	var err error

	switch device {
	case spark.CPU:
		cpuDevice := cpu.NewCpuDevice[T]()
		storage, err = cpuDevice.Full(shape, spark.DTypeOf[T](), value)
		if err != nil {
			panic(err)
		}
	default:
		panic("device not supported")
	}

	return &Tensor[T]{
		id:      NewId(),
		storage: storage,
		layout:  spark.Contiguous(shape),
		isVar:   false,
		dtype:   spark.DTypeOf[T](),
		device:  device,
	}
}

// ZerosLike creates a new tensor with the same shape as the input tensor, filled with zeros.
func (t *Tensor[T]) ZerosLike() *Tensor[T] {
	return Zeros[T](t.layout.Shape(), t.device)
}

// OnesLike creates a new tensor with the same shape as the input tensor, filled with ones.
func (t *Tensor[T]) OnesLike() *Tensor[T] {
	return Ones[T](t.layout.Shape(), t.device)
}

// FullLike creates a new tensor with the same shape as the input tensor, filled with the specified value.
func (t *Tensor[T]) FullLike(value float64) *Tensor[T] {
	return Full[T](value, t.layout.Shape(), t.device)
}

// RandLike creates a new tensor with the same shape as the input tensor, filled with random values.
func (t *Tensor[T]) RandLike(lo, up float64) *Tensor[T] {
	return Rand[T](lo, up, t.layout.Shape(), t.device)
}

// RandNLike creates a new tensor with the same shape as the input tensor, filled with values sampled from a normal distribution.
func (t *Tensor[T]) RandNLike(mean, std float64) *Tensor[T] {
	return RandN[T](mean, std, t.layout.Shape(), t.device)
}

// ID returns the unique identifier of the tensor.
func (t *Tensor[T]) ID() TensorId {
	return t.id
}

// Storage returns the backend storage of the tensor.
func (t *Tensor[T]) Storage() spark.BackendStorage[T] {
	return t.storage
}

// Layout returns the layout of the tensor.
func (t *Tensor[T]) Layout() *spark.Layout {
	return t.layout.Clone()
}

// Op returns the operation that created the tensor.
func (t *Tensor[T]) Op() *Op[T] {
	return t.op
}

// IsVar returns true if the tensor is a variable (requires gradient).
func (t *Tensor[T]) IsVar() bool {
	return t.isVar
}

// DType returns the data type of the tensor.
func (t *Tensor[T]) DType() spark.DType {
	return t.dtype
}

// Device returns the device of the tensor.
func (t *Tensor[T]) Device() spark.Device {
	return t.device
}

// Data returns the underlying data slice. Panics if storage is not CPU-based.
func (t *Tensor[T]) Data() []T {
	return t.storage.Data()
}

// Stride returns the strides of the tensor.
func (t *Tensor[T]) Stride() []int {
	return t.layout.Stride()
}

// Shape returns the shape of the tensor.
func (t *Tensor[T]) Shape() *spark.Shape {
	return t.layout.Shape()
}

// Dims returns the dimensions of the tensor.
func (t *Tensor[T]) Dims() []int {
	return t.layout.Dims()
}

// Dim returns the size of the dimension at the given index.
func (t *Tensor[T]) Dim(dim int) int {
	return t.layout.Dim(dim)
}

// Rank returns the rank of the tensor.
func (t *Tensor[T]) Rank() int {
	return t.layout.Rank()
}

// ElemCount returns the number of elements in the tensor.
func (t *Tensor[T]) ElemCount() int {
	return t.layout.ElemCount()
}

func (t *Tensor[T]) SetStorage(storage spark.BackendStorage[T]) *Tensor[T] {
	t.storage = storage
	return t
}

func (t *Tensor[T]) SetLayout(layout *spark.Layout) *Tensor[T] {
	t.layout = layout.Clone()
	return t
}

func (t *Tensor[T]) SetOp(op *Op[T]) *Tensor[T] {
	t.op = op
	return t
}

func (t *Tensor[T]) SetIsVar(isVar bool) *Tensor[T] {
	t.isVar = isVar
	return t
}

func (t *Tensor[T]) SetDType(dtype spark.DType) *Tensor[T] {
	t.dtype = dtype
	return t
}

func (t *Tensor[T]) SetDevice(device spark.Device) *Tensor[T] {
	t.device = device
	return t
}

func (t *Tensor[T]) RequiresGrad() *Tensor[T] {
	t.isVar = true
	return t
}

func (t *Tensor[T]) Detach() *Tensor[T] {
	return NewFromStorage(t.storage, t.layout, t.dtype, t.device)
}

// Add performs element-wise addition of two tensors.
func (a *Tensor[T]) Add(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, AddForward[T], AddBackward[T])
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
	return ApplyOp([]*Tensor[T]{t, other}, SubForward[T], SubBackward[T])
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
	return ApplyOp([]*Tensor[T]{a, b}, MulForward[T], MulBackward[T])
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
	return ApplyOp([]*Tensor[T]{t, other}, DivForward[T], DivBackward[T])
}

// MustDiv performs element-wise division of two tensors, panicking on error.
func (t *Tensor[T]) MustDiv(other *Tensor[T]) *Tensor[T] {
	t, err := t.Div(other)
	if err != nil {
		panic(err)
	}
	return t
}

// Sqrt computes the square root of each element.
func (t *Tensor[T]) Sqrt() (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{t}, SqrtForward[T], SqrtBackward[T])
}

// MustSqrt computes the square root of each element, panicking on error.
func (t *Tensor[T]) MustSqrt() *Tensor[T] {
	t, err := t.Sqrt()
	if err != nil {
		panic(err)
	}
	return t
}

// BroadcastAdd performs broadcasted addition: result = broadcast(a) + broadcast(b).
func (a *Tensor[T]) BroadcastAdd(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, BroadcastAddForward[T], BroadcastAddBackward[T])
}

// MustBroadcastAdd performs broadcasted addition, panicking on error.
func (a *Tensor[T]) MustBroadcastAdd(b *Tensor[T]) *Tensor[T] {
	t, err := a.BroadcastAdd(b)
	if err != nil {
		panic(err)
	}
	return t
}

// AddScalar adds a scalar value to the tensor.
func (t *Tensor[T]) AddScalar(scalar float64) (*Tensor[T], error) {
	// TODO: Implement scalar addition
	return nil, nil
}

// MulScalar multiplies the tensor by a scalar value.
func (t *Tensor[T]) MulScalar(scalar float64) (*Tensor[T], error) {
	// TODO: Implement scalar multiplication
	return nil, nil
}

// BroadcastAs broadcasts the tensor to the target shape.
func (t *Tensor[T]) BroadcastAs(shape *spark.Shape) (*Tensor[T], error) {
	if t.layout.Shape().Equal(shape) {
		return t, nil
	}

	newLayout, err := t.layout.BroadcastAs(shape)
	if err != nil {
		return nil, err
	}

	return NewFromStorage(t.storage, newLayout, t.dtype, t.device), nil
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

// Backward computes gradients for all variable tensors contributing to the root tensor.
// Returns the gradient store containing all computed gradients.
func (t *Tensor[T]) Backward() (*GradStore[T], error) {
	store := NewGradStore[T]()
	if err := Backward(t, store); err != nil {
		return nil, err
	}
	return store, nil
}

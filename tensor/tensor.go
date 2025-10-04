package tensor

import (
	"sync/atomic"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/internal/cpu"
)

// TensorId is a unique identifier for a tensor.
type TensorId uint64

// counter is a package-level atomic counter for generating unique TensorId values.
var counter uint64

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

func New[T spark.D](array []T, shape *spark.Shape, isVar bool, device spark.Device) *Tensor[T] {
	var storage spark.BackendStorage[T]
	switch device {
	case spark.CPU:
		storage = cpu.New(array)
	default:
		return nil
	}
	return &Tensor[T]{
		id:      NewId(),
		storage: storage,
		layout:  spark.Contiguous(shape),
		isVar:   isVar,
		dtype:   storage.DType(),
		device:  device,
	}
}

func NewTensor[T spark.D](storage spark.BackendStorage[T], layout *spark.Layout, isVar bool, device spark.Device) *Tensor[T] {
	return &Tensor[T]{
		id:      NewId(),
		storage: storage,
		layout:  layout,
		isVar:   isVar,
		dtype:   storage.DType(),
		device:  device,
	}
}

func (t *Tensor[T]) ZerosLike() (*Tensor[T], error) {
	return nil, nil
}

func (t *Tensor[T]) ID() TensorId {
	return t.id
}

func (t *Tensor[T]) IsVar() bool {
	return t.isVar
}

func (t *Tensor[T]) Op() *Op[T] {
	return t.op
}

// Storage returns the backend storage of the tensor.
func (t *Tensor[T]) Storage() spark.BackendStorage[T] {
	return t.storage
}

// Layout returns the layout of the tensor.
func (t *Tensor[T]) Layout() *spark.Layout {
	return t.layout
}

// DType returns the data type of the tensor.
func (t *Tensor[T]) DType() spark.DType {
	return t.dtype
}

// Device returns the device of the tensor.
func (t *Tensor[T]) Device() spark.Device {
	return t.device
}

func (t *Tensor[T]) SetStorage(storage spark.BackendStorage[T]) {
	t.storage = storage
}

func (t *Tensor[T]) SetLayout(layout *spark.Layout) {
	t.layout = layout
}

func (t *Tensor[T]) SetIsVar(isVar bool) {
	t.isVar = isVar
}

func (t *Tensor[T]) SetDType(dtype spark.DType) {
	t.dtype = dtype
}

func (t *Tensor[T]) SetDevice(device spark.Device) {
	t.device = device
}

func (t *Tensor[T]) Backward() error {
	store := NewGradStore[T]()
	return Backward(t, store)
}

func (a *Tensor[T]) Add(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, AddForward[T], AddBackward[T])
}

func (a *Tensor[T]) Mul(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, MulForward[T], MulBackward[T])
}

// Div performs element-wise division of two tensors.
func (t *Tensor[T]) Div(other *Tensor[T]) (*Tensor[T], error) {
	// TODO: Implement tensor division
	return nil, nil
}

// Sub performs element-wise subtraction of two tensors.
func (t *Tensor[T]) Sub(other *Tensor[T]) (*Tensor[T], error) {
	// TODO: Implement tensor subtraction
	return nil, nil
}

// MulScalar multiplies the tensor by a scalar value.
func (t *Tensor[T]) MulScalar(scalar float64) (*Tensor[T], error) {
	// TODO: Implement scalar multiplication
	return nil, nil
}

// AddScalar adds a scalar value to the tensor.
func (t *Tensor[T]) AddScalar(scalar float64) (*Tensor[T], error) {
	// TODO: Implement scalar addition
	return nil, nil
}

// Sqrt computes the square root of each element.
func (t *Tensor[T]) Sqrt() (*Tensor[T], error) {
	// TODO: Implement square root
	return nil, nil
}

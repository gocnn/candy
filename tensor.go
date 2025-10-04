package spark

import (
	"sync/atomic"
)

// TensorId is a unique identifier for a tensor.
type TensorId uint64

// counter is a package-level atomic counter for generating unique TensorId values.
var counter uint64

// NewId generates a new, unique TensorId in a thread-safe manner.
func NewId() TensorId {
	return TensorId(atomic.AddUint64(&counter, 1))
}

type Tensor[T D] struct {
	id      TensorId
	storage BackendStorage[T]
	layout  *Layout
	op      *Op[T]
	isVar   bool
	dtype   DType
	device  Device
}

func NewTensor[T D](storage BackendStorage[T], layout *Layout, isVar bool, dtype DType, device Device) *Tensor[T] {
	return &Tensor[T]{id: NewId(), storage: storage, layout: layout, isVar: isVar, dtype: dtype, device: device}
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
func (t *Tensor[T]) Storage() BackendStorage[T] {
	return t.storage
}

// Layout returns the layout of the tensor.
func (t *Tensor[T]) Layout() *Layout {
	return t.layout
}

// DType returns the data type of the tensor.
func (t *Tensor[T]) DType() DType {
	return t.dtype
}

// Device returns the device of the tensor.
func (t *Tensor[T]) Device() Device {
	return t.device
}

func (t *Tensor[T]) SetStorage(storage BackendStorage[T]) {
	t.storage = storage
}

func (t *Tensor[T]) SetLayout(layout *Layout) {
	t.layout = layout
}

func (t *Tensor[T]) SetIsVar(isVar bool) {
	t.isVar = isVar
}

func (t *Tensor[T]) SetDType(dtype DType) {
	t.dtype = dtype
}

func (t *Tensor[T]) SetDevice(device Device) {
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

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

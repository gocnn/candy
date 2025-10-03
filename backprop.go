package spark

// GradStore is a store for gradients, associating a tensor ID to the corresponding gradient tensor, used for back propagation.
type GradStore[T D] struct {
	m map[TensorId]*Tensor[T] // Private field for encapsulation
}

// NewGradStore creates a new gradient store.
func NewGradStore[T D]() *GradStore[T] {
	return &GradStore[T]{m: make(map[TensorId]*Tensor[T])}
}

// GetByID returns the gradient tensor for the given ID, or nil if not found.
func (s *GradStore[T]) GetByID(id TensorId) *Tensor[T] {
	if t, ok := s.m[id]; ok {
		return t
	}
	return nil
}

// Get returns the gradient tensor for the given tensor, or nil if not found.
func (s *GradStore[T]) Get(tensor *Tensor[T]) *Tensor[T] {
	return s.GetByID(tensor.ID())
}

// GetOrCreate gets the gradient tensor for the given tensor, or creates and inserts a zero-filled tensor
// with the same shape and type if it does not exist.
func (s *GradStore[T]) GetOrCreate(tensor *Tensor[T]) (*Tensor[T], error) {
	id := tensor.ID()
	if _, ok := s.m[id]; !ok {
		grad, err := tensor.ZerosLike()
		if err != nil {
			return nil, err
		}
		s.m[id] = grad
	}
	return s.m[id], nil
}

// SetByID sets the gradient tensor for the given ID, returning the previous value and whether it was replaced.
func (s *GradStore[T]) SetByID(id TensorId, grad *Tensor[T]) (*Tensor[T], bool) {
	prev, ok := s.m[id]
	s.m[id] = grad
	return prev, ok
}

// Set sets the gradient tensor for the given tensor, returning the previous value and whether it was replaced.
func (s *GradStore[T]) Set(tensor *Tensor[T], grad *Tensor[T]) (*Tensor[T], bool) {
	return s.SetByID(tensor.ID(), grad)
}

// DeleteByID deletes the gradient tensor for the given ID, returning it and whether it existed.
func (s *GradStore[T]) DeleteByID(id TensorId) (*Tensor[T], bool) {
	grad, ok := s.m[id]
	if ok {
		delete(s.m, id)
	}
	return grad, ok
}

// Delete deletes the gradient tensor for the given tensor, returning it and whether it existed.
func (s *GradStore[T]) Delete(tensor *Tensor[T]) (*Tensor[T], bool) {
	return s.DeleteByID(tensor.ID())
}

// IDs returns the slice of all stored gradient tensor IDs.
func (s *GradStore[T]) IDs() []TensorId {
	ids := make([]TensorId, 0, len(s.m))
	for id := range s.m {
		ids = append(ids, id)
	}
	return ids
}

package tensor

import (
	"fmt"

	"github.com/gocnn/spark"
)

// GradStore maps tensor IDs to their gradient tensors for backpropagation.
type GradStore[T spark.D] struct {
	m map[TensorID]*Tensor[T]
}

// NewGradStore returns a new gradient store.
func NewGradStore[T spark.D]() *GradStore[T] {
	return &GradStore[T]{m: make(map[TensorID]*Tensor[T])}
}

// GetByID returns the gradient tensor for the given ID, or nil if not found.
func (s *GradStore[T]) GetByID(id TensorID) *Tensor[T] {
	return s.m[id]
}

// Get returns the gradient tensor for the given tensor, or nil if not found.
func (s *GradStore[T]) Get(t *Tensor[T]) *Tensor[T] {
	return s.GetByID(t.ID())
}

// GetOrCreate returns the gradient tensor for the given tensor, creating a zero-filled tensor if none exists.
func (s *GradStore[T]) GetOrCreate(t *Tensor[T]) (*Tensor[T], error) {
	id := t.ID()
	if _, ok := s.m[id]; !ok {
		zero, err := t.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("create zero tensor: %w", err)
		}
		s.m[id] = zero
	}
	return s.m[id], nil
}

// SetByID sets the gradient tensor for the given ID, returning the previous gradient and whether it existed.
func (s *GradStore[T]) SetByID(id TensorID, grad *Tensor[T]) (*Tensor[T], bool) {
	prev, ok := s.m[id]
	s.m[id] = grad
	return prev, ok
}

// Set sets the gradient tensor for the given tensor, returning the previous gradient and whether it existed.
func (s *GradStore[T]) Set(t *Tensor[T], grad *Tensor[T]) (*Tensor[T], bool) {
	return s.SetByID(t.ID(), grad)
}

// DeleteByID deletes the gradient tensor for the given ID, returning it and whether it existed.
func (s *GradStore[T]) DeleteByID(id TensorID) (*Tensor[T], bool) {
	grad, ok := s.m[id]
	if ok {
		delete(s.m, id)
	}
	return grad, ok
}

// Delete deletes the gradient tensor for the given tensor, returning it and whether it existed.
func (s *GradStore[T]) Delete(t *Tensor[T]) (*Tensor[T], bool) {
	return s.DeleteByID(t.ID())
}

// IDs returns all stored gradient tensor IDs.
func (s *GradStore[T]) IDs() []TensorID {
	ids := make([]TensorID, 0, len(s.m))
	for id := range s.m {
		ids = append(ids, id)
	}
	return ids
}

// Clear removes all gradient tensors from the store.
func (s *GradStore[T]) Clear() {
	s.m = make(map[TensorID]*Tensor[T])
}

// Backward computes gradients for all variable tensors contributing to the root tensor.
func Backward[T spark.D](root *Tensor[T], store *GradStore[T]) error {
	if !root.IsVar() {
		return nil
	}

	if store.Get(root) == nil {
		one, err := Ones[T](root.Layout().Shape(), root.Device())
		if err != nil {
			return fmt.Errorf("create ones tensor: %w", err)
		}
		store.Set(root, one)
	}

	vars := make(map[TensorID]*Tensor[T])
	outputs := make(map[TensorID][]*Tensor[T])
	queue := []*Tensor[T]{root}
	visited := make(map[TensorID]bool)

	for len(queue) > 0 {
		t := queue[0]
		queue = queue[1:]
		id := t.ID()
		if visited[id] {
			continue
		}
		visited[id] = true
		if !t.IsVar() {
			continue
		}
		vars[id] = t
		if t.Op() == nil {
			continue
		}
		for _, input := range t.Op().Inputs() {
			if input.IsVar() {
				outputs[input.ID()] = append(outputs[input.ID()], t)
				queue = append(queue, input)
			}
		}
	}

	pending := make(map[TensorID]int, len(vars))
	for id := range vars {
		pending[id] = len(outputs[id])
	}

	backQueue := []*Tensor[T]{root}
	for len(backQueue) > 0 {
		t := backQueue[0]
		backQueue = backQueue[1:]
		if t.Op() == nil {
			continue
		}

		grad, err := store.GetOrCreate(t)
		if err != nil {
			return err
		}
		grads, err := t.Op().Backward(grad, t.Op().Inputs())
		if err != nil {
			return err
		}

		for i, input := range t.Op().Inputs() {
			if !input.IsVar() {
				continue
			}
			id := input.ID()
			existing, err := store.GetOrCreate(input)
			if err != nil {
				return err
			}
			sum, err := existing.Add(grads[i])
			if err != nil {
				return err
			}
			store.SetByID(id, sum)
			pending[id]--
			if pending[id] == 0 {
				backQueue = append(backQueue, input)
			}
		}
	}
	return nil
}

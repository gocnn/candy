package tensor

import "github.com/gocnn/spark"

// GradStore is a store for gradients, associating a tensor ID to the corresponding gradient tensor, used for back propagation.
type GradStore[T spark.D] struct {
	m map[TensorId]*Tensor[T] // Private field for encapsulation
}

// NewGradStore creates a new gradient store.
func NewGradStore[T spark.D]() *GradStore[T] {
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
		grad := tensor.ZerosLike()
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

// Clear removes all gradient tensors from the store.
// This is useful for clearing gradients after each training step or epoch.
func (s *GradStore[T]) Clear() {
	s.m = make(map[TensorId]*Tensor[T])
}

// Backward computes gradients for all variable tensors contributing to the root tensor.
// Root gradient will be automatically initialized to zeros if not already set.
func Backward[T spark.D](root *Tensor[T], store *GradStore[T]) error {
	if !root.IsVar() {
		return nil // No backpropagation needed for non-variable tensors.
	}

	// Initialize root gradient if not already set.
	if store.Get(root) == nil {
		rootGrad := Ones[T](root.Layout().Shape(), root.Device())
		store.Set(root, rootGrad)
	}

	// Collect variable nodes and build reverse graph using BFS.
	vars := make(map[TensorId]*Tensor[T])
	outputs := make(map[TensorId][]*Tensor[T]) // Maps input ID to its output tensors.
	queue := []*Tensor[T]{root}
	visited := make(map[TensorId]bool)
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		id := node.ID()
		if visited[id] {
			continue
		}
		visited[id] = true
		if !node.IsVar() {
			continue
		}
		vars[id] = node
		if node.Op() == nil {
			continue // Leaf node.
		}
		for _, input := range node.Op().Inputs() {
			if input.IsVar() {
				outputs[input.ID()] = append(outputs[input.ID()], node)
				queue = append(queue, input)
			}
		}
	}

	// Track pending dependencies for each variable.
	pending := make(map[TensorId]int, len(vars))
	for id := range vars {
		pending[id] = len(outputs[id])
	}

	// Propagate gradients backward starting from the root.
	backQueue := []*Tensor[T]{root}
	for len(backQueue) > 0 {
		node := backQueue[0]
		backQueue = backQueue[1:]

		if node.Op() == nil {
			continue // Skip leaf nodes - they have no inputs to propagate to.
		}

		// Compute input gradients for the current node.
		nodeGrad, err := store.GetOrCreate(node)
		if err != nil {
			return err
		}
		inputGrads, err := node.Op().Backward(nodeGrad, node.Op().Inputs())
		if err != nil {
			return err
		}

		// Distribute gradients to input nodes.
		for i, input := range node.Op().Inputs() {
			if !input.IsVar() {
				continue
			}
			inputID := input.ID()
			grad := inputGrads[i]
			existingGrad, err := store.GetOrCreate(input)
			if err != nil {
				return err
			}

			// Accumulate gradient.
			newGrad, err := existingGrad.Add(grad)
			if err != nil {
				return err
			}

			store.SetByID(inputID, newGrad)
			pending[inputID]--
			if pending[inputID] == 0 {
				backQueue = append(backQueue, input)
			}
		}
	}
	return nil
}

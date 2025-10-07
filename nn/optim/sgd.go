package optim

import (
	"errors"
	"fmt"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

var _ Optimizer[float32] = (*SGD[float32])(nil)

// SGD implements Stochastic Gradient Descent optimizer.
type SGD[T spark.D] struct {
	vars []*tensor.Tensor[T]
	lr   float64
}

// NewSGD creates a new SGD optimizer with the given variables and learning rate.
func NewSGD[T spark.D](vars []*tensor.Tensor[T], lr float64) *SGD[T] {
	filtered := make([]*tensor.Tensor[T], 0, len(vars))
	for _, v := range vars {
		if v.IsVar() {
			filtered = append(filtered, v)
		}
	}
	return &SGD[T]{vars: filtered, lr: lr}
}

// LearningRate returns the current learning rate.
func (s *SGD[T]) LearningRate() float64 {
	return s.lr
}

// SetLearningRate sets the learning rate.
func (s *SGD[T]) SetLearningRate(lr float64) {
	s.lr = lr
}

// Add adds a variable to be optimized.
func (s *SGD[T]) Add(v *tensor.Tensor[T]) error {
	if !v.IsVar() {
		return errors.New("not a variable")
	}
	s.vars = append(s.vars, v)
	return nil
}

// Vars returns all variables being optimized.
func (s *SGD[T]) Vars() []*tensor.Tensor[T] {
	return s.vars
}

// Optimize performs backward propagation and an SGD step.
func (s *SGD[T]) Optimize(loss *tensor.Tensor[T]) error {
	store := tensor.NewGradStore[T]()
	if err := tensor.Backward(loss, store); err != nil {
		return fmt.Errorf("backward propagation: %w", err)
	}
	return s.Step(store)
}

// Step performs an SGD optimization step.
func (s *SGD[T]) Step(grads *tensor.GradStore[T]) error {
	for _, v := range s.vars {
		grad := grads.Get(v)
		if grad == nil {
			continue
		}

		lrTensor := tensor.Full[T](s.lr, spark.NewShape(), v.Device())
		scaled, err := grad.Mul(lrTensor)
		if err != nil {
			return fmt.Errorf("scale gradient: %w", err)
		}

		updated, err := v.Sub(scaled)
		if err != nil {
			return fmt.Errorf("update variable: %w", err)
		}

		v.SetStorage(updated.Storage())
	}
	return nil
}

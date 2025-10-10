package optim

import (
	"errors"
	"fmt"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

var _ Optimizer[float32] = (*SGD[float32])(nil)
var _ Optimizer[float64] = (*SGD[float64])(nil)

// SGD implements the Stochastic Gradient Descent optimizer.
// Contrary to the PyTorch implementation of SGD, this version does not support momentum.
type SGD[T spark.D] struct {
	vs []*tensor.Tensor[T] // Variables to optimize
	lr float64             // Learning rate
}

// NewSGD creates a new SGD optimizer with the given variables and learning rate.
func NewSGD[T spark.D](vars []*tensor.Tensor[T], lr float64) (*SGD[T], error) {
	vs := make([]*tensor.Tensor[T], 0, len(vars))
	for _, v := range vars {
		if v.IsVar() {
			vs = append(vs, v)
		}
	}
	return &SGD[T]{vs: vs, lr: lr}, nil
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
	s.vs = append(s.vs, v)
	return nil
}

// Vars returns all variables being optimized.
func (s *SGD[T]) Vars() []*tensor.Tensor[T] {
	return s.vs
}

// Optimize performs backward propagation and an SGD step.
func (s *SGD[T]) Optimize(loss *tensor.Tensor[T]) error {
	gs := tensor.NewGradStore[T]()
	if err := tensor.Backward(loss, gs); err != nil {
		return fmt.Errorf("failed to backward: %w", err)
	}
	return s.Step(gs)
}

// MustOptimize performs backward propagation and an SGD step, panics on error.
func (s *SGD[T]) MustOptimize(loss *tensor.Tensor[T]) {
	if err := s.Optimize(loss); err != nil {
		panic(err)
	}
}

// Step performs an SGD optimization step.
func (s *SGD[T]) Step(gs *tensor.GradStore[T]) error {
	for _, v := range s.vs {
		g := gs.Get(v)
		if g == nil {
			continue
		}
		u, err := g.MulScalar(s.lr)
		if err != nil {
			return fmt.Errorf("failed to scale gradient: %w", err)
		}
		vn, err := v.Sub(u)
		if err != nil {
			return fmt.Errorf("failed to update variable: %w", err)
		}
		v.SetStorage(vn.Storage())
	}
	return nil
}

// MustStep performs an SGD optimization step, panics on error.
func (s *SGD[T]) MustStep(gs *tensor.GradStore[T]) {
	if err := s.Step(gs); err != nil {
		panic(err)
	}
}

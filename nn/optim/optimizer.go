package optim

import (
	"github.com/gocnn/candy"
	"github.com/gocnn/candy/tensor"
)

// Optimizer defines the interface for all optimizers.
type Optimizer[T candy.D] interface {
	// Step performs a single optimization step using the provided gradients.
	Step(grads *tensor.GradStore[T]) error

	// MustStep performs a single optimization step, panics on error.
	MustStep(grads *tensor.GradStore[T])

	// Optimize performs backward propagation and an optimization step.
	Optimize(loss *tensor.Tensor[T]) error

	// MustOptimize performs backward propagation and an optimization step, panics on error.
	MustOptimize(loss *tensor.Tensor[T])

	// LearningRate returns the current learning rate.
	LearningRate() float64

	// SetLearningRate sets the learning rate.
	SetLearningRate(lr float64)

	// Add adds a variable to be optimized.
	Add(variable *tensor.Tensor[T]) error

	// Vars returns all variables being optimized.
	Vars() []*tensor.Tensor[T]
}

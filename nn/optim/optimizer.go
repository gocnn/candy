package optim

import (
	"github.com/gocnn/spark"
)

// Optimizer defines the interface for all optimizers.
type Optimizer[T spark.D] interface {
	// Step performs a single optimization step using the provided gradients.
	Step(grads *spark.GradStore[T]) error

	// Optimize performs backward propagation and an optimization step.
	Optimize(loss *spark.Tensor[T]) error

	// LearningRate returns the current learning rate.
	LearningRate() float64

	// SetLearningRate sets the learning rate.
	SetLearningRate(lr float64)

	// Add adds a variable to be optimized.
	Add(variable *spark.Tensor[T]) error

	// Vars returns all variables being optimized.
	Vars() []*spark.Tensor[T]
}

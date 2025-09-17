package op

import (
	"github.com/qntx/spark"
)

var _ Op[float32] = (*OpAdd[float32])(nil)
var _ Op[float64] = (*OpAdd[float64])(nil)

// OpAdd is an operator to perform element-wise sum over two values.
// y = x1 + x2
type OpAdd[T spark.D] struct {
	x1 spark.Tensor[T]
	x2 spark.Tensor[T]
}

// NewAdd returns a new OpAdd Function.
func NewAdd[T spark.D](x1, x2 spark.Tensor[T]) *OpAdd[T] {
	return &OpAdd[T]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *OpAdd[T]) Operands() []spark.Tensor[T] {
	return []spark.Tensor[T]{r.x1, r.x2}
}

// Forward computes the output of the function.
func (r *OpAdd[T]) Forward() (tensor spark.Tensor[T], err error) {
	return r.x1.Add(r.x2)
}

// Backward computes the backward pass.
func (r *OpAdd[T]) Backward(gy spark.Tensor[T]) (err error) {
	if r.x1.RequiresGrad() && gy.Shape().Equals(r.x1.Shape()) {
		r.x1.AccGrad(gy)
	}
	if r.x2.RequiresGrad() && gy.Shape().Equals(r.x2.Shape()) {
		r.x2.AccGrad(gy)
	}

	return nil
}

// Add returns a new operator node as a result of the gradfn.Add function.
// As special case, the first node may be null.
// This help to keep the code as concise as possible e.g. during accumulation.
func Add[T spark.D](x1 spark.Tensor[T], x2 spark.Tensor[T]) spark.Tensor[T] {
	return NewOperator(NewAdd(x1, x2)).Run()
}

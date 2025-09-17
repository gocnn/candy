package op

import (
	"github.com/qntx/spark"
)

// Op represents an operator with automatic differentiation features.
// It's used to define a new operator.
type Op[T spark.D] interface {
	// Forward computes the output of the function.
	Forward() (tensor spark.Tensor[T], err error)
	// Backward computes the backward pass given the gradient of the output.
	Backward(gy spark.Tensor[T]) (err error)
	// Operands returns the list of operands.
	Operands() []spark.Tensor[T]
}

// Operator is a type of node.
// It's used to represent a function with automatic differentiation features.
type Operator[T spark.D] struct {
	// value stores the results of a forward evaluation, as mat.Matrix.
	// It's set by executeForward() goroutine.
	// Use the Value() method to get the actual value.
	// It also contains the accumulated gradients. Use the Grad() method to get them.
	value spark.Tensor[T]
	// op is the operator to be executed.
	op Op[T]
}

// NewOperator returns a new operator node.
func NewOperator[T spark.D](op Op[T]) *Operator[T] {
	return &Operator[T]{
		op: op,
	}
}

func (o *Operator[T]) Run() *Operator[T] {
	v, err := o.op.Forward()
	if err != nil {
		panic(err)
	}
	o.value = v

	return o
}

package op

import (
	"sync"

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

// backwardState is an enumeration type associated to an Operator, to keep
// track of its visited status among different backpropagation phases.
type backwardState = uint32

const (
	// idle reports that gradient propagation is not pending for an
	// operator node.
	//
	// It's the default zero-value state of an operator, and it's also the
	// final value set from the backward step once gradients have been
	// propagated.
	//
	// As soon as a backward operation is performed, the status will change to
	// pending.
	idle backwardState = iota
	// pending is set on an operator node from the preparatory phase
	// of the backward step.
	// It reports that the node has been marked as a candidate for gradients
	// propagation and the number of pendingGrads has been computed.
	//
	// The next logical state is ongoing.
	pending
	// ongoing is set on an operator node from the core phase of the
	// backward step. It reports that the node has been visited once for
	// performing its Operator.backward method.
	//
	// This status remains set until the gradients of all dependents have been
	// resolved, and the node's own gradients have been propagated too.
	// After that, the status is set back to idle.
	ongoing
)

// Operator is a type of node.
// It's used to represent a function with automatic differentiation features.
type Operator[T spark.D] struct {
	// value stores the results of a forward evaluation, as mat.Matrix.
	// It's set by executeForward() goroutine.
	// Use the Value() method to get the actual value.
	// It also contains the accumulated gradients. Use the Grad() method to get them.
	value spark.Tensor[T]
	// onceOperands is used to initialize the operands only once.
	onceOperands sync.Once
	// AutoGradFunction's operands are memoized here after the first request.
	operands []spark.Tensor[T]
	// backwardPass is the backward function to be executed.
	op Op[T]
	// broadcast is the channel used to broadcast the result of the forward pass.
	broadcast chan struct{}
	// broadcastGrad is the channel used to broadcast the result of the backward pass.
	// It is initialized only when the backward pass is performed.
	broadcastGrad chan struct{}
	// pendingGrads is the number of pending gradients to be accumulated. (default: 0)
	pendingGrads int64
	// onceRequiresGrad is used to initialize the requiresGrad only once.
	onceRequiresGrad sync.Once
	// requiresGrad is a flag that indicates whether the operator requires gradients.
	// Use the RequiresGrad() method to get the actual value.
	requiresGrad bool
	// backwardState is the state of the backward pass.
	backwardState backwardState
}

// NewOperator returns a new operator node.
func NewOperator[T spark.D](op Op[T]) *Operator[T] {
	return &Operator[T]{
		op: op,
	}
}

func (r *Operator[T]) Run() spark.Tensor[T] {
	return r.value
}

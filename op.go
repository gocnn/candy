package spark

import "fmt"

type ForwardFunc[T D] func([]*Tensor[T]) (*Tensor[T], error)
type BackwardFunc[T D] func(*Tensor[T], []*Tensor[T]) ([]*Tensor[T], error)

type Op[T D] struct {
	inputs   []*Tensor[T]
	backward BackwardFunc[T]
}

func (op *Op[T]) Inputs() []*Tensor[T] {
	return op.inputs
}

func (op *Op[T]) Backward(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	return op.backward(outputGrad, inputs)
}

// ApplyOp applies a forward function and builds the computation graph.
// This is the core function that enables users to define custom operations.
//
// Parameters:
//   - inputs: Input tensors for the operation
//   - forwardFn: Forward pass function that computes the result
//   - backwardFn: Backward pass function that computes gradients
//
// Returns:
//   - Result tensor with automatic differentiation support
func ApplyOp[T D](inputs []*Tensor[T], forwardFn ForwardFunc[T], backwardFn BackwardFunc[T]) (*Tensor[T], error) {
	// Execute forward pass
	result, err := forwardFn(inputs)
	if err != nil {
		return nil, err
	}

	// Check if any input requires gradient
	needsGrad := false
	for _, input := range inputs {
		if input.IsVar() {
			needsGrad = true
			break
		}
	}

	// Build computation graph if needed
	if needsGrad {
		result.isVar = true
		result.op = &Op[T]{
			inputs:   inputs,
			backward: backwardFn,
		}
	}

	return result, nil
}

// AddForward computes element-wise addition: c = a + b
func AddForward[T D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("AddForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resultLayout := Contiguous(a.layout.Shape())
	resultStorage, err := a.storage.Add(b.storage, a.layout, b.layout, &resultLayout)
	if err != nil {
		return nil, err
	}

	return &Tensor[T]{
		id:      NewId(),
		storage: resultStorage,
		layout:  &resultLayout,
		dtype:   a.dtype,
		device:  a.device,
	}, nil
}

// AddBackward computes gradients for addition: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
func AddBackward[T D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	// For addition, gradient flows unchanged to both inputs
	ga := outputGrad
	gb := outputGrad
	return []*Tensor[T]{ga, gb}, nil
}

func (a *Tensor[T]) Add(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, AddForward[T], AddBackward[T])
}

// MulForward computes element-wise multiplication: c = a * b
func MulForward[T D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MulForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resultLayout := Contiguous(a.layout.Shape())
	resultStorage, err := a.storage.Mul(b.storage, a.layout, b.layout, &resultLayout)
	if err != nil {
		return nil, err
	}

	return &Tensor[T]{
		id:      NewId(),
		storage: resultStorage,
		layout:  &resultLayout,
		dtype:   a.dtype,
		device:  a.device,
	}, nil
}

// MulBackward computes gradients for multiplication: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
func MulBackward[T D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MulBackward expects 2 inputs, got %d", len(inputs))
	}

	// ∂(a*b)/∂a = outputGrad * b
	ga, err := outputGrad.Mul(inputs[1])
	if err != nil {
		return nil, err
	}

	// ∂(a*b)/∂b = outputGrad * a
	gb, err := outputGrad.Mul(inputs[0])
	if err != nil {
		return nil, err
	}

	return []*Tensor[T]{ga, gb}, nil
}

func (a *Tensor[T]) Mul(b *Tensor[T]) (*Tensor[T], error) {
	return ApplyOp([]*Tensor[T]{a, b}, MulForward[T], MulBackward[T])
}

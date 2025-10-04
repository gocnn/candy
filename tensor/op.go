package tensor

import (
	"fmt"

	"github.com/gocnn/spark"
)

type ForwardFunc[T spark.D] func([]*Tensor[T]) (*Tensor[T], error)
type BackwardFunc[T spark.D] func(*Tensor[T], []*Tensor[T]) ([]*Tensor[T], error)

type Op[T spark.D] struct {
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
func ApplyOp[T spark.D](inputs []*Tensor[T], forwardFn ForwardFunc[T], backwardFn BackwardFunc[T]) (*Tensor[T], error) {
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
func AddForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("AddForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resLayout := spark.Contiguous(a.layout.Shape())
	resStorage, err := a.storage.Add(b.storage, a.layout, b.layout, resLayout)
	if err != nil {
		return nil, err
	}

	return NewFromStorage(resStorage, resLayout, a.dtype, a.device), nil
}

// AddBackward computes gradients for addition: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
func AddBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	// For addition, gradient flows unchanged to both inputs
	ga := outputGrad
	gb := outputGrad
	return []*Tensor[T]{ga, gb}, nil
}

// SubForward computes element-wise subtraction: c = a - b
func SubForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("SubForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resLayout := spark.Contiguous(a.layout.Shape())
	resStorage, err := a.storage.Sub(b.storage, a.layout, b.layout, resLayout)
	if err != nil {
		return nil, err
	}

	return NewFromStorage(resStorage, resLayout, a.dtype, a.device), nil
}

// SubBackward computes gradients for subtraction: ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
func SubBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("SubBackward expects 2 inputs, got %d", len(inputs))
	}

	// ∂(a-b)/∂a = outputGrad * 1 = outputGrad
	ga := outputGrad

	// ∂(a-b)/∂b = outputGrad * (-1) = -outputGrad
	negOne := Full[T](-1.0, outputGrad.layout.Shape(), outputGrad.device)
	gbStorage, err := outputGrad.storage.Mul(negOne.storage, outputGrad.layout, negOne.layout, outputGrad.layout)
	if err != nil {
		return nil, err
	}

	gb := NewFromStorage(gbStorage, outputGrad.layout, outputGrad.dtype, outputGrad.device)
	return []*Tensor[T]{ga, gb}, nil
}

// MulForward computes element-wise multiplication: c = a * b
func MulForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MulForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resLayout := spark.Contiguous(a.layout.Shape())
	resStorage, err := a.storage.Mul(b.storage, a.layout, b.layout, resLayout)
	if err != nil {
		return nil, err
	}

	return NewFromStorage(resStorage, resLayout, a.dtype, a.device), nil
}

// MulBackward computes gradients for multiplication: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
func MulBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MulBackward expects 2 inputs, got %d", len(inputs))
	}

	// ∂(a*b)/∂a = outputGrad * b
	gas, err := outputGrad.storage.Mul(inputs[1].storage, outputGrad.layout, inputs[1].layout, outputGrad.layout)
	if err != nil {
		return nil, err
	}

	// ∂(a*b)/∂b = outputGrad * a
	gbs, err := outputGrad.storage.Mul(inputs[0].storage, outputGrad.layout, inputs[0].layout, outputGrad.layout)
	if err != nil {
		return nil, err
	}

	ga := NewFromStorage(gas, outputGrad.layout, outputGrad.dtype, outputGrad.device)
	gb := NewFromStorage(gbs, outputGrad.layout, outputGrad.dtype, outputGrad.device)
	return []*Tensor[T]{ga, gb}, nil
}

// SqrtForward computes element-wise square root: c = sqrt(a)
func SqrtForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SqrtForward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0]
	resStorage, err := a.storage.Sqrt(a.layout)
	if err != nil {
		return nil, err
	}

	return NewFromStorage(resStorage, a.layout.Clone(), a.dtype, a.device), nil
}

// SqrtBackward computes gradients for square root: ∂sqrt(a)/∂a = 1/(2*sqrt(a))
func SqrtBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SqrtBackward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0]

	// Compute sqrt(a) first
	sqrtA, err := a.storage.Sqrt(a.layout)
	if err != nil {
		return nil, err
	}

	// Create tensor with value 2.0
	two := Full[T](2.0, a.layout.Shape(), a.device)

	// Create layout for sqrtA storage (same as input a)
	sqrtALayout := a.layout.Clone()

	// Compute 2 * sqrt(a)
	denominator, err := two.storage.Mul(sqrtA, two.layout, sqrtALayout, two.layout)
	if err != nil {
		return nil, err
	}

	// Compute 1 / (2 * sqrt(a))
	one := Ones[T](a.layout.Shape(), a.device)

	// Create layout for denominator storage (same as two.layout)
	denominatorLayout := two.layout.Clone()
	derivative, err := one.storage.Div(denominator, one.layout, denominatorLayout, one.layout)
	if err != nil {
		return nil, err
	}

	// Create layout for derivative storage (same as one.layout)
	derivativeLayout := one.layout.Clone()

	// Multiply by output gradient
	gaStorage, err := outputGrad.storage.Mul(derivative, outputGrad.layout, derivativeLayout, outputGrad.layout)
	if err != nil {
		return nil, err
	}

	return []*Tensor[T]{NewFromStorage(gaStorage, outputGrad.layout, outputGrad.dtype, outputGrad.device)}, nil
}

// DivForward computes element-wise division: c = a / b
func DivForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("DivForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resLayout := spark.Contiguous(a.layout.Shape())
	resStorage, err := a.storage.Div(b.storage, a.layout, b.layout, resLayout)
	if err != nil {
		return nil, err
	}

	return NewFromStorage(resStorage, resLayout, a.dtype, a.device), nil
}

// DivBackward computes gradients for division: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/(b²)
func DivBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("DivBackward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]

	// ∂(a/b)/∂a = outputGrad * (1/b) = outputGrad / b
	gaStorage, err := outputGrad.storage.Div(b.storage, outputGrad.layout, b.layout, outputGrad.layout)
	if err != nil {
		return nil, err
	}

	ga := NewFromStorage(gaStorage, outputGrad.layout, outputGrad.dtype, outputGrad.device)

	// ∂(a/b)/∂b = outputGrad * (-a/b²)
	// First compute b²
	bSquaredStorage, err := b.storage.Mul(b.storage, b.layout, b.layout, b.layout)
	if err != nil {
		return nil, err
	}

	bSquaredLayout := b.layout.Clone()

	// Then compute a/b²
	aDivBSquaredStorage, err := a.storage.Div(bSquaredStorage, a.layout, bSquaredLayout, a.layout)
	if err != nil {
		return nil, err
	}

	aDivBSquaredLayout := a.layout.Clone()

	// Then multiply by -1
	negOne := Full[T](-1.0, a.layout.Shape(), a.device)

	negADivBSquaredStorage, err := aDivBSquaredStorage.Mul(negOne.storage, aDivBSquaredLayout, negOne.layout, aDivBSquaredLayout)
	if err != nil {
		return nil, err
	}

	// Finally multiply by outputGrad
	gbStorage, err := outputGrad.storage.Mul(negADivBSquaredStorage, outputGrad.layout, aDivBSquaredLayout, outputGrad.layout)
	if err != nil {
		return nil, err
	}

	gb := NewFromStorage(gbStorage, outputGrad.layout, outputGrad.dtype, outputGrad.device)
	return []*Tensor[T]{ga, gb}, nil
}

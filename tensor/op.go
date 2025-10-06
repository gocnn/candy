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

	return NewFrom(resStorage, resLayout, a.dtype, a.device), nil
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

	return NewFrom(resStorage, resLayout, a.dtype, a.device), nil
}

// SubBackward computes gradients for subtraction: ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
func SubBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("SubBackward expects 2 inputs, got %d", len(inputs))
	}

	// ∂(a-b)/∂a = outputGrad
	ga := outputGrad

	// ∂(a-b)/∂b = -outputGrad
	negOne := Full[T](-1.0, outputGrad.Shape(), outputGrad.Device())
	gb, err := negOne.Mul(outputGrad)
	if err != nil {
		return nil, fmt.Errorf("computing gradient for b: %w", err)
	}

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

	return NewFrom(resStorage, resLayout, a.dtype, a.device), nil
}

// MulBackward computes gradients for multiplication: ∂(a*b)/∂a = outputGrad*b, ∂(a*b)/∂b = outputGrad*a
func MulBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MulBackward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0].Detach(), inputs[1].Detach()

	// ∂(a*b)/∂a = outputGrad * b
	ga, err := outputGrad.Mul(b)
	if err != nil {
		return nil, fmt.Errorf("computing gradient for a: %w", err)
	}

	// ∂(a*b)/∂b = outputGrad * a
	gb, err := outputGrad.Mul(a)
	if err != nil {
		return nil, fmt.Errorf("computing gradient for b: %w", err)
	}

	return []*Tensor[T]{ga, gb}, nil
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

	return NewFrom(resStorage, resLayout, a.dtype, a.device), nil
}

// DivBackward computes gradients for division: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/(b²)
func DivBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("DivBackward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0].Detach(), inputs[1].Detach()

	// ∂(a/b)/∂a = outputGrad / b
	ga, err := outputGrad.Div(b)
	if err != nil {
		return nil, fmt.Errorf("computing ga: %w", err)
	}

	// ∂(a/b)/∂b = -outputGrad * a / b²
	bSquared, err := b.Mul(b)
	if err != nil {
		return nil, fmt.Errorf("computing b²: %w", err)
	}

	aDivBSquared, err := a.Div(bSquared)
	if err != nil {
		return nil, fmt.Errorf("computing a/b²: %w", err)
	}

	negOne := Full[T](-1.0, a.Shape(), a.Device())
	negADivBSquared, err := negOne.Mul(aDivBSquared)
	if err != nil {
		return nil, fmt.Errorf("negating a/b²: %w", err)
	}

	gb, err := outputGrad.Mul(negADivBSquared)
	if err != nil {
		return nil, fmt.Errorf("computing gb: %w", err)
	}

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

	return NewFrom(resStorage, a.layout.Clone(), a.dtype, a.device), nil
}

// SqrtBackward computes gradients for square root: ∂sqrt(a)/∂a = 1/(2*sqrt(a))
func SqrtBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SqrtBackward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0].Detach()

	// sqrt(a)
	sqrtA, err := a.Sqrt()
	if err != nil {
		return nil, err
	}

	// 2 * sqrt(a)
	two := Full[T](2.0, a.Shape(), a.Device())
	denominator, err := two.Mul(sqrtA)
	if err != nil {
		return nil, err
	}

	// outputGrad / (2 * sqrt(a))
	inputGrad, err := outputGrad.Div(denominator)
	if err != nil {
		return nil, err
	}

	return []*Tensor[T]{inputGrad}, nil
}

// SumForward computes the sum along the specified dimensions
func SumForward[T spark.D](dims []int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("SumForward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0]

		dims, err := spark.ResolveAxes(dims, input.Shape())
		if err != nil {
			return nil, fmt.Errorf("sum dimensions: %w", err)
		}

		outputDims := make([]int, len(input.Dims()))
		copy(outputDims, input.Dims())
		for _, dim := range dims {
			outputDims[dim] = 1
		}
		outputShape := spark.NewShapeFrom(outputDims)
		outputLayout := spark.Contiguous(outputShape)

		resStorage, err := input.storage.Sum(input.layout, dims)
		if err != nil {
			return nil, err
		}

		return NewFrom(resStorage, outputLayout, input.dtype, input.device), nil
	}
}

// SumBackward computes gradients for sum operation
func SumBackward[T spark.D](dims []int) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("SumBackward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0].Detach()

		// For sum operation, gradient flows back unchanged to all elements
		// that contributed to each sum. This means we need to broadcast
		// the output gradient back to the input shape.
		inputGrad, err := outputGrad.BroadcastAs(input.Shape())
		if err != nil {
			return nil, fmt.Errorf("sum backward broadcast: %w", err)
		}

		return []*Tensor[T]{inputGrad}, nil
	}
}

// BroadcastAddForward computes the broadcasted addition: result = broadcast(a) + broadcast(b).
func BroadcastAddForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("invalid input count: expected 2, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]

	bcastShape, err := a.Shape().BroadcastShapeBinaryOp(b.Shape())
	if err != nil {
		return nil, fmt.Errorf("compute broadcast shape: %w", err)
	}

	aBcast, err := a.BroadcastAs(bcastShape)
	if err != nil {
		return nil, fmt.Errorf("broadcast a: %w", err)
	}
	bBcast, err := b.BroadcastAs(bcastShape)
	if err != nil {
		return nil, fmt.Errorf("broadcast b: %w", err)
	}

	resultLayout := spark.Contiguous(bcastShape)
	result, err := aBcast.storage.Add(bBcast.storage, aBcast.layout, bBcast.layout, resultLayout)
	if err != nil {
		return nil, fmt.Errorf("add: %w", err)
	}

	return NewFrom(result, resultLayout, a.dtype, a.device), nil
}

// BroadcastAddBackward computes gradients for broadcasted addition: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
func BroadcastAddBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("BroadcastAddBackward expects 2 inputs, got %d", len(inputs))
	}

	return []*Tensor[T]{outputGrad, outputGrad}, nil
}

// SqueezeForward creates squeeze forward
func SqueezeForward[T spark.D](dim int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("SqueezeForward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0]
		shape := input.Shape()
		resolvedDim, err := spark.ResolveAxis(dim, shape.Rank())
		if err != nil {
			return nil, fmt.Errorf("squeeze: %w", err)
		}

		// Return shallow copy if dimension is not 1 (PyTorch semantics)
		if shape.Dim(resolvedDim) != 1 {
			return input, nil
		}

		// Build new shape and stride, excluding the specified dimension
		newDims := make([]int, 0, shape.Rank()-1)
		newStrides := make([]int, 0, shape.Rank()-1)
		for i, size := range shape.Dims() {
			if i == resolvedDim {
				continue
			}
			newDims = append(newDims, size)
			newStrides = append(newStrides, input.Stride()[i])
		}

		newShape := spark.NewShapeFrom(newDims)
		newLayout := spark.NewLayout(newShape, newStrides, input.layout.StartOffset())

		return NewFrom(input.storage, newLayout, input.dtype, input.device), nil
	}
}

// SqueezeBackward creates squeeze backward
func SqueezeBackward[T spark.D](dim int) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("SqueezeBackward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0].Detach()
		inputShape := input.Shape()
		resolvedDim, err := spark.ResolveAxis(dim, inputShape.Rank())
		if err != nil {
			return nil, fmt.Errorf("squeeze backward: %w", err)
		}

		// If the dimension is not 1, pass gradient directly
		if inputShape.Dim(resolvedDim) != 1 {
			return []*Tensor[T]{outputGrad}, nil
		}

		// Squeeze's backward is Unsqueeze: restore the removed dimension
		inputGrad, err := outputGrad.Unsqueeze(resolvedDim)
		if err != nil {
			return nil, fmt.Errorf("squeeze backward unsqueeze: %w", err)
		}

		return []*Tensor[T]{inputGrad}, nil
	}
}

// UnsqueezeForward creates unsqueeze forward
func UnsqueezeForward[T spark.D](dim int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("UnsqueezeForward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0]
		shape := input.Shape()
		rank := shape.Rank()

		if dim < 0 {
			dim += rank + 1
		}
		if dim < 0 || dim > rank {
			return nil, fmt.Errorf("unsqueeze: dim out of range [-%d, %d], got %d", rank+1, rank, dim)
		}

		newDims := append(make([]int, 0, rank+1), shape.Dims()[:dim]...)
		newDims = append(newDims, 1)
		newDims = append(newDims, shape.Dims()[dim:]...)

		newStrides := append(make([]int, 0, rank+1), input.Stride()[:dim]...)
		stride := 1
		if dim < rank {
			stride = input.Stride()[dim]
		}
		newStrides = append(newStrides, stride)
		newStrides = append(newStrides, input.Stride()[dim:]...)

		newShape := spark.NewShapeFrom(newDims)
		newLayout := spark.NewLayout(newShape, newStrides, input.layout.StartOffset())

		return NewFrom(input.storage, newLayout, input.dtype, input.device), nil
	}
}

// UnsqueezeBackward creates unsqueeze backward
func UnsqueezeBackward[T spark.D](dim int) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("UnsqueezeBackward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0].Detach()
		inputShape := input.Shape()
		rank := inputShape.Rank()

		originalDim := dim
		if originalDim < 0 {
			originalDim += rank + 1
		}

		inputGrad, err := outputGrad.Squeeze(originalDim)
		if err != nil {
			return nil, fmt.Errorf("unsqueeze backward squeeze: %w", err)
		}

		return []*Tensor[T]{inputGrad}, nil
	}
}

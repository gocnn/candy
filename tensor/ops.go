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

// AffineForward computes affine transformation: y = scale * x + bias
func AffineForward[T spark.D](scale, bias float64) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("AffineForward expects 1 input, got %d", len(inputs))
		}

		a := inputs[0]
		scaleT := T(scale)
		biasT := T(bias)

		resStorage, err := a.storage.Affine(a.layout, scaleT, biasT)
		if err != nil {
			return nil, err
		}

		return NewFrom(resStorage, a.layout.Clone(), a.dtype, a.device), nil
	}
}

// AffineBackward computes gradients for affine transformation: ∂(scale*x + bias)/∂x = scale
func AffineBackward[T spark.D](scale, bias float64) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("AffineBackward expects 1 input, got %d", len(inputs))
		}

		// ∂L/∂x = ∂L/∂y * ∂y/∂x = outputGrad * scale
		scaleTensor := Full[T](T(scale), outputGrad.Shape(), outputGrad.Device())
		inputGrad, err := outputGrad.Mul(scaleTensor)
		if err != nil {
			return nil, err
		}

		return []*Tensor[T]{inputGrad}, nil
	}
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
	gb, err := outputGrad.Neg()
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
	bSquared, err := b.Sqr()
	if err != nil {
		return nil, fmt.Errorf("computing b²: %w", err)
	}

	aDivBSquared, err := a.Div(bSquared)
	if err != nil {
		return nil, fmt.Errorf("computing a/b²: %w", err)
	}

	negADivBSquared, err := aDivBSquared.Neg()
	if err != nil {
		return nil, fmt.Errorf("negating a/b²: %w", err)
	}

	gb, err := outputGrad.Mul(negADivBSquared)
	if err != nil {
		return nil, fmt.Errorf("computing gb: %w", err)
	}

	return []*Tensor[T]{ga, gb}, nil
}

// EqForward computes element-wise equality: result = (a == b)
func EqForward[T spark.D](inputs []*Tensor[T]) (*Tensor[uint8], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("EqForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resLayout := spark.Contiguous(a.Shape())
	resStorage, err := a.storage.Eq(b.storage, a.layout, b.layout, resLayout)
	if err != nil {
		return nil, err
	}

	return New(resStorage.Data(), resLayout.Shape(), a.Device()), nil
}

// NeForward computes element-wise not-equal: result = (a != b)
func NeForward[T spark.D](inputs []*Tensor[T]) (*Tensor[uint8], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("NeForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resLayout := spark.Contiguous(a.Shape())
	resStorage, err := a.storage.Ne(b.storage, a.layout, b.layout, resLayout)
	if err != nil {
		return nil, err
	}

	return New(resStorage.Data(), resLayout.Shape(), a.Device()), nil
}

// LtForward computes element-wise less-than: result = (a < b)
func LtForward[T spark.D](inputs []*Tensor[T]) (*Tensor[uint8], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("LtForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resLayout := spark.Contiguous(a.Shape())
	resStorage, err := a.storage.Lt(b.storage, a.layout, b.layout, resLayout)
	if err != nil {
		return nil, err
	}

	return New(resStorage.Data(), resLayout.Shape(), a.Device()), nil
}

// LeForward computes element-wise less-than-or-equal: result = (a <= b)
func LeForward[T spark.D](inputs []*Tensor[T]) (*Tensor[uint8], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("LeForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resLayout := spark.Contiguous(a.Shape())
	resStorage, err := a.storage.Le(b.storage, a.layout, b.layout, resLayout)
	if err != nil {
		return nil, err
	}

	return New(resStorage.Data(), resLayout.Shape(), a.Device()), nil
}

// GtForward computes element-wise greater-than: result = (a > b)
func GtForward[T spark.D](inputs []*Tensor[T]) (*Tensor[uint8], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("GtForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resLayout := spark.Contiguous(a.Shape())
	resStorage, err := a.storage.Gt(b.storage, a.layout, b.layout, resLayout)
	if err != nil {
		return nil, err
	}

	return New(resStorage.Data(), resLayout.Shape(), a.Device()), nil
}

// GeForward computes element-wise greater-than-or-equal: result = (a >= b)
func GeForward[T spark.D](inputs []*Tensor[T]) (*Tensor[uint8], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("GeForward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	resLayout := spark.Contiguous(a.Shape())
	resStorage, err := a.storage.Ge(b.storage, a.layout, b.layout, resLayout)
	if err != nil {
		return nil, err
	}

	return New(resStorage.Data(), resLayout.Shape(), a.Device()), nil
}

// MatMulForward implements the forward pass for matrix multiplication
func MatMulForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("MatMul requires exactly 2 inputs, got %d", len(inputs))
		}

		lhs, rhs := inputs[0], inputs[1]

		if lhs.Rank() < 2 || rhs.Rank() < 2 {
			return nil, fmt.Errorf("MatMul requires tensors with at least 2 dimensions")
		}

		lhsDims, rhsDims := lhs.Dims(), rhs.Dims()
		if len(lhsDims) != len(rhsDims) {
			return nil, fmt.Errorf("MatMul requires tensors with same rank, got %d vs %d", len(lhsDims), len(rhsDims))
		}

		lhsBatch := spark.NewShapeFrom(lhsDims[:len(lhsDims)-2])
		rhsBatch := spark.NewShapeFrom(rhsDims[:len(rhsDims)-2])
		if !lhsBatch.Equal(rhsBatch) {
			return nil, fmt.Errorf("MatMul batch dimensions mismatch: %v vs %v", lhsBatch, rhsBatch)
		}

		m, k1 := lhsDims[len(lhsDims)-2], lhsDims[len(lhsDims)-1]
		k2, n := rhsDims[len(rhsDims)-2], rhsDims[len(rhsDims)-1]

		if k1 != k2 {
			return nil, fmt.Errorf("MatMul dimension mismatch: %d != %d", k1, k2)
		}

		batchSize := lhsBatch.ElemCount()
		resultDims := append(lhsBatch.Dims(), m, n)

		resultStorage, err := lhs.storage.MatMul(
			lhs.layout,
			rhs.storage,
			rhs.layout,
			batchSize, m, n, k1,
		)
		if err != nil {
			return nil, err
		}

		return NewFrom(resultStorage, spark.Contiguous(spark.NewShapeFrom(resultDims)), lhs.dtype, lhs.device), nil
	}
}

// MatMulBackward implements the backward pass for matrix multiplication
func MatMulBackward[T spark.D]() BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		lhs, rhs := inputs[0].Detach(), inputs[1].Detach()

		// ∂L/∂A = grad_C × B^T
		rhsT, err := rhs.Transpose(-1, -2)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose rhs for lhs gradient: %w", err)
		}

		lhsGrad, err := outputGrad.MatMul(rhsT)
		if err != nil {
			return nil, fmt.Errorf("failed to compute lhs gradient: %w", err)
		}

		// ∂L/∂B = A^T × grad_C
		lhsT, err := lhs.Transpose(-1, -2)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose lhs for rhs gradient: %w", err)
		}

		rhsGrad, err := lhsT.MatMul(outputGrad)
		if err != nil {
			return nil, fmt.Errorf("failed to compute rhs gradient: %w", err)
		}

		return []*Tensor[T]{lhsGrad, rhsGrad}, nil
	}
}

// Conv1dForward computes 1D convolution
func Conv1dForward[T spark.D](params *spark.Conv1DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("Conv1dForward expects 2 inputs, got %d", len(inputs))
		}

		input, kernel := inputs[0], inputs[1]

		if input.Rank() != 3 || kernel.Rank() != 3 {
			return nil, fmt.Errorf("input and kernel must be 3D tensors")
		}

		outLen := params.OutLen()
		outputShape := spark.NewShapeFrom([]int{params.Batch, params.OutCh, outLen})
		outputLayout := spark.Contiguous(outputShape)

		resStorage, err := input.storage.Conv1d(input.layout, kernel.storage, kernel.layout, params)
		if err != nil {
			return nil, err
		}

		return NewFrom(resStorage, outputLayout, input.dtype, input.device), nil
	}
}

// Conv1dBackward computes gradients for 1D convolution
func Conv1dBackward[T spark.D](params *spark.Conv1DParams) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		input, kernel := inputs[0].Detach(), inputs[1].Detach()

		gradLIn := outputGrad.Shape().Dims()[2]
		kSize := params.KSize
		outSize := (gradLIn-1)*params.Stride + params.Dilate*(kSize-1) + 1 - 2*params.Pad
		outPadding := params.InLen - outSize

		inputGradParams := &spark.ConvT1DParams{
			Batch:  params.Batch,
			InCh:   params.OutCh,
			InLen:  gradLIn,
			OutCh:  params.InCh,
			KSize:  params.KSize,
			Stride: params.Stride,
			Pad:    params.Pad,
			OutPad: outPadding,
			Dilate: params.Dilate,
		}

		inputGrad, err := outputGrad.ConvTranspose1d(kernel, inputGradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute input gradient: %v", err)
		}

		inputT, err := input.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		outputGradT, err := outputGrad.Transpose(0, 1)
		if err != nil {
			return nil, err
		}

		outGradLen := outputGrad.Shape().Dims()[2]
		kernelGradParams := &spark.Conv1DParams{
			Batch:  params.InCh,
			InCh:   params.Batch,
			InLen:  params.InLen,
			OutCh:  params.OutCh,
			KSize:  outGradLen,
			Stride: params.Stride,
			Pad:    params.Pad,
			Dilate: params.Dilate,
		}

		kernelGradT, err := inputT.Conv1d(outputGradT, kernelGradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute kernel gradient: %v", err)
		}

		kernelGrad, err := kernelGradT.Transpose(0, 1)
		if err != nil {
			return nil, err
		}

		return []*Tensor[T]{inputGrad, kernelGrad}, nil
	}
}

// ConvTranspose1dForward computes 1D transposed convolution
func ConvTranspose1dForward[T spark.D](params *spark.ConvT1DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("ConvTranspose1dForward expects 2 inputs, got %d", len(inputs))
		}

		input, kernel := inputs[0], inputs[1]

		if input.Rank() != 3 || kernel.Rank() != 3 {
			return nil, fmt.Errorf("input and kernel must be 3D tensors")
		}
		outLen := params.OutLen()
		outputShape := spark.NewShapeFrom([]int{params.Batch, params.OutCh, outLen})
		outputLayout := spark.Contiguous(outputShape)

		resStorage, err := input.storage.ConvTranspose1d(input.layout, kernel.storage, kernel.layout, params)
		if err != nil {
			return nil, err
		}

		return NewFrom(resStorage, outputLayout, input.dtype, input.device), nil
	}
}

// ConvTranspose1dBackward computes gradients for 1D transposed convolution
// Based on Rust Candle implementation
func ConvTranspose1dBackward[T spark.D](params *spark.ConvT1DParams) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("ConvTranspose1dBackward expects 2 inputs, got %d", len(inputs))
		}

		input, kernel := inputs[0].Detach(), inputs[1].Detach()

		inputGradParams := &spark.Conv1DParams{
			Batch:  params.Batch,
			InCh:   params.OutCh,
			InLen:  outputGrad.Shape().Dims()[2],
			OutCh:  params.InCh,
			KSize:  params.KSize,
			Stride: params.Stride,
			Pad:    params.Pad,
			Dilate: params.Dilate,
		}

		inputGrad, err := outputGrad.Conv1d(kernel, inputGradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute input gradient: %v", err)
		}

		outputGradT, err := outputGrad.Transpose(0, 1)
		if err != nil {
			return nil, err
		}

		inputT, err := input.Transpose(0, 1)
		if err != nil {
			return nil, err
		}

		outGradLen := outputGrad.Shape().Dims()[2]

		kernelGradParams := &spark.Conv1DParams{
			Batch:  params.OutCh,
			InCh:   params.Batch,
			InLen:  params.InLen,
			OutCh:  params.InCh,
			KSize:  outGradLen,
			Stride: params.Dilate,
			Pad:    params.Pad,
			Dilate: params.Stride,
		}

		kernelGradT, err := outputGradT.Conv1d(inputT, kernelGradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute kernel gradient: %v", err)
		}

		kernelGrad, err := kernelGradT.Transpose(0, 1)
		if err != nil {
			return nil, err
		}

		return []*Tensor[T]{inputGrad, kernelGrad}, nil
	}
}

// Conv2dForward computes 2D convolution
func Conv2dForward[T spark.D](params *spark.Conv2DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("Conv2dForward expects 2 inputs, got %d", len(inputs))
		}

		input, kernel := inputs[0], inputs[1]

		if input.Rank() != 4 || kernel.Rank() != 4 {
			return nil, fmt.Errorf("input and kernel must be 4D tensors")
		}

		hOut := params.OutH()
		wOut := params.OutW()
		outputShape := spark.NewShapeFrom([]int{params.Batch, params.OutCh, hOut, wOut})
		outputLayout := spark.Contiguous(outputShape)

		resStorage, err := input.storage.Conv2d(input.layout, kernel.storage, kernel.layout, params)
		if err != nil {
			return nil, err
		}

		return NewFrom(resStorage, outputLayout, input.dtype, input.device), nil
	}
}

func Conv2dBackward[T spark.D](params *spark.Conv2DParams) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		input, kernel := inputs[0].Detach(), inputs[1].Detach()

		gradH := outputGrad.Shape().Dims()[2]
		gradW := outputGrad.Shape().Dims()[3]
		kH := params.KH
		kW := params.KW

		outSizeH := (gradH-1)*params.Stride + params.Dilate*(kH-1) + 1 - 2*params.Pad
		outSizeW := (gradW-1)*params.Stride + params.Dilate*(kW-1) + 1 - 2*params.Pad
		outPadH := params.InH - outSizeH
		outPadW := params.InW - outSizeW

		outPad := max(outPadH, outPadW)

		inputGradParams := &spark.ConvT2DParams{
			Batch:  params.Batch,
			InCh:   params.OutCh,
			InH:    gradH,
			InW:    gradW,
			OutCh:  params.InCh,
			KH:     params.KH,
			KW:     params.KW,
			Stride: params.Stride,
			Pad:    params.Pad,
			OutPad: outPad,
			Dilate: params.Dilate,
		}

		inputGrad, err := outputGrad.ConvTranspose2d(kernel, inputGradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute input gradient: %v", err)
		}

		inputT, err := input.Transpose(0, 1)
		if err != nil {
			return nil, err
		}

		outputGradT, err := outputGrad.Transpose(0, 1)
		if err != nil {
			return nil, err
		}

		outGradH := outputGrad.Shape().Dims()[2]
		outGradW := outputGrad.Shape().Dims()[3]
		kernelGradParams := &spark.Conv2DParams{
			Batch:  params.InCh,
			InCh:   params.Batch,
			InH:    params.InH,
			InW:    params.InW,
			OutCh:  params.OutCh,
			KH:     outGradH,
			KW:     outGradW,
			Stride: params.Stride,
			Pad:    params.Pad,
			Dilate: params.Dilate,
		}

		kernelGradT, err := inputT.Conv2d(outputGradT, kernelGradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute kernel gradient: %v", err)
		}

		kernelGrad, err := kernelGradT.Transpose(0, 1)
		if err != nil {
			return nil, err
		}

		return []*Tensor[T]{inputGrad, kernelGrad}, nil
	}
}

// ConvTranspose2dForward computes 2D transposed convolution
func ConvTranspose2dForward[T spark.D](params *spark.ConvT2DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("ConvTranspose2dForward expects 2 inputs, got %d", len(inputs))
		}

		input, kernel := inputs[0], inputs[1]

		if input.Rank() != 4 || kernel.Rank() != 4 {
			return nil, fmt.Errorf("input and kernel must be 4D tensors")
		}

		hOut := params.OutH()
		wOut := params.OutW()
		outputShape := spark.NewShapeFrom([]int{params.Batch, params.OutCh, hOut, wOut})
		outputLayout := spark.Contiguous(outputShape)

		resStorage, err := input.storage.ConvTranspose2d(input.layout, kernel.storage, kernel.layout, params)
		if err != nil {
			return nil, err
		}

		return NewFrom(resStorage, outputLayout, input.dtype, input.device), nil
	}
}

// ConvTranspose2dBackward computes gradients for 2D transposed convolution
// Based on Rust Candle implementation
func ConvTranspose2dBackward[T spark.D](params *spark.ConvT2DParams) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		input, kernel := inputs[0].Detach(), inputs[1].Detach()

		inputGradParams := &spark.Conv2DParams{
			Batch:  params.Batch,
			InCh:   params.OutCh,
			InH:    outputGrad.Shape().Dims()[2],
			InW:    outputGrad.Shape().Dims()[3],
			OutCh:  params.InCh,
			KH:     params.KH,
			KW:     params.KW,
			Stride: params.Stride,
			Pad:    params.Pad,
			Dilate: params.Dilate,
		}

		inputGrad, err := outputGrad.Conv2d(kernel, inputGradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute input gradient: %v", err)
		}

		outputGradT, err := outputGrad.Transpose(0, 1)
		if err != nil {
			return nil, err
		}

		inputT, err := input.Transpose(0, 1)
		if err != nil {
			return nil, err
		}

		kernelGradParams := &spark.Conv2DParams{
			Batch:  params.OutCh,
			InCh:   params.Batch,
			InH:    params.InH,
			InW:    params.InW,
			OutCh:  params.InCh,
			KH:     outputGrad.Shape().Dims()[2],
			KW:     outputGrad.Shape().Dims()[3],
			Stride: params.Dilate,
			Pad:    params.Pad,
			Dilate: params.Stride,
		}

		kernelGradT, err := outputGradT.Conv2d(inputT, kernelGradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute kernel gradient: %v", err)
		}

		kernelGrad, err := kernelGradT.Transpose(0, 1)
		if err != nil {
			return nil, err
		}

		return []*Tensor[T]{inputGrad, kernelGrad}, nil
	}
}

// AvgPool2dForward computes 2D average pooling
func AvgPool2dForward[T spark.D](params *spark.Pool2DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("AvgPool2dForward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0]

		if input.Rank() != 4 {
			return nil, fmt.Errorf("input must be 4D tensor for AvgPool2d, got %dD", input.Rank())
		}

		hOut := params.OutH()
		wOut := params.OutW()
		outputShape := spark.NewShapeFrom([]int{params.Batch, params.Ch, hOut, wOut})
		outputLayout := spark.Contiguous(outputShape)

		resStorage, err := input.storage.AvgPool2d(input.layout, params)
		if err != nil {
			return nil, err
		}

		return NewFrom(resStorage, outputLayout, input.dtype, input.device), nil
	}
}

// AvgPool2dBackward computes gradients for 2D average pooling
// Based on Rust Candle implementation: only supports kernel_size == stride
func AvgPool2dBackward[T spark.D](params *spark.Pool2DParams) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		input := inputs[0].Detach()

		if params.KH != params.HStride || params.KW != params.WStride {
			return nil, fmt.Errorf("AvgPool2dBackward only supports kernel_size == stride, got kh=%d, stride_h=%d, kw=%d, stride_w=%d",
				params.KH, params.HStride, params.KW, params.WStride)
		}

		dims := input.Shape().Dims()
		batch, ch, h, w := dims[0], dims[1], dims[2], dims[3]
		outDims := outputGrad.Shape().Dims()
		outH, outW := outDims[2], outDims[3]

		upsampleParams := &spark.UpsampleParams{
			Batch:  batch,
			Ch:     ch,
			InH:    outH,
			InW:    outW,
			HOut:   h,
			WOut:   w,
			HScale: float64(h) / float64(outH),
			WScale: float64(w) / float64(outW),
		}

		gradArg, err := outputGrad.UpsampleNearest2d(upsampleParams)
		if err != nil {
			return nil, fmt.Errorf("failed to upsample gradient: %v", err)
		}
		poolSize := float64(params.KH * params.KW)
		scale := T(1.0 / poolSize)
		scaleT := Full[T](scale, gradArg.Shape(), gradArg.Device())

		gradArg, err = gradArg.Mul(scaleT)
		if err != nil {
			return nil, fmt.Errorf("failed to scale gradient: %v", err)
		}

		return []*Tensor[T]{gradArg}, nil
	}
}

// UpsampleNearest2dForward computes 2D nearest neighbor upsampling
func UpsampleNearest2dForward[T spark.D](params *spark.UpsampleParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("UpsampleNearest2dForward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0]

		if input.Rank() != 4 {
			return nil, fmt.Errorf("input must be 4D tensor for UpsampleNearest2d, got %dD", input.Rank())
		}

		outputShape := spark.NewShapeFrom([]int{params.Batch, params.Ch, params.HOut, params.WOut})
		outputLayout := spark.Contiguous(outputShape)

		resStorage, err := input.storage.UpsampleNearest2d(input.layout, params)
		if err != nil {
			return nil, err
		}

		return NewFrom(resStorage, outputLayout, input.dtype, input.device), nil
	}
}

// UpsampleNearest2dBackward computes gradients for 2D nearest neighbor upsampling
// Based on Rust Candle implementation with restrictions:
// 1. Only supports integer scale factors
// 2. Only supports uniform scaling (scale_h == scale_w)
func UpsampleNearest2dBackward[T spark.D](params *spark.UpsampleParams) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		input := inputs[0].Detach()

		dims := input.Shape().Dims()
		c, h, w := dims[1], dims[2], dims[3]
		targetH, targetW := params.HOut, params.WOut

		if targetH%h != 0 || targetW%w != 0 {
			return nil, fmt.Errorf("UpsampleNearest2dBackward only supports integer scale factors, got target=(%d,%d), input=(%d,%d)",
				targetH, targetW, h, w)
		}

		scaleH := targetH / h
		scaleW := targetW / w

		if scaleH != scaleW {
			return nil, fmt.Errorf("UpsampleNearest2dBackward only supports uniform scaling, got scale_h=%d, scale_w=%d",
				scaleH, scaleW)
		}

		kernelData := make([]T, c*scaleH*scaleW)
		for i := range kernelData {
			kernelData[i] = T(1.0)
		}
		kernel := New(kernelData, spark.NewShape(c, 1, scaleH, scaleW), input.Device())

		convParams := &spark.Conv2DParams{
			Batch:  params.Batch,
			InCh:   c,
			InH:    targetH,
			InW:    targetW,
			OutCh:  c,
			KH:     scaleH,
			KW:     scaleW,
			Stride: scaleH, // stride = scale
			Pad:    0,
			Dilate: 1,
		}

		inputGrad, err := outputGrad.Conv2d(kernel, convParams)
		if err != nil {
			return nil, fmt.Errorf("failed to aggregate gradients: %v", err)
		}

		return []*Tensor[T]{inputGrad}, nil
	}
}

// SoftmaxForward computes softmax activation along the last dimension
// softmax(x_i) = exp(x_i) / sum(exp(x_j))
func SoftmaxForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SoftmaxForward expects 1 input, got %d", len(inputs))
	}

	input := inputs[0]

	resStorage, err := input.storage.Softmax(input.layout)
	if err != nil {
		return nil, err
	}

	return NewFrom(resStorage, input.layout.Clone(), input.dtype, input.device), nil
}

// SoftmaxBackward computes gradients for softmax
// grad_input = softmax * (grad_output - sum(grad_output * softmax))
func SoftmaxBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SoftmaxBackward expects 1 input, got %d", len(inputs))
	}

	input := inputs[0].Detach()

	softmaxOutput, err := input.Softmax()
	if err != nil {
		return nil, fmt.Errorf("failed to compute softmax in backward: %v", err)
	}

	gradMulSoftmax, err := outputGrad.Mul(softmaxOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to multiply grad_output and softmax: %v", err)
	}

	rank := input.Rank()
	lastDim := rank - 1

	sumGrad, err := gradMulSoftmax.Sum([]int{lastDim}, false)
	if err != nil {
		return nil, fmt.Errorf("failed to sum along last dimension: %v", err)
	}

	sumGradBroadcast, err := sumGrad.BroadcastAs(input.Shape())
	if err != nil {
		return nil, fmt.Errorf("failed to broadcast sum: %v", err)
	}

	gradDiff, err := outputGrad.Sub(sumGradBroadcast)
	if err != nil {
		return nil, fmt.Errorf("failed to compute gradient difference: %v", err)
	}

	inputGrad, err := softmaxOutput.Mul(gradDiff)
	if err != nil {
		return nil, fmt.Errorf("failed to compute input gradient: %v", err)
	}

	return []*Tensor[T]{inputGrad}, nil
}

// ReluForward computes element-wise ReLU: relu(x) = max(0, x)
func ReluForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("ReluForward expects 1 input, got %d", len(inputs))
	}

	input := inputs[0]

	resStorage, err := input.storage.Relu(input.layout)
	if err != nil {
		return nil, err
	}

	return NewFrom(resStorage, input.layout.Clone(), input.dtype, input.device), nil
}

// ReluBackward computes gradients for ReLU
// grad_input = grad_output * (input > 0 ? 1 : 0)
func ReluBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("ReluBackward expects 1 input, got %d", len(inputs))
	}

	input := inputs[0].Detach()
	zero := Full[T](0.0, input.Shape(), input.Device())

	// Compute mask: input > 0 returns uint8 (0 or 1)
	mask, err := input.Gt(zero)
	if err != nil {
		return nil, fmt.Errorf("failed to compute mask: %v", err)
	}

	var maskFloat *Tensor[T]
	switch outputGrad.DType() {
	case spark.F32:
		maskF32, err := mask.ToFloat32()
		if err != nil {
			return nil, err
		}
		maskFloat = any(maskF32).(*Tensor[T])
	case spark.F64:
		maskF64, err := mask.ToFloat64()
		if err != nil {
			return nil, err
		}
		maskFloat = any(maskF64).(*Tensor[T])
	default:
		return nil, fmt.Errorf("unsupported dtype for ReLU: %v", outputGrad.DType())
	}

	// Use WhereCond: mask ? outputGrad : zeros
	zeros := Zeros[T](outputGrad.Shape(), outputGrad.Device())
	inputGrad, err := maskFloat.WhereCond(outputGrad, zeros)
	if err != nil {
		return nil, fmt.Errorf("failed to compute input gradient: %v", err)
	}

	return []*Tensor[T]{inputGrad}, nil
}

// WhereCondForward computes conditional selection: result = condition ? trueVal : falseVal
func WhereCondForward[T spark.D](condition *Tensor[T]) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("WhereCondForward expects 2 inputs (trueVal, falseVal), got %d", len(inputs))
		}

		trueVal := inputs[0]
		falseVal := inputs[1]

		finalShape, err := condition.Shape().BroadcastShapeBinaryOp(trueVal.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast condition and trueVal shapes: %v", err)
		}

		finalShape, err = finalShape.BroadcastShapeBinaryOp(falseVal.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast with falseVal shape: %v", err)
		}

		condLayout, err := condition.layout.BroadcastAs(finalShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast condition layout: %v", err)
		}

		tLayout, err := trueVal.layout.BroadcastAs(finalShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast trueVal layout: %v", err)
		}

		fLayout, err := falseVal.layout.BroadcastAs(finalShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast falseVal layout: %v", err)
		}

		resStorage, err := condition.storage.WhereCond(
			condLayout,
			trueVal.storage, tLayout,
			falseVal.storage, fLayout,
		)
		if err != nil {
			return nil, err
		}

		return NewFrom(resStorage, spark.Contiguous(finalShape), trueVal.dtype, trueVal.device), nil
	}
}

// WhereCondBackward computes gradients for conditional selection
// Gradient flows only through the selected branch based on condition
func WhereCondBackward[T spark.D](condition *Tensor[T]) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("WhereCondBackward expects 2 inputs (trueVal, falseVal), got %d", len(inputs))
		}

		outputGradDetached := outputGrad.Detach()
		zeros := Zeros[T](outputGrad.Shape(), outputGrad.Device())
		trueGradStorage, err := condition.storage.WhereCond(
			condition.layout,
			outputGradDetached.storage, outputGradDetached.layout,
			zeros.storage, zeros.layout,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to compute trueVal gradient: %v", err)
		}
		trueGrad := NewFrom(trueGradStorage, outputGrad.layout.Clone(), outputGrad.dtype, outputGrad.device)

		falseGradStorage, err := condition.storage.WhereCond(
			condition.layout,
			zeros.storage, zeros.layout,
			outputGradDetached.storage, outputGradDetached.layout,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to compute falseVal gradient: %v", err)
		}
		falseGrad := NewFrom(falseGradStorage, outputGrad.layout.Clone(), outputGrad.dtype, outputGrad.device)

		return []*Tensor[T]{trueGrad, falseGrad}, nil
	}
}

// NegForward computes element-wise negation: c = -a
func NegForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("NegForward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0]
	resStorage, err := a.storage.Neg(a.layout)
	if err != nil {
		return nil, err
	}

	return NewFrom(resStorage, a.layout.Clone(), a.dtype, a.device), nil
}

// NegBackward computes gradients for negation: ∂(-x)/∂x = -1
func NegBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("NegBackward expects 1 input, got %d", len(inputs))
	}

	// ∂(-x)/∂x = -outputGrad
	inputGrad, err := outputGrad.Neg()
	if err != nil {
		return nil, err
	}

	return []*Tensor[T]{inputGrad}, nil
}

// LogForward computes element-wise natural logarithm: c = log(a)
func LogForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("LogForward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0]
	resStorage, err := a.storage.Log(a.layout)
	if err != nil {
		return nil, err
	}

	return NewFrom(resStorage, a.layout.Clone(), a.dtype, a.device), nil
}

// LogBackward computes gradients for natural logarithm: ∂log(x)/∂x = 1/x
func LogBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("LogBackward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0].Detach()

	// 1 / a
	one := Full[T](1.0, a.Shape(), a.Device())
	recipA, err := one.Div(a)
	if err != nil {
		return nil, err
	}

	// outputGrad / a
	inputGrad, err := outputGrad.Mul(recipA)
	if err != nil {
		return nil, err
	}

	return []*Tensor[T]{inputGrad}, nil
}

// AbsForward computes element-wise absolute value: c = |a|
func AbsForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("AbsForward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0]
	resStorage, err := a.storage.Abs(a.layout)
	if err != nil {
		return nil, err
	}

	return NewFrom(resStorage, a.layout.Clone(), a.dtype, a.device), nil
}

// AbsBackward computes gradients for absolute value: ∂|a|/∂a = sign(a)
func AbsBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("AbsBackward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0].Detach()

	// sign(a): 1 if a > 0, -1 if a < 0, 0 if a == 0
	signA, err := a.Sign()
	if err != nil {
		return nil, err
	}

	// outputGrad * sign(a)
	inputGrad, err := outputGrad.Mul(signA)
	if err != nil {
		return nil, err
	}

	return []*Tensor[T]{inputGrad}, nil
}

// SqrForward computes element-wise square: c = a²
func SqrForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SqrForward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0]
	resStorage, err := a.storage.Sqr(a.layout)
	if err != nil {
		return nil, err
	}

	return NewFrom(resStorage, a.layout.Clone(), a.dtype, a.device), nil
}

// SqrBackward computes gradients for square: ∂(a²)/∂a = 2a
func SqrBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SqrBackward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0].Detach()

	// 2 * a
	two := Full[T](2.0, a.Shape(), a.Device())
	twoA, err := two.Mul(a)
	if err != nil {
		return nil, err
	}

	// outputGrad * 2a
	inputGrad, err := outputGrad.Mul(twoA)
	if err != nil {
		return nil, err
	}

	return []*Tensor[T]{inputGrad}, nil
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

// SigmoidForward computes element-wise sigmoid activation: σ(x) = 1/(1+e^(-x))
func SigmoidForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SigmoidForward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0]
	resStorage, err := a.storage.Sigmoid(a.layout)
	if err != nil {
		return nil, err
	}

	return NewFrom(resStorage, a.layout.Clone(), a.dtype, a.device), nil
}

// SigmoidBackward computes gradients for sigmoid: ∂σ(x)/∂x = σ(x) * (1 - σ(x))
func SigmoidBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SigmoidBackward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0].Detach()

	sigmoidA, err := a.Sigmoid()
	if err != nil {
		return nil, err
	}

	one := Full[T](1.0, sigmoidA.Shape(), sigmoidA.Device())
	oneMinusSigmoid, err := one.Sub(sigmoidA)
	if err != nil {
		return nil, err
	}

	sigmoidDerivative, err := sigmoidA.Mul(oneMinusSigmoid)
	if err != nil {
		return nil, err
	}

	inputGrad, err := outputGrad.Mul(sigmoidDerivative)
	if err != nil {
		return nil, err
	}

	return []*Tensor[T]{inputGrad}, nil
}

// SignForward computes element-wise sign function: sign(x) = {1 if x>0, 0 if x=0, -1 if x<0}
func SignForward[T spark.D](inputs []*Tensor[T]) (*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SignForward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0]
	resStorage, err := a.storage.Sign(a.layout)
	if err != nil {
		return nil, err
	}

	return NewFrom(resStorage, a.layout.Clone(), a.dtype, a.device), nil
}

// SignBackward computes gradients for sign function
// Note: sign function has zero gradient almost everywhere (except at x=0 where it's undefined)
// Following PyTorch convention, we return zero gradient
func SignBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SignBackward expects 1 input, got %d", len(inputs))
	}

	a := inputs[0].Detach()

	zeroGrad := a.ZerosLike()

	return []*Tensor[T]{zeroGrad}, nil
}

// SumForward computes the sum along the specified dimensions
func SumForward[T spark.D](dims []int, keepdim bool) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("SumForward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0]

		resolvedDims, err := spark.ResolveAxes(dims, input.Shape())
		if err != nil {
			return nil, fmt.Errorf("sum dimensions: %w", err)
		}

		resStorage, err := input.storage.Sum(input.layout, resolvedDims)
		if err != nil {
			return nil, err
		}

		outputDims := make([]int, len(input.Dims()))
		copy(outputDims, input.Dims())
		for _, dim := range resolvedDims {
			outputDims[dim] = 1
		}
		outputShape := spark.NewShapeFrom(outputDims)

		sum := NewFrom(resStorage, spark.Contiguous(outputShape), input.dtype, input.device)

		if keepdim {
			return sum, nil
		} else {
			return sum.SqueezeDims(resolvedDims)
		}
	}
}

// SumBackward computes gradients for sum operation
func SumBackward[T spark.D](dims []int, keepdim bool) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("SumBackward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0].Detach()
		grad := outputGrad

		resolvedDims, err := spark.ResolveAxes(dims, input.Shape())
		if err != nil {
			return nil, fmt.Errorf("sum backward resolve dims: %w", err)
		}

		if !keepdim {
			targetDims := make([]int, len(input.Dims()))
			copy(targetDims, input.Dims())
			for _, dim := range resolvedDims {
				targetDims[dim] = 1
			}

			grad, err = grad.Reshape(targetDims...)
			if err != nil {
				return nil, fmt.Errorf("sum backward reshape: %w", err)
			}
		}

		inputGrad, err := grad.BroadcastAs(input.Shape())
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

// BroadcastAddBackward computes gradients for broadcasted addition
func BroadcastAddBackward[T spark.D](outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("BroadcastAddBackward expects 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0].Detach(), inputs[1].Detach()

	gradA, err := reduceBroadcastGrad(outputGrad, a.Dims())
	if err != nil {
		return nil, err
	}

	gradB, err := reduceBroadcastGrad(outputGrad, b.Dims())
	if err != nil {
		return nil, err
	}

	return []*Tensor[T]{gradA, gradB}, nil
}

// reduceBroadcastGrad reduces gradient from broadcasted shape back to original shape
func reduceBroadcastGrad[T spark.D](grad *Tensor[T], argDims []int) (*Tensor[T], error) {
	nodeDims := grad.Dims()

	leftDims := len(nodeDims) - len(argDims)
	sumDims := make([]int, 0, len(nodeDims))

	for i := range leftDims {
		sumDims = append(sumDims, i)
	}

	for i, argDim := range argDims {
		if nodeDims[i+leftDims] != argDim {
			sumDims = append(sumDims, i+leftDims)
		}
	}

	result := grad
	var err error
	if len(sumDims) > 0 {
		if result, err = grad.SumKeepDim(sumDims); err != nil {
			return nil, fmt.Errorf("sum_keepdim failed: %w", err)
		}
	}

	for range leftDims {
		if result, err = result.Squeeze(0); err != nil {
			return nil, fmt.Errorf("squeeze failed: %w", err)
		}
	}

	return result, nil
}

// TransposeForward computes tensor transpose by swapping two dimensions
func TransposeForward[T spark.D](dim1, dim2 int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("TransposeForward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0]

		// Resolve dimensions (handle negative indices)
		resolvedDim1, err := spark.ResolveAxis(dim1, input.Rank())
		if err != nil {
			return nil, fmt.Errorf("transpose dim1: %w", err)
		}

		resolvedDim2, err := spark.ResolveAxis(dim2, input.Rank())
		if err != nil {
			return nil, fmt.Errorf("transpose dim2: %w", err)
		}

		// If dimensions are the same, return a copy
		if resolvedDim1 == resolvedDim2 {
			return input, nil
		}

		// Create transposed layout
		newLayout, err := input.layout.Transpose(resolvedDim1, resolvedDim2)
		if err != nil {
			return nil, fmt.Errorf("transpose layout: %w", err)
		}

		// Return new tensor with transposed layout (zero-copy)
		return NewFrom(input.storage, newLayout, input.dtype, input.device), nil
	}
}

// TransposeBackward computes gradients for transpose operation
func TransposeBackward[T spark.D](dim1, dim2 int) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("TransposeBackward expects 1 input, got %d", len(inputs))
		}

		// For transpose, the gradient is simply the transpose of the output gradient
		// This is because transpose is its own inverse: transpose(transpose(x)) = x
		inputGrad, err := outputGrad.Transpose(dim1, dim2)
		if err != nil {
			return nil, fmt.Errorf("transpose backward: %w", err)
		}

		return []*Tensor[T]{inputGrad}, nil
	}
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

// ReshapeForward creates reshape forward operation
func ReshapeForward[T spark.D](newShape *spark.Shape) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("ReshapeForward expects 1 input, got %d", len(inputs))
		}

		input := inputs[0]

		if newShape.ElemCount() != input.layout.Shape().ElemCount() {
			return nil, fmt.Errorf("reshape: element count mismatch")
		}

		// For now, only support contiguous tensors
		// Non-contiguous tensors should be made contiguous first
		// TODO: support
		if !input.layout.IsContiguous() {
			return nil, fmt.Errorf("reshape: non-contiguous tensors not supported yet")
		}

		newLayout := spark.ContiguousWithOffset(newShape, input.layout.StartOffset())
		return NewFrom(input.storage, newLayout, input.dtype, input.device), nil
	}
}

// ReshapeBackward creates reshape backward operation
func ReshapeBackward[T spark.D](originalShape *spark.Shape) BackwardFunc[T] {
	return func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("ReshapeBackward expects 1 input, got %d", len(inputs))
		}

		// Reshape gradient back to original shape
		inputGrad, err := outputGrad.Reshape(originalShape.Dims()...)
		if err != nil {
			return nil, fmt.Errorf("reshape backward: %w", err)
		}

		return []*Tensor[T]{inputGrad}, nil
	}
}

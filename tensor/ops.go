package tensor

import (
	"fmt"

	"github.com/gocnn/spark"
)

// ForwardFunc defines a forward pass function for tensor operations.
type ForwardFunc[T spark.D] func([]*Tensor[T]) (*Tensor[T], error)

// BackwardFunc defines a backward pass function for computing gradients.
type BackwardFunc[T spark.D] func(*Tensor[T], []*Tensor[T]) ([]*Tensor[T], error)

// Op holds the computation graph node for automatic differentiation.
type Op[T spark.D] struct {
	inputs   []*Tensor[T]
	backward BackwardFunc[T]
}

// Inputs returns the input tensors of the operation.
func (op *Op[T]) Inputs() []*Tensor[T] {
	return op.inputs
}

// Backward computes the gradients for the operation.
func (op *Op[T]) Backward(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
	return op.backward(grad, inputs)
}

// ApplyOp applies a forward function and builds the computation graph if needed.
func ApplyOp[T spark.D](inputs []*Tensor[T], forward ForwardFunc[T], backward BackwardFunc[T]) (*Tensor[T], error) {
	result, err := forward(inputs)
	if err != nil {
		return nil, err
	}

	for _, input := range inputs {
		if input.IsVar() {
			result.isVar = true
			result.op = &Op[T]{inputs: inputs, backward: backward}
			break
		}
	}

	return result, nil
}

// ReduceBroadcastGrad reduces a broadcasted gradient to the target shape.
func ReduceBroadcastGrad[T spark.D](g *Tensor[T], dims []int) (*Tensor[T], error) {
	gDims := g.Dims()
	offset := len(gDims) - len(dims)
	sumDims := make([]int, 0, len(gDims))
	for i := range offset {
		sumDims = append(sumDims, i)
	}
	for i, d := range dims {
		if gDims[i+offset] != d {
			sumDims = append(sumDims, i+offset)
		}
	}
	r := g
	var err error
	if len(sumDims) > 0 {
		r, err = g.SumKeepDim(sumDims)
		if err != nil {
			return nil, fmt.Errorf("failed to sum dims: %w", err)
		}
	}
	for range offset {
		r, err = r.Squeeze(0)
		if err != nil {
			return nil, fmt.Errorf("failed to squeeze dim: %w", err)
		}
	}
	return r, nil
}

// AffineForward returns a ForwardFunc for affine transformation: y = scale * x + bias.
func AffineForward[T spark.D](scale, bias float64) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Affine(x.layout, T(scale), T(bias))
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// AffineBackward returns a BackwardFunc for affine transformation gradients: ∂y/∂x = scale.
func AffineBackward[T spark.D](scale, bias float64) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		scaleTensor, err := Full[T](scale, grad.Shape(), grad.Device())
		if err != nil {
			return nil, err
		}
		dx, err := grad.Mul(scaleTensor)
		if err != nil {
			return nil, err
		}
		return []*Tensor[T]{dx}, nil
	}
}

// AddForward returns a ForwardFunc for element-wise addition: c = a + b.
func AddForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		layout := spark.Contiguous(a.layout.Shape())
		data, err := a.storage.Add(b.storage, a.layout, b.layout, layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, layout, a.dtype, a.device), nil
	}
}

// AddBackward returns a BackwardFunc for addition gradients: ∂c/∂a = 1, ∂c/∂b = 1.
func AddBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		return []*Tensor[T]{grad, grad}, nil
	}
}

// SubForward returns a ForwardFunc for element-wise subtraction: c = a - b.
func SubForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		layout := spark.Contiguous(a.layout.Shape())
		data, err := a.storage.Sub(b.storage, a.layout, b.layout, layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, layout, a.dtype, a.device), nil
	}
}

// SubBackward returns a BackwardFunc for subtraction gradients: ∂c/∂a = 1, ∂c/∂b = -1.
func SubBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		db, err := grad.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate gradient: %w", err)
		}
		return []*Tensor[T]{grad, db}, nil
	}
}

// MulForward returns a ForwardFunc for element-wise multiplication: c = a * b.
func MulForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		layout := spark.Contiguous(a.layout.Shape())
		data, err := a.storage.Mul(b.storage, a.layout, b.layout, layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, layout, a.dtype, a.device), nil
	}
}

// MulBackward returns a BackwardFunc for multiplication gradients: ∂c/∂a = b, ∂c/∂b = a.
func MulBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0].Detach(), inputs[1].Detach()
		da, err := grad.Mul(b)
		if err != nil {
			return nil, fmt.Errorf("failed to compute da: %w", err)
		}
		db, err := grad.Mul(a)
		if err != nil {
			return nil, fmt.Errorf("failed to compute db: %w", err)
		}
		return []*Tensor[T]{da, db}, nil
	}
}

// DivForward returns a ForwardFunc for element-wise division: c = a / b.
func DivForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		layout := spark.Contiguous(a.layout.Shape())
		data, err := a.storage.Div(b.storage, a.layout, b.layout, layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, layout, a.dtype, a.device), nil
	}
}

// DivBackward returns a BackwardFunc for division gradients: ∂c/∂a = 1/b, ∂c/∂b = -a/b².
func DivBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0].Detach(), inputs[1].Detach()
		da, err := grad.Div(b)
		if err != nil {
			return nil, fmt.Errorf("failed to compute da: %w", err)
		}
		b2, err := b.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute b²: %w", err)
		}
		aOverB2, err := a.Div(b2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute a/b²: %w", err)
		}
		negAOverB2, err := aOverB2.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate a/b²: %w", err)
		}
		db, err := grad.Mul(negAOverB2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute db: %w", err)
		}
		return []*Tensor[T]{da, db}, nil
	}
}

// MaxForward returns a ForwardFunc for element-wise maximum: c = max(a, b).
func MaxForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		layout := spark.Contiguous(a.layout.Shape())
		data, err := a.storage.Max(b.storage, a.layout, b.layout, layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, layout, a.dtype, a.device), nil
	}
}

// MaxBackward returns a BackwardFunc for maximum gradients: ∂c/∂a = (a >= b) ? grad : 0, ∂c/∂b = (b > a) ? grad : 0.
func MaxBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0].Detach(), inputs[1].Detach()

		// Create masks: a_mask = (a >= b), b_mask = (b > a)
		aMask, err := a.Ge(b)
		if err != nil {
			return nil, fmt.Errorf("failed to compute a >= b: %w", err)
		}
		bMask, err := b.Gt(a)
		if err != nil {
			return nil, fmt.Errorf("failed to compute b > a: %w", err)
		}

		zeros, err := Zeros[T](grad.Shape(), grad.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		da, err := aMask.WhereCond(grad, zeros)
		if err != nil {
			return nil, fmt.Errorf("failed to compute da: %w", err)
		}
		db, err := bMask.WhereCond(grad, zeros)
		if err != nil {
			return nil, fmt.Errorf("failed to compute db: %w", err)
		}

		return []*Tensor[T]{da, db}, nil
	}
}

// MinForward returns a ForwardFunc for element-wise minimum: c = min(a, b).
func MinForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		layout := spark.Contiguous(a.layout.Shape())
		data, err := a.storage.Min(b.storage, a.layout, b.layout, layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, layout, a.dtype, a.device), nil
	}
}

// MinBackward returns a BackwardFunc for minimum gradients: ∂c/∂a = (a <= b) ? grad : 0, ∂c/∂b = (b < a) ? grad : 0.
func MinBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0].Detach(), inputs[1].Detach()

		// Create masks: a_mask = (a <= b), b_mask = (b < a)
		aMask, err := a.Le(b)
		if err != nil {
			return nil, fmt.Errorf("failed to compute a <= b: %w", err)
		}
		bMask, err := b.Lt(a)
		if err != nil {
			return nil, fmt.Errorf("failed to compute b < a: %w", err)
		}

		zeros, err := Zeros[T](grad.Shape(), grad.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		da, err := aMask.WhereCond(grad, zeros)
		if err != nil {
			return nil, fmt.Errorf("failed to compute da: %w", err)
		}
		db, err := bMask.WhereCond(grad, zeros)
		if err != nil {
			return nil, fmt.Errorf("failed to compute db: %w", err)
		}

		return []*Tensor[T]{da, db}, nil
	}
}

// EqForward returns a ForwardFunc for element-wise equality comparison.
func EqForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]

		// Validate shapes are compatible
		if !a.Shape().Equal(b.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", a.Shape(), b.Shape())
		}

		// Perform comparison using storage layer
		data, err := a.storage.Eq(b.storage, a.layout, b.layout, spark.Contiguous(a.Shape()))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}

		// Result is always uint8 for comparison operations
		return NewFrom(data, spark.Contiguous(a.Shape()), spark.U8, a.device), nil
	}
}

// EqBackward returns a BackwardFunc for equality comparison (zero gradients).
func EqBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}

		// Comparison operations have zero gradients
		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}

		return []*Tensor[T]{da, db}, nil
	}
}

// NeForward returns a ForwardFunc for element-wise inequality comparison.
func NeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]

		if !a.Shape().Equal(b.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", a.Shape(), b.Shape())
		}

		data, err := a.storage.Ne(b.storage, a.layout, b.layout, spark.Contiguous(a.Shape()))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}

		return NewFrom(data, spark.Contiguous(a.Shape()), spark.U8, a.device), nil
	}
}

// NeBackward returns a BackwardFunc for inequality comparison (zero gradients).
func NeBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}

		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}

		return []*Tensor[T]{da, db}, nil
	}
}

// LtForward returns a ForwardFunc for element-wise less-than comparison.
func LtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]

		if !a.Shape().Equal(b.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", a.Shape(), b.Shape())
		}

		data, err := a.storage.Lt(b.storage, a.layout, b.layout, spark.Contiguous(a.Shape()))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}

		return NewFrom(data, spark.Contiguous(a.Shape()), spark.U8, a.device), nil
	}
}

// LtBackward returns a BackwardFunc for less-than comparison (zero gradients).
func LtBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}

		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}

		return []*Tensor[T]{da, db}, nil
	}
}

// LeForward returns a ForwardFunc for element-wise less-than-or-equal comparison.
func LeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]

		if !a.Shape().Equal(b.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", a.Shape(), b.Shape())
		}

		data, err := a.storage.Le(b.storage, a.layout, b.layout, spark.Contiguous(a.Shape()))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}

		return NewFrom(data, spark.Contiguous(a.Shape()), spark.U8, a.device), nil
	}
}

// LeBackward returns a BackwardFunc for less-than-or-equal comparison (zero gradients).
func LeBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}

		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}

		return []*Tensor[T]{da, db}, nil
	}
}

// GtForward returns a ForwardFunc for element-wise greater-than comparison.
func GtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]

		if !a.Shape().Equal(b.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", a.Shape(), b.Shape())
		}

		data, err := a.storage.Gt(b.storage, a.layout, b.layout, spark.Contiguous(a.Shape()))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}

		return NewFrom(data, spark.Contiguous(a.Shape()), spark.U8, a.device), nil
	}
}

// GtBackward returns a BackwardFunc for greater-than comparison (zero gradients).
func GtBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}

		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}

		return []*Tensor[T]{da, db}, nil
	}
}

// GeForward returns a ForwardFunc for element-wise greater-than-or-equal comparison.
func GeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]

		if !a.Shape().Equal(b.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", a.Shape(), b.Shape())
		}

		data, err := a.storage.Ge(b.storage, a.layout, b.layout, spark.Contiguous(a.Shape()))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}

		return NewFrom(data, spark.Contiguous(a.Shape()), spark.U8, a.device), nil
	}
}

// GeBackward returns a BackwardFunc for greater-than-or-equal comparison (zero gradients).
func GeBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}

		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}

		return []*Tensor[T]{da, db}, nil
	}
}

// BroadcastAddForward returns a ForwardFunc for broadcasted addition: a + b.
func BroadcastAddForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		bcastShape, err := a.Shape().BroadcastShapeBinaryOp(b.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		aBcast, err := a.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a: %w", err)
		}
		bBcast, err := b.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b: %w", err)
		}
		data, err := aBcast.storage.Add(bBcast.storage, aBcast.layout, bBcast.layout, spark.Contiguous(bcastShape))
		if err != nil {
			return nil, fmt.Errorf("failed to add: %w", err)
		}
		return NewFrom(data, spark.Contiguous(bcastShape), a.dtype, a.device), nil
	}
}

// BroadcastAddBackward returns a BackwardFunc for broadcasted addition gradients.
func BroadcastAddBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0].Detach(), inputs[1].Detach()
		da, err := ReduceBroadcastGrad(grad, a.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to reduce grad for a: %w", err)
		}
		db, err := ReduceBroadcastGrad(grad, b.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to reduce grad for b: %w", err)
		}
		return []*Tensor[T]{da, db}, nil
	}
}

// BroadcastSubForward returns a ForwardFunc for broadcasted subtraction: a - b.
func BroadcastSubForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		bcastShape, err := a.Shape().BroadcastShapeBinaryOp(b.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		aBcast, err := a.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a: %w", err)
		}
		bBcast, err := b.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b: %w", err)
		}
		data, err := aBcast.storage.Sub(bBcast.storage, aBcast.layout, bBcast.layout, spark.Contiguous(bcastShape))
		if err != nil {
			return nil, fmt.Errorf("failed to sub: %w", err)
		}
		return NewFrom(data, spark.Contiguous(bcastShape), a.dtype, a.device), nil
	}
}

// BroadcastSubBackward returns a BackwardFunc for broadcasted subtraction gradients.
func BroadcastSubBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0].Detach(), inputs[1].Detach()
		da, err := ReduceBroadcastGrad(grad, a.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to reduce grad for a: %w", err)
		}
		// For subtraction: ∂(a-b)/∂b = -1, so negate the gradient
		negGrad, err := grad.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate grad: %w", err)
		}
		db, err := ReduceBroadcastGrad(negGrad, b.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to reduce grad for b: %w", err)
		}
		return []*Tensor[T]{da, db}, nil
	}
}

// BroadcastMulForward returns a ForwardFunc for broadcasted multiplication: a * b.
func BroadcastMulForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		bcastShape, err := a.Shape().BroadcastShapeBinaryOp(b.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		aBcast, err := a.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a: %w", err)
		}
		bBcast, err := b.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b: %w", err)
		}
		data, err := aBcast.storage.Mul(bBcast.storage, aBcast.layout, bBcast.layout, spark.Contiguous(bcastShape))
		if err != nil {
			return nil, fmt.Errorf("failed to mul: %w", err)
		}
		return NewFrom(data, spark.Contiguous(bcastShape), a.dtype, a.device), nil
	}
}

// BroadcastMulBackward returns a BackwardFunc for broadcasted multiplication gradients.
func BroadcastMulBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0].Detach(), inputs[1].Detach()

		// ∂(a*b)/∂a = b, so grad_a = grad * b
		bBcast, err := b.BroadcastAs(grad.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b for grad_a: %w", err)
		}
		gradA, err := grad.Mul(bBcast)
		if err != nil {
			return nil, fmt.Errorf("failed to compute grad_a: %w", err)
		}
		da, err := ReduceBroadcastGrad(gradA, a.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to reduce grad for a: %w", err)
		}

		// ∂(a*b)/∂b = a, so grad_b = grad * a
		aBcast, err := a.BroadcastAs(grad.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a for grad_b: %w", err)
		}
		gradB, err := grad.Mul(aBcast)
		if err != nil {
			return nil, fmt.Errorf("failed to compute grad_b: %w", err)
		}
		db, err := ReduceBroadcastGrad(gradB, b.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to reduce grad for b: %w", err)
		}

		return []*Tensor[T]{da, db}, nil
	}
}

// BroadcastDivForward returns a ForwardFunc for broadcasted division: a / b.
func BroadcastDivForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		bcastShape, err := a.Shape().BroadcastShapeBinaryOp(b.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		aBcast, err := a.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a: %w", err)
		}
		bBcast, err := b.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b: %w", err)
		}
		data, err := aBcast.storage.Div(bBcast.storage, aBcast.layout, bBcast.layout, spark.Contiguous(bcastShape))
		if err != nil {
			return nil, fmt.Errorf("failed to div: %w", err)
		}
		return NewFrom(data, spark.Contiguous(bcastShape), a.dtype, a.device), nil
	}
}

// BroadcastDivBackward returns a BackwardFunc for broadcasted division gradients.
func BroadcastDivBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0].Detach(), inputs[1].Detach()

		// ∂(a/b)/∂a = 1/b, so grad_a = grad / b
		bBcast, err := b.BroadcastAs(grad.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b for grad_a: %w", err)
		}
		gradA, err := grad.Div(bBcast)
		if err != nil {
			return nil, fmt.Errorf("failed to compute grad_a: %w", err)
		}
		da, err := ReduceBroadcastGrad(gradA, a.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to reduce grad for a: %w", err)
		}

		// ∂(a/b)/∂b = -a/b², so grad_b = -grad * a / b²
		aBcast, err := a.BroadcastAs(grad.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a for grad_b: %w", err)
		}
		bSqr, err := bBcast.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute b²: %w", err)
		}
		gradB, err := grad.Mul(aBcast)
		if err != nil {
			return nil, fmt.Errorf("failed to compute grad*a: %w", err)
		}
		gradB, err = gradB.Div(bSqr)
		if err != nil {
			return nil, fmt.Errorf("failed to compute grad*a/b²: %w", err)
		}
		gradB, err = gradB.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate grad_b: %w", err)
		}
		db, err := ReduceBroadcastGrad(gradB, b.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to reduce grad for b: %w", err)
		}

		return []*Tensor[T]{da, db}, nil
	}
}

// BroadcastEqForward returns a ForwardFunc for broadcasted equality: a == b.
func BroadcastEqForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		bcastShape, err := a.Shape().BroadcastShapeBinaryOp(b.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		aBcast, err := a.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a: %w", err)
		}
		bBcast, err := b.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b: %w", err)
		}
		data, err := aBcast.storage.Eq(bBcast.storage, aBcast.layout, bBcast.layout, spark.Contiguous(bcastShape))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}
		return NewFrom(data, spark.Contiguous(bcastShape), spark.U8, a.device), nil
	}
}

// BroadcastEqBackward returns a BackwardFunc for broadcasted equality (zero gradients).
func BroadcastEqBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}
		return []*Tensor[T]{da, db}, nil
	}
}

// BroadcastNeForward returns a ForwardFunc for broadcasted inequality: a != b.
func BroadcastNeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		bcastShape, err := a.Shape().BroadcastShapeBinaryOp(b.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		aBcast, err := a.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a: %w", err)
		}
		bBcast, err := b.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b: %w", err)
		}
		data, err := aBcast.storage.Ne(bBcast.storage, aBcast.layout, bBcast.layout, spark.Contiguous(bcastShape))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}
		return NewFrom(data, spark.Contiguous(bcastShape), spark.U8, a.device), nil
	}
}

// BroadcastNeBackward returns a BackwardFunc for broadcasted inequality (zero gradients).
func BroadcastNeBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}
		return []*Tensor[T]{da, db}, nil
	}
}

// BroadcastLtForward returns a ForwardFunc for broadcasted less-than: a < b.
func BroadcastLtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		bcastShape, err := a.Shape().BroadcastShapeBinaryOp(b.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		aBcast, err := a.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a: %w", err)
		}
		bBcast, err := b.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b: %w", err)
		}
		data, err := aBcast.storage.Lt(bBcast.storage, aBcast.layout, bBcast.layout, spark.Contiguous(bcastShape))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}
		return NewFrom(data, spark.Contiguous(bcastShape), spark.U8, a.device), nil
	}
}

// BroadcastLtBackward returns a BackwardFunc for broadcasted less-than (zero gradients).
func BroadcastLtBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}
		return []*Tensor[T]{da, db}, nil
	}
}

// BroadcastLeForward returns a ForwardFunc for broadcasted less-equal: a <= b.
func BroadcastLeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		bcastShape, err := a.Shape().BroadcastShapeBinaryOp(b.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		aBcast, err := a.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a: %w", err)
		}
		bBcast, err := b.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b: %w", err)
		}
		data, err := aBcast.storage.Le(bBcast.storage, aBcast.layout, bBcast.layout, spark.Contiguous(bcastShape))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}
		return NewFrom(data, spark.Contiguous(bcastShape), spark.U8, a.device), nil
	}
}

// BroadcastLeBackward returns a BackwardFunc for broadcasted less-equal (zero gradients).
func BroadcastLeBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}
		return []*Tensor[T]{da, db}, nil
	}
}

// BroadcastGtForward returns a ForwardFunc for broadcasted greater-than: a > b.
func BroadcastGtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		bcastShape, err := a.Shape().BroadcastShapeBinaryOp(b.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		aBcast, err := a.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a: %w", err)
		}
		bBcast, err := b.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b: %w", err)
		}
		data, err := aBcast.storage.Gt(bBcast.storage, aBcast.layout, bBcast.layout, spark.Contiguous(bcastShape))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}
		return NewFrom(data, spark.Contiguous(bcastShape), spark.U8, a.device), nil
	}
}

// BroadcastGtBackward returns a BackwardFunc for broadcasted greater-than (zero gradients).
func BroadcastGtBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}
		return []*Tensor[T]{da, db}, nil
	}
}

// BroadcastGeForward returns a ForwardFunc for broadcasted greater-equal: a >= b.
func BroadcastGeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		bcastShape, err := a.Shape().BroadcastShapeBinaryOp(b.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		aBcast, err := a.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast a: %w", err)
		}
		bBcast, err := b.BroadcastAs(bcastShape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast b: %w", err)
		}
		data, err := aBcast.storage.Ge(bBcast.storage, aBcast.layout, bBcast.layout, spark.Contiguous(bcastShape))
		if err != nil {
			return nil, fmt.Errorf("failed to compare: %w", err)
		}
		return NewFrom(data, spark.Contiguous(bcastShape), spark.U8, a.device), nil
	}
}

// BroadcastGeBackward returns a BackwardFunc for broadcasted greater-equal (zero gradients).
func BroadcastGeBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		da, err := a.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for a: %w", err)
		}
		db, err := b.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("failed to create zero grad for b: %w", err)
		}
		return []*Tensor[T]{da, db}, nil
	}
}

// MatMulForward returns a ForwardFunc for matrix multiplication: a @ b.
func MatMulForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0], inputs[1]
		if a.Rank() < 2 || b.Rank() < 2 {
			return nil, fmt.Errorf("tensors must have at least 2 dims")
		}
		aDims, bDims := a.Dims(), b.Dims()
		if len(aDims) != len(bDims) {
			return nil, fmt.Errorf("tensors must have same rank: %d vs %d", len(aDims), len(bDims))
		}
		batchShape := spark.NewShapeFrom(aDims[:len(aDims)-2])
		if !batchShape.Equal(spark.NewShapeFrom(bDims[:len(bDims)-2])) {
			return nil, fmt.Errorf("batch dims mismatch: %v vs %v", batchShape, spark.NewShapeFrom(bDims[:len(bDims)-2]))
		}
		m, k1 := aDims[len(aDims)-2], aDims[len(aDims)-1]
		k2, n := bDims[len(bDims)-2], bDims[len(bDims)-1]
		if k1 != k2 {
			return nil, fmt.Errorf("inner dims mismatch: %d != %d", k1, k2)
		}
		batchSize := batchShape.ElemCount()
		resultDims := append(batchShape.Dims(), m, n)
		data, err := a.storage.MatMul(a.layout, b.storage, b.layout, batchSize, m, n, k1)
		if err != nil {
			return nil, fmt.Errorf("failed to matmul: %w", err)
		}
		return NewFrom(data, spark.Contiguous(spark.NewShapeFrom(resultDims)), a.dtype, a.device), nil
	}
}

// MatMulBackward returns a BackwardFunc for matrix multiplication gradients.
func MatMulBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		a, b := inputs[0].Detach(), inputs[1].Detach()
		bT, err := b.Transpose(-1, -2)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose b: %w", err)
		}
		da, err := grad.MatMul(bT)
		if err != nil {
			return nil, fmt.Errorf("failed to compute da: %w", err)
		}
		aT, err := a.Transpose(-1, -2)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose a: %w", err)
		}
		db, err := aT.MatMul(grad)
		if err != nil {
			return nil, fmt.Errorf("failed to compute db: %w", err)
		}
		return []*Tensor[T]{da, db}, nil
	}
}

// Conv1dForward returns a ForwardFunc for 1D convolution.
func Conv1dForward[T spark.D](params *spark.Conv1DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 3 || w.Rank() != 3 {
			return nil, fmt.Errorf("tensors must be 3D")
		}
		outLen := params.OutLen()
		shape := spark.NewShapeFrom([]int{params.Batch, params.OutCh, outLen})
		layout := spark.Contiguous(shape)
		data, err := x.storage.Conv1d(x.layout, w.storage, w.layout, params)
		if err != nil {
			return nil, fmt.Errorf("failed to conv1d: %w", err)
		}
		return NewFrom(data, layout, x.dtype, x.device), nil
	}
}

// Conv1dBackward returns a BackwardFunc for 1D convolution gradients.
func Conv1dBackward[T spark.D](params *spark.Conv1DParams) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0].Detach(), inputs[1].Detach()
		gradLen := grad.Shape().Dims()[2]
		kSize := params.KSize
		outSize := (gradLen-1)*params.Stride + params.Dilate*(kSize-1) + 1 - 2*params.Pad
		outPadding := params.InLen - outSize
		gradParams := &spark.ConvT1DParams{
			Batch:  params.Batch,
			InCh:   params.OutCh,
			InLen:  gradLen,
			OutCh:  params.InCh,
			KSize:  params.KSize,
			Stride: params.Stride,
			Pad:    params.Pad,
			OutPad: outPadding,
			Dilate: params.Dilate,
		}
		dx, err := grad.ConvTranspose1d(w, gradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		xT, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose x: %w", err)
		}
		gradT, err := grad.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose grad: %w", err)
		}
		kernelParams := &spark.Conv1DParams{
			Batch:  params.InCh,
			InCh:   params.Batch,
			InLen:  params.InLen,
			OutCh:  params.OutCh,
			KSize:  gradLen,
			Stride: params.Stride,
			Pad:    params.Pad,
			Dilate: params.Dilate,
		}
		dwT, err := xT.Conv1d(gradT, kernelParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dwT: %w", err)
		}
		dw, err := dwT.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose dwT: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// ConvTranspose1dForward returns a ForwardFunc for 1D transposed convolution.
func ConvTranspose1dForward[T spark.D](params *spark.ConvT1DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 3 || w.Rank() != 3 {
			return nil, fmt.Errorf("tensors must be 3D")
		}
		outLen := params.OutLen()
		shape := spark.NewShapeFrom([]int{params.Batch, params.OutCh, outLen})
		layout := spark.Contiguous(shape)
		data, err := x.storage.ConvTranspose1d(x.layout, w.storage, w.layout, params)
		if err != nil {
			return nil, fmt.Errorf("failed to convTranspose1d: %w", err)
		}
		return NewFrom(data, layout, x.dtype, x.device), nil
	}
}

// ConvTranspose1dBackward returns a BackwardFunc for 1D transposed convolution gradients.
func ConvTranspose1dBackward[T spark.D](params *spark.ConvT1DParams) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0].Detach(), inputs[1].Detach()
		gradParams := &spark.Conv1DParams{
			Batch:  params.Batch,
			InCh:   params.OutCh,
			InLen:  grad.Shape().Dims()[2],
			OutCh:  params.InCh,
			KSize:  params.KSize,
			Stride: params.Stride,
			Pad:    params.Pad,
			Dilate: params.Dilate,
		}
		dx, err := grad.Conv1d(w, gradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		gradT, err := grad.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose grad: %w", err)
		}
		xT, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose x: %w", err)
		}
		kernelParams := &spark.Conv1DParams{
			Batch:  params.OutCh,
			InCh:   params.Batch,
			InLen:  params.InLen,
			OutCh:  params.InCh,
			KSize:  grad.Shape().Dims()[2],
			Stride: params.Dilate,
			Pad:    params.Pad,
			Dilate: params.Stride,
		}
		dwT, err := gradT.Conv1d(xT, kernelParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dwT: %w", err)
		}
		dw, err := dwT.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose dwT: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// Conv2dForward returns a ForwardFunc for 2D convolution.
func Conv2dForward[T spark.D](params *spark.Conv2DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 4 || w.Rank() != 4 {
			return nil, fmt.Errorf("tensors must be 4D")
		}
		hOut, wOut := params.OutH(), params.OutW()
		shape := spark.NewShapeFrom([]int{params.Batch, params.OutCh, hOut, wOut})
		layout := spark.Contiguous(shape)
		data, err := x.storage.Conv2d(x.layout, w.storage, w.layout, params)
		if err != nil {
			return nil, fmt.Errorf("failed to conv2d: %w", err)
		}
		return NewFrom(data, layout, x.dtype, x.device), nil
	}
}

// Conv2dBackward returns a BackwardFunc for 2D convolution gradients.
func Conv2dBackward[T spark.D](params *spark.Conv2DParams) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0].Detach(), inputs[1].Detach()
		gradH, gradW := grad.Shape().Dims()[2], grad.Shape().Dims()[3]
		kH, kW := params.KH, params.KW
		outSizeH := (gradH-1)*params.Stride + params.Dilate*(kH-1) + 1 - 2*params.Pad
		outSizeW := (gradW-1)*params.Stride + params.Dilate*(kW-1) + 1 - 2*params.Pad
		outPadH := params.InH - outSizeH
		outPadW := params.InW - outSizeW
		outPad := max(outPadH, outPadW)
		gradParams := &spark.ConvT2DParams{
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
		dx, err := grad.ConvTranspose2d(w, gradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		xT, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose x: %w", err)
		}
		gradT, err := grad.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose grad: %w", err)
		}
		kernelParams := &spark.Conv2DParams{
			Batch:  params.InCh,
			InCh:   params.Batch,
			InH:    params.InH,
			InW:    params.InW,
			OutCh:  params.OutCh,
			KH:     gradH,
			KW:     gradW,
			Stride: params.Stride,
			Pad:    params.Pad,
			Dilate: params.Dilate,
		}
		dwT, err := xT.Conv2d(gradT, kernelParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dwT: %w", err)
		}
		dw, err := dwT.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose dwT: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// ConvTranspose2dForward returns a ForwardFunc for 2D transposed convolution.
func ConvTranspose2dForward[T spark.D](params *spark.ConvT2DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 4 || w.Rank() != 4 {
			return nil, fmt.Errorf("tensors must be 4D")
		}
		hOut, wOut := params.OutH(), params.OutW()
		shape := spark.NewShapeFrom([]int{params.Batch, params.OutCh, hOut, wOut})
		layout := spark.Contiguous(shape)
		data, err := x.storage.ConvTranspose2d(x.layout, w.storage, w.layout, params)
		if err != nil {
			return nil, fmt.Errorf("failed to convTranspose2d: %w", err)
		}
		return NewFrom(data, layout, x.dtype, x.device), nil
	}
}

// ConvTranspose2dBackward returns a BackwardFunc for 2D transposed convolution gradients.
func ConvTranspose2dBackward[T spark.D](params *spark.ConvT2DParams) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0].Detach(), inputs[1].Detach()
		gradParams := &spark.Conv2DParams{
			Batch:  params.Batch,
			InCh:   params.OutCh,
			InH:    grad.Shape().Dims()[2],
			InW:    grad.Shape().Dims()[3],
			OutCh:  params.InCh,
			KH:     params.KH,
			KW:     params.KW,
			Stride: params.Stride,
			Pad:    params.Pad,
			Dilate: params.Dilate,
		}
		dx, err := grad.Conv2d(w, gradParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		gradT, err := grad.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose grad: %w", err)
		}
		xT, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose x: %w", err)
		}
		kernelParams := &spark.Conv2DParams{
			Batch:  params.OutCh,
			InCh:   params.Batch,
			InH:    params.InH,
			InW:    params.InW,
			OutCh:  params.InCh,
			KH:     grad.Shape().Dims()[2],
			KW:     grad.Shape().Dims()[3],
			Stride: params.Dilate,
			Pad:    params.Pad,
			Dilate: params.Stride,
		}
		dwT, err := gradT.Conv2d(xT, kernelParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dwT: %w", err)
		}
		dw, err := dwT.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose dwT: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// AvgPool2dForward returns a ForwardFunc for 2D average pooling.
func AvgPool2dForward[T spark.D](params *spark.Pool2DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Rank() != 4 {
			return nil, fmt.Errorf("tensor must be 4D, got %dD", x.Rank())
		}
		shape := spark.NewShapeFrom([]int{params.Batch, params.Ch, params.OutH(), params.OutW()})
		data, err := x.storage.AvgPool2d(x.layout, params)
		if err != nil {
			return nil, fmt.Errorf("failed to avgpool2d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(shape), x.dtype, x.device), nil
	}
}

// AvgPool2dBackward returns a BackwardFunc for 2D average pooling gradients.
func AvgPool2dBackward[T spark.D](params *spark.Pool2DParams) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		if params.KH != params.HStride || params.KW != params.WStride {
			return nil, fmt.Errorf("kernel size must equal stride: kh=%d, stride_h=%d, kw=%d, stride_w=%d",
				params.KH, params.HStride, params.KW, params.WStride)
		}
		x := inputs[0].Detach()
		dims := x.Dims()
		batch, ch, h, w := dims[0], dims[1], dims[2], dims[3]
		outH, outW := grad.Dims()[2], grad.Dims()[3]
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
		dx, err := grad.UpsampleNearest2d(upsampleParams)
		if err != nil {
			return nil, fmt.Errorf("failed to upsample grad: %w", err)
		}
		scale := 1.0 / float64(params.KH*params.KW)
		scaleTensor, err := Full[T](scale, dx.Shape(), dx.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create scale tensor: %w", err)
		}
		dx, err = dx.Mul(scaleTensor)
		if err != nil {
			return nil, fmt.Errorf("failed to scale grad: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// UpsampleNearest2dForward returns a ForwardFunc for 2D nearest neighbor upsampling.
func UpsampleNearest2dForward[T spark.D](params *spark.UpsampleParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Rank() != 4 {
			return nil, fmt.Errorf("tensor must be 4D, got %dD", x.Rank())
		}
		shape := spark.NewShapeFrom([]int{params.Batch, params.Ch, params.HOut, params.WOut})
		data, err := x.storage.UpsampleNearest2d(x.layout, params)
		if err != nil {
			return nil, fmt.Errorf("failed to upsampleNearest2d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(shape), x.dtype, x.device), nil
	}
}

// UpsampleNearest2dBackward returns a BackwardFunc for 2D nearest neighbor upsampling gradients.
func UpsampleNearest2dBackward[T spark.D](params *spark.UpsampleParams) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		dims := x.Dims()
		c, h, w := dims[1], dims[2], dims[3]
		targetH, targetW := params.HOut, params.WOut
		if targetH%h != 0 || targetW%w != 0 {
			return nil, fmt.Errorf("non-integer scale factors: target=(%d,%d), input=(%d,%d)", targetH, targetW, h, w)
		}
		scaleH, scaleW := targetH/h, targetW/w
		if scaleH != scaleW {
			return nil, fmt.Errorf("non-uniform scaling: scale_h=%d, scale_w=%d", scaleH, scaleW)
		}
		kernel, err := Full[T](1.0, spark.NewShape(c, 1, scaleH, scaleW), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create kernel: %w", err)
		}
		convParams := &spark.Conv2DParams{
			Batch:  params.Batch,
			InCh:   c,
			InH:    targetH,
			InW:    targetW,
			OutCh:  c,
			KH:     scaleH,
			KW:     scaleW,
			Stride: scaleH,
			Pad:    0,
			Dilate: 1,
		}
		dx, err := grad.Conv2d(kernel, convParams)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SoftmaxForward returns a ForwardFunc for softmax along the last dimension.
func SoftmaxForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Softmax(x.layout)
		if err != nil {
			return nil, fmt.Errorf("failed to softmax: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SoftmaxBackward returns a BackwardFunc for softmax gradients.
func SoftmaxBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		softmaxX, err := x.Softmax()
		if err != nil {
			return nil, fmt.Errorf("failed to compute softmax: %w", err)
		}
		gradSoftmax, err := grad.Mul(softmaxX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute grad * softmax: %w", err)
		}
		sumGrad, err := gradSoftmax.Sum([]int{x.Rank() - 1})
		if err != nil {
			return nil, fmt.Errorf("failed to sum grad: %w", err)
		}
		sumGradBcast, err := sumGrad.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast sum: %w", err)
		}
		gradDiff, err := grad.Sub(sumGradBcast)
		if err != nil {
			return nil, fmt.Errorf("failed to compute grad difference: %w", err)
		}
		dx, err := softmaxX.Mul(gradDiff)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// WhereCondForward returns a ForwardFunc for conditional selection: condition ? trueVal : falseVal.
func WhereCondForward[T spark.D](cond *Tensor[T]) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		tVal, fVal := inputs[0], inputs[1]
		shape, err := cond.Shape().BroadcastShapeBinaryOp(tVal.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast with trueVal: %w", err)
		}
		shape, err = shape.BroadcastShapeBinaryOp(fVal.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast with falseVal: %w", err)
		}
		condLayout, err := cond.layout.BroadcastAs(shape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast cond: %w", err)
		}
		tLayout, err := tVal.layout.BroadcastAs(shape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast trueVal: %w", err)
		}
		fLayout, err := fVal.layout.BroadcastAs(shape)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast falseVal: %w", err)
		}
		data, err := cond.storage.WhereCond(condLayout, tVal.storage, tLayout, fVal.storage, fLayout)
		if err != nil {
			return nil, fmt.Errorf("failed to whereCond: %w", err)
		}
		return NewFrom(data, spark.Contiguous(shape), tVal.dtype, tVal.device), nil
	}
}

// WhereCondBackward returns a BackwardFunc for conditional selection gradients.
func WhereCondBackward[T spark.D](cond *Tensor[T]) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		gradD := grad.Detach()
		zeros, err := Zeros[T](grad.Shape(), grad.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		tGrad, err := cond.storage.WhereCond(cond.layout, gradD.storage, gradD.layout, zeros.storage, zeros.layout)
		if err != nil {
			return nil, fmt.Errorf("failed to compute trueVal grad: %w", err)
		}
		fGrad, err := cond.storage.WhereCond(cond.layout, zeros.storage, zeros.layout, gradD.storage, gradD.layout)
		if err != nil {
			return nil, fmt.Errorf("failed to compute falseVal grad: %w", err)
		}
		return []*Tensor[T]{
			NewFrom(tGrad, grad.layout.Clone(), grad.dtype, grad.device),
			NewFrom(fGrad, grad.layout.Clone(), grad.dtype, grad.device),
		}, nil
	}
}

// CopyForward returns a ForwardFunc for tensor cloning: y = x.
func CopyForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		layout := spark.Contiguous(x.Shape())
		data, err := x.storage.Copy(layout, x.storage)
		if err != nil {
			return nil, fmt.Errorf("failed to clone: %w", err)
		}
		return NewFrom(data, layout, x.dtype, x.device), nil
	}
}

// CopyBackward returns a BackwardFunc for clone gradients: ∂y/∂x = 1.
func CopyBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		return []*Tensor[T]{grad}, nil
	}
}

// NegForward returns a ForwardFunc for element-wise negation: -x.
func NegForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Neg(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// NegBackward returns a BackwardFunc for negation gradients: ∂(-x)/∂x = -1.
func NegBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		dx, err := grad.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// RecipForward returns a ForwardFunc for element-wise reciprocal: 1/x.
func RecipForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Recip(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// RecipBackward returns a BackwardFunc for reciprocal gradients: ∂(1/x)/∂x = -1/x².
func RecipBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to square x: %w", err)
		}
		one, err := Full[T](1.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		recipX2, err := one.Div(x2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1/x²: %w", err)
		}
		negRecipX2, err := recipX2.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate 1/x²: %w", err)
		}
		dx, err := grad.Mul(negRecipX2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ExpForward returns a ForwardFunc for element-wise exponential: exp(x).
func ExpForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Exp(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// ExpBackward returns a BackwardFunc for exponential gradients: ∂exp(x)/∂x = exp(x).
func ExpBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		expX, err := x.Exp()
		if err != nil {
			return nil, fmt.Errorf("failed to compute exp(x): %w", err)
		}
		dx, err := grad.Mul(expX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// LogForward returns a ForwardFunc for element-wise natural logarithm: log(x).
func LogForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Log(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// LogBackward returns a BackwardFunc for logarithm gradients: ∂log(x)/∂x = 1/x.
func LogBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		one, err := Full[T](1.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		recipX, err := one.Div(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1/x: %w", err)
		}
		dx, err := grad.Mul(recipX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SinForward returns a ForwardFunc for element-wise sine: sin(x).
func SinForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sin(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SinBackward returns a BackwardFunc for sine gradients: ∂sin(x)/∂x = cos(x).
func SinBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		cosX, err := x.Cos()
		if err != nil {
			return nil, fmt.Errorf("failed to compute cos(x): %w", err)
		}
		dx, err := grad.Mul(cosX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// CosForward returns a ForwardFunc for element-wise cosine: cos(x).
func CosForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Cos(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// CosBackward returns a BackwardFunc for cosine gradients: ∂cos(x)/∂x = -sin(x).
func CosBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		sinX, err := x.Sin()
		if err != nil {
			return nil, fmt.Errorf("failed to compute sin(x): %w", err)
		}
		negSinX, err := sinX.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate sin(x): %w", err)
		}
		dx, err := grad.Mul(negSinX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// TanhForward returns a ForwardFunc for element-wise hyperbolic tangent: tanh(x).
func TanhForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Tanh(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// TanhBackward returns a BackwardFunc for hyperbolic tangent gradients: ∂tanh(x)/∂x = 1 - tanh²(x).
func TanhBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		tanhX, err := x.Tanh()
		if err != nil {
			return nil, fmt.Errorf("failed to compute tanh(x): %w", err)
		}
		tanhX2, err := tanhX.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute tanh²(x): %w", err)
		}
		one, err := Full[T](1.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		deriv, err := one.Sub(tanhX2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1 - tanh²(x): %w", err)
		}
		dx, err := grad.Mul(deriv)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ErfForward returns a ForwardFunc for element-wise error function: erf(x).
func ErfForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Erf(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// ErfBackward returns a BackwardFunc for error function gradients: ∂erf(x)/∂x = (2/√π) * exp(-x²).
func ErfBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²: %w", err)
		}
		negX2, err := x2.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate x²: %w", err)
		}
		expNegX2, err := negX2.Exp()
		if err != nil {
			return nil, fmt.Errorf("failed to compute exp(-x²): %w", err)
		}
		coeff, err := Full[T](1.1283791670955126, x.Shape(), x.Device()) // 2/√π
		if err != nil {
			return nil, fmt.Errorf("failed to create coeff: %w", err)
		}
		deriv, err := coeff.Mul(expNegX2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute (2/√π) * exp(-x²): %w", err)
		}
		dx, err := grad.Mul(deriv)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// CeilForward returns a ForwardFunc for element-wise ceiling: ceil(x).
func CeilForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Ceil(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// CeilBackward returns a BackwardFunc for ceiling gradients: ∂ceil(x)/∂x = 0.
func CeilBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		zeros, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		return []*Tensor[T]{zeros}, nil
	}
}

// FloorForward returns a ForwardFunc for element-wise floor: floor(x).
func FloorForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Floor(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// FloorBackward returns a BackwardFunc for floor gradients: ∂floor(x)/∂x = 0.
func FloorBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		zeros, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		return []*Tensor[T]{zeros}, nil
	}
}

// RoundForward returns a ForwardFunc for element-wise rounding: round(x).
func RoundForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Round(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// RoundBackward returns a BackwardFunc for rounding gradients: ∂round(x)/∂x = 0.
func RoundBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		zeros, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		return []*Tensor[T]{zeros}, nil
	}
}

// NormcdfForward returns a ForwardFunc for element-wise normal CDF: Φ(x).
func NormcdfForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Normcdf(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// NormcdfBackward returns a BackwardFunc for normal CDF gradients: ∂Φ(x)/∂x = φ(x) = (1/√(2π)) * exp(-x²/2).
func NormcdfBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²: %w", err)
		}
		half, err := Full[T](0.5, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create half: %w", err)
		}
		negHalfX2, err := x2.Mul(half)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²/2: %w", err)
		}
		negHalfX2, err = negHalfX2.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate x²/2: %w", err)
		}
		expTerm, err := negHalfX2.Exp()
		if err != nil {
			return nil, fmt.Errorf("failed to compute exp(-x²/2): %w", err)
		}
		coeff, err := Full[T](0.3989422804014327, x.Shape(), x.Device()) // 1/√(2π)
		if err != nil {
			return nil, fmt.Errorf("failed to create coeff: %w", err)
		}
		deriv, err := coeff.Mul(expTerm)
		if err != nil {
			return nil, fmt.Errorf("failed to compute (1/√(2π)) * exp(-x²/2): %w", err)
		}
		dx, err := grad.Mul(deriv)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// AbsForward returns a ForwardFunc for element-wise absolute value: |x|.
func AbsForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Abs(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// AbsBackward returns a BackwardFunc for absolute value gradients: ∂|x|/∂x = sign(x).
func AbsBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		signX, err := x.Sign()
		if err != nil {
			return nil, fmt.Errorf("failed to compute sign(x): %w", err)
		}
		dx, err := grad.Mul(signX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SqrForward returns a ForwardFunc for element-wise square: x².
func SqrForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sqr(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SqrBackward returns a BackwardFunc for square gradients: ∂(x²)/∂x = 2x.
func SqrBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		twoX, err := Full[T](2.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create twoX: %w", err)
		}
		twoX, err = twoX.Mul(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 2x: %w", err)
		}
		dx, err := grad.Mul(twoX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SqrtForward returns a ForwardFunc for element-wise square root: √x.
func SqrtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sqrt(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SqrtBackward returns a BackwardFunc for square root gradients: ∂√x/∂x = 1/(2√x).
func SqrtBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		sqrtX, err := x.Sqrt()
		if err != nil {
			return nil, fmt.Errorf("failed to compute √x: %w", err)
		}
		denom, err := Full[T](2.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create denom: %w", err)
		}
		denom, err = denom.Mul(sqrtX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 2√x: %w", err)
		}
		dx, err := grad.Div(denom)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// GeluForward returns a ForwardFunc for element-wise GELU activation: gelu(x).
func GeluForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Gelu(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// GeluBackward returns a BackwardFunc for GELU gradients.
func GeluBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		sqrt2OverPi, err := Full[T](0.7978845608028654, x.Shape(), x.Device()) // √(2/π)
		if err != nil {
			return nil, fmt.Errorf("failed to create sqrt2OverPi: %w", err)
		}
		c, err := Full[T](0.044715, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create c: %w", err)
		}
		half, err := Full[T](0.5, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create half: %w", err)
		}
		one, err := Full[T](1.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		three, err := Full[T](3.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create three: %w", err)
		}
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²: %w", err)
		}
		x3, err := x2.Mul(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x³: %w", err)
		}
		cx3, err := c.Mul(x3)
		if err != nil {
			return nil, fmt.Errorf("failed to compute c*x³: %w", err)
		}
		inner, err := x.Add(cx3)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x + c*x³: %w", err)
		}
		tanhArg, err := sqrt2OverPi.Mul(inner)
		if err != nil {
			return nil, fmt.Errorf("failed to compute √(2/π)*(x + c*x³): %w", err)
		}
		tanhX, err := tanhArg.Tanh()
		if err != nil {
			return nil, fmt.Errorf("failed to compute tanh(...): %w", err)
		}
		onePlusTanh, err := one.Add(tanhX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1 + tanh(...): %w", err)
		}
		firstTerm, err := half.Mul(onePlusTanh)
		if err != nil {
			return nil, fmt.Errorf("failed to compute first term: %w", err)
		}
		tanh2, err := tanhX.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute tanh²(...): %w", err)
		}
		sech2, err := one.Sub(tanh2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute sech²(...): %w", err)
		}
		threeC, err := three.Mul(c)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 3*c: %w", err)
		}
		threeCx2, err := threeC.Mul(x2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 3*c*x²: %w", err)
		}
		onePlusThreeCx2, err := one.Add(threeCx2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1 + 3*c*x²: %w", err)
		}
		secondTerm, err := half.Mul(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute half*x: %w", err)
		}
		secondTerm, err = secondTerm.Mul(sech2)
		if err != nil {
			return nil, fmt.Errorf("failed to multiply by sech²: %w", err)
		}
		secondTerm, err = secondTerm.Mul(sqrt2OverPi)
		if err != nil {
			return nil, fmt.Errorf("failed to multiply by √(2/π): %w", err)
		}
		secondTerm, err = secondTerm.Mul(onePlusThreeCx2)
		if err != nil {
			return nil, fmt.Errorf("failed to multiply by (1 + 3*c*x²): %w", err)
		}
		deriv, err := firstTerm.Add(secondTerm)
		if err != nil {
			return nil, fmt.Errorf("failed to compute derivative: %w", err)
		}
		dx, err := grad.Mul(deriv)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// GeluErfForward returns a ForwardFunc for ERF-based GELU activation: gelu_erf(x).
func GeluErfForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.GeluErf(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// GeluErfBackward returns a BackwardFunc for ERF-based GELU gradients.
func GeluErfBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		half, err := Full[T](0.5, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create half: %w", err)
		}
		one, err := Full[T](1.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		sqrt2, err := Full[T](1.4142135623730951, x.Shape(), x.Device()) // √2
		if err != nil {
			return nil, fmt.Errorf("failed to create sqrt2: %w", err)
		}
		xOverSqrt2, err := x.Div(sqrt2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x/√2: %w", err)
		}
		erfVal, err := xOverSqrt2.Erf()
		if err != nil {
			return nil, fmt.Errorf("failed to compute erf(x/√2): %w", err)
		}
		onePlusErf, err := one.Add(erfVal)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1 + erf(...): %w", err)
		}
		firstTerm, err := half.Mul(onePlusErf)
		if err != nil {
			return nil, fmt.Errorf("failed to compute first term: %w", err)
		}
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²: %w", err)
		}
		x2Half, err := x2.Mul(half)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²/2: %w", err)
		}
		negX2Half, err := x2Half.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate x²/2: %w", err)
		}
		expTerm, err := negX2Half.Exp()
		if err != nil {
			return nil, fmt.Errorf("failed to compute exp(-x²/2): %w", err)
		}
		twoOverSqrt2Pi, err := Full[T](0.7978845608028654, x.Shape(), x.Device()) // 2/√(2π)
		if err != nil {
			return nil, fmt.Errorf("failed to create twoOverSqrt2Pi: %w", err)
		}
		secondTerm, err := half.Mul(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute half*x: %w", err)
		}
		secondTerm, err = secondTerm.Mul(twoOverSqrt2Pi)
		if err != nil {
			return nil, fmt.Errorf("failed to multiply by 2/√(2π): %w", err)
		}
		secondTerm, err = secondTerm.Mul(expTerm)
		if err != nil {
			return nil, fmt.Errorf("failed to multiply by exp(-x²/2): %w", err)
		}
		deriv, err := firstTerm.Add(secondTerm)
		if err != nil {
			return nil, fmt.Errorf("failed to compute derivative: %w", err)
		}
		dx, err := grad.Mul(deriv)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ReluForward returns a ForwardFunc for element-wise ReLU: max(0, x).
func ReluForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Relu(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// ReluBackward returns a BackwardFunc for ReLU gradients: grad * (x > 0 ? 1 : 0).
func ReluBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		zero, err := Full[T](0.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zero: %w", err)
		}
		mask, err := x.Gt(zero)
		if err != nil {
			return nil, fmt.Errorf("failed to compute mask: %w", err)
		}
		zeros, err := Zeros[T](grad.Shape(), grad.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		dx, err := mask.WhereCond(grad, zeros)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// EluForward returns a ForwardFunc for element-wise ELU activation: elu(x, alpha).
func EluForward[T spark.D](alpha float64) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Elu(x.layout, T(alpha))
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// EluBackward returns a BackwardFunc for ELU gradients: 1 if x >= 0, alpha * exp(x) if x < 0.
func EluBackward[T spark.D](alpha float64) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		zero, err := Full[T](0.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zero: %w", err)
		}
		mask, err := x.Ge(zero)
		if err != nil {
			return nil, fmt.Errorf("failed to compute mask: %w", err)
		}
		one, err := Full[T](1.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		expX, err := x.Exp()
		if err != nil {
			return nil, fmt.Errorf("failed to compute exp(x): %w", err)
		}
		alphaT, err := Full[T](alpha, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create alphaT: %w", err)
		}
		alphaExpX, err := alphaT.Mul(expX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute alpha*exp(x): %w", err)
		}
		deriv, err := mask.WhereCond(one, alphaExpX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute derivative: %w", err)
		}
		dx, err := grad.Mul(deriv)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SiluForward returns a ForwardFunc for element-wise SiLU activation: x * sigmoid(x).
func SiluForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Silu(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SiluBackward returns a BackwardFunc for SiLU gradients: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
func SiluBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		sigmoidX, err := x.Sigmoid()
		if err != nil {
			return nil, fmt.Errorf("failed to compute sigmoid(x): %w", err)
		}
		one, err := Full[T](1.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		oneMinusSigmoid, err := one.Sub(sigmoidX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1 - sigmoid(x): %w", err)
		}
		xSigmoid, err := x.Mul(sigmoidX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x*sigmoid(x): %w", err)
		}
		xSigmoidTerm, err := xSigmoid.Mul(oneMinusSigmoid)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x*sigmoid(x)*(1 - sigmoid(x)): %w", err)
		}
		deriv, err := sigmoidX.Add(xSigmoidTerm)
		if err != nil {
			return nil, fmt.Errorf("failed to compute derivative: %w", err)
		}
		dx, err := grad.Mul(deriv)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// PowfForward returns a ForwardFunc for element-wise power: x^param.
func PowfForward[T spark.D](param float64) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Powf(x.layout, T(param))
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// PowfBackward returns a BackwardFunc for power gradients: ∂(x^param)/∂x = param * x^(param-1).
func PowfBackward[T spark.D](param float64) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		paramMinusOne := param - 1.0
		xPowParamM1, err := x.Powf(paramMinusOne)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x^(param-1): %w", err)
		}
		deriv, err := Full[T](param, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create param: %w", err)
		}
		deriv, err = deriv.Mul(xPowParamM1)
		if err != nil {
			return nil, fmt.Errorf("failed to compute param * x^(param-1): %w", err)
		}
		dx, err := grad.Mul(deriv)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SigmoidForward returns a ForwardFunc for element-wise sigmoid: σ(x) = 1/(1+e^(-x)).
func SigmoidForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sigmoid(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SigmoidBackward returns a BackwardFunc for sigmoid gradients: ∂σ(x)/∂x = σ(x) * (1 - σ(x)).
func SigmoidBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		sigmoidX, err := x.Sigmoid()
		if err != nil {
			return nil, fmt.Errorf("failed to compute sigmoid(x): %w", err)
		}
		oneMinusSigmoid, err := Full[T](1.0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create oneMinusSigmoid: %w", err)
		}
		oneMinusSigmoid, err = oneMinusSigmoid.Sub(sigmoidX)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1 - sigmoid(x): %w", err)
		}
		deriv, err := sigmoidX.Mul(oneMinusSigmoid)
		if err != nil {
			return nil, fmt.Errorf("failed to compute derivative: %w", err)
		}
		dx, err := grad.Mul(deriv)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SignForward returns a ForwardFunc for element-wise sign: 1 if x > 0, 0 if x = 0, -1 if x < 0.
func SignForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sign(x.layout)
		if err != nil {
			return nil, err
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SignBackward returns a BackwardFunc for sign gradients: ∂sign(x)/∂x = 0.
func SignBackward[T spark.D]() BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		zeros, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		return []*Tensor[T]{zeros}, nil
	}
}

// SumDimForward returns a ForwardFunc for sum along specified dimensions.
func SumDimForward[T spark.D](dims []int, keepdim bool) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		resolvedDims, err := spark.ResolveAxes(dims, x.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to resolve dims: %w", err)
		}
		data, err := x.storage.Sum(x.layout, resolvedDims)
		if err != nil {
			return nil, err
		}
		outputDims := make([]int, len(x.Dims()))
		copy(outputDims, x.Dims())
		for _, dim := range resolvedDims {
			outputDims[dim] = 1
		}
		outputShape := spark.NewShapeFrom(outputDims)
		sum := NewFrom(data, spark.Contiguous(outputShape), x.dtype, x.device)
		if keepdim {
			return sum, nil
		}
		return sum.SqueezeDims(resolvedDims)
	}
}

// SumDimBackward returns a BackwardFunc for sum gradients.
func SumDimBackward[T spark.D](dims []int, keepdim bool) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		resolvedDims, err := spark.ResolveAxes(dims, x.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to resolve dims: %w", err)
		}
		targetGrad := grad
		if !keepdim {
			targetDims := make([]int, len(x.Dims()))
			copy(targetDims, x.Dims())
			for _, dim := range resolvedDims {
				targetDims[dim] = 1
			}
			targetGrad, err = grad.Reshape(targetDims...)
			if err != nil {
				return nil, fmt.Errorf("failed to reshape grad: %w", err)
			}
		}
		dx, err := targetGrad.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast grad: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// TransposeForward returns a ForwardFunc for transposing dimensions.
func TransposeForward[T spark.D](dim1, dim2 int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		d1, err := spark.ResolveAxis(dim1, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("failed to resolve dim1: %w", err)
		}
		d2, err := spark.ResolveAxis(dim2, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("failed to resolve dim2: %w", err)
		}
		if d1 == d2 {
			return x, nil
		}
		layout, err := x.layout.Transpose(d1, d2)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose layout: %w", err)
		}
		return NewFrom(x.storage, layout, x.dtype, x.device), nil
	}
}

// TransposeBackward returns a BackwardFunc for transpose gradients.
func TransposeBackward[T spark.D](dim1, dim2 int) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		dx, err := grad.Transpose(dim1, dim2)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose grad: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SqueezeForward returns a ForwardFunc for squeezing a dimension.
func SqueezeForward[T spark.D](dim int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		rank := x.Rank()
		d, err := spark.ResolveAxis(dim, rank)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve dim: %w", err)
		}
		if x.Shape().Dim(d) != 1 {
			return x, nil
		}
		dims := make([]int, 0, rank-1)
		strides := make([]int, 0, rank-1)
		for i := range rank {
			if i == d {
				continue
			}
			dims = append(dims, x.Dims()[i])
			strides = append(strides, x.Stride()[i])
		}
		shape := spark.NewShapeFrom(dims)
		layout := spark.NewLayout(shape, strides, x.layout.StartOffset())
		return NewFrom(x.storage, layout, x.dtype, x.device), nil
	}
}

// SqueezeBackward returns a BackwardFunc for squeeze gradients.
func SqueezeBackward[T spark.D](dim int) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		rank := x.Rank()
		d, err := spark.ResolveAxis(dim, rank)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve dim: %w", err)
		}
		if x.Shape().Dim(d) != 1 {
			return []*Tensor[T]{grad}, nil
		}
		dx, err := grad.Unsqueeze(d)
		if err != nil {
			return nil, fmt.Errorf("failed to unsqueeze grad: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// UnsqueezeForward returns a ForwardFunc for unsqueezing a dimension.
func UnsqueezeForward[T spark.D](dim int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		rank := x.Rank()
		d := dim
		if d < 0 {
			d += rank + 1
		}
		if d < 0 || d > rank {
			return nil, fmt.Errorf("dim out of range [-%d, %d], got %d", rank+1, rank, dim)
		}
		dims := make([]int, 0, rank+1)
		dims = append(dims, x.Dims()[:d]...)
		dims = append(dims, 1)
		dims = append(dims, x.Dims()[d:]...)
		strides := make([]int, 0, rank+1)
		strides = append(strides, x.Stride()[:d]...)
		stride := 1
		if d < rank {
			stride = x.Stride()[d]
		}
		strides = append(strides, stride)
		strides = append(strides, x.Stride()[d:]...)
		shape := spark.NewShapeFrom(dims)
		layout := spark.NewLayout(shape, strides, x.layout.StartOffset())
		return NewFrom(x.storage, layout, x.dtype, x.device), nil
	}
}

// UnsqueezeBackward returns a BackwardFunc for unsqueeze gradients.
func UnsqueezeBackward[T spark.D](dim int) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		rank := x.Rank()
		d := dim
		if d < 0 {
			d += rank + 1
		}
		dx, err := grad.Squeeze(d)
		if err != nil {
			return nil, fmt.Errorf("failed to squeeze grad: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ReshapeForward returns a ForwardFunc for reshaping to new shape.
func ReshapeForward[T spark.D](newShape *spark.Shape) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if newShape.ElemCount() != x.layout.Shape().ElemCount() {
			return nil, fmt.Errorf("element count mismatch: %d != %d", newShape.ElemCount(), x.layout.Shape().ElemCount())
		}
		if !x.layout.IsContiguous() {
			return nil, fmt.Errorf("non-contiguous tensors not supported")
		}
		layout := spark.ContiguousWithOffset(newShape, x.layout.StartOffset())
		return NewFrom(x.storage, layout, x.dtype, x.device), nil
	}
}

// ReshapeBackward returns a BackwardFunc for reshape gradients.
func ReshapeBackward[T spark.D](origShape *spark.Shape) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		dx, err := grad.Reshape(origShape.Dims()...)
		if err != nil {
			return nil, fmt.Errorf("failed to reshape grad: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// BroadcastAsForward returns a ForwardFunc for broadcasting to a target shape.
func BroadcastAsForward[T spark.D](s *spark.Shape) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Shape().Equal(s) {
			return x, nil
		}
		l, err := x.layout.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast shape %v to %v: %w", x.Shape(), s, err)
		}
		st, err := x.storage.Clone()
		if err != nil {
			return nil, fmt.Errorf("failed to clone storage: %w", err)
		}
		return NewFrom(st, l, x.dtype, x.device), nil
	}
}

// BroadcastAsBackward returns a BackwardFunc for broadcasting gradients.
func BroadcastAsBackward[T spark.D](s *spark.Shape) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		if g.Shape().Equal(s) {
			return []*Tensor[T]{g}, nil
		}
		dx, err := ReduceBroadcastGrad(g, s.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to reduce grad: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

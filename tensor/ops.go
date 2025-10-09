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
	gd := g.Dims()
	n := len(gd) - len(dims)
	sd := make([]int, 0, len(gd))
	for i := range n {
		sd = append(sd, i)
	}
	for i, d := range dims {
		if gd[i+n] != d {
			sd = append(sd, i+n)
		}
	}
	r := g
	if len(sd) > 0 {
		var err error
		r, err = g.SumKeep(sd)
		if err != nil {
			return nil, fmt.Errorf("reduce broadcast grad: failed to sum dims: %w", err)
		}
	}
	for range n {
		var err error
		r, err = r.Squeeze(0)
		if err != nil {
			return nil, fmt.Errorf("reduce broadcast grad: failed to squeeze: %w", err)
		}
	}
	return r, nil
}

// AffineForward returns a ForwardFunc for affine transformation: y = scale * x + bias.
func AffineForward[T spark.D](scale, bias float64) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("affine forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Affine(x.layout, T(scale), T(bias))
		if err != nil {
			return nil, fmt.Errorf("affine forward: failed to compute affine: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// AffineBackward returns a BackwardFunc for affine transformation gradients: ∂y/∂x = scale.
func AffineBackward[T spark.D](scale, bias float64) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("affine backward: expected 1 input, got %d", len(inputs))
		}
		dx, err := g.MulScalar(scale)
		if err != nil {
			return nil, fmt.Errorf("affine backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// AddForward returns a ForwardFunc for element-wise addition: x + y.
func AddForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("add forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Add(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("add forward: failed to add: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// AddBackward returns a BackwardFunc for addition gradients: ∂z/∂x = 1, ∂z/∂y = 1.
func AddBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("add backward: expected 2 inputs, got %d", len(inputs))
		}
		return []*Tensor[T]{g, g}, nil
	}
}

// SubForward returns a ForwardFunc for element-wise subtraction: x - y.
func SubForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("sub forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Sub(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("sub forward: failed to subtract: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// SubBackward returns a BackwardFunc for subtraction gradients: ∂z/∂x = 1, ∂z/∂y = -1.
func SubBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("sub backward: expected 2 inputs, got %d", len(inputs))
		}
		dy, err := g.Neg()
		if err != nil {
			return nil, fmt.Errorf("sub backward: failed to negate: %w", err)
		}
		return []*Tensor[T]{g, dy}, nil
	}
}

// MulForward returns a ForwardFunc for element-wise multiplication: x * y.
func MulForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("mul forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Mul(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("mul forward: failed to multiply: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// MulBackward returns a BackwardFunc for multiplication gradients: ∂z/∂x = y, ∂z/∂y = x.
func MulBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("mul backward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		dx, err := g.Mul(y)
		if err != nil {
			return nil, fmt.Errorf("mul backward: failed to compute dx: %w", err)
		}
		dy, err := g.Mul(x)
		if err != nil {
			return nil, fmt.Errorf("mul backward: failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// DivForward returns a ForwardFunc for element-wise division: x / y.
func DivForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("div forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Div(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("div forward: failed to divide: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// DivBackward returns a BackwardFunc for division gradients: ∂z/∂x = 1/y, ∂z/∂y = -x/y².
func DivBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("div backward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		dx, err := g.Div(y)
		if err != nil {
			return nil, fmt.Errorf("div backward: failed to compute dx: %w", err)
		}
		y2, err := y.Sqr()
		if err != nil {
			return nil, fmt.Errorf("div backward: failed to compute y²: %w", err)
		}
		xy2, err := x.Div(y2)
		if err != nil {
			return nil, fmt.Errorf("div backward: failed to compute x/y²: %w", err)
		}
		dy, err := xy2.Neg()
		if err != nil {
			return nil, fmt.Errorf("div backward: failed to negate x/y²: %w", err)
		}
		dy, err = g.Mul(dy)
		if err != nil {
			return nil, fmt.Errorf("div backward: failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// MaximumForward returns a ForwardFunc for element-wise maximum: max(x, y).
func MaximumForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("maximum forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Maximum(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("maximum forward: failed to compute max: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// MaximumBackward returns a BackwardFunc for maximum gradients: ∂z/∂x = (x >= y) ? g : 0, ∂z/∂y = (y > x) ? g : 0.
func MaximumBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("maximum backward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		mx, err := x.Ge(y)
		if err != nil {
			return nil, fmt.Errorf("maximum backward: failed to compute x >= y: %w", err)
		}
		my, err := y.Gt(x)
		if err != nil {
			return nil, fmt.Errorf("maximum backward: failed to compute y > x: %w", err)
		}
		z, err := Zeros[T](g.Shape(), g.Device())
		if err != nil {
			return nil, fmt.Errorf("maximum backward: failed to create zeros: %w", err)
		}
		dx, err := mx.WhereCond(g, z)
		if err != nil {
			return nil, fmt.Errorf("maximum backward: failed to compute dx: %w", err)
		}
		dy, err := my.WhereCond(g, z)
		if err != nil {
			return nil, fmt.Errorf("maximum backward: failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// MinimumForward returns a ForwardFunc for element-wise minimum: min(x, y).
func MinimumForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("minimum forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Minimum(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("minimum forward: failed to compute min: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// MinimumBackward returns a BackwardFunc for minimum gradients: ∂z/∂x = (x <= y) ? g : 0, ∂z/∂y = (y < x) ? g : 0.
func MinimumBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("minimum backward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		mx, err := x.Le(y)
		if err != nil {
			return nil, fmt.Errorf("minimum backward: failed to compute x <= y: %w", err)
		}
		my, err := y.Lt(x)
		if err != nil {
			return nil, fmt.Errorf("minimum backward: failed to compute y < x: %w", err)
		}
		z, err := Zeros[T](g.Shape(), g.Device())
		if err != nil {
			return nil, fmt.Errorf("minimum backward: failed to create zeros: %w", err)
		}
		dx, err := mx.WhereCond(g, z)
		if err != nil {
			return nil, fmt.Errorf("minimum backward: failed to compute dx: %w", err)
		}
		dy, err := my.WhereCond(g, z)
		if err != nil {
			return nil, fmt.Errorf("minimum backward: failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// EqForward returns a ForwardFunc for element-wise equality: x == y.
func EqForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("eq forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("eq forward: shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Eq(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("eq forward: failed to compute eq: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// EqBackward returns a BackwardFunc for equality gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func EqBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("eq backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("eq backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("eq backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// NeForward returns a ForwardFunc for element-wise inequality: x != y.
func NeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("ne forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("ne forward: shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Ne(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("ne forward: failed to compute ne: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// NeBackward returns a BackwardFunc for inequality gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func NeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("ne backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("ne backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("ne backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// LtForward returns a ForwardFunc for element-wise less-than: x < y.
func LtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("lt forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("lt forward: shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Lt(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("lt forward: failed to compute lt: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// LtBackward returns a BackwardFunc for less-than gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func LtBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("lt backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("lt backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("lt backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// LeForward returns a ForwardFunc for element-wise less-than-or-equal: x <= y.
func LeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("le forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("le forward: shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Le(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("le forward: failed to compute le: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// LeBackward returns a BackwardFunc for less-than-or-equal gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func LeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("le backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("le backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("le backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// GtForward returns a ForwardFunc for element-wise greater-than: x > y.
func GtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("gt forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("gt forward: shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Gt(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("gt forward: failed to compute gt: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// GtBackward returns a BackwardFunc for greater-than gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func GtBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("gt backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("gt backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("gt backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// GeForward returns a ForwardFunc for element-wise greater-than-or-equal: x >= y.
func GeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("ge forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("ge forward: shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Ge(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("ge forward: failed to compute ge: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// GeBackward returns a BackwardFunc for greater-than-or-equal gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func GeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("ge backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("ge backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("ge backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastAddForward returns a ForwardFunc for broadcasted addition: x + y.
func BroadcastAddForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast add forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast add forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast add forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast add forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Add(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast add forward: failed to add: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// BroadcastAddBackward returns a BackwardFunc for broadcasted addition gradients: ∂z/∂x = 1, ∂z/∂y = 1.
func BroadcastAddBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast add backward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		dx, err := ReduceBroadcastGrad(g, x.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast add backward: failed to compute dx: %w", err)
		}
		dy, err := ReduceBroadcastGrad(g, y.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast add backward: failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastSubForward returns a ForwardFunc for broadcasted subtraction: x - y.
func BroadcastSubForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast sub forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast sub forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast sub forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast sub forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Sub(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast sub forward: failed to subtract: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// BroadcastSubBackward returns a BackwardFunc for broadcasted subtraction gradients: ∂z/∂x = 1, ∂z/∂y = -1.
func BroadcastSubBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast sub backward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		dx, err := ReduceBroadcastGrad(g, x.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast sub backward: failed to compute dx: %w", err)
		}
		ng, err := g.Neg()
		if err != nil {
			return nil, fmt.Errorf("broadcast sub backward: failed to negate grad: %w", err)
		}
		dy, err := ReduceBroadcastGrad(ng, y.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast sub backward: failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastMulForward returns a ForwardFunc for broadcasted multiplication: x * y.
func BroadcastMulForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast mul forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast mul forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast mul forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast mul forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Mul(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast mul forward: failed to multiply: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// BroadcastMulBackward returns a BackwardFunc for broadcasted multiplication gradients: ∂z/∂x = y, ∂z/∂y = x.
func BroadcastMulBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast mul backward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		yb, err := y.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast mul backward: failed to broadcast y for dx: %w", err)
		}
		gx, err := g.Mul(yb)
		if err != nil {
			return nil, fmt.Errorf("broadcast mul backward: failed to compute g*y: %w", err)
		}
		dx, err := ReduceBroadcastGrad(gx, x.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast mul backward: failed to compute dx: %w", err)
		}
		xb, err := x.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast mul backward: failed to broadcast x for dy: %w", err)
		}
		gy, err := g.Mul(xb)
		if err != nil {
			return nil, fmt.Errorf("broadcast mul backward: failed to compute g*x: %w", err)
		}
		dy, err := ReduceBroadcastGrad(gy, y.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast mul backward: failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastDivForward returns a ForwardFunc for broadcasted division: x / y.
func BroadcastDivForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast div forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast div forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast div forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast div forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Div(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast div forward: failed to divide: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// BroadcastDivBackward returns a BackwardFunc for broadcasted division gradients: ∂z/∂x = 1/y, ∂z/∂y = -x/y².
func BroadcastDivBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast div backward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		yb, err := y.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast div backward: failed to broadcast y for dx: %w", err)
		}
		gx, err := g.Div(yb)
		if err != nil {
			return nil, fmt.Errorf("broadcast div backward: failed to compute g/y: %w", err)
		}
		dx, err := ReduceBroadcastGrad(gx, x.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast div backward: failed to compute dx: %w", err)
		}
		y2, err := yb.Sqr()
		if err != nil {
			return nil, fmt.Errorf("broadcast div backward: failed to compute y²: %w", err)
		}
		xb, err := x.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast div backward: failed to broadcast x for dy: %w", err)
		}
		gy, err := g.Mul(xb)
		if err != nil {
			return nil, fmt.Errorf("broadcast div backward: failed to compute g*x: %w", err)
		}
		gy, err = gy.Div(y2)
		if err != nil {
			return nil, fmt.Errorf("broadcast div backward: failed to compute g*x/y²: %w", err)
		}
		gy, err = gy.Neg()
		if err != nil {
			return nil, fmt.Errorf("broadcast div backward: failed to negate g*x/y²: %w", err)
		}
		dy, err := ReduceBroadcastGrad(gy, y.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast div backward: failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastMaximumForward returns a ForwardFunc for broadcasted maximum: max(x, y).
func BroadcastMaximumForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast maximum forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Maximum(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum forward: failed to compute max: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// BroadcastMaximumBackward returns a BackwardFunc for broadcasted maximum gradients.
func BroadcastMaximumBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast maximum backward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		xb, err := x.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to broadcast y: %w", err)
		}
		o, err := xb.Maximum(yb)
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to compute output: %w", err)
		}
		mx, err := o.Eq(xb)
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to create mask_x: %w", err)
		}
		my, err := o.Eq(yb)
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to create mask_y: %w", err)
		}
		d, err := mx.Add(my)
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to compute denom: %w", err)
		}
		gx, err := mx.Mul(g)
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to compute gx: %w", err)
		}
		gx, err = gx.Div(d)
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to divide gx: %w", err)
		}
		gy, err := my.Mul(g)
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to compute gy: %w", err)
		}
		gy, err = gy.Div(d)
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to divide gy: %w", err)
		}
		dx, err := ReduceBroadcastGrad(gx, x.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to compute dx: %w", err)
		}
		dy, err := ReduceBroadcastGrad(gy, y.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast maximum backward: failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastMinimumForward returns a ForwardFunc for broadcasted minimum: min(x, y).
func BroadcastMinimumForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast minimum forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Minimum(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum forward: failed to compute min: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// BroadcastMinimumBackward returns a BackwardFunc for broadcasted minimum gradients.
func BroadcastMinimumBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast minimum backward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		xb, err := x.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to broadcast y: %w", err)
		}
		o, err := xb.Minimum(yb)
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to compute output: %w", err)
		}
		mx, err := o.Eq(xb)
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to create mask_x: %w", err)
		}
		my, err := o.Eq(yb)
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to create mask_y: %w", err)
		}
		d, err := mx.Add(my)
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to compute denom: %w", err)
		}
		gx, err := mx.Mul(g)
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to compute gx: %w", err)
		}
		gx, err = gx.Div(d)
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to divide gx: %w", err)
		}
		gy, err := my.Mul(g)
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to compute gy: %w", err)
		}
		gy, err = gy.Div(d)
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to divide gy: %w", err)
		}
		dx, err := ReduceBroadcastGrad(gx, x.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to compute dx: %w", err)
		}
		dy, err := ReduceBroadcastGrad(gy, y.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast minimum backward: failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastEqForward returns a ForwardFunc for broadcasted equality: x == y.
func BroadcastEqForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast eq forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast eq forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast eq forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast eq forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Eq(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast eq forward: failed to compute eq: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastEqBackward returns a BackwardFunc for broadcasted equality gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastEqBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast eq backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast eq backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast eq backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastNeForward returns a ForwardFunc for broadcasted inequality: x != y.
func BroadcastNeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast ne forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast ne forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast ne forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast ne forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Ne(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast ne forward: failed to compute ne: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastNeBackward returns a BackwardFunc for broadcasted inequality gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastNeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast ne backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast ne backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast ne backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastLtForward returns a ForwardFunc for broadcasted less-than: x < y.
func BroadcastLtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast lt forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast lt forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast lt forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast lt forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Lt(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast lt forward: failed to compute lt: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastLtBackward returns a BackwardFunc for broadcasted less-than gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastLtBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast lt backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast lt backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast lt backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastLeForward returns a ForwardFunc for broadcasted less-than-or-equal: x <= y.
func BroadcastLeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast le forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast le forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast le forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast le forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Le(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast le forward: failed to compute le: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastLeBackward returns a BackwardFunc for broadcasted less-than-or-equal gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastLeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast le backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast le backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast le backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastGtForward returns a ForwardFunc for broadcasted greater-than: x > y.
func BroadcastGtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast gt forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast gt forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast gt forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast gt forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Gt(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast gt forward: failed to compute gt: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastGtBackward returns a BackwardFunc for broadcasted greater-than gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastGtBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast gt backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast gt backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast gt backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastGeForward returns a ForwardFunc for broadcasted greater-than-or-equal: x >= y.
func BroadcastGeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast ge forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("broadcast ge forward: failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast ge forward: failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast ge forward: failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Ge(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("broadcast ge forward: failed to compute ge: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastGeBackward returns a BackwardFunc for broadcasted greater-than-or-equal gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastGeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("broadcast ge backward: expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast ge backward: failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("broadcast ge backward: failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// MatMulForward returns a ForwardFunc for matrix multiplication: x @ y.
func MatMulForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("matmul forward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if x.Rank() < 2 || y.Rank() < 2 {
			return nil, fmt.Errorf("matmul forward: tensors must have rank >= 2")
		}
		xd, yd := x.Dims(), y.Dims()
		if len(xd) != len(yd) {
			return nil, fmt.Errorf("matmul forward: tensors must have same rank: %d vs %d", len(xd), len(yd))
		}
		bs := spark.NewShapeFrom(xd[:len(xd)-2])
		if !bs.Equal(spark.NewShapeFrom(yd[:len(yd)-2])) {
			return nil, fmt.Errorf("matmul forward: batch dims mismatch: %v vs %v", bs, spark.NewShapeFrom(yd[:len(yd)-2]))
		}
		m, k := xd[len(xd)-2], xd[len(xd)-1]
		if k != yd[len(yd)-2] {
			return nil, fmt.Errorf("matmul forward: inner dims mismatch: %d vs %d", k, yd[len(yd)-2])
		}
		s := spark.NewShapeFrom(append(bs.Dims(), m, yd[len(yd)-1]))
		data, err := x.storage.MatMul(x.layout, y.storage, y.layout, bs.ElemCount(), m, yd[len(yd)-1], k)
		if err != nil {
			return nil, fmt.Errorf("matmul forward: failed to matmul: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// MatMulBackward returns a BackwardFunc for matrix multiplication gradients: ∂z/∂x = g @ yᵀ, ∂z/∂y = xᵀ @ g.
func MatMulBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("matmul backward: expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		yt, err := y.Transpose(-1, -2)
		if err != nil {
			return nil, fmt.Errorf("matmul backward: failed to transpose y: %w", err)
		}
		dx, err := g.MatMul(yt)
		if err != nil {
			return nil, fmt.Errorf("matmul backward: failed to compute dx: %w", err)
		}
		xt, err := x.Transpose(-1, -2)
		if err != nil {
			return nil, fmt.Errorf("matmul backward: failed to transpose x: %w", err)
		}
		dy, err := xt.MatMul(g)
		if err != nil {
			return nil, fmt.Errorf("matmul backward: failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// Conv1dForward returns a ForwardFunc for 1D convolution.
func Conv1dForward[T spark.D](p *spark.Conv1DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("conv1d forward: expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 3 || w.Rank() != 3 {
			return nil, fmt.Errorf("conv1d forward: tensors must be 3D")
		}
		s := spark.NewShapeFrom([]int{p.Batch, p.OutCh, p.OutLen()})
		data, err := x.storage.Conv1d(x.layout, w.storage, w.layout, p)
		if err != nil {
			return nil, fmt.Errorf("conv1d forward: failed to conv1d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// Conv1dBackward returns a BackwardFunc for 1D convolution gradients.
func Conv1dBackward[T spark.D](p *spark.Conv1DParams) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("conv1d backward: expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0].Detach(), inputs[1].Detach()
		l := g.Dims()[2]
		o := (l-1)*p.Stride + p.Dilate*(p.KSize-1) + 1 - 2*p.Pad
		gradP := &spark.ConvT1DParams{
			Batch:  p.Batch,
			InCh:   p.OutCh,
			InLen:  l,
			OutCh:  p.InCh,
			KSize:  p.KSize,
			Stride: p.Stride,
			Pad:    p.Pad,
			OutPad: p.InLen - o,
			Dilate: p.Dilate,
		}
		dx, err := g.ConvTranspose1d(w, gradP)
		if err != nil {
			return nil, fmt.Errorf("conv1d backward: failed to compute dx: %w", err)
		}
		xt, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("conv1d backward: failed to transpose x: %w", err)
		}
		gt, err := g.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("conv1d backward: failed to transpose grad: %w", err)
		}
		kernelP := &spark.Conv1DParams{
			Batch:  p.InCh,
			InCh:   p.Batch,
			InLen:  p.InLen,
			OutCh:  p.OutCh,
			KSize:  l,
			Stride: p.Stride,
			Pad:    p.Pad,
			Dilate: p.Dilate,
		}
		dwt, err := xt.Conv1d(gt, kernelP)
		if err != nil {
			return nil, fmt.Errorf("conv1d backward: failed to compute dwt: %w", err)
		}
		dw, err := dwt.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("conv1d backward: failed to transpose dwt: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// ConvTranspose1dForward returns a ForwardFunc for 1D transposed convolution.
func ConvTranspose1dForward[T spark.D](p *spark.ConvT1DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("convTranspose1d forward: expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 3 || w.Rank() != 3 {
			return nil, fmt.Errorf("convTranspose1d forward: tensors must be 3D")
		}
		s := spark.NewShapeFrom([]int{p.Batch, p.OutCh, p.OutLen()})
		data, err := x.storage.ConvTranspose1d(x.layout, w.storage, w.layout, p)
		if err != nil {
			return nil, fmt.Errorf("convTranspose1d forward: failed to convTranspose1d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// ConvTranspose1dBackward returns a BackwardFunc for 1D transposed convolution gradients.
func ConvTranspose1dBackward[T spark.D](p *spark.ConvT1DParams) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("convTranspose1d backward: expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0].Detach(), inputs[1].Detach()
		gradP := &spark.Conv1DParams{
			Batch:  p.Batch,
			InCh:   p.OutCh,
			InLen:  g.Dims()[2],
			OutCh:  p.InCh,
			KSize:  p.KSize,
			Stride: p.Stride,
			Pad:    p.Pad,
			Dilate: p.Dilate,
		}
		dx, err := g.Conv1d(w, gradP)
		if err != nil {
			return nil, fmt.Errorf("convTranspose1d backward: failed to compute dx: %w", err)
		}
		gt, err := g.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("convTranspose1d backward: failed to transpose grad: %w", err)
		}
		xt, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("convTranspose1d backward: failed to transpose x: %w", err)
		}
		kernelP := &spark.Conv1DParams{
			Batch:  p.OutCh,
			InCh:   p.Batch,
			InLen:  p.InLen,
			OutCh:  p.InCh,
			KSize:  g.Dims()[2],
			Stride: p.Dilate,
			Pad:    p.Pad,
			Dilate: p.Stride,
		}
		dwt, err := gt.Conv1d(xt, kernelP)
		if err != nil {
			return nil, fmt.Errorf("convTranspose1d backward: failed to compute dwt: %w", err)
		}
		dw, err := dwt.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("convTranspose1d backward: failed to transpose dwt: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// Conv2dForward returns a ForwardFunc for 2D convolution.
func Conv2dForward[T spark.D](p *spark.Conv2DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("conv2d forward: expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 4 || w.Rank() != 4 {
			return nil, fmt.Errorf("conv2d forward: tensors must be 4D")
		}
		s := spark.NewShapeFrom([]int{p.Batch, p.OutCh, p.OutH(), p.OutW()})
		data, err := x.storage.Conv2d(x.layout, w.storage, w.layout, p)
		if err != nil {
			return nil, fmt.Errorf("conv2d forward: failed to conv2d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// Conv2dBackward returns a BackwardFunc for 2D convolution gradients.
func Conv2dBackward[T spark.D](p *spark.Conv2DParams) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("conv2d backward: expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0].Detach(), inputs[1].Detach()
		gh, gw := g.Dims()[2], g.Dims()[3]
		oh := (gh-1)*p.Stride + p.Dilate*(p.KH-1) + 1 - 2*p.Pad
		ow := (gw-1)*p.Stride + p.Dilate*(p.KW-1) + 1 - 2*p.Pad
		gradP := &spark.ConvT2DParams{
			Batch:  p.Batch,
			InCh:   p.OutCh,
			InH:    gh,
			InW:    gw,
			OutCh:  p.InCh,
			KH:     p.KH,
			KW:     p.KW,
			Stride: p.Stride,
			Pad:    p.Pad,
			OutPad: max(p.InH-oh, p.InW-ow),
			Dilate: p.Dilate,
		}
		dx, err := g.ConvTranspose2d(w, gradP)
		if err != nil {
			return nil, fmt.Errorf("conv2d backward: failed to compute dx: %w", err)
		}
		xt, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("conv2d backward: failed to transpose x: %w", err)
		}
		gt, err := g.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("conv2d backward: failed to transpose grad: %w", err)
		}
		kernelP := &spark.Conv2DParams{
			Batch:  p.InCh,
			InCh:   p.Batch,
			InH:    p.InH,
			InW:    p.InW,
			OutCh:  p.OutCh,
			KH:     gh,
			KW:     gw,
			Stride: p.Stride,
			Pad:    p.Pad,
			Dilate: p.Dilate,
		}
		dwt, err := xt.Conv2d(gt, kernelP)
		if err != nil {
			return nil, fmt.Errorf("conv2d backward: failed to compute dwt: %w", err)
		}
		dw, err := dwt.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("conv2d backward: failed to transpose dwt: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// ConvTranspose2dForward returns a ForwardFunc for 2D transposed convolution.
func ConvTranspose2dForward[T spark.D](p *spark.ConvT2DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("convTranspose2d forward: expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 4 || w.Rank() != 4 {
			return nil, fmt.Errorf("convTranspose2d forward: tensors must be 4D")
		}
		s := spark.NewShapeFrom([]int{p.Batch, p.OutCh, p.OutH(), p.OutW()})
		data, err := x.storage.ConvTranspose2d(x.layout, w.storage, w.layout, p)
		if err != nil {
			return nil, fmt.Errorf("convTranspose2d forward: failed to convTranspose2d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// ConvTranspose2dBackward returns a BackwardFunc for 2D transposed convolution gradients.
func ConvTranspose2dBackward[T spark.D](p *spark.ConvT2DParams) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("convTranspose2d backward: expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0].Detach(), inputs[1].Detach()
		gradP := &spark.Conv2DParams{
			Batch:  p.Batch,
			InCh:   p.OutCh,
			InH:    g.Dims()[2],
			InW:    g.Dims()[3],
			OutCh:  p.InCh,
			KH:     p.KH,
			KW:     p.KW,
			Stride: p.Stride,
			Pad:    p.Pad,
			Dilate: p.Dilate,
		}
		dx, err := g.Conv2d(w, gradP)
		if err != nil {
			return nil, fmt.Errorf("convTranspose2d backward: failed to compute dx: %w", err)
		}
		gt, err := g.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("convTranspose2d backward: failed to transpose grad: %w", err)
		}
		xt, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("convTranspose2d backward: failed to transpose x: %w", err)
		}
		kernelP := &spark.Conv2DParams{
			Batch:  p.OutCh,
			InCh:   p.Batch,
			InH:    p.InH,
			InW:    p.InW,
			OutCh:  p.InCh,
			KH:     g.Dims()[2],
			KW:     g.Dims()[3],
			Stride: p.Dilate,
			Pad:    p.Pad,
			Dilate: p.Stride,
		}
		dwt, err := gt.Conv2d(xt, kernelP)
		if err != nil {
			return nil, fmt.Errorf("convTranspose2d backward: failed to compute dwt: %w", err)
		}
		dw, err := dwt.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("convTranspose2d backward: failed to transpose dwt: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// AvgPool2dForward returns a ForwardFunc for 2D average pooling.
func AvgPool2dForward[T spark.D](kH, kW, sH, sW int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("avgPool2d forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Rank() != 4 {
			return nil, fmt.Errorf("avgPool2d forward: tensor must be 4D, got %dD", x.Rank())
		}
		b, c, h, w, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("avgPool2d forward: expected 4D tensor for avg_pool2d, got: %w", err)
		}
		if h < kH || w < kW {
			return nil, fmt.Errorf("avgPool2d forward: kernel size (%d,%d) larger than input (%d,%d)", kH, kW, h, w)
		}
		if kH <= 0 || kW <= 0 || sH <= 0 || sW <= 0 {
			return nil, fmt.Errorf("kernel and stride must be positive")
		}
		hOut := (h-kH)/sH + 1
		wOut := (w-kW)/sW + 1
		shape := spark.NewShapeFrom([]int{b, c, hOut, wOut})
		data, err := x.storage.AvgPool2d(x.layout, kH, kW, sH, sW)
		if err != nil {
			return nil, fmt.Errorf("failed to avgpool2d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(shape), x.dtype, x.device), nil
	}
}

// AvgPool2dBackward returns a BackwardFunc for 2D average pooling gradients.
func AvgPool2dBackward[T spark.D](kH, kW, sH, sW int) BackwardFunc[T] {
	return func(grad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("avgPool2d backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		_, _, h, w, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("avgPool2d backward: expected 4D tensor for avg_pool2d, got: %w", err)
		}
		dx, err := grad.UpsampleNearest2d(h, w)
		if err != nil {
			return nil, fmt.Errorf("avgPool2d backward: failed to upsample grad: %w", err)
		}
		dx, err = dx.MulScalar(1.0 / float64(kH*kW))
		if err != nil {
			return nil, fmt.Errorf("avgPool2d backward: failed to scale grad: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// MaxPool2dForward returns a ForwardFunc for 2D max pooling.
func MaxPool2dForward[T spark.D](kH, kW, sH, sW int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("maxPool2d forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Rank() != 4 {
			return nil, fmt.Errorf("maxPool2d forward: tensor must be 4D, got %dD", x.Rank())
		}
		b, c, h, w, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("maxPool2d forward: failed to get 4D shape: %w", err)
		}
		if h < kH || w < kW {
			return nil, fmt.Errorf("maxPool2d forward: kernel (%d,%d) larger than input (%d,%d)", kH, kW, h, w)
		}
		if kH <= 0 || kW <= 0 || sH <= 0 || sW <= 0 {
			return nil, fmt.Errorf("kernel and stride must be positive")
		}
		hOut := (h-kH)/sH + 1
		wOut := (w-kW)/sW + 1
		shape := spark.NewShapeFrom([]int{b, c, hOut, wOut})
		data, err := x.storage.MaxPool2d(x.layout, kH, kW, sH, sW)
		if err != nil {
			return nil, fmt.Errorf("maxPool2d forward: failed to maxpool2d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(shape), x.dtype, x.device), nil
	}
}

// MaxPool2dBackward returns a BackwardFunc for 2D max pooling gradients.
func MaxPool2dBackward[T spark.D](kH, kW, sH, sW int) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("maxPool2d backward: expected 1 input, got %d", len(inputs))
		}
		if kH != sH || kW != sW {
			return nil, fmt.Errorf("maxPool2d backward: kernel must equal stride: kH=%d, sH=%d, kW=%d, sW=%d", kH, sH, kW, sW)
		}
		x := inputs[0].Detach()
		_, _, h, w, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("maxPool2d backward: failed to get 4D shape: %w", err)
		}
		p, err := x.MaxPool2d(kH, kW, sH, sW)
		if err != nil {
			return nil, fmt.Errorf("maxPool2d backward: failed to compute maxpool: %w", err)
		}
		pu, err := p.UpsampleNearest2d(h, w)
		if err != nil {
			return nil, fmt.Errorf("maxPool2d backward: failed to upsample maxpool: %w", err)
		}
		m, err := x.Eq(pu)
		if err != nil {
			return nil, fmt.Errorf("maxPool2d backward: failed to create mask: %w", err)
		}
		ma, err := m.AvgPool2d(kH, kW, sH, sW)
		if err != nil {
			return nil, fmt.Errorf("maxPool2d backward: failed to average mask: %w", err)
		}
		sg, err := g.Mul(ma)
		if err != nil {
			return nil, fmt.Errorf("maxPool2d backward: failed to scale grad: %w", err)
		}
		gu, err := sg.UpsampleNearest2d(h, w)
		if err != nil {
			return nil, fmt.Errorf("maxPool2d backward: failed to upsample grad: %w", err)
		}
		dx, err := gu.Mul(m)
		if err != nil {
			return nil, fmt.Errorf("maxPool2d backward: failed to apply mask: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// UpsampleNearest2dForward returns a ForwardFunc for 2D nearest neighbor upsampling.
func UpsampleNearest2dForward[T spark.D](h, w int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("upsampleNearest2d forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Rank() != 4 {
			return nil, fmt.Errorf("upsampleNearest2d forward: tensor must be 4D, got %dD", x.Rank())
		}
		if h <= 0 || w <= 0 {
			return nil, fmt.Errorf("upsampleNearest2d forward: target dims must be positive, got (%d,%d)", h, w)
		}
		b, c, _, _, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("upsampleNearest2d forward: failed to get 4D shape: %w", err)
		}
		shape := spark.NewShapeFrom([]int{b, c, h, w})
		data, err := x.storage.UpsampleNearest2d(x.layout, h, w)
		if err != nil {
			return nil, fmt.Errorf("upsampleNearest2d forward: failed to upsample: %w", err)
		}
		return NewFrom(data, spark.Contiguous(shape), x.dtype, x.device), nil
	}
}

// UpsampleNearest2dBackward returns a BackwardFunc for 2D nearest neighbor upsampling gradients.
func UpsampleNearest2dBackward[T spark.D](h, w int) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("upsampleNearest2d backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		_, _, srcH, srcW, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("upsampleNearest2d backward: failed to get 4D shape: %w", err)
		}
		if h%srcH != 0 || w%srcW != 0 {
			return nil, fmt.Errorf("upsampleNearest2d backward: non-integer scales: target=(%d,%d), input=(%d,%d)", h, w, srcH, srcW)
		}
		sh, sw := h/srcH, w/srcW
		if sh != sw {
			return nil, fmt.Errorf("upsampleNearest2d backward: non-uniform scales: scale_h=%d, scale_w=%d", sh, sw)
		}
		p, err := g.AvgPool2d(sh, sw, sh, sw)
		if err != nil {
			return nil, fmt.Errorf("upsampleNearest2d backward: failed to avgpool grad: %w", err)
		}
		dx, err := p.MulScalar(float64(sh * sw))
		if err != nil {
			return nil, fmt.Errorf("upsampleNearest2d backward: failed to scale grad: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// GatherForward returns a ForwardFunc for gathering elements along a dimension.
func GatherForward[T spark.D](dim int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("gather forward: expected 2 inputs, got %d", len(inputs))
		}
		x, idx := inputs[0], inputs[1]
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("gather forward: failed to resolve dim: %w", err)
		}
		data, err := x.storage.Gather(x.layout, idx.storage, idx.layout, d)
		if err != nil {
			return nil, fmt.Errorf("gather forward: failed to gather: %w", err)
		}
		s := append(append([]int{}, x.Dims()[:d]...), idx.Dims()...)
		s = append(s, x.Dims()[d+1:]...)
		return NewFrom(data, spark.Contiguous(spark.NewShapeFrom(s)), x.dtype, x.device), nil
	}
}

// GatherBackward returns a BackwardFunc for gather gradients.
func GatherBackward[T spark.D](dim int) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("gather backward: expected 2 inputs, got %d", len(inputs))
		}
		x, idx := inputs[0].Detach(), inputs[1]
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("gather backward: failed to resolve dim: %w", err)
		}
		dx, err := Zeros[T](x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("gather backward: failed to create input grad: %w", err)
		}
		data, err := dx.storage.ScatterAdd(dx.layout, idx.storage, idx.layout, g.storage, g.layout, d)
		if err != nil {
			return nil, fmt.Errorf("gather backward: failed to scatter-add: %w", err)
		}
		dx = NewFrom(data, dx.layout, dx.dtype, dx.device)
		ig, err := Zeros[T](idx.Shape(), idx.Device())
		if err != nil {
			return nil, fmt.Errorf("gather backward: failed to create idx grad: %w", err)
		}
		return []*Tensor[T]{dx, ig}, nil
	}
}

// ScatterForward returns a ForwardFunc for scattering source elements into a destination tensor along a dimension.
func ScatterForward[T spark.D](dim int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 3 {
			return nil, fmt.Errorf("scatter forward: expected 3 inputs, got %d", len(inputs))
		}
		x, idx, y := inputs[0], inputs[1], inputs[2]
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("scatter forward: failed to resolve dim: %w", err)
		}
		data, err := x.storage.Scatter(x.layout, idx.storage, idx.layout, y.storage, y.layout, d)
		if err != nil {
			return nil, fmt.Errorf("scatter forward: failed to scatter: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// ScatterBackward returns a BackwardFunc for scatter gradients.
func ScatterBackward[T spark.D](dim int) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 3 {
			return nil, fmt.Errorf("scatter backward: expected 3 inputs, got %d", len(inputs))
		}
		idx := inputs[1].Detach()
		d, err := spark.ResolveAxis(dim, inputs[0].Rank())
		if err != nil {
			return nil, fmt.Errorf("scatter backward: failed to resolve dim: %w", err)
		}
		sg, err := g.Gather(idx, d)
		if err != nil {
			return nil, fmt.Errorf("scatter backward: failed to gather src grad: %w", err)
		}
		ig, err := Zeros[T](idx.Shape(), idx.Device())
		if err != nil {
			return nil, fmt.Errorf("scatter backward: failed to create idx grad: %w", err)
		}
		return []*Tensor[T]{g, ig, sg}, nil
	}
}

func ScatterAddForward[T spark.D](dim int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 3 {
			return nil, fmt.Errorf("scatterAdd forward: expected 3 inputs, got %d", len(inputs))
		}
		x, idx, y := inputs[0], inputs[1], inputs[2]
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("scatterAdd forward: failed to resolve dim: %w", err)
		}
		data, err := x.storage.ScatterAdd(x.layout, idx.storage, idx.layout, y.storage, y.layout, d)
		if err != nil {
			return nil, fmt.Errorf("scatterAdd forward: failed to scatter-add: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// ScatterAddBackward returns a BackwardFunc for scatter-add gradients.
func ScatterAddBackward[T spark.D](dim int) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 3 {
			return nil, fmt.Errorf("scatterAdd backward: expected 3 inputs, got %d", len(inputs))
		}
		idx := inputs[1].Detach()
		d, err := spark.ResolveAxis(dim, inputs[0].Rank())
		if err != nil {
			return nil, fmt.Errorf("scatterAdd backward: failed to resolve dim: %w", err)
		}
		sg, err := g.Gather(idx, d)
		if err != nil {
			return nil, fmt.Errorf("scatterAdd backward: failed to gather src grad: %w", err)
		}
		ig, err := Zeros[T](idx.Shape(), idx.Device())
		if err != nil {
			return nil, fmt.Errorf("scatterAdd backward: failed to create idx grad: %w", err)
		}
		return []*Tensor[T]{g, ig, sg}, nil
	}
}

// ReduceSumForward returns a ForwardFunc for summing along specified dimensions.
func ReduceSumForward[T spark.D](dims []int, keepdim bool) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("reduceSum forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		d, err := spark.ResolveAxes(dims, x.Shape())
		if err != nil {
			return nil, fmt.Errorf("reduceSum forward: failed to resolve dims: %w", err)
		}
		data, err := x.storage.Sum(x.layout, d)
		if err != nil {
			return nil, fmt.Errorf("reduceSum forward: failed to sum: %w", err)
		}
		s := make([]int, len(x.Dims()))
		copy(s, x.Dims())
		for _, i := range d {
			s[i] = 1
		}
		shape := spark.NewShapeFrom(s)
		r := NewFrom(data, spark.Contiguous(shape), x.dtype, x.device)
		if keepdim {
			return r, nil
		}
		return r.SqueezeDims(d)
	}
}

// ReduceSumBackward returns a BackwardFunc for sum gradients.
func ReduceSumBackward[T spark.D](dims []int, keepdim bool) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("reduceSum backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d, err := spark.ResolveAxes(dims, x.Shape())
		if err != nil {
			return nil, fmt.Errorf("reduceSum backward: failed to resolve dims: %w", err)
		}
		r := g
		if !keepdim {
			s := make([]int, len(x.Dims()))
			copy(s, x.Dims())
			for _, i := range d {
				s[i] = 1
			}
			r, err = g.Reshape(s...)
			if err != nil {
				return nil, fmt.Errorf("reduceSum backward: failed to reshape: %w", err)
			}
		}
		dx, err := r.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("reduceSum backward: failed to broadcast: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ReduceMeanForward returns a ForwardFunc for computing the mean along specified dimensions.
func ReduceMeanForward[T spark.D](dims []int, keepdim bool) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("reduceMean forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		d, err := spark.ResolveAxes(dims, x.Shape())
		if err != nil {
			return nil, fmt.Errorf("reduceMean forward: failed to resolve dims: %w", err)
		}
		n := 1
		for _, i := range d {
			n *= x.Dims()[i]
		}
		s, err := x.ReduceSum(d, keepdim)
		if err != nil {
			return nil, fmt.Errorf("reduceMean forward: failed to sum: %w", err)
		}
		m, err := s.DivScalar(float64(n))
		if err != nil {
			return nil, fmt.Errorf("reduceMean forward: failed to divide: %w", err)
		}
		return m, nil
	}
}

// ReduceMeanBackward returns a BackwardFunc for mean gradients along specified dimensions.
func ReduceMeanBackward[T spark.D](dims []int, keepdim bool) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("reduceMean backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d, err := spark.ResolveAxes(dims, x.Shape())
		if err != nil {
			return nil, fmt.Errorf("reduceMean backward: failed to resolve dims: %w", err)
		}
		n := 1
		for _, i := range d {
			n *= x.Dims()[i]
		}
		m, err := g.MulScalar(1.0 / float64(n))
		if err != nil {
			return nil, fmt.Errorf("reduceMean backward: failed to scale: %w", err)
		}
		if !keepdim {
			s := make([]int, len(x.Dims()))
			copy(s, x.Dims())
			for _, i := range d {
				s[i] = 1
			}
			m, err = m.Reshape(s...)
			if err != nil {
				return nil, fmt.Errorf("reduceMean backward: failed to reshape: %w", err)
			}
		}
		dx, err := m.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("reduceMean backward: failed to broadcast: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ReduceMinForward returns a ForwardFunc for computing the minimum along a specified dimension.
func ReduceMinForward[T spark.D](dim int, keepdim bool) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("reduceMin forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("reduceMin forward: failed to resolve dim: %w", err)
		}
		data, err := x.storage.Min(x.layout, d)
		if err != nil {
			return nil, fmt.Errorf("reduceMin forward: failed to compute min: %w", err)
		}
		s := make([]int, len(x.Dims()))
		copy(s, x.Dims())
		s[d] = 1
		r := NewFrom(data, spark.Contiguous(spark.NewShapeFrom(s)), x.dtype, x.device)
		if keepdim {
			return r, nil
		}
		return r.SqueezeDims([]int{d})
	}
}

// ReduceMinBackward returns a BackwardFunc for minimum gradients along a specified dimension.
func ReduceMinBackward[T spark.D](dim int, keepdim bool) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("reduceMin backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("reduceMin backward: failed to resolve dim: %w", err)
		}
		m, err := x.Min(d)
		if err != nil {
			return nil, fmt.Errorf("reduceMin backward: failed to compute min: %w", err)
		}
		mb := m
		if !keepdim {
			mb, err = m.Unsqueeze(d)
			if err != nil {
				return nil, fmt.Errorf("reduceMin backward: failed to unsqueeze: %w", err)
			}
		}
		n, err := x.BroadcastEq(mb)
		if err != nil {
			return nil, fmt.Errorf("reduceMin backward: failed to compute mask: %w", err)
		}
		r := g
		if !keepdim {
			s := make([]int, len(x.Dims()))
			copy(s, x.Dims())
			s[d] = 1
			r, err = g.Reshape(s...)
			if err != nil {
				return nil, fmt.Errorf("reduceMin backward: failed to reshape: %w", err)
			}
		}
		gb, err := r.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("reduceMin backward: failed to broadcast: %w", err)
		}
		dx, err := gb.Mul(n)
		if err != nil {
			return nil, fmt.Errorf("reduceMin backward: failed to apply mask: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ReduceMaxForward returns a ForwardFunc for computing the maximum along a specified dimension.
func ReduceMaxForward[T spark.D](dim int, keepdim bool) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("reduceMax forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("reduceMax forward: failed to resolve dim: %w", err)
		}
		data, err := x.storage.Max(x.layout, d)
		if err != nil {
			return nil, fmt.Errorf("reduceMax forward: failed to compute max: %w", err)
		}
		s := make([]int, len(x.Dims()))
		copy(s, x.Dims())
		s[d] = 1
		// fmt.Printf("DEBUG ReduceMaxForward: x.shape=%v, d=%d, keepdim=%v, intermediate_shape=%v\n", x.Shape(), d, keepdim, s)
		r := NewFrom(data, spark.Contiguous(spark.NewShapeFrom(s)), x.dtype, x.device)
		if keepdim {
			return r, nil
		}
		// fmt.Printf("DEBUG ReduceMaxForward: final_shape=%v\n", s)
		r.SqueezeDims([]int{d})
		// fmt.Printf("DEBUG ReduceMaxForward: final_shape=%v\n", r.Shape())
		return r, nil
	}
}

// ReduceMaxBackward returns a BackwardFunc for maximum gradients along a specified dimension.
func ReduceMaxBackward[T spark.D](dim int, keepdim bool) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("reduceMax backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("reduceMax backward: failed to resolve dim: %w", err)
		}
		m, err := x.Max(d)
		if err != nil {
			return nil, fmt.Errorf("reduceMax backward: failed to compute max: %w", err)
		}
		mb := m
		if !keepdim {
			mb, err = m.Unsqueeze(d)
			if err != nil {
				return nil, fmt.Errorf("reduceMax backward: failed to unsqueeze: %w", err)
			}
		}
		n, err := x.BroadcastEq(mb)
		if err != nil {
			return nil, fmt.Errorf("reduceMax backward: failed to compute mask: %w", err)
		}
		r := g
		if !keepdim {
			s := make([]int, len(x.Dims()))
			copy(s, x.Dims())
			s[d] = 1
			r, err = g.Reshape(s...)
			if err != nil {
				return nil, fmt.Errorf("reduceMax backward: failed to reshape: %w", err)
			}
		}
		gb, err := r.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("reduceMax backward: failed to broadcast: %w", err)
		}
		dx, err := gb.Mul(n)
		if err != nil {
			return nil, fmt.Errorf("reduceMax backward: failed to apply mask: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// FastMinForward returns a ForwardFunc for minimum over the last dimension.
func FastMinForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("fastMin forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Rank() == 0 {
			return nil, fmt.Errorf("fastMin forward: cannot reduce scalar")
		}
		data, err := x.storage.FastMin(x.layout)
		if err != nil {
			return nil, fmt.Errorf("fastMin forward: failed to compute min: %w", err)
		}
		s := make([]int, len(x.Dims()))
		copy(s, x.Dims())
		s[len(s)-1] = 1
		return NewFrom(data, spark.Contiguous(spark.NewShapeFrom(s)), x.dtype, x.device), nil
	}
}

// FastMinBackward returns a BackwardFunc for minimum gradients over the last dimension.
func FastMinBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("fastMin backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d := x.Rank() - 1
		m, err := x.FastMin()
		if err != nil {
			return nil, fmt.Errorf("fastMin backward: failed to compute min: %w", err)
		}
		mb, err := m.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("fastMin backward: failed to broadcast min: %w", err)
		}
		n, err := x.Eq(mb)
		if err != nil {
			return nil, fmt.Errorf("fastMin backward: failed to compute mask: %w", err)
		}
		s, err := n.Sum([]int{d})
		if err != nil {
			return nil, fmt.Errorf("fastMin backward: failed to sum mask: %w", err)
		}
		sb, err := s.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("fastMin backward: failed to broadcast sum: %w", err)
		}
		gb, err := g.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("fastMin backward: failed to broadcast grad: %w", err)
		}
		dx, err := n.Mul(gb)
		if err != nil {
			return nil, fmt.Errorf("fastMin backward: failed to apply mask: %w", err)
		}
		dx, err = dx.Div(sb)
		if err != nil {
			return nil, fmt.Errorf("fastMin backward: failed to normalize: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// FastMaxForward returns a ForwardFunc for maximum over the last dimension.
func FastMaxForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("fastMax forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Rank() == 0 {
			return nil, fmt.Errorf("fastMax forward: cannot reduce scalar")
		}
		data, err := x.storage.FastMax(x.layout)
		if err != nil {
			return nil, fmt.Errorf("fastMax forward: failed to compute max: %w", err)
		}
		s := make([]int, len(x.Dims()))
		copy(s, x.Dims())
		s[len(s)-1] = 1
		return NewFrom(data, spark.Contiguous(spark.NewShapeFrom(s)), x.dtype, x.device), nil
	}
}

// FastMaxBackward returns a BackwardFunc for maximum gradients over the last dimension.
func FastMaxBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("fastMax backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d := x.Rank() - 1
		m, err := x.FastMax()
		if err != nil {
			return nil, fmt.Errorf("fastMax backward: failed to compute max: %w", err)
		}
		mb, err := m.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("fastMax backward: failed to broadcast max: %w", err)
		}
		n, err := x.Eq(mb)
		if err != nil {
			return nil, fmt.Errorf("fastMax backward: failed to compute mask: %w", err)
		}
		s, err := n.Sum([]int{d})
		if err != nil {
			return nil, fmt.Errorf("fastMax backward: failed to sum mask: %w", err)
		}
		sb, err := s.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("fastMax backward: failed to broadcast sum: %w", err)
		}
		gb, err := g.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("fastMax backward: failed to broadcast grad: %w", err)
		}
		dx, err := n.Mul(gb)
		if err != nil {
			return nil, fmt.Errorf("fastMax backward: failed to apply mask: %w", err)
		}
		dx, err = dx.Div(sb)
		if err != nil {
			return nil, fmt.Errorf("fastMax backward: failed to normalize: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// FastSoftmaxForward returns a ForwardFunc for softmax along the last dimension.
func FastSoftmaxForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("fastSoftmax forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.FastSoftmax(x.layout)
		if err != nil {
			return nil, fmt.Errorf("fastSoftmax forward: failed to compute softmax: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// FastSoftmaxBackward returns a BackwardFunc for softmax gradients.
func FastSoftmaxBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("fastSoftmax backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.FastSoftmax()
		if err != nil {
			return nil, fmt.Errorf("fastSoftmax backward: failed to compute softmax: %w", err)
		}
		gs, err := g.Mul(s)
		if err != nil {
			return nil, fmt.Errorf("fastSoftmax backward: failed to compute g*s: %w", err)
		}
		r, err := gs.Sum([]int{x.Rank() - 1})
		if err != nil {
			return nil, fmt.Errorf("fastSoftmax backward: failed to sum: %w", err)
		}
		rb, err := r.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("fastSoftmax backward: failed to broadcast: %w", err)
		}
		d, err := g.Sub(rb)
		if err != nil {
			return nil, fmt.Errorf("fastSoftmax backward: failed to compute g-rb: %w", err)
		}
		dx, err := s.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("fastSoftmax backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// WhereCondForward returns a ForwardFunc for conditional selection: c ? t : f.
func WhereCondForward[T spark.D](c *Tensor[T]) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("whereCond forward: expected 2 inputs, got %d", len(inputs))
		}
		t, f := inputs[0], inputs[1]
		s, err := c.Shape().BroadcastShapeBinaryOp(t.Shape())
		if err != nil {
			return nil, fmt.Errorf("whereCond forward: failed to broadcast with true: %w", err)
		}
		s, err = s.BroadcastShapeBinaryOp(f.Shape())
		if err != nil {
			return nil, fmt.Errorf("whereCond forward: failed to broadcast with false: %w", err)
		}
		cl, err := c.layout.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("whereCond forward: failed to broadcast cond: %w", err)
		}
		tl, err := t.layout.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("whereCond forward: failed to broadcast true: %w", err)
		}
		fl, err := f.layout.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("whereCond forward: failed to broadcast false: %w", err)
		}
		data, err := c.storage.WhereCond(cl, t.storage, tl, f.storage, fl)
		if err != nil {
			return nil, fmt.Errorf("whereCond forward: failed to compute where: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), t.dtype, t.device), nil
	}
}

// WhereCondBackward returns a BackwardFunc for conditional selection gradients.
func WhereCondBackward[T spark.D](c *Tensor[T]) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("whereCond backward: expected 2 inputs, got %d", len(inputs))
		}
		gd := g.Detach()
		z, err := Zeros[T](g.Shape(), g.Device())
		if err != nil {
			return nil, fmt.Errorf("whereCond backward: failed to create zeros: %w", err)
		}
		dt, err := c.storage.WhereCond(c.layout, gd.storage, gd.layout, z.storage, z.layout)
		if err != nil {
			return nil, fmt.Errorf("whereCond backward: failed to compute dt: %w", err)
		}
		df, err := c.storage.WhereCond(c.layout, z.storage, z.layout, gd.storage, gd.layout)
		if err != nil {
			return nil, fmt.Errorf("whereCond backward: failed to compute df: %w", err)
		}
		return []*Tensor[T]{
			NewFrom(dt, g.layout.Clone(), g.dtype, g.device),
			NewFrom(df, g.layout.Clone(), g.dtype, g.device),
		}, nil
	}
}

// CopyForward returns a ForwardFunc for tensor cloning: y = x.
func CopyForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("copy forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Copy(s, x.storage)
		if err != nil {
			return nil, fmt.Errorf("copy forward: failed to copy: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// CopyBackward returns a BackwardFunc for clone gradients: ∂y/∂x = 1.
func CopyBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("copy backward: expected 1 input, got %d", len(inputs))
		}
		return []*Tensor[T]{g}, nil
	}
}

// NegForward returns a ForwardFunc for element-wise negation: -x.
func NegForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("neg forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Neg(x.layout)
		if err != nil {
			return nil, fmt.Errorf("neg forward: failed to compute neg: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// NegBackward returns a BackwardFunc for negation gradients: ∂(-x)/∂x = -1.
func NegBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("neg backward: expected 1 input, got %d", len(inputs))
		}
		dx, err := g.Neg()
		if err != nil {
			return nil, fmt.Errorf("neg backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// RecipForward returns a ForwardFunc for element-wise reciprocal: 1/x.
func RecipForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("recip forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Recip(x.layout)
		if err != nil {
			return nil, fmt.Errorf("recip forward: failed to compute recip: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// RecipBackward returns a BackwardFunc for reciprocal gradients: ∂(1/x)/∂x = -1/x².
func RecipBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("recip backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("recip backward: failed to compute x²: %w", err)
		}
		dx, err := g.Div(x2)
		if err != nil {
			return nil, fmt.Errorf("recip backward: failed to compute dx: %w", err)
		}
		dx, err = dx.Neg()
		if err != nil {
			return nil, fmt.Errorf("recip backward: failed to negate dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ExpForward returns a ForwardFunc for element-wise exponential: exp(x).
func ExpForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("exp forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Exp(x.layout)
		if err != nil {
			return nil, fmt.Errorf("exp forward: failed to compute exp: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// ExpBackward returns a BackwardFunc for exponential gradients: ∂exp(x)/∂x = exp(x).
func ExpBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("exp backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		e, err := x.Exp()
		if err != nil {
			return nil, fmt.Errorf("exp backward: failed to compute exp: %w", err)
		}
		dx, err := g.Mul(e)
		if err != nil {
			return nil, fmt.Errorf("exp backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// LogForward returns a ForwardFunc for element-wise natural logarithm: log(x).
func LogForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("log forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Log(x.layout)
		if err != nil {
			return nil, fmt.Errorf("log forward: failed to compute log: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// LogBackward returns a BackwardFunc for logarithm gradients: ∂log(x)/∂x = 1/x.
func LogBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("log backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		dx, err := g.Div(x)
		if err != nil {
			return nil, fmt.Errorf("log backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SinForward returns a ForwardFunc for element-wise sine: sin(x).
func SinForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("sin forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sin(x.layout)
		if err != nil {
			return nil, fmt.Errorf("sin forward: failed to compute sin: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SinBackward returns a BackwardFunc for sine gradients: ∂sin(x)/∂x = cos(x).
func SinBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("sin backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		c, err := x.Cos()
		if err != nil {
			return nil, fmt.Errorf("sin backward: failed to compute cos: %w", err)
		}
		dx, err := g.Mul(c)
		if err != nil {
			return nil, fmt.Errorf("sin backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// CosForward returns a ForwardFunc for element-wise cosine: cos(x).
func CosForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("cos forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Cos(x.layout)
		if err != nil {
			return nil, fmt.Errorf("cos forward: failed to compute cos: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// CosBackward returns a BackwardFunc for cosine gradients: ∂cos(x)/∂x = -sin(x).
func CosBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("cos backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.Sin()
		if err != nil {
			return nil, fmt.Errorf("cos backward: failed to compute sin: %w", err)
		}
		s, err = s.Neg()
		if err != nil {
			return nil, fmt.Errorf("cos backward: failed to negate: %w", err)
		}
		dx, err := g.Mul(s)
		if err != nil {
			return nil, fmt.Errorf("cos backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// TanhForward returns a ForwardFunc for element-wise hyperbolic tangent: tanh(x).
func TanhForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("tanh forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Tanh(x.layout)
		if err != nil {
			return nil, fmt.Errorf("tanh forward: failed to compute tanh: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// TanhBackward returns a BackwardFunc for hyperbolic tangent gradients: ∂tanh(x)/∂x = 1 - tanh²(x).
func TanhBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("tanh backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		t, err := x.Tanh()
		if err != nil {
			return nil, fmt.Errorf("tanh backward: failed to compute tanh: %w", err)
		}
		t2, err := t.Sqr()
		if err != nil {
			return nil, fmt.Errorf("tanh backward: failed to compute tanh²: %w", err)
		}
		d, err := t2.Neg()
		if err != nil {
			return nil, fmt.Errorf("tanh backward: failed to negate: %w", err)
		}
		d, err = d.AddScalar(1.0)
		if err != nil {
			return nil, fmt.Errorf("tanh backward: failed to add one: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("tanh backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ErfForward returns a ForwardFunc for element-wise error function: erf(x).
func ErfForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("erf forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Erf(x.layout)
		if err != nil {
			return nil, fmt.Errorf("erf forward: failed to compute erf: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// ErfBackward returns a BackwardFunc for error function gradients: ∂erf(x)/∂x = (2/√π)exp(-x²).
func ErfBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("erf backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("erf backward: failed to compute x²: %w", err)
		}
		n, err := x2.Neg()
		if err != nil {
			return nil, fmt.Errorf("erf backward: failed to negate: %w", err)
		}
		e, err := n.Exp()
		if err != nil {
			return nil, fmt.Errorf("erf backward: failed to compute exp: %w", err)
		}
		d, err := e.MulScalar(1.1283791670955126) // (2/√π) * exp(-x²)
		if err != nil {
			return nil, fmt.Errorf("erf backward: failed to scale: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("erf backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// CeilForward returns a ForwardFunc for element-wise ceiling: ceil(x).
func CeilForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("ceil forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Ceil(x.layout)
		if err != nil {
			return nil, fmt.Errorf("ceil forward: failed to compute ceil: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// CeilBackward returns a BackwardFunc for ceiling gradients: ∂ceil(x)/∂x = 0.
func CeilBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("ceil backward: expected 1 input, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("ceil backward: failed to create zeros: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// FloorForward returns a ForwardFunc for element-wise floor: floor(x).
func FloorForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("floor forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Floor(x.layout)
		if err != nil {
			return nil, fmt.Errorf("floor forward: failed to compute floor: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// FloorBackward returns a BackwardFunc for floor gradients: ∂floor(x)/∂x = 0.
func FloorBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("floor backward: expected 1 input, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("floor backward: failed to create zeros: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// RoundForward returns a ForwardFunc for element-wise rounding: round(x).
func RoundForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("round forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Round(x.layout)
		if err != nil {
			return nil, fmt.Errorf("round forward: failed to compute round: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// RoundBackward returns a BackwardFunc for rounding gradients: ∂round(x)/∂x = 0.
func RoundBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("round backward: expected 1 input, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("round backward: failed to create zeros: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// NormcdfForward returns a ForwardFunc for element-wise normal CDF: Φ(x).
func NormcdfForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("normcdf forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Normcdf(x.layout)
		if err != nil {
			return nil, fmt.Errorf("normcdf forward: failed to compute normcdf: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// NormcdfBackward returns a BackwardFunc for normal CDF gradients: ∂Φ(x)/∂x = φ(x) = (1/√(2π))exp(-x²/2).
func NormcdfBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("normcdf backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("normcdf backward: failed to compute x²: %w", err)
		}
		n, err := x2.MulScalar(0.5)
		if err != nil {
			return nil, fmt.Errorf("normcdf backward: failed to compute x²/2: %w", err)
		}
		n, err = n.Neg()
		if err != nil {
			return nil, fmt.Errorf("normcdf backward: failed to negate: %w", err)
		}
		e, err := n.Exp()
		if err != nil {
			return nil, fmt.Errorf("normcdf backward: failed to compute exp: %w", err)
		}
		d, err := e.MulScalar(0.3989422804014327) // (1/√(2π)) * exp(-x²/2)
		if err != nil {
			return nil, fmt.Errorf("normcdf backward: failed to scale: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("normcdf backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// AbsForward returns a ForwardFunc for element-wise absolute value: |x|.
func AbsForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("abs forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Abs(x.layout)
		if err != nil {
			return nil, fmt.Errorf("abs forward: failed to compute abs: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// AbsBackward returns a BackwardFunc for absolute value gradients: ∂|x|/∂x = sign(x).
func AbsBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("abs backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.Sign()
		if err != nil {
			return nil, fmt.Errorf("abs backward: failed to compute sign: %w", err)
		}
		dx, err := g.Mul(s)
		if err != nil {
			return nil, fmt.Errorf("abs backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SqrForward returns a ForwardFunc for element-wise square: x².
func SqrForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("sqr forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sqr(x.layout)
		if err != nil {
			return nil, fmt.Errorf("sqr forward: failed to compute sqr: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SqrBackward returns a BackwardFunc for square gradients: ∂(x²)/∂x = 2x.
func SqrBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("sqr backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d, err := x.MulScalar(2.0)
		if err != nil {
			return nil, fmt.Errorf("sqr backward: failed to compute 2x: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("sqr backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SqrtForward returns a ForwardFunc for element-wise square root: √x.
func SqrtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("sqrt forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sqrt(x.layout)
		if err != nil {
			return nil, fmt.Errorf("sqrt forward: failed to compute sqrt: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SqrtBackward returns a BackwardFunc for square root gradients: ∂√x/∂x = 1/(2√x).
func SqrtBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("sqrt backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.Sqrt()
		if err != nil {
			return nil, fmt.Errorf("sqrt backward: failed to compute sqrt: %w", err)
		}
		d, err := s.MulScalar(2.0)
		if err != nil {
			return nil, fmt.Errorf("sqrt backward: failed to compute 2√x: %w", err)
		}
		dx, err := g.Div(d)
		if err != nil {
			return nil, fmt.Errorf("sqrt backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// GeluForward returns a ForwardFunc for element-wise GELU: gelu(x).
func GeluForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("gelu forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Gelu(x.layout)
		if err != nil {
			return nil, fmt.Errorf("gelu forward: failed to compute gelu: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// GeluBackward returns a BackwardFunc for GELU gradients.
func GeluBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("gelu backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute x²: %w", err)
		}
		x3, err := x2.Mul(x)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute x³: %w", err)
		}
		cx3, err := x3.MulScalar(0.044715)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute c*x³: %w", err)
		}
		i, err := x.Add(cx3)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute x+c*x³: %w", err)
		}
		a, err := i.MulScalar(0.7978845608028654) // √(2/π)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute a: %w", err)
		}
		t, err := a.Tanh()
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute tanh: %w", err)
		}
		p, err := t.AddScalar(1.0)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute 1+tanh: %w", err)
		}
		f, err := p.MulScalar(0.5)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute first: %w", err)
		}
		t2, err := t.Sqr()
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute tanh²: %w", err)
		}
		o, err := t2.Neg()
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to negate: %w", err)
		}
		o, err = o.AddScalar(1.0)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute sech²: %w", err)
		}
		n, err := x2.MulScalar(0.134145) // 3*c
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute 3*c*x²: %w", err)
		}
		n, err = n.AddScalar(1.0)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute 1+3*c*x²: %w", err)
		}
		r, err := x.MulScalar(0.5)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute half*x: %w", err)
		}
		r, err = r.Mul(o)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute sech² term: %w", err)
		}
		r, err = r.MulScalar(0.7978845608028654)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute sqrt2/pi term: %w", err)
		}
		r, err = r.Mul(n)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute second: %w", err)
		}
		d, err := f.Add(r)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("gelu backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// GeluErfForward returns a ForwardFunc for element-wise GELU (erf-based): gelu_erf(x).
func GeluErfForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("gelu erf forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.GeluErf(x.layout)
		if err != nil {
			return nil, fmt.Errorf("gelu erf forward: failed to compute gelu_erf: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// GeluErfBackward returns a BackwardFunc for GELU (erf-based) gradients.
func GeluErfBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("gelu erf backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		xs, err := x.DivScalar(1.4142135623730951) // x/√2
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute x/√2: %w", err)
		}
		e, err := xs.Erf()
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute erf: %w", err)
		}
		p, err := e.AddScalar(1.0)
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute 1+erf: %w", err)
		}
		f, err := p.MulScalar(0.5)
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute first: %w", err)
		}
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute x²: %w", err)
		}
		x2, err = x2.MulScalar(0.5)
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute x²/2: %w", err)
		}
		n, err := x2.Neg()
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to negate: %w", err)
		}
		e, err = n.Exp()
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute exp: %w", err)
		}
		r, err := x.MulScalar(0.5)
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute half*x: %w", err)
		}
		r, err = r.MulScalar(0.7978845608028654)
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute coeff term: %w", err)
		}
		r, err = r.Mul(e)
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute exp term: %w", err)
		}
		d, err := f.Add(r)
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("gelu erf backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ReluForward returns a ForwardFunc for element-wise ReLU: max(0, x).
func ReluForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("relu forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Relu(x.layout)
		if err != nil {
			return nil, fmt.Errorf("relu forward: failed to compute relu: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// ReluBackward returns a BackwardFunc for ReLU gradients: ∂relu(x)/∂x = x > 0 ? 1 : 0.
func ReluBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("relu backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		z, err := Zeros[T](x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("relu backward: failed to create zero: %w", err)
		}
		m, err := x.Gt(z)
		if err != nil {
			return nil, fmt.Errorf("relu backward: failed to compute mask: %w", err)
		}
		z, err = Zeros[T](g.Shape(), g.Device())
		if err != nil {
			return nil, fmt.Errorf("relu backward: failed to create zeros: %w", err)
		}
		dx, err := m.WhereCond(g, z)
		if err != nil {
			return nil, fmt.Errorf("relu backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// EluForward returns a ForwardFunc for element-wise ELU: x if x >= 0, alpha*(exp(x)-1) if x < 0.
func EluForward[T spark.D](alpha float64) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("elu forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Elu(x.layout, T(alpha))
		if err != nil {
			return nil, fmt.Errorf("elu forward: failed to compute elu: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// EluBackward returns a BackwardFunc for ELU gradients: ∂elu(x)/∂x = 1 if x >= 0, alpha*exp(x) if x < 0.
func EluBackward[T spark.D](alpha float64) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("elu backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		z, err := Zeros[T](x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("elu backward: failed to create zero: %w", err)
		}
		m, err := x.Ge(z)
		if err != nil {
			return nil, fmt.Errorf("elu backward: failed to compute mask: %w", err)
		}
		o, err := Ones[T](x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("elu backward: failed to create one: %w", err)
		}
		e, err := x.Exp()
		if err != nil {
			return nil, fmt.Errorf("elu backward: failed to compute exp: %w", err)
		}
		ae, err := e.MulScalar(alpha)
		if err != nil {
			return nil, fmt.Errorf("elu backward: failed to compute alpha*exp: %w", err)
		}
		d, err := m.WhereCond(o, ae)
		if err != nil {
			return nil, fmt.Errorf("elu backward: failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("elu backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SiluForward returns a ForwardFunc for element-wise SiLU: x * sigmoid(x).
func SiluForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("silu forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Silu(x.layout)
		if err != nil {
			return nil, fmt.Errorf("silu forward: failed to compute silu: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SiluBackward returns a BackwardFunc for SiLU gradients: ∂silu(x)/∂x = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x)).
func SiluBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("silu backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.Sigmoid()
		if err != nil {
			return nil, fmt.Errorf("silu backward: failed to compute sigmoid: %w", err)
		}
		n, err := s.Neg()
		if err != nil {
			return nil, fmt.Errorf("silu backward: failed to negate: %w", err)
		}
		n, err = n.AddScalar(1.0)
		if err != nil {
			return nil, fmt.Errorf("silu backward: failed to compute 1-sigmoid: %w", err)
		}
		xs, err := x.Mul(s)
		if err != nil {
			return nil, fmt.Errorf("silu backward: failed to compute x*sigmoid: %w", err)
		}
		r, err := xs.Mul(n)
		if err != nil {
			return nil, fmt.Errorf("silu backward: failed to compute x*sigmoid*(1-sigmoid): %w", err)
		}
		d, err := s.Add(r)
		if err != nil {
			return nil, fmt.Errorf("silu backward: failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("silu backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// PowfForward returns a ForwardFunc for element-wise power: x^param.
func PowfForward[T spark.D](p float64) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("powf forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Powf(x.layout, T(p))
		if err != nil {
			return nil, fmt.Errorf("powf forward: failed to compute pow: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// PowfBackward returns a BackwardFunc for power gradients: ∂(x^param)/∂x = param*x^(param-1).
func PowfBackward[T spark.D](p float64) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("powf backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		e, err := x.Powf(p - 1)
		if err != nil {
			return nil, fmt.Errorf("powf backward: failed to compute x^(param-1): %w", err)
		}
		d, err := e.MulScalar(p)
		if err != nil {
			return nil, fmt.Errorf("powf backward: failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("powf backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SigmoidForward returns a ForwardFunc for element-wise sigmoid: σ(x) = 1/(1+exp(-x)).
func SigmoidForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("sigmoid forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sigmoid(x.layout)
		if err != nil {
			return nil, fmt.Errorf("sigmoid forward: failed to compute sigmoid: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SigmoidBackward returns a BackwardFunc for sigmoid gradients: ∂σ(x)/∂x = σ(x)*(1-σ(x)).
func SigmoidBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("sigmoid backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.Sigmoid()
		if err != nil {
			return nil, fmt.Errorf("sigmoid backward: failed to compute sigmoid: %w", err)
		}
		n, err := s.Neg()
		if err != nil {
			return nil, fmt.Errorf("sigmoid backward: failed to negate: %w", err)
		}
		n, err = n.AddScalar(1.0)
		if err != nil {
			return nil, fmt.Errorf("sigmoid backward: failed to compute 1-sigmoid: %w", err)
		}
		d, err := s.Mul(n)
		if err != nil {
			return nil, fmt.Errorf("sigmoid backward: failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("sigmoid backward: failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SignForward returns a ForwardFunc for element-wise sign: sign(x).
func SignForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("sign forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sign(x.layout)
		if err != nil {
			return nil, fmt.Errorf("sign forward: failed to compute sign: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SignBackward returns a BackwardFunc for sign gradients: ∂sign(x)/∂x = 0.
func SignBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("sign backward: expected 1 input, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("sign backward: failed to create zeros: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// TransposeForward returns a ForwardFunc for transposing dimensions.
func TransposeForward[T spark.D](dim1, dim2 int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("transpose forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		d1, err := spark.ResolveAxis(dim1, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("transpose forward: failed to resolve dim1: %w", err)
		}
		d2, err := spark.ResolveAxis(dim2, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("transpose forward: failed to resolve dim2: %w", err)
		}
		if d1 == d2 {
			return x, nil
		}
		l, err := x.layout.Transpose(d1, d2)
		if err != nil {
			return nil, fmt.Errorf("transpose forward: failed to transpose: %w", err)
		}
		return NewFrom(x.storage, l, x.dtype, x.device), nil
	}
}

// TransposeBackward returns a BackwardFunc for transpose gradients.
func TransposeBackward[T spark.D](dim1, dim2 int) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("transpose backward: expected 1 input, got %d", len(inputs))
		}
		dx, err := g.Transpose(dim1, dim2)
		if err != nil {
			return nil, fmt.Errorf("transpose backward: failed to transpose: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SqueezeForward returns a ForwardFunc for squeezing a dimension.
func SqueezeForward[T spark.D](dim int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("squeeze forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("squeeze forward: failed to resolve dim: %w", err)
		}
		if x.Dim(d) != 1 {
			return x, nil
		}
		s := make([]int, 0, x.Rank()-1)
		t := make([]int, 0, x.Rank()-1)
		for i := 0; i < x.Rank(); i++ {
			if i == d {
				continue
			}
			s = append(s, x.Dims()[i])
			t = append(t, x.Stride()[i])
		}
		return NewFrom(x.storage, spark.NewLayout(spark.NewShapeFrom(s), t, x.layout.StartOffset()), x.dtype, x.device), nil
	}
}

// SqueezeBackward returns a BackwardFunc for squeeze gradients.
func SqueezeBackward[T spark.D](dim int) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("squeeze backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("squeeze backward: failed to resolve dim: %w", err)
		}
		if x.Dim(d) != 1 {
			return []*Tensor[T]{g}, nil
		}
		dx, err := g.Unsqueeze(d)
		if err != nil {
			return nil, fmt.Errorf("squeeze backward: failed to unsqueeze: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// UnsqueezeForward returns a ForwardFunc for unsqueezing a dimension.
func UnsqueezeForward[T spark.D](dim int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("unsqueeze forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		d := dim
		if d < 0 {
			d += x.Rank() + 1
		}
		if d < 0 || d > x.Rank() {
			return nil, fmt.Errorf("unsqueeze forward: dim out of range [-%d, %d], got %d", x.Rank()+1, x.Rank(), dim)
		}
		s := make([]int, 0, x.Rank()+1)
		t := make([]int, 0, x.Rank()+1)
		s = append(s, x.Dims()[:d]...)
		s = append(s, 1)
		s = append(s, x.Dims()[d:]...)
		t = append(t, x.Stride()[:d]...)
		stride := 1
		if d < x.Rank() {
			stride = x.Stride()[d]
		}
		t = append(t, stride)
		t = append(t, x.Stride()[d:]...)
		return NewFrom(x.storage, spark.NewLayout(spark.NewShapeFrom(s), t, x.layout.StartOffset()), x.dtype, x.device), nil
	}
}

// UnsqueezeBackward returns a BackwardFunc for unsqueeze gradients.
func UnsqueezeBackward[T spark.D](dim int) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("unsqueeze backward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d := dim
		if d < 0 {
			d += x.Rank() + 1
		}
		dx, err := g.Squeeze(d)
		if err != nil {
			return nil, fmt.Errorf("unsqueeze backward: failed to squeeze: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ReshapeForward returns a ForwardFunc for reshaping to a new shape.
func ReshapeForward[T spark.D](s *spark.Shape) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("reshape forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if s.ElemCount() != x.Shape().ElemCount() {
			return nil, fmt.Errorf("reshape forward: element count mismatch: %d vs %d", s.ElemCount(), x.Shape().ElemCount())
		}
		if !x.layout.IsContiguous() {
			return nil, fmt.Errorf("reshape forward: non-contiguous tensor not supported")
		}
		return NewFrom(x.storage, spark.ContiguousWithOffset(s, x.layout.StartOffset()), x.dtype, x.device), nil
	}
}

// ReshapeBackward returns a BackwardFunc for reshape gradients.
func ReshapeBackward[T spark.D](s *spark.Shape) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("reshape backward: expected 1 input, got %d", len(inputs))
		}
		dx, err := g.Reshape(s.Dims()...)
		if err != nil {
			return nil, fmt.Errorf("reshape backward: failed to reshape: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// BroadcastAsForward returns a ForwardFunc for broadcasting to a target shape.
func BroadcastAsForward[T spark.D](s *spark.Shape) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("broadcast as forward: expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Shape().Equal(s) {
			return x, nil
		}
		l, err := x.layout.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("broadcast as forward: failed to broadcast %v to %v: %w", x.Shape(), s, err)
		}
		data, err := x.storage.Clone()
		if err != nil {
			return nil, fmt.Errorf("broadcast as forward: failed to clone: %w", err)
		}
		return NewFrom(data, l, x.dtype, x.device), nil
	}
}

// BroadcastAsBackward returns a BackwardFunc for broadcasting gradients.
func BroadcastAsBackward[T spark.D](s *spark.Shape) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("broadcast as backward: expected 1 input, got %d", len(inputs))
		}
		if g.Shape().Equal(s) {
			return []*Tensor[T]{g}, nil
		}
		dx, err := ReduceBroadcastGrad(g, s.Dims())
		if err != nil {
			return nil, fmt.Errorf("broadcast as backward: failed to reduce: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

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
		r, err = g.SumKeepDim(sd)
		if err != nil {
			return nil, fmt.Errorf("failed to sum dims: %w", err)
		}
	}
	for range n {
		var err error
		r, err = r.Squeeze(0)
		if err != nil {
			return nil, fmt.Errorf("failed to squeeze: %w", err)
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
			return nil, fmt.Errorf("failed to compute affine: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// AffineBackward returns a BackwardFunc for affine transformation gradients: ∂y/∂x = scale.
func AffineBackward[T spark.D](scale, bias float64) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		s, err := Full[T](scale, g.Shape(), g.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create scale: %w", err)
		}
		dx, err := g.Mul(s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// AddForward returns a ForwardFunc for element-wise addition: x + y.
func AddForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Add(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to add: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// AddBackward returns a BackwardFunc for addition gradients: ∂z/∂x = 1, ∂z/∂y = 1.
func AddBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		return []*Tensor[T]{g, g}, nil
	}
}

// SubForward returns a ForwardFunc for element-wise subtraction: x - y.
func SubForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Sub(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to subtract: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// SubBackward returns a BackwardFunc for subtraction gradients: ∂z/∂x = 1, ∂z/∂y = -1.
func SubBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dy, err := g.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate: %w", err)
		}
		return []*Tensor[T]{g, dy}, nil
	}
}

// MulForward returns a ForwardFunc for element-wise multiplication: x * y.
func MulForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Mul(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to multiply: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// MulBackward returns a BackwardFunc for multiplication gradients: ∂z/∂x = y, ∂z/∂y = x.
func MulBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		dx, err := g.Mul(y)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		dy, err := g.Mul(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// DivForward returns a ForwardFunc for element-wise division: x / y.
func DivForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Div(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to divide: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// DivBackward returns a BackwardFunc for division gradients: ∂z/∂x = 1/y, ∂z/∂y = -x/y².
func DivBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		dx, err := g.Div(y)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		y2, err := y.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute y²: %w", err)
		}
		xy2, err := x.Div(y2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x/y²: %w", err)
		}
		dy, err := xy2.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate x/y²: %w", err)
		}
		dy, err = g.Mul(dy)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// MaxForward returns a ForwardFunc for element-wise maximum: max(x, y).
func MaxForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Max(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute max: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// MaxBackward returns a BackwardFunc for maximum gradients: ∂z/∂x = (x >= y) ? g : 0, ∂z/∂y = (y > x) ? g : 0.
func MaxBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		mx, err := x.Ge(y)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x >= y: %w", err)
		}
		my, err := y.Gt(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute y > x: %w", err)
		}
		z, err := Zeros[T](g.Shape(), g.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		dx, err := mx.WhereCond(g, z)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		dy, err := my.WhereCond(g, z)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// MinForward returns a ForwardFunc for element-wise minimum: min(x, y).
func MinForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Min(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute min: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// MinBackward returns a BackwardFunc for minimum gradients: ∂z/∂x = (x <= y) ? g : 0, ∂z/∂y = (y < x) ? g : 0.
func MinBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		mx, err := x.Le(y)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x <= y: %w", err)
		}
		my, err := y.Lt(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute y < x: %w", err)
		}
		z, err := Zeros[T](g.Shape(), g.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		dx, err := mx.WhereCond(g, z)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		dy, err := my.WhereCond(g, z)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// EqForward returns a ForwardFunc for element-wise equality: x == y.
func EqForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Eq(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute eq: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// EqBackward returns a BackwardFunc for equality gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func EqBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// NeForward returns a ForwardFunc for element-wise inequality: x != y.
func NeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Ne(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute ne: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// NeBackward returns a BackwardFunc for inequality gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func NeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// LtForward returns a ForwardFunc for element-wise less-than: x < y.
func LtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Lt(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute lt: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// LtBackward returns a BackwardFunc for less-than gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func LtBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// LeForward returns a ForwardFunc for element-wise less-than-or-equal: x <= y.
func LeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Le(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute le: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// LeBackward returns a BackwardFunc for less-than-or-equal gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func LeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// GtForward returns a ForwardFunc for element-wise greater-than: x > y.
func GtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Gt(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute gt: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// GtBackward returns a BackwardFunc for greater-than gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func GtBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// GeForward returns a ForwardFunc for element-wise greater-than-or-equal: x >= y.
func GeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if !x.Shape().Equal(y.Shape()) {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", x.Shape(), y.Shape())
		}
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Ge(y.storage, x.layout, y.layout, s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute ge: %w", err)
		}
		return NewFrom(data, s, spark.U8, x.device), nil
	}
}

// GeBackward returns a BackwardFunc for greater-than-or-equal gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func GeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastAddForward returns a ForwardFunc for broadcasted addition: x + y.
func BroadcastAddForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Add(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("failed to add: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// BroadcastAddBackward returns a BackwardFunc for broadcasted addition gradients: ∂z/∂x = 1, ∂z/∂y = 1.
func BroadcastAddBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		dx, err := ReduceBroadcastGrad(g, x.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		dy, err := ReduceBroadcastGrad(g, y.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastSubForward returns a ForwardFunc for broadcasted subtraction: x - y.
func BroadcastSubForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Sub(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("failed to subtract: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// BroadcastSubBackward returns a BackwardFunc for broadcasted subtraction gradients: ∂z/∂x = 1, ∂z/∂y = -1.
func BroadcastSubBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		dx, err := ReduceBroadcastGrad(g, x.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		ng, err := g.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate grad: %w", err)
		}
		dy, err := ReduceBroadcastGrad(ng, y.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastMulForward returns a ForwardFunc for broadcasted multiplication: x * y.
func BroadcastMulForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Mul(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("failed to multiply: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// BroadcastMulBackward returns a BackwardFunc for broadcasted multiplication gradients: ∂z/∂x = y, ∂z/∂y = x.
func BroadcastMulBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		yb, err := y.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y for dx: %w", err)
		}
		gx, err := g.Mul(yb)
		if err != nil {
			return nil, fmt.Errorf("failed to compute g*y: %w", err)
		}
		dx, err := ReduceBroadcastGrad(gx, x.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		xb, err := x.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x for dy: %w", err)
		}
		gy, err := g.Mul(xb)
		if err != nil {
			return nil, fmt.Errorf("failed to compute g*x: %w", err)
		}
		dy, err := ReduceBroadcastGrad(gy, y.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastDivForward returns a ForwardFunc for broadcasted division: x / y.
func BroadcastDivForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Div(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("failed to divide: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// BroadcastDivBackward returns a BackwardFunc for broadcasted division gradients: ∂z/∂x = 1/y, ∂z/∂y = -x/y².
func BroadcastDivBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		yb, err := y.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y for dx: %w", err)
		}
		gx, err := g.Div(yb)
		if err != nil {
			return nil, fmt.Errorf("failed to compute g/y: %w", err)
		}
		dx, err := ReduceBroadcastGrad(gx, x.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		y2, err := yb.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute y²: %w", err)
		}
		xb, err := x.BroadcastAs(g.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x for dy: %w", err)
		}
		gy, err := g.Mul(xb)
		if err != nil {
			return nil, fmt.Errorf("failed to compute g*x: %w", err)
		}
		gy, err = gy.Div(y2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute g*x/y²: %w", err)
		}
		gy, err = gy.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate g*x/y²: %w", err)
		}
		dy, err := ReduceBroadcastGrad(gy, y.Dims())
		if err != nil {
			return nil, fmt.Errorf("failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastEqForward returns a ForwardFunc for broadcasted equality: x == y.
func BroadcastEqForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Eq(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("failed to compute eq: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastEqBackward returns a BackwardFunc for broadcasted equality gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastEqBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastNeForward returns a ForwardFunc for broadcasted inequality: x != y.
func BroadcastNeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Ne(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("failed to compute ne: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastNeBackward returns a BackwardFunc for broadcasted inequality gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastNeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastLtForward returns a ForwardFunc for broadcasted less-than: x < y.
func BroadcastLtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Lt(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("failed to compute lt: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastLtBackward returns a BackwardFunc for broadcasted less-than gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastLtBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastLeForward returns a ForwardFunc for broadcasted less-than-or-equal: x <= y.
func BroadcastLeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Le(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("failed to compute le: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastLeBackward returns a BackwardFunc for broadcasted less-than-or-equal gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastLeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastGtForward returns a ForwardFunc for broadcasted greater-than: x > y.
func BroadcastGtForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Gt(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("failed to compute gt: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastGtBackward returns a BackwardFunc for broadcasted greater-than gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastGtBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// BroadcastGeForward returns a ForwardFunc for broadcasted greater-than-or-equal: x >= y.
func BroadcastGeForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		s, err := x.Shape().BroadcastShapeBinaryOp(y.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to compute broadcast shape: %w", err)
		}
		xb, err := x.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast x: %w", err)
		}
		yb, err := y.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast y: %w", err)
		}
		data, err := xb.storage.Ge(yb.storage, xb.layout, yb.layout, spark.Contiguous(s))
		if err != nil {
			return nil, fmt.Errorf("failed to compute ge: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), spark.U8, x.device), nil
	}
}

// BroadcastGeBackward returns a BackwardFunc for broadcasted greater-than-or-equal gradients: ∂z/∂x = 0, ∂z/∂y = 0.
func BroadcastGeBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dx: %w", err)
		}
		dy, err := Zeros[T](inputs[1].Shape(), inputs[1].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// MatMulForward returns a ForwardFunc for matrix multiplication: x @ y.
func MatMulForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0], inputs[1]
		if x.Rank() < 2 || y.Rank() < 2 {
			return nil, fmt.Errorf("tensors must have rank >= 2")
		}
		xd, yd := x.Dims(), y.Dims()
		if len(xd) != len(yd) {
			return nil, fmt.Errorf("tensors must have same rank: %d vs %d", len(xd), len(yd))
		}
		bs := spark.NewShapeFrom(xd[:len(xd)-2])
		if !bs.Equal(spark.NewShapeFrom(yd[:len(yd)-2])) {
			return nil, fmt.Errorf("batch dims mismatch: %v vs %v", bs, spark.NewShapeFrom(yd[:len(yd)-2]))
		}
		m, k := xd[len(xd)-2], xd[len(xd)-1]
		if k != yd[len(yd)-2] {
			return nil, fmt.Errorf("inner dims mismatch: %d vs %d", k, yd[len(yd)-2])
		}
		s := spark.NewShapeFrom(append(bs.Dims(), m, yd[len(yd)-1]))
		data, err := x.storage.MatMul(x.layout, y.storage, y.layout, bs.ElemCount(), m, yd[len(yd)-1], k)
		if err != nil {
			return nil, fmt.Errorf("failed to matmul: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// MatMulBackward returns a BackwardFunc for matrix multiplication gradients: ∂z/∂x = g @ yᵀ, ∂z/∂y = xᵀ @ g.
func MatMulBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, y := inputs[0].Detach(), inputs[1].Detach()
		yt, err := y.Transpose(-1, -2)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose y: %w", err)
		}
		dx, err := g.MatMul(yt)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		xt, err := x.Transpose(-1, -2)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose x: %w", err)
		}
		dy, err := xt.MatMul(g)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dy: %w", err)
		}
		return []*Tensor[T]{dx, dy}, nil
	}
}

// Conv1dForward returns a ForwardFunc for 1D convolution.
func Conv1dForward[T spark.D](p *spark.Conv1DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 3 || w.Rank() != 3 {
			return nil, fmt.Errorf("tensors must be 3D")
		}
		s := spark.NewShapeFrom([]int{p.Batch, p.OutCh, p.OutLen()})
		data, err := x.storage.Conv1d(x.layout, w.storage, w.layout, p)
		if err != nil {
			return nil, fmt.Errorf("failed to conv1d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// Conv1dBackward returns a BackwardFunc for 1D convolution gradients.
func Conv1dBackward[T spark.D](p *spark.Conv1DParams) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
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
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		xt, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose x: %w", err)
		}
		gt, err := g.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose grad: %w", err)
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
			return nil, fmt.Errorf("failed to compute dwt: %w", err)
		}
		dw, err := dwt.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose dwt: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// ConvTranspose1dForward returns a ForwardFunc for 1D transposed convolution.
func ConvTranspose1dForward[T spark.D](p *spark.ConvT1DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 3 || w.Rank() != 3 {
			return nil, fmt.Errorf("tensors must be 3D")
		}
		s := spark.NewShapeFrom([]int{p.Batch, p.OutCh, p.OutLen()})
		data, err := x.storage.ConvTranspose1d(x.layout, w.storage, w.layout, p)
		if err != nil {
			return nil, fmt.Errorf("failed to convTranspose1d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// ConvTranspose1dBackward returns a BackwardFunc for 1D transposed convolution gradients.
func ConvTranspose1dBackward[T spark.D](p *spark.ConvT1DParams) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
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
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		gt, err := g.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose grad: %w", err)
		}
		xt, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose x: %w", err)
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
			return nil, fmt.Errorf("failed to compute dwt: %w", err)
		}
		dw, err := dwt.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose dwt: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// Conv2dForward returns a ForwardFunc for 2D convolution.
func Conv2dForward[T spark.D](p *spark.Conv2DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 4 || w.Rank() != 4 {
			return nil, fmt.Errorf("tensors must be 4D")
		}
		s := spark.NewShapeFrom([]int{p.Batch, p.OutCh, p.OutH(), p.OutW()})
		data, err := x.storage.Conv2d(x.layout, w.storage, w.layout, p)
		if err != nil {
			return nil, fmt.Errorf("failed to conv2d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// Conv2dBackward returns a BackwardFunc for 2D convolution gradients.
func Conv2dBackward[T spark.D](p *spark.Conv2DParams) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
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
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		xt, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose x: %w", err)
		}
		gt, err := g.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose grad: %w", err)
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
			return nil, fmt.Errorf("failed to compute dwt: %w", err)
		}
		dw, err := dwt.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose dwt: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// ConvTranspose2dForward returns a ForwardFunc for 2D transposed convolution.
func ConvTranspose2dForward[T spark.D](p *spark.ConvT2DParams) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		x, w := inputs[0], inputs[1]
		if x.Rank() != 4 || w.Rank() != 4 {
			return nil, fmt.Errorf("tensors must be 4D")
		}
		s := spark.NewShapeFrom([]int{p.Batch, p.OutCh, p.OutH(), p.OutW()})
		data, err := x.storage.ConvTranspose2d(x.layout, w.storage, w.layout, p)
		if err != nil {
			return nil, fmt.Errorf("failed to convTranspose2d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), x.dtype, x.device), nil
	}
}

// ConvTranspose2dBackward returns a BackwardFunc for 2D transposed convolution gradients.
func ConvTranspose2dBackward[T spark.D](p *spark.ConvT2DParams) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
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
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		gt, err := g.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose grad: %w", err)
		}
		xt, err := x.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose x: %w", err)
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
			return nil, fmt.Errorf("failed to compute dwt: %w", err)
		}
		dw, err := dwt.Transpose(0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose dwt: %w", err)
		}
		return []*Tensor[T]{dx, dw}, nil
	}
}

// AvgPool2dForward returns a ForwardFunc for 2D average pooling.
func AvgPool2dForward[T spark.D](kH, kW, sH, sW int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Rank() != 4 {
			return nil, fmt.Errorf("tensor must be 4D, got %dD", x.Rank())
		}
		b, c, h, w, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("expected 4D tensor for avg_pool2d, got: %w", err)
		}
		if h < kH || w < kW {
			return nil, fmt.Errorf("kernel size (%d,%d) larger than input (%d,%d)", kH, kW, h, w)
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
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		_, _, h, w, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("expected 4D tensor for avg_pool2d, got: %w", err)
		}
		dx, err := grad.UpsampleNearest2d(h, w)
		if err != nil {
			return nil, fmt.Errorf("failed to upsample grad: %w", err)
		}
		scale := 1.0 / float64(kH*kW)
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

// MaxPool2dForward returns a ForwardFunc for 2D max pooling.
func MaxPool2dForward[T spark.D](kH, kW, sH, sW int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Rank() != 4 {
			return nil, fmt.Errorf("tensor must be 4D, got %dD", x.Rank())
		}
		b, c, h, w, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("failed to get 4D shape: %w", err)
		}
		if h < kH || w < kW {
			return nil, fmt.Errorf("kernel (%d,%d) larger than input (%d,%d)", kH, kW, h, w)
		}
		if kH <= 0 || kW <= 0 || sH <= 0 || sW <= 0 {
			return nil, fmt.Errorf("kernel and stride must be positive")
		}
		hOut := (h-kH)/sH + 1
		wOut := (w-kW)/sW + 1
		shape := spark.NewShapeFrom([]int{b, c, hOut, wOut})
		data, err := x.storage.MaxPool2d(x.layout, kH, kW, sH, sW)
		if err != nil {
			return nil, fmt.Errorf("failed to maxpool2d: %w", err)
		}
		return NewFrom(data, spark.Contiguous(shape), x.dtype, x.device), nil
	}
}

// MaxPool2dBackward returns a BackwardFunc for 2D max pooling gradients.
func MaxPool2dBackward[T spark.D](kH, kW, sH, sW int) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		if kH != sH || kW != sW {
			return nil, fmt.Errorf("kernel must equal stride: kH=%d, sH=%d, kW=%d, sW=%d", kH, sH, kW, sW)
		}
		x := inputs[0].Detach()
		_, _, h, w, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("failed to get 4D shape: %w", err)
		}
		p, err := x.MaxPool2d(kH, kW, sH, sW)
		if err != nil {
			return nil, fmt.Errorf("failed to compute maxpool: %w", err)
		}
		pu, err := p.UpsampleNearest2d(h, w)
		if err != nil {
			return nil, fmt.Errorf("failed to upsample maxpool: %w", err)
		}
		m, err := x.Eq(pu)
		if err != nil {
			return nil, fmt.Errorf("failed to create mask: %w", err)
		}
		ma, err := m.AvgPool2d(kH, kW, sH, sW)
		if err != nil {
			return nil, fmt.Errorf("failed to average mask: %w", err)
		}
		sg, err := g.Mul(ma)
		if err != nil {
			return nil, fmt.Errorf("failed to scale grad: %w", err)
		}
		gu, err := sg.UpsampleNearest2d(h, w)
		if err != nil {
			return nil, fmt.Errorf("failed to upsample grad: %w", err)
		}
		dx, err := gu.Mul(m)
		if err != nil {
			return nil, fmt.Errorf("failed to apply mask: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// UpsampleNearest2dForward returns a ForwardFunc for 2D nearest neighbor upsampling.
func UpsampleNearest2dForward[T spark.D](h, w int) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if x.Rank() != 4 {
			return nil, fmt.Errorf("tensor must be 4D, got %dD", x.Rank())
		}
		if h <= 0 || w <= 0 {
			return nil, fmt.Errorf("target dims must be positive, got (%d,%d)", h, w)
		}
		b, c, _, _, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("failed to get 4D shape: %w", err)
		}
		shape := spark.NewShapeFrom([]int{b, c, h, w})
		data, err := x.storage.UpsampleNearest2d(x.layout, h, w)
		if err != nil {
			return nil, fmt.Errorf("failed to upsample: %w", err)
		}
		return NewFrom(data, spark.Contiguous(shape), x.dtype, x.device), nil
	}
}

// UpsampleNearest2dBackward returns a BackwardFunc for 2D nearest neighbor upsampling gradients.
func UpsampleNearest2dBackward[T spark.D](h, w int) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		_, _, srcH, srcW, err := x.Shape().Dims4()
		if err != nil {
			return nil, fmt.Errorf("failed to get 4D shape: %w", err)
		}
		if h%srcH != 0 || w%srcW != 0 {
			return nil, fmt.Errorf("non-integer scales: target=(%d,%d), input=(%d,%d)", h, w, srcH, srcW)
		}
		sh, sw := h/srcH, w/srcW
		if sh != sw {
			return nil, fmt.Errorf("non-uniform scales: scale_h=%d, scale_w=%d", sh, sw)
		}
		p, err := g.AvgPool2d(sh, sw, sh, sw)
		if err != nil {
			return nil, fmt.Errorf("failed to avgpool grad: %w", err)
		}
		s, err := Full[T](float64(sh*sw), p.Shape(), p.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create scale: %w", err)
		}
		dx, err := p.Mul(s)
		if err != nil {
			return nil, fmt.Errorf("failed to scale grad: %w", err)
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
			return nil, fmt.Errorf("failed to compute softmax: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SoftmaxBackward returns a BackwardFunc for softmax gradients.
func SoftmaxBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.Softmax()
		if err != nil {
			return nil, fmt.Errorf("failed to compute softmax: %w", err)
		}
		gs, err := g.Mul(s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute g*s: %w", err)
		}
		r, err := gs.Sum([]int{x.Rank() - 1})
		if err != nil {
			return nil, fmt.Errorf("failed to sum: %w", err)
		}
		rb, err := r.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast: %w", err)
		}
		d, err := g.Sub(rb)
		if err != nil {
			return nil, fmt.Errorf("failed to compute g-rb: %w", err)
		}
		dx, err := s.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// WhereCondForward returns a ForwardFunc for conditional selection: c ? t : f.
func WhereCondForward[T spark.D](c *Tensor[T]) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		t, f := inputs[0], inputs[1]
		s, err := c.Shape().BroadcastShapeBinaryOp(t.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast with true: %w", err)
		}
		s, err = s.BroadcastShapeBinaryOp(f.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast with false: %w", err)
		}
		cl, err := c.layout.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast cond: %w", err)
		}
		tl, err := t.layout.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast true: %w", err)
		}
		fl, err := f.layout.BroadcastAs(s)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast false: %w", err)
		}
		data, err := c.storage.WhereCond(cl, t.storage, tl, f.storage, fl)
		if err != nil {
			return nil, fmt.Errorf("failed to compute where: %w", err)
		}
		return NewFrom(data, spark.Contiguous(s), t.dtype, t.device), nil
	}
}

// WhereCondBackward returns a BackwardFunc for conditional selection gradients.
func WhereCondBackward[T spark.D](c *Tensor[T]) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("expected 2 inputs, got %d", len(inputs))
		}
		gd := g.Detach()
		z, err := Zeros[T](g.Shape(), g.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		dt, err := c.storage.WhereCond(c.layout, gd.storage, gd.layout, z.storage, z.layout)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dt: %w", err)
		}
		df, err := c.storage.WhereCond(c.layout, z.storage, z.layout, gd.storage, gd.layout)
		if err != nil {
			return nil, fmt.Errorf("failed to compute df: %w", err)
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
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		s := spark.Contiguous(x.Shape())
		data, err := x.storage.Copy(s, x.storage)
		if err != nil {
			return nil, fmt.Errorf("failed to copy: %w", err)
		}
		return NewFrom(data, s, x.dtype, x.device), nil
	}
}

// CopyBackward returns a BackwardFunc for clone gradients: ∂y/∂x = 1.
func CopyBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		return []*Tensor[T]{g}, nil
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
			return nil, fmt.Errorf("failed to compute neg: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// NegBackward returns a BackwardFunc for negation gradients: ∂(-x)/∂x = -1.
func NegBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		dx, err := g.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
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
			return nil, fmt.Errorf("failed to compute recip: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// RecipBackward returns a BackwardFunc for reciprocal gradients: ∂(1/x)/∂x = -1/x².
func RecipBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²: %w", err)
		}
		r, err := Full[T](1, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		r, err = r.Div(x2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1/x²: %w", err)
		}
		r, err = r.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate: %w", err)
		}
		dx, err := g.Mul(r)
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
			return nil, fmt.Errorf("failed to compute exp: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// ExpBackward returns a BackwardFunc for exponential gradients: ∂exp(x)/∂x = exp(x).
func ExpBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		e, err := x.Exp()
		if err != nil {
			return nil, fmt.Errorf("failed to compute exp: %w", err)
		}
		dx, err := g.Mul(e)
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
			return nil, fmt.Errorf("failed to compute log: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// LogBackward returns a BackwardFunc for logarithm gradients: ∂log(x)/∂x = 1/x.
func LogBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		r, err := Full[T](1, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		r, err = r.Div(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1/x: %w", err)
		}
		dx, err := g.Mul(r)
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
			return nil, fmt.Errorf("failed to compute sin: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SinBackward returns a BackwardFunc for sine gradients: ∂sin(x)/∂x = cos(x).
func SinBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		c, err := x.Cos()
		if err != nil {
			return nil, fmt.Errorf("failed to compute cos: %w", err)
		}
		dx, err := g.Mul(c)
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
			return nil, fmt.Errorf("failed to compute cos: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// CosBackward returns a BackwardFunc for cosine gradients: ∂cos(x)/∂x = -sin(x).
func CosBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.Sin()
		if err != nil {
			return nil, fmt.Errorf("failed to compute sin: %w", err)
		}
		s, err = s.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate: %w", err)
		}
		dx, err := g.Mul(s)
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
			return nil, fmt.Errorf("failed to compute tanh: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// TanhBackward returns a BackwardFunc for hyperbolic tangent gradients: ∂tanh(x)/∂x = 1 - tanh²(x).
func TanhBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		t, err := x.Tanh()
		if err != nil {
			return nil, fmt.Errorf("failed to compute tanh: %w", err)
		}
		t2, err := t.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute tanh²: %w", err)
		}
		d, err := Full[T](1, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		d, err = d.Sub(t2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1-tanh²: %w", err)
		}
		dx, err := g.Mul(d)
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
			return nil, fmt.Errorf("failed to compute erf: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// ErfBackward returns a BackwardFunc for error function gradients: ∂erf(x)/∂x = (2/√π)exp(-x²).
func ErfBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²: %w", err)
		}
		n, err := x2.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate: %w", err)
		}
		e, err := n.Exp()
		if err != nil {
			return nil, fmt.Errorf("failed to compute exp: %w", err)
		}
		c, err := Full[T](1.1283791670955126, x.Shape(), x.Device()) // 2/√π
		if err != nil {
			return nil, fmt.Errorf("failed to create coeff: %w", err)
		}
		d, err := c.Mul(e)
		if err != nil {
			return nil, fmt.Errorf("failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
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
			return nil, fmt.Errorf("failed to compute ceil: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// CeilBackward returns a BackwardFunc for ceiling gradients: ∂ceil(x)/∂x = 0.
func CeilBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		return []*Tensor[T]{dx}, nil
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
			return nil, fmt.Errorf("failed to compute floor: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// FloorBackward returns a BackwardFunc for floor gradients: ∂floor(x)/∂x = 0.
func FloorBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		return []*Tensor[T]{dx}, nil
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
			return nil, fmt.Errorf("failed to compute round: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// RoundBackward returns a BackwardFunc for rounding gradients: ∂round(x)/∂x = 0.
func RoundBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		return []*Tensor[T]{dx}, nil
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
			return nil, fmt.Errorf("failed to compute normcdf: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// NormcdfBackward returns a BackwardFunc for normal CDF gradients: ∂Φ(x)/∂x = φ(x) = (1/√(2π))exp(-x²/2).
func NormcdfBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²: %w", err)
		}
		h, err := Full[T](0.5, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create half: %w", err)
		}
		n, err := x2.Mul(h)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²/2: %w", err)
		}
		n, err = n.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate: %w", err)
		}
		e, err := n.Exp()
		if err != nil {
			return nil, fmt.Errorf("failed to compute exp: %w", err)
		}
		c, err := Full[T](0.3989422804014327, x.Shape(), x.Device()) // 1/√(2π)
		if err != nil {
			return nil, fmt.Errorf("failed to create coeff: %w", err)
		}
		d, err := c.Mul(e)
		if err != nil {
			return nil, fmt.Errorf("failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
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
			return nil, fmt.Errorf("failed to compute abs: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// AbsBackward returns a BackwardFunc for absolute value gradients: ∂|x|/∂x = sign(x).
func AbsBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.Sign()
		if err != nil {
			return nil, fmt.Errorf("failed to compute sign: %w", err)
		}
		dx, err := g.Mul(s)
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
			return nil, fmt.Errorf("failed to compute sqr: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SqrBackward returns a BackwardFunc for square gradients: ∂(x²)/∂x = 2x.
func SqrBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d, err := Full[T](2, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create two: %w", err)
		}
		d, err = d.Mul(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 2x: %w", err)
		}
		dx, err := g.Mul(d)
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
			return nil, fmt.Errorf("failed to compute sqrt: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SqrtBackward returns a BackwardFunc for square root gradients: ∂√x/∂x = 1/(2√x).
func SqrtBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.Sqrt()
		if err != nil {
			return nil, fmt.Errorf("failed to compute sqrt: %w", err)
		}
		d, err := Full[T](2, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create two: %w", err)
		}
		d, err = d.Mul(s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 2√x: %w", err)
		}
		dx, err := g.Div(d)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// GeluForward returns a ForwardFunc for element-wise GELU: gelu(x).
func GeluForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Gelu(x.layout)
		if err != nil {
			return nil, fmt.Errorf("failed to compute gelu: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// GeluBackward returns a BackwardFunc for GELU gradients.
func GeluBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		c, err := Full[T](0.044715, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create coeff: %w", err)
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
		i, err := x.Add(cx3)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x+c*x³: %w", err)
		}
		s, err := Full[T](0.7978845608028654, x.Shape(), x.Device()) // √(2/π)
		if err != nil {
			return nil, fmt.Errorf("failed to create sqrt2/pi: %w", err)
		}
		a, err := s.Mul(i)
		if err != nil {
			return nil, fmt.Errorf("failed to compute a: %w", err)
		}
		t, err := a.Tanh()
		if err != nil {
			return nil, fmt.Errorf("failed to compute tanh: %w", err)
		}
		o, err := Full[T](1, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		p, err := o.Add(t)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1+tanh: %w", err)
		}
		h, err := Full[T](0.5, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create half: %w", err)
		}
		f, err := h.Mul(p)
		if err != nil {
			return nil, fmt.Errorf("failed to compute first: %w", err)
		}
		t2, err := t.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute tanh²: %w", err)
		}
		o, err = o.Sub(t2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute sech²: %w", err)
		}
		n, err := Full[T](3, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create three: %w", err)
		}
		n, err = n.Mul(c)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 3*c: %w", err)
		}
		n, err = n.Mul(x2)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 3*c*x²: %w", err)
		}
		n, err = o.Add(n)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1+3*c*x²: %w", err)
		}
		r, err := h.Mul(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute half*x: %w", err)
		}
		r, err = r.Mul(o)
		if err != nil {
			return nil, fmt.Errorf("failed to compute sech² term: %w", err)
		}
		r, err = r.Mul(s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute sqrt2/pi term: %w", err)
		}
		r, err = r.Mul(n)
		if err != nil {
			return nil, fmt.Errorf("failed to compute second: %w", err)
		}
		d, err := f.Add(r)
		if err != nil {
			return nil, fmt.Errorf("failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// GeluErfForward returns a ForwardFunc for element-wise GELU (erf-based): gelu_erf(x).
func GeluErfForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.GeluErf(x.layout)
		if err != nil {
			return nil, fmt.Errorf("failed to compute gelu_erf: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// GeluErfBackward returns a BackwardFunc for GELU (erf-based) gradients.
func GeluErfBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		h, err := Full[T](0.5, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create half: %w", err)
		}
		o, err := Full[T](1, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		s, err := Full[T](1.4142135623730951, x.Shape(), x.Device()) // √2
		if err != nil {
			return nil, fmt.Errorf("failed to create sqrt2: %w", err)
		}
		xs, err := x.Div(s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x/√2: %w", err)
		}
		e, err := xs.Erf()
		if err != nil {
			return nil, fmt.Errorf("failed to compute erf: %w", err)
		}
		p, err := o.Add(e)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1+erf: %w", err)
		}
		f, err := h.Mul(p)
		if err != nil {
			return nil, fmt.Errorf("failed to compute first: %w", err)
		}
		x2, err := x.Sqr()
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²: %w", err)
		}
		x2, err = x2.Mul(h)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x²/2: %w", err)
		}
		n, err := x2.Neg()
		if err != nil {
			return nil, fmt.Errorf("failed to negate: %w", err)
		}
		e, err = n.Exp()
		if err != nil {
			return nil, fmt.Errorf("failed to compute exp: %w", err)
		}
		c, err := Full[T](0.7978845608028654, x.Shape(), x.Device()) // 2/√(2π)
		if err != nil {
			return nil, fmt.Errorf("failed to create coeff: %w", err)
		}
		r, err := h.Mul(x)
		if err != nil {
			return nil, fmt.Errorf("failed to compute half*x: %w", err)
		}
		r, err = r.Mul(c)
		if err != nil {
			return nil, fmt.Errorf("failed to compute coeff term: %w", err)
		}
		r, err = r.Mul(e)
		if err != nil {
			return nil, fmt.Errorf("failed to compute exp term: %w", err)
		}
		d, err := f.Add(r)
		if err != nil {
			return nil, fmt.Errorf("failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
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
			return nil, fmt.Errorf("failed to compute relu: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// ReluBackward returns a BackwardFunc for ReLU gradients: ∂relu(x)/∂x = x > 0 ? 1 : 0.
func ReluBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		z, err := Full[T](0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zero: %w", err)
		}
		m, err := x.Gt(z)
		if err != nil {
			return nil, fmt.Errorf("failed to compute mask: %w", err)
		}
		z, err = Zeros[T](g.Shape(), g.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		dx, err := m.WhereCond(g, z)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// EluForward returns a ForwardFunc for element-wise ELU: x if x >= 0, alpha*(exp(x)-1) if x < 0.
func EluForward[T spark.D](alpha float64) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Elu(x.layout, T(alpha))
		if err != nil {
			return nil, fmt.Errorf("failed to compute elu: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// EluBackward returns a BackwardFunc for ELU gradients: ∂elu(x)/∂x = 1 if x >= 0, alpha*exp(x) if x < 0.
func EluBackward[T spark.D](alpha float64) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		z, err := Full[T](0, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zero: %w", err)
		}
		m, err := x.Ge(z)
		if err != nil {
			return nil, fmt.Errorf("failed to compute mask: %w", err)
		}
		o, err := Full[T](1, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		e, err := x.Exp()
		if err != nil {
			return nil, fmt.Errorf("failed to compute exp: %w", err)
		}
		a, err := Full[T](alpha, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create alpha: %w", err)
		}
		ae, err := a.Mul(e)
		if err != nil {
			return nil, fmt.Errorf("failed to compute alpha*exp: %w", err)
		}
		d, err := m.WhereCond(o, ae)
		if err != nil {
			return nil, fmt.Errorf("failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SiluForward returns a ForwardFunc for element-wise SiLU: x * sigmoid(x).
func SiluForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Silu(x.layout)
		if err != nil {
			return nil, fmt.Errorf("failed to compute silu: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SiluBackward returns a BackwardFunc for SiLU gradients: ∂silu(x)/∂x = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x)).
func SiluBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.Sigmoid()
		if err != nil {
			return nil, fmt.Errorf("failed to compute sigmoid: %w", err)
		}
		o, err := Full[T](1, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		n, err := o.Sub(s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1-sigmoid: %w", err)
		}
		xs, err := x.Mul(s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x*sigmoid: %w", err)
		}
		r, err := xs.Mul(n)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x*sigmoid*(1-sigmoid): %w", err)
		}
		d, err := s.Add(r)
		if err != nil {
			return nil, fmt.Errorf("failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// PowfForward returns a ForwardFunc for element-wise power: x^param.
func PowfForward[T spark.D](p float64) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Powf(x.layout, T(p))
		if err != nil {
			return nil, fmt.Errorf("failed to compute pow: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// PowfBackward returns a BackwardFunc for power gradients: ∂(x^param)/∂x = param*x^(param-1).
func PowfBackward[T spark.D](p float64) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		e, err := x.Powf(p - 1)
		if err != nil {
			return nil, fmt.Errorf("failed to compute x^(param-1): %w", err)
		}
		d, err := Full[T](p, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create param: %w", err)
		}
		d, err = d.Mul(e)
		if err != nil {
			return nil, fmt.Errorf("failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SigmoidForward returns a ForwardFunc for element-wise sigmoid: σ(x) = 1/(1+exp(-x)).
func SigmoidForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sigmoid(x.layout)
		if err != nil {
			return nil, fmt.Errorf("failed to compute sigmoid: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SigmoidBackward returns a BackwardFunc for sigmoid gradients: ∂σ(x)/∂x = σ(x)*(1-σ(x)).
func SigmoidBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		s, err := x.Sigmoid()
		if err != nil {
			return nil, fmt.Errorf("failed to compute sigmoid: %w", err)
		}
		o, err := Full[T](1, x.Shape(), x.Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create one: %w", err)
		}
		n, err := o.Sub(s)
		if err != nil {
			return nil, fmt.Errorf("failed to compute 1-sigmoid: %w", err)
		}
		d, err := s.Mul(n)
		if err != nil {
			return nil, fmt.Errorf("failed to compute deriv: %w", err)
		}
		dx, err := g.Mul(d)
		if err != nil {
			return nil, fmt.Errorf("failed to compute dx: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SignForward returns a ForwardFunc for element-wise sign: sign(x).
func SignForward[T spark.D]() ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		data, err := x.storage.Sign(x.layout)
		if err != nil {
			return nil, fmt.Errorf("failed to compute sign: %w", err)
		}
		return NewFrom(data, x.layout.Clone(), x.dtype, x.device), nil
	}
}

// SignBackward returns a BackwardFunc for sign gradients: ∂sign(x)/∂x = 0.
func SignBackward[T spark.D]() BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		dx, err := Zeros[T](inputs[0].Shape(), inputs[0].Device())
		if err != nil {
			return nil, fmt.Errorf("failed to create zeros: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// SumDimForward returns a ForwardFunc for summing along specified dimensions.
func SumDimForward[T spark.D](dims []int, keepdim bool) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		d, err := spark.ResolveAxes(dims, x.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to resolve dims: %w", err)
		}
		data, err := x.storage.Sum(x.layout, d)
		if err != nil {
			return nil, fmt.Errorf("failed to sum: %w", err)
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

// SumDimBackward returns a BackwardFunc for sum gradients.
func SumDimBackward[T spark.D](dims []int, keepdim bool) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d, err := spark.ResolveAxes(dims, x.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to resolve dims: %w", err)
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
				return nil, fmt.Errorf("failed to reshape: %w", err)
			}
		}
		dx, err := r.BroadcastAs(x.Shape())
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast: %w", err)
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
		l, err := x.layout.Transpose(d1, d2)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose: %w", err)
		}
		return NewFrom(x.storage, l, x.dtype, x.device), nil
	}
}

// TransposeBackward returns a BackwardFunc for transpose gradients.
func TransposeBackward[T spark.D](dim1, dim2 int) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		dx, err := g.Transpose(dim1, dim2)
		if err != nil {
			return nil, fmt.Errorf("failed to transpose: %w", err)
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
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("failed to resolve dim: %w", err)
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
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d, err := spark.ResolveAxis(dim, x.Rank())
		if err != nil {
			return nil, fmt.Errorf("failed to resolve dim: %w", err)
		}
		if x.Dim(d) != 1 {
			return []*Tensor[T]{g}, nil
		}
		dx, err := g.Unsqueeze(d)
		if err != nil {
			return nil, fmt.Errorf("failed to unsqueeze: %w", err)
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
		d := dim
		if d < 0 {
			d += x.Rank() + 1
		}
		if d < 0 || d > x.Rank() {
			return nil, fmt.Errorf("dim out of range [-%d, %d], got %d", x.Rank()+1, x.Rank(), dim)
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
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0].Detach()
		d := dim
		if d < 0 {
			d += x.Rank() + 1
		}
		dx, err := g.Squeeze(d)
		if err != nil {
			return nil, fmt.Errorf("failed to squeeze: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

// ReshapeForward returns a ForwardFunc for reshaping to a new shape.
func ReshapeForward[T spark.D](s *spark.Shape) ForwardFunc[T] {
	return func(inputs []*Tensor[T]) (*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		x := inputs[0]
		if s.ElemCount() != x.Shape().ElemCount() {
			return nil, fmt.Errorf("element count mismatch: %d vs %d", s.ElemCount(), x.Shape().ElemCount())
		}
		if !x.layout.IsContiguous() {
			return nil, fmt.Errorf("non-contiguous tensor not supported")
		}
		return NewFrom(x.storage, spark.ContiguousWithOffset(s, x.layout.StartOffset()), x.dtype, x.device), nil
	}
}

// ReshapeBackward returns a BackwardFunc for reshape gradients.
func ReshapeBackward[T spark.D](s *spark.Shape) BackwardFunc[T] {
	return func(g *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("expected 1 input, got %d", len(inputs))
		}
		dx, err := g.Reshape(s.Dims()...)
		if err != nil {
			return nil, fmt.Errorf("failed to reshape: %w", err)
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
			return nil, fmt.Errorf("failed to broadcast %v to %v: %w", x.Shape(), s, err)
		}
		data, err := x.storage.Clone()
		if err != nil {
			return nil, fmt.Errorf("failed to clone: %w", err)
		}
		return NewFrom(data, l, x.dtype, x.device), nil
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
			return nil, fmt.Errorf("failed to reduce: %w", err)
		}
		return []*Tensor[T]{dx}, nil
	}
}

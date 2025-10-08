package loss

import (
	"fmt"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

// NLL computes the negative log likelihood loss for log probabilities.
func NLL[T spark.D](x, y *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	xs, ys := x.Shape(), y.Shape()
	if ys.Rank() != 1 {
		return nil, fmt.Errorf("target must be 1D, got %dD", ys.Rank())
	}
	n := ys.Dim(0)
	if xs.Rank() != 2 || xs.Dim(0) != n {
		return nil, fmt.Errorf("input must be 2D with batch size %d, got shape %v", n, xs)
	}
	yt, err := y.Unsqueeze(1)
	if err != nil {
		return nil, fmt.Errorf("failed to unsqueeze target: %w", err)
	}
	g, err := x.Gather(yt, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to gather: %w", err)
	}
	s, err := g.SumAll()
	if err != nil {
		return nil, fmt.Errorf("failed to sum: %w", err)
	}
	m, err := s.Affine(-1.0/float64(n), 0)
	if err != nil {
		return nil, fmt.Errorf("failed to scale: %w", err)
	}
	return m, nil
}

// CrossEntropy computes the cross-entropy loss for logits.
func CrossEntropy[T spark.D](x, y *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if x.Rank() != 2 {
		return nil, fmt.Errorf("input must be 2D, got %dD", x.Rank())
	}
	z, err := x.LogSoftmax(1)
	if err != nil {
		return nil, fmt.Errorf("failed to compute log_softmax: %w", err)
	}
	return NLL(z, y)
}

// MSE computes the mean squared error loss between input and target.
func MSE[T spark.D](x, y *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	diff, err := x.Sub(y)
	if err != nil {
		return nil, fmt.Errorf("failed to compute x - y: %w", err)
	}
	sqr, err := diff.Sqr()
	if err != nil {
		return nil, fmt.Errorf("failed to square: %w", err)
	}
	mean, err := sqr.MeanAll()
	if err != nil {
		return nil, fmt.Errorf("failed to compute mean: %w", err)
	}
	return mean, nil
}

// BCE computes the binary cross-entropy loss for logits.
func BCE[T spark.D](x, y *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	// Stable formula: BCE = 1/N * Σ [max(x,0) - x*y + log(1 + exp(-|x|))]
	zero, err := tensor.Full[T](0.0, x.Shape(), x.Device())
	if err != nil {
		return nil, fmt.Errorf("failed to create zero tensor: %w", err)
	}
	c, err := x.Gt(zero)
	if err != nil {
		return nil, fmt.Errorf("failed to compute condition: %w", err)
	}
	m, err := c.WhereCond(x, zero) // max(x, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to compute max(x, 0): %w", err)
	}
	p, err := x.Mul(y) // x*y
	if err != nil {
		return nil, fmt.Errorf("failed to compute x*y: %w", err)
	}
	a, err := x.Abs() // |x|
	if err != nil {
		return nil, fmt.Errorf("failed to compute |x|: %w", err)
	}
	n, err := a.Neg() // -|x|
	if err != nil {
		return nil, fmt.Errorf("failed to compute -|x|: %w", err)
	}
	e, err := n.Exp() // exp(-|x|)
	if err != nil {
		return nil, fmt.Errorf("failed to compute exp(-|x|): %w", err)
	}
	one, err := tensor.Full[T](1.0, e.Shape(), e.Device())
	if err != nil {
		return nil, fmt.Errorf("failed to create one tensor: %w", err)
	}
	t, err := one.Add(e) // 1 + exp(-|x|)
	if err != nil {
		return nil, fmt.Errorf("failed to compute 1 + exp(-|x|): %w", err)
	}
	l, err := t.Log() // log(1 + exp(-|x|))
	if err != nil {
		return nil, fmt.Errorf("failed to compute log(1 + exp(-|x|)): %w", err)
	}
	z, err := m.Sub(p) // max(x, 0) - x*y
	if err != nil {
		return nil, fmt.Errorf("failed to compute max(x, 0) - x*y: %w", err)
	}
	s, err := z.Add(l) // max(x, 0) - x*y + log(1 + exp(-|x|))
	if err != nil {
		return nil, fmt.Errorf("failed to compute sum: %w", err)
	}
	mn, err := s.MeanAll() // 1/N * Σ
	if err != nil {
		return nil, fmt.Errorf("failed to compute mean: %w", err)
	}
	return mn, nil
}

// L1Loss computes the L1 (mean absolute error) loss between input and target.
func L1Loss[T spark.D](x, y *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	diff, err := x.Sub(y)
	if err != nil {
		return nil, fmt.Errorf("failed to compute x - y: %w", err)
	}
	abs, err := diff.Abs()
	if err != nil {
		return nil, fmt.Errorf("failed to compute abs: %w", err)
	}
	mean, err := abs.MeanAll()
	if err != nil {
		return nil, fmt.Errorf("failed to compute mean: %w", err)
	}
	return mean, nil
}

// SmoothL1Loss computes the Smooth L1 (Huber) loss between input and target.
func SmoothL1Loss[T spark.D](x, y *tensor.Tensor[T], beta float64) (*tensor.Tensor[T], error) {
	if beta <= 0 {
		beta = 1.0
	}
	d, err := x.Sub(y)
	if err != nil {
		return nil, fmt.Errorf("failed to compute x - y: %w", err)
	}
	a, err := d.Abs()
	if err != nil {
		return nil, fmt.Errorf("failed to compute abs: %w", err)
	}
	b, err := tensor.Full[T](beta, a.Shape(), a.Device())
	if err != nil {
		return nil, fmt.Errorf("failed to create beta tensor: %w", err)
	}
	c, err := a.Lt(b)
	if err != nil {
		return nil, fmt.Errorf("failed to compute condition: %w", err)
	}
	s, err := d.Sqr()
	if err != nil {
		return nil, fmt.Errorf("failed to square: %w", err)
	}
	b, err = tensor.Full[T](0.5/beta, s.Shape(), s.Device())
	if err != nil {
		return nil, fmt.Errorf("failed to create scale: %w", err)
	}
	q, err := s.Mul(b)
	if err != nil {
		return nil, fmt.Errorf("failed to compute quadratic part: %w", err)
	}
	b, err = tensor.Full[T](-0.5*beta, a.Shape(), a.Device())
	if err != nil {
		return nil, fmt.Errorf("failed to create offset: %w", err)
	}
	l, err := a.Add(b)
	if err != nil {
		return nil, fmt.Errorf("failed to compute linear part: %w", err)
	}
	z, err := c.WhereCond(q, l)
	if err != nil {
		return nil, fmt.Errorf("failed to select loss: %w", err)
	}
	m, err := z.MeanAll()
	if err != nil {
		return nil, fmt.Errorf("failed to compute mean: %w", err)
	}
	return m, nil
}

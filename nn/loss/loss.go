package loss

import (
	"fmt"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/tensor"
)

// NLL computes the negative log likelihood loss for log probabilities.
func NLL[T candy.D](x, y *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	xs, ys := x.Shape(), y.Shape()
	if ys.Rank() != 1 {
		return nil, fmt.Errorf("nll loss: target must be 1D, got %dD", ys.Rank())
	}
	n := ys.Dim(0)
	if xs.Rank() != 2 || xs.Dim(0) != n {
		return nil, fmt.Errorf("nll loss: input must be 2D with batch size %d, got shape %v", n, xs)
	}
	yt, err := y.Unsqueeze(1)
	if err != nil {
		return nil, fmt.Errorf("nll loss: failed to unsqueeze target: %w", err)
	}
	g, err := x.Gather(yt, 1)
	if err != nil {
		return nil, fmt.Errorf("nll loss: failed to gather: %w", err)
	}
	s, err := g.SumAll()
	if err != nil {
		return nil, fmt.Errorf("nll loss: failed to sum: %w", err)
	}
	m, err := s.Affine(-1.0/float64(n), 0)
	if err != nil {
		return nil, fmt.Errorf("nll loss: failed to scale: %w", err)
	}
	return m, nil
}

// MustNLL computes the negative log likelihood loss for log probabilities.
func MustNLL[T candy.D](x, y *tensor.Tensor[T]) *tensor.Tensor[T] {
	ls, err := NLL(x, y)
	if err != nil {
		panic(err)
	}
	return ls
}

// CrossEntropy computes the cross-entropy loss for logits.
func CrossEntropy[T candy.D](x, y *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if x.Rank() != 2 {
		return nil, fmt.Errorf("cross entropy loss: input must be 2D, got %dD", x.Rank())
	}
	z, err := x.LogSoftmax(1)
	if err != nil {
		return nil, fmt.Errorf("cross entropy loss: failed to compute log_softmax: %w", err)
	}
	return NLL(z, y)
}

// MustCrossEntropy computes the cross-entropy loss for logits.
func MustCrossEntropy[T candy.D](x, y *tensor.Tensor[T]) *tensor.Tensor[T] {
	ls, err := CrossEntropy(x, y)
	if err != nil {
		panic(err)
	}
	return ls
}

// MSE computes the mean squared error loss between input and target.
func MSE[T candy.D](x, y *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	diff, err := x.Sub(y)
	if err != nil {
		return nil, fmt.Errorf("mse loss: failed to compute x - y: %w", err)
	}
	sqr, err := diff.Sqr()
	if err != nil {
		return nil, fmt.Errorf("mse loss: failed to square: %w", err)
	}
	mean, err := sqr.MeanAll()
	if err != nil {
		return nil, fmt.Errorf("mse loss: failed to compute mean: %w", err)
	}
	return mean, nil
}

// MustMSE computes the mean squared error loss between input and target.
func MustMSE[T candy.D](x, y *tensor.Tensor[T]) *tensor.Tensor[T] {
	ls, err := MSE(x, y)
	if err != nil {
		panic(err)
	}
	return ls
}

// BCE computes the binary cross-entropy loss for logits.
// Stable formula: BCE = 1/N * Σ [max(x,0) - x*y + log(1 + exp(-|x|))]
func BCE[T candy.D](x, y *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	zero, err := tensor.Zeros[T](x.Shape(), x.Device())
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to create zero tensor: %w", err)
	}
	c, err := x.Gt(zero)
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to compute condition: %w", err)
	}
	m, err := c.WhereCond(x, zero) // max(x, 0)
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to compute max(x, 0): %w", err)
	}
	p, err := x.Mul(y) // x*y
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to compute x*y: %w", err)
	}
	a, err := x.Abs() // |x|
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to compute |x|: %w", err)
	}
	n, err := a.Neg() // -|x|
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to compute -|x|: %w", err)
	}
	e, err := n.Exp() // exp(-|x|)
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to compute exp(-|x|): %w", err)
	}
	t, err := e.AddScalar(1.0) // 1 + exp(-|x|)
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to compute 1 + exp(-|x|): %w", err)
	}
	l, err := t.Log() // log(1 + exp(-|x|))
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to compute log(1 + exp(-|x|)): %w", err)
	}
	z, err := m.Sub(p) // max(x, 0) - x*y
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to compute max(x, 0) - x*y: %w", err)
	}
	s, err := z.Add(l) // max(x, 0) - x*y + log(1 + exp(-|x|))
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to compute sum: %w", err)
	}
	mn, err := s.MeanAll() // 1/N * Σ
	if err != nil {
		return nil, fmt.Errorf("bce loss: failed to compute mean: %w", err)
	}
	return mn, nil
}

// MustBCE computes the binary cross-entropy loss for logits.
func MustBCE[T candy.D](x, y *tensor.Tensor[T]) *tensor.Tensor[T] {
	ls, err := BCE(x, y)
	if err != nil {
		panic(err)
	}
	return ls
}

// L1Loss computes the L1 (mean absolute error) loss between input and target.
func L1Loss[T candy.D](x, y *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	diff, err := x.Sub(y)
	if err != nil {
		return nil, fmt.Errorf("l1 loss: failed to compute x - y: %w", err)
	}
	abs, err := diff.Abs()
	if err != nil {
		return nil, fmt.Errorf("l1 loss: failed to compute abs: %w", err)
	}
	mean, err := abs.MeanAll()
	if err != nil {
		return nil, fmt.Errorf("l1 loss: failed to compute mean: %w", err)
	}
	return mean, nil
}

// MustL1Loss computes the L1 (mean absolute error) loss between input and target.
func MustL1Loss[T candy.D](x, y *tensor.Tensor[T]) *tensor.Tensor[T] {
	ls, err := L1Loss(x, y)
	if err != nil {
		panic(err)
	}
	return ls
}

// SmoothL1Loss computes the Smooth L1 (Huber) loss between input and target.
func SmoothL1Loss[T candy.D](x, y *tensor.Tensor[T], beta float64) (*tensor.Tensor[T], error) {
	if beta <= 0 {
		beta = 1.0
	}
	d, err := x.Sub(y)
	if err != nil {
		return nil, fmt.Errorf("smooth l1 loss: failed to compute x - y: %w", err)
	}
	a, err := d.Abs()
	if err != nil {
		return nil, fmt.Errorf("smooth l1 loss: failed to compute abs: %w", err)
	}
	b, err := tensor.Full[T](beta, a.Shape(), a.Device())
	if err != nil {
		return nil, fmt.Errorf("smooth l1 loss: failed to create beta tensor: %w", err)
	}
	c, err := a.Lt(b)
	if err != nil {
		return nil, fmt.Errorf("smooth l1 loss: failed to compute condition: %w", err)
	}
	s, err := d.Sqr()
	if err != nil {
		return nil, fmt.Errorf("smooth l1 loss: failed to square: %w", err)
	}
	q, err := s.MulScalar(0.5 / beta)
	if err != nil {
		return nil, fmt.Errorf("smooth l1 loss: failed to compute quadratic part: %w", err)
	}
	l, err := a.AddScalar(-0.5 * beta)
	if err != nil {
		return nil, fmt.Errorf("smooth l1 loss: failed to compute linear part: %w", err)
	}
	z, err := c.WhereCond(q, l)
	if err != nil {
		return nil, fmt.Errorf("smooth l1 loss: failed to select loss: %w", err)
	}
	m, err := z.MeanAll()
	if err != nil {
		return nil, fmt.Errorf("smooth l1 loss: failed to compute mean: %w", err)
	}
	return m, nil
}

// MustSmoothL1Loss computes the Smooth L1 (Huber) loss between input and target.
func MustSmoothL1Loss[T candy.D](x, y *tensor.Tensor[T], beta float64) *tensor.Tensor[T] {
	ls, err := SmoothL1Loss(x, y, beta)
	if err != nil {
		panic(err)
	}
	return ls
}

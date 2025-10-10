package nn

import (
	"fmt"
	"math"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

// Linear represents a linear transformation layer: y = xW^T + b.
type Linear[T spark.D] struct {
	w *tensor.Tensor[T] // Weight tensor
	b *tensor.Tensor[T] // Bias tensor (optional)
}

// NewLinear creates a new linear layer with given weight and optional bias.
func NewLinear[T spark.D](w, b *tensor.Tensor[T]) *Linear[T] {
	return &Linear[T]{w: w, b: b}
}

// NewLinearLayer creates a linear layer with PyTorch-style Kaiming Uniform initialization.
func NewLinearLayer[T spark.D](inDim, outDim int, bias bool, device spark.Device) *Linear[T] {
	bound := math.Sqrt(1.0 / float64(inDim))
	w, err := tensor.Rand[T](-bound, bound, spark.NewShape(outDim, inDim), device)
	if err != nil {
		panic(fmt.Errorf("failed to create weight: %w", err))
	}
	w.SetIsVar(true)
	var b *tensor.Tensor[T]
	if bias {
		bBound := 1.0 / math.Sqrt(float64(inDim))
		b, err = tensor.Rand[T](-bBound, bBound, spark.NewShape(outDim), device)
		if err != nil {
			panic(fmt.Errorf("failed to create bias: %w", err))
		}
		b.SetIsVar(true)
	}
	return NewLinear(w, b)
}

// NewLinearNoBias creates a linear layer without bias.
func NewLinearNoBias[T spark.D](inDim, outDim int, device spark.Device) *Linear[T] {
	return NewLinearLayer[T](inDim, outDim, false, device)
}

// Weight returns the weight tensor.
func (l *Linear[T]) Weight() *tensor.Tensor[T] {
	return l.w
}

// Bias returns the bias tensor (may be nil).
func (l *Linear[T]) Bias() *tensor.Tensor[T] {
	return l.b
}

// Forward applies the linear transformation: y = xW^T + b.
func (l *Linear[T]) Forward(x *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	dims := x.Dims()
	rank := len(dims)
	if rank < 2 {
		return nil, fmt.Errorf("input rank must be >= 2, got %d", rank)
	}

	wt, err := l.w.Transpose(-1, -2)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose weight: %w", err)
	}

	var r *tensor.Tensor[T]
	switch {
	case x.Layout().IsContiguous():
		// Reshape input for efficient matmul
		bs := dims[:rank-1]
		k := dims[rank-1]
		n := int64(1)
		for _, d := range bs {
			n *= int64(d)
		}
		xr, err := x.Reshape(int(n), k)
		if err != nil {
			return nil, fmt.Errorf("failed to reshape input: %w", err)
		}
		r, err = xr.MatMul(wt)
		if err != nil {
			return nil, fmt.Errorf("failed to matmul: %w", err)
		}
		s := append(bs, l.w.Dim(0))
		r, err = r.Reshape(s...)
		if err != nil {
			return nil, fmt.Errorf("failed to reshape output: %w", err)
		}
	default:
		// Broadcast weight for non-contiguous input
		wb, err := wt.BroadcastLeft(dims[:rank-1]...)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast weight: %w", err)
		}
		r, err = x.MatMul(wb)
		if err != nil {
			return nil, fmt.Errorf("failed to matmul: %w", err)
		}
	}

	if l.b != nil {
		r, err = r.BroadcastAdd(l.b)
		if err != nil {
			return nil, fmt.Errorf("failed to add bias: %w", err)
		}
	}
	return r, nil
}

// MustForward applies the linear transformation, panicking on error.
func (l *Linear[T]) MustForward(x *tensor.Tensor[T]) *tensor.Tensor[T] {
	r, err := l.Forward(x)
	if err != nil {
		panic(fmt.Errorf("failed forward: %w", err))
	}
	return r
}

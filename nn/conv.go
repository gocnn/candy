package nn

import (
	"fmt"
	"math"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

// Conv2d represents a 2D convolutional layer: y = conv2d(x, w) + b.
type Conv2d[T spark.D] struct {
	w      *tensor.Tensor[T]   // Weight tensor
	b      *tensor.Tensor[T]   // Bias tensor
	params *spark.Conv2DParams // Convolution parameters
}

// NewConv2d creates a new 2D convolutional layer with Xavier initialization.
func NewConv2d[T spark.D](inCh, outCh, kSize, stride, pad int, device spark.Device) *Conv2d[T] {
	std := math.Sqrt(2.0 / float64(inCh*kSize*kSize))
	w, err := tensor.RandN[T](0, std, spark.NewShape(outCh, inCh, kSize, kSize), device)
	if err != nil {
		panic(fmt.Errorf("failed to create weight: %w", err))
	}
	w.SetIsVar(true)
	b, err := tensor.Zeros[T](spark.NewShape(outCh), device)
	if err != nil {
		panic(fmt.Errorf("failed to create bias: %w", err))
	}
	b.SetIsVar(true)
	return &Conv2d[T]{
		w: w,
		b: b,
		params: &spark.Conv2DParams{
			Batch:  1, // Updated dynamically
			InCh:   inCh,
			OutCh:  outCh,
			KH:     kSize,
			KW:     kSize,
			Stride: stride,
			Pad:    pad,
			Dilate: 1,
		},
	}
}

// Forward applies the convolutional layer.
func (c *Conv2d[T]) Forward(x *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(x.Dims()) != 4 {
		return nil, fmt.Errorf("expected 4D input, got %dD", len(x.Dims()))
	}
	c.params.Batch, c.params.InH, c.params.InW = x.Dim(0), x.Dim(2), x.Dim(3)
	r, err := x.Conv2d(c.w, c.params)
	if err != nil {
		return nil, fmt.Errorf("failed to conv2d: %w", err)
	}
	br, err := c.b.Reshape(1, c.params.OutCh, 1, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to reshape bias: %w", err)
	}
	r, err = r.BroadcastAdd(br)
	if err != nil {
		return nil, fmt.Errorf("failed to add bias: %w", err)
	}
	return r, nil
}

// MustForward applies the convolutional layer.
func (c *Conv2d[T]) MustForward(x *tensor.Tensor[T]) *tensor.Tensor[T] {
	r, err := c.Forward(x)
	if err != nil {
		panic(err)
	}
	return r
}

// Parameters returns the trainable parameters.
func (c *Conv2d[T]) Parameters() []*tensor.Tensor[T] {
	return []*tensor.Tensor[T]{c.w, c.b}
}

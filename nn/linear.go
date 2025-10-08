package nn

import (
	"fmt"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

// Linear represents a linear transformation layer: y = xW^T + b
type Linear[T spark.D] struct {
	weight *tensor.Tensor[T]
	bias   *tensor.Tensor[T] // can be nil for no bias
}

// NewLinear creates a new linear layer with given weight and optional bias
func NewLinear[T spark.D](weight *tensor.Tensor[T], bias *tensor.Tensor[T]) *Linear[T] {
	return &Linear[T]{
		weight: weight,
		bias:   bias,
	}
}

// NewLinearLayer creates a linear layer with random initialization
func NewLinearLayer[T spark.D](inDim, outDim int, bias bool, device spark.Device) *Linear[T] {
	bound := 1.0 / float64(inDim)
	weight, err := tensor.RandN[T](0.0, bound, spark.NewShape(outDim, inDim), device)
	if err != nil {
		panic(err)
	}

	var biasT *tensor.Tensor[T]
	if bias {
		var err error
		biasT, err = tensor.Rand[T](-bound, bound, spark.NewShape(outDim), device)
		if err != nil {
			panic(err)
		}
	}

	return NewLinear(weight, biasT)
}

// NewLinearNoBias creates a linear layer without bias
func NewLinearNoBias[T spark.D](inDim, outDim int, device spark.Device) *Linear[T] {
	return NewLinearLayer[T](inDim, outDim, false, device)
}

// Weight returns the weight tensor
func (l *Linear[T]) Weight() *tensor.Tensor[T] {
	return l.weight
}

// Bias returns the bias tensor (can be nil)
func (l *Linear[T]) Bias() *tensor.Tensor[T] {
	return l.bias
}

// Forward applies the linear transformation: y = xW^T + b
func (l *Linear[T]) Forward(x *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	// Handle different input dimensions for efficiency
	dims := x.Shape().Dims()

	var result *tensor.Tensor[T]
	var err error

	switch len(dims) {
	case 4: // [b1, b2, m, k]
		b1, b2, m, k := dims[0], dims[1], dims[2], dims[3]
		if x.Layout().IsContiguous() {
			// Efficient path: reshape and use standard matmul
			w, err := l.weight.Transpose(-1, -2) // transpose weight
			if err != nil {
				return nil, err
			}

			reshaped, err := x.Reshape(b1*b2*m, k)
			if err != nil {
				return nil, err
			}

			matmulResult, err := reshaped.MatMul(w)
			if err != nil {
				return nil, err
			}

			outFeatures := l.weight.Shape().Dim(0)
			result, err = matmulResult.Reshape(b1, b2, m, outFeatures)
			if err != nil {
				return nil, err
			}
		} else {
			// Non-contiguous path: use broadcasted matmul
			w, err := l.weight.BroadcastLeft(b1, b2)
			if err != nil {
				return nil, err
			}

			w, err = w.Transpose(-1, -2)
			if err != nil {
				return nil, err
			}

			result, err = x.MatMul(w)
			if err != nil {
				return nil, err
			}
		}

	case 3: // [bsize, m, k]
		bsize, m, k := dims[0], dims[1], dims[2]
		if x.Layout().IsContiguous() {
			// Efficient path
			w, err := l.weight.Transpose(-1, -2)
			if err != nil {
				return nil, err
			}

			reshaped, err := x.Reshape(bsize*m, k)
			if err != nil {
				return nil, err
			}

			matmulResult, err := reshaped.MatMul(w)
			if err != nil {
				return nil, err
			}

			outFeatures := l.weight.Shape().Dim(0)
			result, err = matmulResult.Reshape(bsize, m, outFeatures)
			if err != nil {
				return nil, err
			}
		} else {
			// Non-contiguous path
			w, err := l.weight.BroadcastLeft(bsize)
			if err != nil {
				return nil, err
			}

			w, err = w.Transpose(-1, -2)
			if err != nil {
				return nil, err
			}

			result, err = x.MatMul(w)
			if err != nil {
				return nil, err
			}
		}

	default: // 2D or other cases: standard matmul
		w, err := l.weight.Transpose(-1, -2)
		if err != nil {
			return nil, err
		}

		result, err = x.MatMul(w)
		if err != nil {
			return nil, err
		}
	}

	// Add bias if present
	if l.bias != nil {
		result, err = result.BroadcastAdd(l.bias)
		if err != nil {
			return nil, fmt.Errorf("failed to add bias: %w", err)
		}
	}

	return result, nil
}

// MustForward applies the linear transformation, panicking on error
func (l *Linear[T]) MustForward(x *tensor.Tensor[T]) *tensor.Tensor[T] {
	result, err := l.Forward(x)
	if err != nil {
		panic(err)
	}
	return result
}

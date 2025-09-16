package goml

// Scalar represents a scalar value that can be used in tensor operations
type Scalar interface {
	// Float64 returns the scalar value as float64
	Float64() float64
	// Float32 returns the scalar value as float32 (may lose precision)
	Float32() float32
	// DType returns the original data type of the scalar
	DType() DType
}

// scalar is the internal implementation of Scalar
type scalar[T D] struct {
	value T
}

// NewScalar creates a new Scalar from a numeric value
func NewScalar[T D](value T) Scalar {
	return scalar[T]{value: value}
}

// NewScalarFrom creates a Scalar from various numeric types
func NewScalarFromFloat32(value float32) Scalar {
	return scalar[float32]{value: value}
}

func NewScalarFromFloat64(value float64) Scalar {
	return scalar[float64]{value: value}
}

func NewScalarFromInt(value int) Scalar {
	return scalar[float64]{value: float64(value)}
}

func NewZeroScalar[T D]() Scalar {
	return scalar[T]{value: T(0)}
}

func NewOneScalar[T D]() Scalar {
	return scalar[T]{value: T(1)}
}

func (s scalar[T]) Float64() float64 {
	return float64(s.value)
}

func (s scalar[T]) Float32() float32 {
	return float32(s.value)
}

func (s scalar[T]) DType() DType {
	switch any(T(0)).(type) {
	case float32:
		return Float32
	case float64:
		return Float64
	default:
		return Float32 // fallback
	}
}

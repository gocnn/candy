package goml

// DType represents the data type of tensors (enum for runtime use)
type DType int

const (
	Float32 DType = iota
	Float64
)

func (dt DType) String() string {
	switch dt {
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	default:
		return "unknown"
	}
}

// BitSize returns the size in bits of the data type
func (dt DType) BitSize() int {
	switch dt {
	case Float32:
		return 32
	case Float64:
		return 64
	default:
		return 0
	}
}

// D is the type constraint for matrices defined in this package.
type D interface {
	float32 | float64
}

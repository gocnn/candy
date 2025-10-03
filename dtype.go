package spark

// D is the type constraint for matrices defined in this package.
type D interface {
	float32 | float64
}

type DType int

const (
	F32 DType = iota
	F64
	F16
	BF16
	U8
	U32
	I64
)

func (d DType) String() string {
	switch d {
	case F32:
		return "f32"
	case F64:
		return "f64"
	case F16:
		return "f16"
	case BF16:
		return "bf16"
	case U8:
		return "u8"
	case U32:
		return "u32"
	case I64:
		return "i64"
	default:
		return "unknown"
	}
}

// DTypeOf returns the DType corresponding to the given type parameter
func DTypeOf[T D]() DType {
	var zero T
	switch any(zero).(type) {
	case float32:
		return F32
	case float64:
		return F64
	default:
		// This should never happen due to type constraint
		panic("unsupported type for DTypeOf")
	}
}

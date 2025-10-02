package spark

// CmpOp represents comparison operations
type CmpOp int

const (
	CmpEq CmpOp = iota
	CmpNe
	CmpLe
	CmpGe
	CmpLt
	CmpGt
)

// String returns the string representation of the comparison operation
func (op CmpOp) String() string {
	switch op {
	case CmpEq:
		return "eq"
	case CmpNe:
		return "ne"
	case CmpLe:
		return "le"
	case CmpGe:
		return "ge"
	case CmpLt:
		return "lt"
	case CmpGt:
		return "gt"
	default:
		return "unknown"
	}
}

// ReduceOp represents reduction operations
type ReduceOp int

const (
	ReduceSum ReduceOp = iota
	ReduceMin
	ReduceMax
	ReduceArgMin
	ReduceArgMax
)

// String returns the string representation of the reduce operation
func (op ReduceOp) String() string {
	switch op {
	case ReduceSum:
		return "sum"
	case ReduceMin:
		return "min"
	case ReduceMax:
		return "max"
	case ReduceArgMin:
		return "argmin"
	case ReduceArgMax:
		return "argmax"
	default:
		return "unknown"
	}
}

// BinaryOp represents binary operations that return the same type as input
type BinaryOp int

const (
	BinaryAdd BinaryOp = iota
	BinaryMul
	BinarySub
	BinaryDiv
	BinaryMaximum
	BinaryMinimum
)

// String returns the string representation of the binary operation
func (op BinaryOp) String() string {
	switch op {
	case BinaryAdd:
		return "add"
	case BinaryMul:
		return "mul"
	case BinarySub:
		return "sub"
	case BinaryDiv:
		return "div"
	case BinaryMaximum:
		return "maximum"
	case BinaryMinimum:
		return "minimum"
	default:
		return "unknown"
	}
}

// UnaryOp represents unary operations with no arguments
type UnaryOp int

const (
	UnaryExp UnaryOp = iota
	UnaryLog
	UnarySin
	UnaryCos
	UnaryAbs
	UnaryNeg
	UnaryRecip
	UnarySqr
	UnarySqrt
	UnaryGelu
	UnaryGeluErf
	UnaryErf
	UnaryRelu
	UnarySilu
	UnaryTanh
	UnaryFloor
	UnaryCeil
	UnaryRound
	UnarySign
)

// String returns the string representation of the unary operation
func (op UnaryOp) String() string {
	switch op {
	case UnaryExp:
		return "exp"
	case UnaryLog:
		return "log"
	case UnarySin:
		return "sin"
	case UnaryCos:
		return "cos"
	case UnaryAbs:
		return "abs"
	case UnaryNeg:
		return "neg"
	case UnaryRecip:
		return "recip"
	case UnarySqr:
		return "sqr"
	case UnarySqrt:
		return "sqrt"
	case UnaryGelu:
		return "gelu"
	case UnaryGeluErf:
		return "gelu_erf"
	case UnaryErf:
		return "erf"
	case UnaryRelu:
		return "relu"
	case UnarySilu:
		return "silu"
	case UnaryTanh:
		return "tanh"
	case UnaryFloor:
		return "floor"
	case UnaryCeil:
		return "ceil"
	case UnaryRound:
		return "round"
	case UnarySign:
		return "sign"
	default:
		return "unknown"
	}
}

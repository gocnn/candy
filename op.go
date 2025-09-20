package spark

import "math"

// Tensor represents a tensor with storage and layout.
// It is an interface to allow different implementations.
type Tensor interface {
	TrackOp() bool // Indicates if the tensor tracks operations for backpropagation.
	Clone() Tensor // Creates a deep copy of the tensor.
}

// CmpOp represents comparison operations.
type CmpOp int

const (
	CmpOpEq CmpOp = iota // Equal
	CmpOpNe              // Not equal
	CmpOpLe              // Less than or equal
	CmpOpGe              // Greater than or equal
	CmpOpLt              // Less than
	CmpOpGt              // Greater than
)

// ReduceOp represents reduction operations.
type ReduceOp int

const (
	ReduceOpSum    ReduceOp = iota // Sum
	ReduceOpMin                    // Minimum
	ReduceOpMax                    // Maximum
	ReduceOpArgMin                 // Index of minimum
	ReduceOpArgMax                 // Index of maximum
)

// Name returns the string name of the reduction operation.
func (r ReduceOp) Name() string {
	switch r {
	case ReduceOpSum:
		return "sum"
	case ReduceOpMin:
		return "min"
	case ReduceOpMax:
		return "max"
	case ReduceOpArgMin:
		return "argmin"
	case ReduceOpArgMax:
		return "argmax"
	default:
		return "unknown"
	}
}

// BinaryOp represents binary operations that return the same type as inputs.
type BinaryOp int

const (
	BinaryOpAdd     BinaryOp = iota // Addition
	BinaryOpMul                     // Multiplication
	BinaryOpSub                     // Subtraction
	BinaryOpDiv                     // Division
	BinaryOpMaximum                 // Element-wise maximum
	BinaryOpMinimum                 // Element-wise minimum
)

// UnaryOp represents unary operations with no arguments.
type UnaryOp int

const (
	UnaryOpExp     UnaryOp = iota // Exponential
	UnaryOpLog                    // Natural logarithm
	UnaryOpSin                    // Sine
	UnaryOpCos                    // Cosine
	UnaryOpAbs                    // Absolute value
	UnaryOpNeg                    // Negation
	UnaryOpRecip                  // Reciprocal
	UnaryOpSqr                    // Square
	UnaryOpSqrt                   // Square root
	UnaryOpGelu                   // GELU activation (tanh approximation)
	UnaryOpGeluErf                // GELU activation (erf-based)
	UnaryOpErf                    // Error function
	UnaryOpRelu                   // ReLU activation
	UnaryOpSilu                   // SiLU activation
	UnaryOpTanh                   // Hyperbolic tangent
	UnaryOpFloor                  // Floor
	UnaryOpCeil                   // Ceil
	UnaryOpRound                  // Round
	UnaryOpSign                   // Sign
)

// UnaryOpT defines the interface for unary operations across different data types.
type UnaryOpT interface {
	Name() string   // Returns the operation name (e.g., "exp").
	Kernel() string // Returns the kernel name (e.g., "uexp").
	ApplyF32(x float32) float32
	ApplyF64(x float64) float64
	// ApplyF16(x float16) float16 // Placeholder for half-precision types if needed.
	// ApplyBF16(x bf16) bf16
	// ApplyF8E4M3(x f8e4m3) f8e4m3
	// ApplyU8(x uint8) uint8
	ApplyU32(x uint32) uint32
	ApplyI64(x int64) int64
	// Vectorized operations can be added if needed (e.g., for MKL or Accelerate).
}

// BinaryOpT defines the interface for binary operations across different data types.
type BinaryOpT interface {
	Name() string   // Returns the operation name (e.g., "add").
	Kernel() string // Returns the kernel name (e.g., "badd").
	ApplyF32(x, y float32) float32
	ApplyF64(x, y float64) float64
	// ApplyF16(x, y float16) float16
	// ApplyBF16(x, y bf16) bf16
	// ApplyF8E4M3(x, y f8e4m3) f8e4m3
	// ApplyU8(x, y uint8) uint8
	ApplyU32(x, y uint32) uint32
	ApplyI64(x, y int64) int64
	// Vectorized operations can be added if needed.
}

// Op represents a tensor operation for computation and backpropagation.
type Op struct {
	Type OpType
}

// OpType defines the specific operation variants.
type OpType interface {
	// Define a method to identify the operation type if needed.
}

// Concrete operation types
type (
	BinaryOpStruct struct {
		Lhs, Rhs Tensor
		Op       BinaryOp
	}
	UnaryOpStruct struct {
		Arg Tensor
		Op  UnaryOp
	}
	CmpOpStruct struct {
		Lhs, Rhs Tensor
		Op       CmpOp
	}
	ReduceOpStruct struct {
		Arg  Tensor
		Op   ReduceOp
		Dims []int
	}
	MatmulOp struct {
		Lhs, Rhs Tensor
	}
	GatherOp struct {
		Arg, Indices Tensor
		Dim          int
	}
	ScatterOp struct {
		Dst, Indices, Values Tensor
		Dim                  int
	}
	ScatterAddOp struct {
		Dst, Indices, Values Tensor
		Dim                  int
	}
	IndexSelectOp struct {
		Arg, Indices Tensor
		Dim          int
	}
	IndexAddOp struct {
		Dst, Indices, Values Tensor
		Dim                  int
	}
	WhereCondOp struct {
		Cond, TrueVal, FalseVal Tensor
	}
	Conv1DOp struct {
		Arg, Kernel               Tensor
		Padding, Stride, Dilation int
	}
	ConvTranspose1DOp struct {
		Arg, Kernel                              Tensor
		Padding, OutputPadding, Stride, Dilation int
	}
	Conv2DOp struct {
		Arg, Kernel               Tensor
		Padding, Stride, Dilation int
	}
	ConvTranspose2DOp struct {
		Arg, Kernel                              Tensor
		Padding, OutputPadding, Stride, Dilation int
	}
	AvgPool2DOp struct {
		Arg                Tensor
		KernelSize, Stride [2]int
	}
	MaxPool2DOp struct {
		Arg                Tensor
		KernelSize, Stride [2]int
	}
	UpsampleNearest1DOp struct {
		Arg        Tensor
		TargetSize int
	}
	UpsampleNearest2DOp struct {
		Arg              Tensor
		TargetH, TargetW int
	}
	CatOp struct {
		Tensors []Tensor
		Dim     int
	}
	AffineOp struct {
		Arg      Tensor
		Mul, Add float64
	}
	ToDTypeOp struct {
		Arg Tensor
	}
	CopyOp struct {
		Arg Tensor
	}
	BroadcastOp struct {
		Arg Tensor
	}
	NarrowOp struct {
		Arg             Tensor
		Dim, Start, Len int
	}
	SliceScatter0Op struct {
		Dst, Src Tensor
		Dim      int
	}
	ReshapeOp struct {
		Arg Tensor
	}
	ToDeviceOp struct {
		Arg Tensor
	}
	TransposeOp struct {
		Arg        Tensor
		Dim1, Dim2 int
	}
	PermuteOp struct {
		Arg  Tensor
		Dims []int
	}
	EluOp struct {
		Arg   Tensor
		Alpha float64
	}
	PowfOp struct {
		Arg   Tensor
		Power float64
	}
)

// BackpropOp wraps an optional Op for backpropagation tracking.
type BackpropOp struct {
	op *Op
}

// None creates a BackpropOp with no operation.
func None() BackpropOp {
	return BackpropOp{nil}
}

// New1 creates a BackpropOp for a unary operation if the argument tracks operations.
func New1(arg Tensor, f func(Tensor) Op) BackpropOp {
	if arg.TrackOp() {
		return BackpropOp{&Op{f(arg.Clone())}}
	}
	return BackpropOp{nil}
}

// New2 creates a BackpropOp for a binary operation if any argument tracks operations.
func New2(arg1, arg2 Tensor, f func(Tensor, Tensor) Op) BackpropOp {
	if arg1.TrackOp() || arg2.TrackOp() {
		return BackpropOp{&Op{f(arg1.Clone(), arg2.Clone())}}
	}
	return BackpropOp{nil}
}

// New3 creates a BackpropOp for a ternary operation if any argument tracks operations.
func New3(arg1, arg2, arg3 Tensor, f func(Tensor, Tensor, Tensor) Op) BackpropOp {
	if arg1.TrackOp() || arg2.TrackOp() || arg3.TrackOp() {
		return BackpropOp{&Op{f(arg1.Clone(), arg2.Clone(), arg3.Clone())}}
	}
	return BackpropOp{nil}
}

// New creates a BackpropOp for an operation with multiple arguments.
func New(args []Tensor, f func([]Tensor) Op) BackpropOp {
	for _, arg := range args {
		if arg.TrackOp() {
			cloned := make([]Tensor, len(args))
			for i, a := range args {
				cloned[i] = a.Clone()
			}
			return BackpropOp{&Op{f(cloned)}}
		}
	}
	return BackpropOp{nil}
}

// IsNone checks if the BackpropOp contains no operation.
func (b BackpropOp) IsNone() bool {
	return b.op == nil
}

// Op returns the wrapped operation, if any.
func (b BackpropOp) Op() *Op {
	return b.op
}

// Unary operation implementations
type (
	ExpOp     struct{}
	LogOp     struct{}
	SinOp     struct{}
	CosOp     struct{}
	TanhOp    struct{}
	NegOp     struct{}
	RecipOp   struct{}
	SqrOp     struct{}
	SqrtOp    struct{}
	GeluOp    struct{}
	GeluErfOp struct{}
	ErfOp     struct{}
	ReluOp    struct{}
	SiluOp    struct{}
	AbsOp     struct{}
	FloorOp   struct{}
	CeilOp    struct{}
	RoundOp   struct{}
	SignOp    struct{}
)

// Binary operation implementations
type (
	AddOp     struct{}
	SubOp     struct{}
	MulOp     struct{}
	DivOp     struct{}
	MaximumOp struct{}
	MinimumOp struct{}
)

// Constants for GELU
const (
	sqrtTwoOverPiF32 = 0.79788456080286535587989211986876373
	sqrtTwoOverPiF64 = 0.79788456080286535587989211986876373
)

// UnaryOpT implementations
func (ExpOp) Name() string               { return "exp" }
func (ExpOp) Kernel() string             { return "uexp" }
func (ExpOp) ApplyF32(x float32) float32 { return float32(math.Exp(float64(x))) }
func (ExpOp) ApplyF64(x float64) float64 { return math.Exp(x) }
func (ExpOp) ApplyU8(x uint8) uint8      { panic("no unary op for u8") }
func (ExpOp) ApplyU32(x uint32) uint32   { panic("no unary op for u32") }
func (ExpOp) ApplyI64(x int64) int64     { panic("no unary op for i64") }

func (LogOp) Name() string               { return "log" }
func (LogOp) Kernel() string             { return "ulog" }
func (LogOp) ApplyF32(x float32) float32 { return float32(math.Log(float64(x))) }
func (LogOp) ApplyF64(x float64) float64 { return math.Log(x) }
func (LogOp) ApplyU8(x uint8) uint8      { panic("no unary op for u8") }
func (LogOp) ApplyU32(x uint32) uint32   { panic("no unary op for u32") }
func (LogOp) ApplyI64(x int64) int64     { panic("no unary op for i64") }

func (SinOp) Name() string               { return "sin" }
func (SinOp) Kernel() string             { return "usin" }
func (SinOp) ApplyF32(x float32) float32 { return float32(math.Sin(float64(x))) }
func (SinOp) ApplyF64(x float64) float64 { return math.Sin(x) }
func (SinOp) ApplyU8(x uint8) uint8      { panic("no unary op for u8") }
func (SinOp) ApplyU32(x uint32) uint32   { panic("no unary op for u32") }
func (SinOp) ApplyI64(x int64) int64     { panic("no unary op for i64") }

func (CosOp) Name() string               { return "cos" }
func (CosOp) Kernel() string             { return "ucos" }
func (CosOp) ApplyF32(x float32) float32 { return float32(math.Cos(float64(x))) }
func (CosOp) ApplyF64(x float64) float64 { return math.Cos(x) }
func (CosOp) ApplyU8(x uint8) uint8      { panic("no unary op for u8") }
func (CosOp) ApplyU32(x uint32) uint32   { panic("no unary op for u32") }
func (CosOp) ApplyI64(x int64) int64     { panic("no unary op for i64") }

func (TanhOp) Name() string               { return "tanh" }
func (TanhOp) Kernel() string             { return "utanh" }
func (TanhOp) ApplyF32(x float32) float32 { return float32(math.Tanh(float64(x))) }
func (TanhOp) ApplyF64(x float64) float64 { return math.Tanh(x) }
func (TanhOp) ApplyU8(x uint8) uint8      { panic("no unary op for u8") }
func (TanhOp) ApplyU32(x uint32) uint32   { panic("no unary op for u32") }
func (TanhOp) ApplyI64(x int64) int64     { panic("no unary op for i64") }

func (NegOp) Name() string               { return "neg" }
func (NegOp) Kernel() string             { return "uneg" }
func (NegOp) ApplyF32(x float32) float32 { return -x }
func (NegOp) ApplyF64(x float64) float64 { return -x }
func (NegOp) ApplyU8(x uint8) uint8      { panic("no unary op for u8") }
func (NegOp) ApplyU32(x uint32) uint32   { panic("no unary op for u32") }
func (NegOp) ApplyI64(x int64) int64     { return -x }

func (RecipOp) Name() string               { return "recip" }
func (RecipOp) Kernel() string             { return "urecip" }
func (RecipOp) ApplyF32(x float32) float32 { return 1 / x }
func (RecipOp) ApplyF64(x float64) float64 { return 1 / x }
func (RecipOp) ApplyU8(x uint8) uint8      { panic("no unary op for u8") }
func (RecipOp) ApplyU32(x uint32) uint32   { panic("no unary op for u32") }
func (RecipOp) ApplyI64(x int64) int64     { panic("no unary op for i64") }

func (SqrOp) Name() string               { return "sqr" }
func (SqrOp) Kernel() string             { return "usqr" }
func (SqrOp) ApplyF32(x float32) float32 { return x * x }
func (SqrOp) ApplyF64(x float64) float64 { return x * x }
func (SqrOp) ApplyU8(x uint8) uint8      { panic("no unary op for u8") }
func (SqrOp) ApplyU32(x uint32) uint32   { panic("no unary op for u32") }
func (SqrOp) ApplyI64(x int64) int64     { panic("no unary op for i64") }

func (SqrtOp) Name() string               { return "sqrt" }
func (SqrtOp) Kernel() string             { return "usqrt" }
func (SqrtOp) ApplyF32(x float32) float32 { return float32(math.Sqrt(float64(x))) }
func (SqrtOp) ApplyF64(x float64) float64 { return math.Sqrt(x) }
func (SqrtOp) ApplyU8(x uint8) uint8      { panic("no unary op for u8") }
func (SqrtOp) ApplyU32(x uint32) uint32   { panic("no unary op for u32") }
func (SqrtOp) ApplyI64(x int64) int64     { panic("no unary op for i64") }

func (GeluOp) Name() string   { return "gelu" }
func (GeluOp) Kernel() string { return "ugelu" }
func (GeluOp) ApplyF32(x float32) float32 {
	return 0.5 * x * (1.0 + float32(math.Tanh(float64(sqrtTwoOverPiF32*x*(1.0+0.044715*x*x)))))
}
func (GeluOp) ApplyF64(x float64) float64 {
	return 0.5 * x * (1.0 + math.Tanh(sqrtTwoOverPiF64*x*(1.0+0.044715*x*x)))
}
func (GeluOp) ApplyU8(x uint8) uint8    { return 0 }
func (GeluOp) ApplyU32(x uint32) uint32 { return 0 }
func (GeluOp) ApplyI64(x int64) int64   { return 0 }

func (GeluErfOp) Name() string   { return "gelu_erf" }
func (GeluErfOp) Kernel() string { return "ugelu_erf" }
func (GeluErfOp) ApplyF32(x float32) float32 {
	return float32((math.Erf(float64(x)/math.Sqrt2) + 1.0) * 0.5 * float64(x))
}
func (GeluErfOp) ApplyF64(x float64) float64 {
	return (math.Erf(x/math.Sqrt2) + 1.0) * 0.5 * x
}
func (GeluErfOp) ApplyU8(x uint8) uint8    { return 0 }
func (GeluErfOp) ApplyU32(x uint32) uint32 { return 0 }
func (GeluErfOp) ApplyI64(x int64) int64   { return 0 }

func (ErfOp) Name() string               { return "erf" }
func (ErfOp) Kernel() string             { return "uerf" }
func (ErfOp) ApplyF32(x float32) float32 { return float32(math.Erf(float64(x))) }
func (ErfOp) ApplyF64(x float64) float64 { return math.Erf(x) }
func (ErfOp) ApplyU8(x uint8) uint8      { return 0 }
func (ErfOp) ApplyU32(x uint32) uint32   { return 0 }
func (ErfOp) ApplyI64(x int64) int64     { return 0 }

func (ReluOp) Name() string   { return "relu" }
func (ReluOp) Kernel() string { return "urelu" }
func (ReluOp) ApplyF32(x float32) float32 {
	if x > 0 {
		return x
	} else {
		return 0
	}
}
func (ReluOp) ApplyF64(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0
	}
}
func (ReluOp) ApplyU8(x uint8) uint8    { return x }
func (ReluOp) ApplyU32(x uint32) uint32 { return x }
func (ReluOp) ApplyI64(x int64) int64 {
	if x > 0 {
		return x
	} else {
		return 0
	}
}

func (SiluOp) Name() string               { return "silu" }
func (SiluOp) Kernel() string             { return "usilu" }
func (SiluOp) ApplyF32(x float32) float32 { return x / (1.0 + float32(math.Exp(-float64(x)))) }
func (SiluOp) ApplyF64(x float64) float64 { return x / (1.0 + math.Exp(-x)) }
func (SiluOp) ApplyU8(x uint8) uint8      { return 0 }
func (SiluOp) ApplyU32(x uint32) uint32   { return 0 }
func (SiluOp) ApplyI64(x int64) int64     { return 0 }

func (AbsOp) Name() string               { return "abs" }
func (AbsOp) Kernel() string             { return "uabs" }
func (AbsOp) ApplyF32(x float32) float32 { return float32(math.Abs(float64(x))) }
func (AbsOp) ApplyF64(x float64) float64 { return math.Abs(x) }
func (AbsOp) ApplyU8(x uint8) uint8      { return x }
func (AbsOp) ApplyU32(x uint32) uint32   { return x }
func (AbsOp) ApplyI64(x int64) int64 {
	if x < 0 {
		return -x
	} else {
		return x
	}
}

func (FloorOp) Name() string               { return "floor" }
func (FloorOp) Kernel() string             { return "ufloor" }
func (FloorOp) ApplyF32(x float32) float32 { return float32(math.Floor(float64(x))) }
func (FloorOp) ApplyF64(x float64) float64 { return math.Floor(x) }
func (FloorOp) ApplyU8(x uint8) uint8      { return x }
func (FloorOp) ApplyU32(x uint32) uint32   { return x }
func (FloorOp) ApplyI64(x int64) int64     { return x }

func (CeilOp) Name() string               { return "ceil" }
func (CeilOp) Kernel() string             { return "uceil" }
func (CeilOp) ApplyF32(x float32) float32 { return float32(math.Ceil(float64(x))) }
func (CeilOp) ApplyF64(x float64) float64 { return math.Ceil(x) }
func (CeilOp) ApplyU8(x uint8) uint8      { return x }
func (CeilOp) ApplyU32(x uint32) uint32   { return x }
func (CeilOp) ApplyI64(x int64) int64     { return x }

func (RoundOp) Name() string               { return "round" }
func (RoundOp) Kernel() string             { return "uround" }
func (RoundOp) ApplyF32(x float32) float32 { return float32(math.Round(float64(x))) }
func (RoundOp) ApplyF64(x float64) float64 { return math.Round(x) }
func (RoundOp) ApplyU8(x uint8) uint8      { return x }
func (RoundOp) ApplyU32(x uint32) uint32   { return x }
func (RoundOp) ApplyI64(x int64) int64     { return x }

func (SignOp) Name() string   { return "sign" }
func (SignOp) Kernel() string { return "usign" }
func (SignOp) ApplyF32(x float32) float32 {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
}
func (SignOp) ApplyF64(x float64) float64 {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
}
func (SignOp) ApplyU8(x uint8) uint8 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}
func (SignOp) ApplyU32(x uint32) uint32 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}
func (SignOp) ApplyI64(x int64) int64 {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	} else {
		return 0
	}
}

// BinaryOpT implementations
func (AddOp) Name() string                  { return "add" }
func (AddOp) Kernel() string                { return "badd" }
func (AddOp) ApplyF32(x, y float32) float32 { return x + y }
func (AddOp) ApplyF64(x, y float64) float64 { return x + y }
func (AddOp) ApplyU8(x, y uint8) uint8      { return x + y }
func (AddOp) ApplyU32(x, y uint32) uint32   { return x + y }
func (AddOp) ApplyI64(x, y int64) int64     { return x + y }

func (SubOp) Name() string                  { return "sub" }
func (SubOp) Kernel() string                { return "bsub" }
func (SubOp) ApplyF32(x, y float32) float32 { return x - y }
func (SubOp) ApplyF64(x, y float64) float64 { return x - y }
func (SubOp) ApplyU8(x, y uint8) uint8      { return x - y }
func (SubOp) ApplyU32(x, y uint32) uint32   { return x - y }
func (SubOp) ApplyI64(x, y int64) int64     { return x - y }

func (MulOp) Name() string                  { return "mul" }
func (MulOp) Kernel() string                { return "bmul" }
func (MulOp) ApplyF32(x, y float32) float32 { return x * y }
func (MulOp) ApplyF64(x, y float64) float64 { return x * y }
func (MulOp) ApplyU8(x, y uint8) uint8      { return x * y }
func (MulOp) ApplyU32(x, y uint32) uint32   { return x * y }
func (MulOp) ApplyI64(x, y int64) int64     { return x * y }

func (DivOp) Name() string                  { return "div" }
func (DivOp) Kernel() string                { return "bdiv" }
func (DivOp) ApplyF32(x, y float32) float32 { return x / y }
func (DivOp) ApplyF64(x, y float64) float64 { return x / y }
func (DivOp) ApplyU8(x, y uint8) uint8      { return x / y }
func (DivOp) ApplyU32(x, y uint32) uint32   { return x / y }
func (DivOp) ApplyI64(x, y int64) int64     { return x / y }

func (MaximumOp) Name() string   { return "maximum" }
func (MaximumOp) Kernel() string { return "bmaximum" }
func (MaximumOp) ApplyF32(x, y float32) float32 {
	if x < y {
		return y
	} else {
		return x
	}
}
func (MaximumOp) ApplyF64(x, y float64) float64 {
	if x < y {
		return y
	} else {
		return x
	}
}
func (MaximumOp) ApplyU8(x, y uint8) uint8 {
	if x < y {
		return y
	} else {
		return x
	}
}
func (MaximumOp) ApplyU32(x, y uint32) uint32 {
	if x < y {
		return y
	} else {
		return x
	}
}
func (MaximumOp) ApplyI64(x, y int64) int64 {
	if x < y {
		return y
	} else {
		return x
	}
}

func (MinimumOp) Name() string   { return "minimum" }
func (MinimumOp) Kernel() string { return "bminimum" }
func (MinimumOp) ApplyF32(x, y float32) float32 {
	if x > y {
		return y
	} else {
		return x
	}
}
func (MinimumOp) ApplyF64(x, y float64) float64 {
	if x > y {
		return y
	} else {
		return x
	}
}
func (MinimumOp) ApplyU8(x, y uint8) uint8 {
	if x > y {
		return y
	} else {
		return x
	}
}
func (MinimumOp) ApplyU32(x, y uint32) uint32 {
	if x > y {
		return y
	} else {
		return x
	}
}
func (MinimumOp) ApplyI64(x, y int64) int64 {
	if x > y {
		return y
	} else {
		return x
	}
}

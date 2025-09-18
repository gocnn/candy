package spark

// Tensor represents a multi-dimensional array with generic type T.
type Tensor[T D] interface {
	// Basic Properties
	DType() DType
	Layout() Layout
	Device() Device
	Size() int
	RequiresGrad() bool
	Clone() Tensor[T]

	// Convenience methods for shape access
	Shape() Shape
	// Dims() []int
	// Ndim() int
	// Rank() int

	// Layout operations
	// IsContiguous() bool
	// IsFortranContiguous() bool
	// Contiguous() Tensor[T]
	// Transpose(dim1, dim2 int) Tensor[T]
	// Permute(dims []int) Tensor[T]
	// Narrow(dim, start, length int) Tensor[T]

	// Tensor Creation
	FullLike(value T) Tensor[T]
	OnesLike() Tensor[T]
	ZerosLike() Tensor[T]
	RandLike(lo, hi T) Tensor[T]
	RandnLike(mean, std T) Tensor[T]

	// Shape Manipulation
	Reshape(dims ...int) Tensor[T]
	// Squeeze(dim ...int) (Tensor[T], error)
	// Unsqueeze(dim int) (Tensor[T], error)
	// T() (Tensor[T], error)
	// Flatten() (Tensor[T], error)
	// FlattenFrom(dim int) (Tensor[T], error)
	// FlattenTo(dim int) (Tensor[T], error)
	// Repeat(repeats []int) (Tensor[T], error)
	// Expand(shape Shape) (Tensor[T], error)
	// PadWithZeros(dim, before, after int) (Tensor[T], error)
	// PadWithSame(dim, before, after int) (Tensor[T], error)

	// Indexing and Slicing
	// Narrow(dim, start, length int) (Tensor[T], error)
	// Get(index ...int) (Tensor[T], error)
	// GetOnDim(dim, index int) (Tensor[T], error)
	// IndexSelect(dim int, indices []int) (Tensor[T], error)
	// Gather(dim int, indices []int) (Tensor[T], error)
	// Scatter(dim int, indices []int, src Tensor[T]) (Tensor[T], error)
	// ScatterAdd(dim int, indices []int, src Tensor[T]) (Tensor[T], error)
	// ScatterSet(dim int, indices []int, src Tensor[T]) (Tensor[T], error)
	// IndexAdd(dim int, indices []int, src Tensor[T]) (Tensor[T], error)
	// Chunk(chunks, dim int) ([]Tensor[T], error)
	// Split(sizes []int, dim int) ([]Tensor[T], error)
	// SliceAssign(dim, start, length int, src Tensor[T]) (Tensor[T], error)
	// SliceSet(dim, start, length int, src Tensor[T]) (Tensor[T], error)

	// Arithmetic Operations
	Add(other Tensor[T]) (Tensor[T], error)
	// Sub(other Tensor[T]) (Tensor[T], error)
	// Mul(other Tensor[T]) (Tensor[T], error)
	// Div(other Tensor[T]) (Tensor[T], error)
	// Pow(other Tensor[T]) (Tensor[T], error)
	// Powf(exponent T) (Tensor[T], error)
	// MatMul(other Tensor[T]) (Tensor[T], error)
	// Dot(other Tensor[T]) (Tensor[T], error)
	// Affine(mul, add T) (Tensor[T], error)

	// Broadcasting Operations
	// BroadcastAs(shape Shape) (Tensor[T], error)
	// BroadcastAdd(other Tensor[T]) (Tensor[T], error)
	// BroadcastSub(other Tensor[T]) (Tensor[T], error)
	// BroadcastMul(other Tensor[T]) (Tensor[T], error)
	// BroadcastDiv(other Tensor[T]) (Tensor[T], error)
	// BroadcastPow(other Tensor[T]) (Tensor[T], error)
	// BroadcastMaximum(other Tensor[T]) (Tensor[T], error)
	// BroadcastMinimum(other Tensor[T]) (Tensor[T], error)

	// Element-wise Operations
	// Neg() (Tensor[T], error)
	// Abs() (Tensor[T], error)
	// Sign() (Tensor[T], error)
	// Recip() (Tensor[T], error)
	// Sqr() (Tensor[T], error)
	// Sqrt() (Tensor[T], error)
	// Exp() (Tensor[T], error)
	// Log() (Tensor[T], error)
	// Sin() (Tensor[T], error)
	// Cos() (Tensor[T], error)
	// Tanh() (Tensor[T], error)
	// Erf() (Tensor[T], error)
	// Ceil() (Tensor[T], error)
	// Floor() (Tensor[T], error)
	// Round() (Tensor[T], error)
	// RoundTo(decimals int) (Tensor[T], error)

	// Activation Functions
	// Relu() (Tensor[T], error)
	// Gelu() (Tensor[T], error)
	// GeluErf() (Tensor[T], error)
	// Elu() (Tensor[T], error)
	// Silu() (Tensor[T], error)

	// Reduction Operations
	// Sum(dim ...int) (Tensor[T], error)
	// SumKeepDim(dim ...int) (Tensor[T], error)
	// SumAll() (T, error)
	// Mean(dim ...int) (Tensor[T], error)
	// MeanKeepDim(dim ...int) (Tensor[T], error)
	// MeanAll() (T, error)
	// Max(dim ...int) (Tensor[T], error)
	// MaxKeepDim(dim ...int) (Tensor[T], error)
	// MaxAll() (T, error)
	// Min(dim ...int) (Tensor[T], error)
	// MinKeepDim(dim ...int) (Tensor[T], error)
	// MinAll() (T, error)
	// Var(dim int) (Tensor[T], error)
	// VarKeepDim(dim int) (Tensor[T], error)
	// VarAll() (T, error)
	// Cumsum(dim int) (Tensor[T], error)
	// LogSumExp(dim ...int) (Tensor[T], error)
	// ArgMax(dim int) ([]int, error)
	// ArgMaxKeepDim(dim int) ([]int, error)
	// ArgMin(dim int) ([]int, error)
	// ArgMinKeepDim(dim int) ([]int, error)

	// Comparison Operations
	// Eq(other Tensor[T]) (Tensor[T], error)
	// Ne(other Tensor[T]) (Tensor[T], error)
	// Lt(other Tensor[T]) (Tensor[T], error)
	// Le(other Tensor[T]) (Tensor[T], error)
	// Gt(other Tensor[T]) (Tensor[T], error)
	// Ge(other Tensor[T]) (Tensor[T], error)
	// BroadcastEq(other Tensor[T]) (Tensor[T], error)
	// BroadcastNe(other Tensor[T]) (Tensor[T], error)
	// BroadcastLt(other Tensor[T]) (Tensor[T], error)
	// BroadcastLe(other Tensor[T]) (Tensor[T], error)
	// BroadcastGt(other Tensor[T]) (Tensor[T], error)
	// BroadcastGe(other Tensor[T]) (Tensor[T], error)
	// Cmp(other Tensor[T], op string) (Tensor[T], error)

	// Sorting and Conditional Operations
	// Sort(dim int) (Tensor[T], Tensor[T], error)
	// ArgSort(dim int) (Tensor[T], error)
	// Clamp(min, max T) (Tensor[T], error)
	// Where(cond Tensor[T], x, y Tensor[T]) (Tensor[T], error)

	// Convolution and Pooling
	// Conv1D(kernel Tensor[T], stride, padding int) (Tensor[T], error)
	// Conv2D(kernel Tensor[T], stride, padding []int) (Tensor[T], error)
	// ConvTranspose1D(kernel Tensor[T], stride, padding int) (Tensor[T], error)
	// ConvTranspose2D(kernel Tensor[T], stride, padding []int) (Tensor[T], error)
	// AvgPool2D(kernel, stride []int) (Tensor[T], error)
	// MaxPool2D(kernel, stride []int) (Tensor[T], error)
	// Interpolate1D(size int) (Tensor[T], error)
	// Interpolate2D(h, w int) (Tensor[T], error)
	// UpsampleNearest1D(size int) (Tensor[T], error)
	// UpsampleNearest2D(h, w int) (Tensor[T], error)

	// Specialized Operations
	// Embedding(indices []int) (Tensor[T], error)
	// MeshGrid(tensors []Tensor[T]) ([]Tensor[T], error)

	// Gradient and Memory Operations
	// Backward() (Tensor[T], error)
	AccGrad(gy Tensor[T]) error
	// TrackOp() error
	// ID() string
	// ToSlice() ([]T, error)
	// ToDType(dtype DType) (Tensor[T], error)
	// Copy() (Tensor[T], error)
	// Detach() (Tensor[T], error)
	// ConstSet(value T) (Tensor[T], error)
	// OneSet() (Tensor[T], error)
	// ZeroSet() (Tensor[T], error)

	// File Operations
	// ReadNpy(filePath string) (Tensor[T], error)
	// ReadNpz(filePath string) (Tensor[T], error)
	// ReadNpzByName(filePath, name string) (Tensor[T], error)
	// WriteNpy(filePath string) error
	// WriteNpz(filePath string) error
	// SaveSafetensors(filePath string) error
	// WriteBytes() ([]byte, error)

	// Miscellaneous Operations
	// Flip(dims []int) (Tensor[T], error)
	// Roll(shift, dim int) (Tensor[T], error)
	// Tril2(offset int) (Tensor[T], error)
	// Triu2(offset int) (Tensor[T], error)
	// NormalizeAxis(dim int) (Tensor[T], error)
}

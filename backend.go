package spark

// BackendStorage defines operations for tensor storage management.
type BackendStorage[T D] interface {
	// Clone creates a deep copy of the storage.
	Clone() (BackendStorage[T], error)

	// Data returns a copy of the underlying data.
	Data() []T

	// Device returns the associated device.
	Device() BackendDevice[T]

	// DType returns the data type of the storage.
	DType() DType

	// Affine applies an affine transformation (scale * x + bias) to the storage.
	Affine(layout *Layout, scale, bias T) (BackendStorage[T], error)

	// Add performs element-wise addition between this and another storage.
	Add(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// Sub performs element-wise subtraction between this and another storage.
	Sub(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// Mul performs element-wise multiplication between this and another storage.
	Mul(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// Div performs element-wise division between this and another storage.
	Div(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// Maximum performs element-wise maximum of two tensors.
	Maximum(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// Minimum performs element-wise minimum of two tensors.
	Minimum(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// Eq performs element-wise equality comparison of two tensors.
	Eq(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// Ne performs element-wise not-equal comparison of two tensors.
	Ne(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// Lt performs element-wise less-than comparison of two tensors.
	Lt(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// Le performs element-wise less-than-or-equal comparison of two tensors.
	Le(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// Gt performs element-wise greater-than comparison of two tensors.
	Gt(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// Ge performs element-wise greater-than-or-equal comparison of two tensors.
	Ge(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[T], error)

	// EqU8 performs element-wise equality comparison of two tensors.
	EqU8(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[uint8], error)

	// NeU8 performs element-wise not-equal comparison of two tensors.
	NeU8(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[uint8], error)

	// LtU8 performs element-wise less-than comparison of two tensors.
	LtU8(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[uint8], error)

	// LeU8 performs element-wise less-than-or-equal comparison of two tensors.
	LeU8(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[uint8], error)

	// GtU8 performs element-wise greater-than comparison of two tensors.
	GtU8(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[uint8], error)

	// GeU8 performs element-wise greater-than-or-equal comparison of two tensors.
	GeU8(rhs BackendStorage[T], lhsLayout, rhsLayout, resLayout *Layout) (BackendStorage[uint8], error)

	// ToDtype performs type conversion to the specified target type.
	ToDtype(layout *Layout, dtype DType) (any, error)

	// MatMul performs matrix multiplication: C = A * B
	MatMul(lhsLayout *Layout, rhs BackendStorage[T], rhsLayout *Layout, b, m, n, k int) (BackendStorage[T], error)

	// Conv1d performs 1D convolution using im2col + BLAS for supported types.
	Conv1d(layout *Layout, kernel BackendStorage[T], kernelLayout *Layout, params *Conv1DParams) (BackendStorage[T], error)

	// ConvTranspose1d performs 1D transposed convolution (deconvolution) for supported types.
	ConvTranspose1d(layout *Layout, kernel BackendStorage[T], kernelLayout *Layout, params *ConvT1DParams) (BackendStorage[T], error)

	// Conv2d performs 2D convolution using im2col + BLAS for supported types.
	Conv2d(layout *Layout, kernel BackendStorage[T], kernelLayout *Layout, params *Conv2DParams) (BackendStorage[T], error)

	// ConvTranspose2d performs 2D transposed convolution (deconvolution) for supported types.
	ConvTranspose2d(layout *Layout, kernel BackendStorage[T], kernelLayout *Layout, params *ConvT2DParams) (BackendStorage[T], error)

	// AvgPool2d performs 2D average pooling for supported types.
	AvgPool2d(layout *Layout, kH, kW, sH, sW int) (BackendStorage[T], error)

	// MaxPool2d performs 2D max pooling for supported types.
	MaxPool2d(layout *Layout, kH, kW, sH, sW int) (BackendStorage[T], error)

	// UpsampleNearest2d performs 2D nearest neighbor upsampling for supported types.
	UpsampleNearest2d(layout *Layout, targetH, targetW int) (BackendStorage[T], error)

	// ConstSet sets all elements to a constant value for supported types.
	ConstSet(layout *Layout, val T) error

	// Copy2d copies a 2D region from source to destination for supported types.
	Copy2d(dst BackendStorage[T], d1, d2, srcStride1, dstStride1, srcOffset, dstOffset int) error

	// FastSum computes the sum over the last dimension.
	FastSum(layout *Layout) (BackendStorage[T], error)

	// FastMin computes the minimum over the last dimension.
	FastMin(layout *Layout) (BackendStorage[T], error)

	// FastMax computes the maximum over the last dimension.
	FastMax(layout *Layout) (BackendStorage[T], error)

	// FastArgmin computes the indices of minimum values over the last dimension.
	FastArgmin(layout *Layout) (BackendStorage[uint32], error)

	// FastArgmax computes the indices of maximum values over the last dimension.
	FastArgmax(layout *Layout) (BackendStorage[uint32], error)

	// Sum performs summation along specified dimensions.
	Sum(layout *Layout, dims []int) (BackendStorage[T], error)

	// Min computes the minimum over the specified dimension.
	Min(layout *Layout, dim int) (BackendStorage[T], error)

	// Max computes the maximum over the specified dimension.
	Max(layout *Layout, dim int) (BackendStorage[T], error)

	// Argmin computes the index of minimum over the specified dimension.
	Argmin(layout *Layout, dim int) (BackendStorage[uint32], error)

	// Argmax computes the index of maximum over the specified dimension.
	Argmax(layout *Layout, dim int) (BackendStorage[uint32], error)

	// FastFastSoftmax performs softmax along the last dimension.
	FastSoftmax(layout *Layout) (BackendStorage[T], error)

	// FastRmsNorm performs RMS normalization along the last dimension.
	FastRmsNorm(layout *Layout, alpha BackendStorage[T], alphaLayout *Layout, eps T) (BackendStorage[T], error)

	// FastLayerNorm performs Layer normalization along the last dimension.
	FastLayerNorm(layout *Layout, alpha BackendStorage[T], alphaLayout *Layout, beta BackendStorage[T], betaLayout *Layout, eps T) (BackendStorage[T], error)

	// RopeI performs rotary position embedding (rope_i variant).
	RopeI(layout *Layout, cos BackendStorage[T], cosLayout *Layout, sin BackendStorage[T], sinLayout *Layout) (BackendStorage[T], error)

	// Rope performs rotary position embedding (rope variant).
	Rope(layout *Layout, cos BackendStorage[T], cosLayout *Layout, sin BackendStorage[T], sinLayout *Layout) (BackendStorage[T], error)

	// RopeThd performs rotary position embedding (rope_thd variant).
	RopeThd(layout *Layout, cos BackendStorage[T], cosLayout *Layout, sin BackendStorage[T], sinLayout *Layout) (BackendStorage[T], error)

	// WhereCond performs element-wise selection based on condition.
	WhereCond(condLayout *Layout, t BackendStorage[T], tLayout *Layout, f BackendStorage[T], fLayout *Layout) (BackendStorage[T], error)

	// Copy performs element-wise copy operation.
	Copy(layout *Layout, src BackendStorage[T]) (BackendStorage[T], error)

	// Neg performs element-wise negation operation.
	Neg(layout *Layout) (BackendStorage[T], error)

	// Recip performs element-wise reciprocal operation.
	Recip(layout *Layout) (BackendStorage[T], error)

	// Exp performs element-wise exponential operation.
	Exp(layout *Layout) (BackendStorage[T], error)

	// Log performs element-wise logarithm operation.
	Log(layout *Layout) (BackendStorage[T], error)

	// Sin performs element-wise sine operation.
	Sin(layout *Layout) (BackendStorage[T], error)

	// Cos performs element-wise cosine operation.
	Cos(layout *Layout) (BackendStorage[T], error)

	// Tanh performs element-wise hyperbolic tangent operation.
	Tanh(layout *Layout) (BackendStorage[T], error)

	// Erf performs element-wise error function operation.
	Erf(layout *Layout) (BackendStorage[T], error)

	// Ceil performs element-wise ceiling operation.
	Ceil(layout *Layout) (BackendStorage[T], error)

	// Floor performs element-wise floor operation.
	Floor(layout *Layout) (BackendStorage[T], error)

	// Round performs element-wise round operation.
	Round(layout *Layout) (BackendStorage[T], error)

	// Normcdf performs element-wise normal CDF operation.
	Normcdf(layout *Layout) (BackendStorage[T], error)

	// Abs performs element-wise absolute value operation.
	Abs(layout *Layout) (BackendStorage[T], error)

	// Sqr performs element-wise square operation.
	Sqr(layout *Layout) (BackendStorage[T], error)

	// Sqrt performs element-wise square root operation.
	Sqrt(layout *Layout) (BackendStorage[T], error)

	// Gelu performs element-wise GELU activation operation.
	Gelu(layout *Layout) (BackendStorage[T], error)

	// GeluErf performs element-wise GELU (ERF-based) activation operation.
	GeluErf(layout *Layout) (BackendStorage[T], error)

	// Relu performs element-wise ReLU activation operation.
	Relu(layout *Layout) (BackendStorage[T], error)

	// Elu performs element-wise ELU activation operation with parameter alpha.
	Elu(layout *Layout, alpha T) (BackendStorage[T], error)

	// Silu performs element-wise SiLU (Swish) activation operation.
	Silu(layout *Layout) (BackendStorage[T], error)

	// Powf performs element-wise power operation with parameter param.
	Powf(layout *Layout, param T) (BackendStorage[T], error)

	// Sign performs element-wise sign operation.
	Sign(layout *Layout) (BackendStorage[T], error)

	// Sigmoid performs element-wise sigmoid activation operation.
	Sigmoid(layout *Layout) (BackendStorage[T], error)
}

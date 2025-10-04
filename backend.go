package spark

// // BackendStorage defines operations for tensor storage management.
type BackendStorage[T D] interface {

	// TryClone creates a deep copy of the storage with the given layout.
	TryClone() (BackendStorage[T], error)

	// DType returns the data type of the storage.
	DType() DType

	// Device returns the associated device.
	Device() BackendDevice[T]

	// Affine applies an affine transformation (scale * x + bias) to the storage.
	Affine(*Layout, T, T) (BackendStorage[T], error)

	// Add performs element-wise addition between this and another storage.
	Add(BackendStorage[T], *Layout, *Layout, *Layout) (BackendStorage[T], error)

	// Mul performs element-wise multiplication between this and another storage.
	Mul(BackendStorage[T], *Layout, *Layout, *Layout) (BackendStorage[T], error)

	// // Powf raises each element to the given power.
	// Powf(*Layout, T) (BackendStorage[T], error)

	// // Elu applies the ELU activation function with the given alpha.
	// Elu(*Layout, T) (BackendStorage[T], error)

	// // ReduceOp applies a reduction operation (e.g., sum, max) along specified dimensions.
	// ReduceOp(ReduceOp, *Layout, []int) (BackendStorage[T], error)

	// // Cmp performs a comparison operation between this and another storage.
	// Cmp(CmpOp, BackendStorage[T], *Layout, *Layout) (BackendStorage[T], error)

	// // ToDType converts the storage to the specified data type.
	// ToDType(*Layout, DType) (BackendStorage[T], error)

	// // UnaryImpl applies a unary operation to the storage.
	// UnaryImpl(UnaryOp, *Layout) (BackendStorage[T], error)

	// // BinaryImpl applies a binary operation between this and another storage.
	// BinaryImpl(BinaryOp, BackendStorage[T], *Layout, *Layout) (BackendStorage[T], error)

	// WhereCond applies a conditional operation: if cond then true_value else false_value.
	// 	WhereCond(condLayout *Layout, trueValue BackendStorage[T], trueLayout *Layout, falseValue BackendStorage[T], falseLayout *Layout) (BackendStorage[T], error)

	// 	// Conv1D performs a 1D convolution with the given kernel and parameters.
	// 	Conv1D(layout *Layout, kernel BackendStorage[T], kernelLayout *Layout, params *ParamsConv1D) (BackendStorage[T], error)

	// 	// ConvTranspose1D performs a 1D transposed convolution with the given kernel and parameters.
	// 	ConvTranspose1D(layout *Layout, kernel BackendStorage[T], kernelLayout *Layout, params *ParamsConvTranspose1D) (BackendStorage[T], error)

	// 	// Conv2D performs a 2D convolution with the given kernel and parameters.
	// 	Conv2D(layout *Layout, kernel BackendStorage[T], kernelLayout *Layout, params *ParamsConv2D) (BackendStorage[T], error)

	// 	// ConvTranspose2D performs a 2D transposed convolution with the given kernel and parameters.
	// 	ConvTranspose2D(layout *Layout, kernel BackendStorage[T], kernelLayout *Layout, params *ParamsConvTranspose2D) (BackendStorage[T], error)

	// 	// AvgPool2D applies 2D average pooling with the given kernel size and strides.
	// 	AvgPool2D(layout *Layout, kernelSize, strides [2]int) (BackendStorage[T], error)

	// 	// MaxPool2D applies 2D max pooling with the given kernel size and strides.
	// 	MaxPool2D(layout *Layout, kernelSize, strides [2]int) (BackendStorage[T], error)

	// 	// UpsampleNearest1D performs 1D nearest-neighbor upsampling to the specified output size.
	// 	UpsampleNearest1D(layout *Layout, outSize int) (BackendStorage[T], error)

	// 	// UpsampleNearest2D performs 2D nearest-neighbor upsampling to the specified output size.
	// 	UpsampleNearest2D(layout *Layout, outH, outW int) (BackendStorage[T], error)

	// 	// Gather collects elements along the specified dimension using indices.
	// 	Gather(layout *Layout, indices BackendStorage[T], indicesLayout *Layout, dim int) (BackendStorage[T], error)

	// 	// ScatterSet scatters values into this storage at the specified indices along the given dimension.
	// 	ScatterSet(layout *Layout, indices BackendStorage[T], indicesLayout *Layout, values BackendStorage[T], valuesLayout *Layout, dim int) error

	// 	// ScatterAddSet adds values into this storage at the specified indices along the given dimension.
	// 	ScatterAddSet(layout *Layout, indices BackendStorage[T], indicesLayout *Layout, values BackendStorage[T], valuesLayout *Layout, dim int) error

	// 	// IndexSelect selects elements along the specified dimension using indices.
	// 	IndexSelect(indices BackendStorage[T], indicesLayout, layout *Layout, dim int) (BackendStorage[T], error)

	// 	// IndexAdd adds values into this storage at the specified indices along the given dimension.
	// 	IndexAdd(layout *Layout, indices BackendStorage[T], indicesLayout *Layout, values BackendStorage[T], valuesLayout *Layout, dim int) (BackendStorage[T], error)

	// 	// Matmul performs matrix multiplication with the given right-hand side and dimensions.
	// 	Matmul(rhs BackendStorage[T], dims [4]int, lhsLayout, rhsLayout *Layout) (BackendStorage[T], error)

	// 	// CopyStridedSrc copies data from this storage to a destination with strided source layout.
	// 	CopyStridedSrc(dst BackendStorage[T], dstOffset int, srcLayout *Layout) error

	// 	// Copy2D copies a 2D region from this storage to a destination with specified strides and offsets.
	// 	Copy2D(dst BackendStorage[T], d1, d2, srcStride1, dstStride1, srcOffset, dstOffset int) error

	// // ConstSet sets all elements in the storage to the given scalar value.
	// ConstSet(scalar interface{}, layout *Layout) error
}

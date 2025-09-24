package spark

// DeviceLocation represents the location of a device (e.g., CPU, GPU, device ID).
type DeviceLocation interface {
	String() string
}

// // BackendStorage defines operations for tensor storage management.
// type BackendStorage[T D] interface {
// 	// Device returns the associated device.
// 	Device() BackendDevice[T]

// 	// DType returns the data type of the storage.
// 	DType() DType

// 	// TryClone creates a deep copy of the storage with the given layout.
// 	TryClone(layout *Layout) (BackendStorage[T], error)

// 	// ToCpuStorage converts the storage to CPU-based storage.
// 	ToCpuStorage() (CpuStorage, error)

// 	// Affine applies an affine transformation (scale * x + bias) to the storage.
// 	Affine(layout *Layout, scale, bias float64) (BackendStorage[T], error)

// 	// Powf raises each element to the given power.
// 	Powf(layout *Layout, power float64) (BackendStorage[T], error)

// 	// Elu applies the ELU activation function with the given alpha.
// 	Elu(layout *Layout, alpha float64) (BackendStorage[T], error)

// 	// ReduceOp applies a reduction operation (e.g., sum, max) along specified dimensions.
// 	ReduceOp(op ReduceOp, layout *Layout, dims []int) (BackendStorage[T], error)

// 	// Cmp performs a comparison operation between this and another storage.
// 	Cmp(op CmpOp, rhs BackendStorage[T], lhsLayout, rhsLayout *Layout) (BackendStorage[T], error)

// 	// ToDType converts the storage to the specified data type.
// 	ToDType(layout *Layout, dtype DType) (BackendStorage[T], error)

// 	// UnaryImpl applies a unary operation to the storage.
// 	UnaryImpl(op UnaryOpT, layout *Layout) (BackendStorage[T], error)

// 	// BinaryImpl applies a binary operation between this and another storage.
// 	BinaryImpl(op BinaryOpT, rhs BackendStorage[T], lhsLayout, rhsLayout *Layout) (BackendStorage[T], error)

// 	// WhereCond applies a conditional operation: if cond then true_value else false_value.
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

// 	// ConstSet sets all elements in the storage to the given scalar value.
// 	ConstSet(scalar interface{}, layout *Layout) error
// }

// BackendDevice defines operations for device management.
// type BackendDevice[T D] interface {
// 	// Location returns the device location (e.g., CPU, GPU, device ID).
// 	Location() DeviceLocation

// 	// SameDevice checks if two devices are the same.
// 	SameDevice(other BackendDevice[T]) bool

// 	// NewStorage creates a new storage with zeros for the given shape and data type.
// 	NewStorage(shape *Shape, dtype DType) (BackendStorage[T], error)

// 	// AllocUninit allocates uninitialized storage for the given shape and data type.
// 	// The caller must ensure the storage is initialized after allocation.
// 	AllocUninit(shape *Shape, dtype DType) (BackendStorage[T], error)

// 	// StorageFromSlice creates storage from a slice of values.
// 	StorageFromSlice(data interface{}) (BackendStorage[T], error)

// 	// StorageFromCpuStorage creates storage from CPU-based storage.
// 	StorageFromCpuStorage(cpuStorage CpuStorage) (BackendStorage[T], error)

// 	// StorageFromCpuStorageOwned creates storage by taking ownership of CPU-based storage.
// 	StorageFromCpuStorageOwned(cpuStorage CpuStorage) (BackendStorage[T], error)

// 	// RandUniform creates storage with random values from a uniform distribution.
// 	RandUniform(shape *Shape, dtype DType, low, high float64) (BackendStorage[T], error)

// 	// RandNormal creates storage with random values from a normal distribution.
// 	RandNormal(shape *Shape, dtype DType, mean, std float64) (BackendStorage[T], error)

// 	// SetSeed sets the random seed for the device.
// 	SetSeed(seed uint64) error

// 	// Synchronize blocks until all operations on the device are complete.
// 	Synchronize() error
// }

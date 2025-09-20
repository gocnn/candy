package spark

// CpuStorage represents CPU-based storage for tensor data.
// Implementations should define the actual storage structure.
type CpuStorage interface {
	// Define methods as needed for CPU storage access.
}

// DeviceLocation represents the location of a device (e.g., CPU, GPU, device ID).
type DeviceLocation interface {
	String() string
}

// BackendStorage defines operations for tensor storage management.
type BackendStorage interface {
	// Device returns the associated device.
	Device() BackendDevice

	// DType returns the data type of the storage.
	DType() DType

	// TryClone creates a deep copy of the storage with the given layout.
	TryClone(layout *Layout) (BackendStorage, error)

	// ToCpuStorage converts the storage to CPU-based storage.
	ToCpuStorage() (CpuStorage, error)

	// Affine applies an affine transformation (scale * x + bias) to the storage.
	Affine(layout *Layout, scale, bias float64) (BackendStorage, error)

	// Powf raises each element to the given power.
	Powf(layout *Layout, power float64) (BackendStorage, error)

	// Elu applies the ELU activation function with the given alpha.
	Elu(layout *Layout, alpha float64) (BackendStorage, error)

	// ReduceOp applies a reduction operation (e.g., sum, max) along specified dimensions.
	ReduceOp(op ReduceOp, layout *Layout, dims []int) (BackendStorage, error)

	// Cmp performs a comparison operation between this and another storage.
	Cmp(op CmpOp, rhs BackendStorage, lhsLayout, rhsLayout *Layout) (BackendStorage, error)

	// ToDType converts the storage to the specified data type.
	ToDType(layout *Layout, dtype DType) (BackendStorage, error)

	// UnaryImpl applies a unary operation to the storage.
	UnaryImpl(op UnaryOpT, layout *Layout) (BackendStorage, error)

	// BinaryImpl applies a binary operation between this and another storage.
	BinaryImpl(op BinaryOpT, rhs BackendStorage, lhsLayout, rhsLayout *Layout) (BackendStorage, error)

	// WhereCond applies a conditional operation: if cond then true_value else false_value.
	WhereCond(condLayout *Layout, trueValue BackendStorage, trueLayout *Layout, falseValue BackendStorage, falseLayout *Layout) (BackendStorage, error)

	// Conv1D performs a 1D convolution with the given kernel and parameters.
	Conv1D(layout *Layout, kernel BackendStorage, kernelLayout *Layout, params *ParamsConv1D) (BackendStorage, error)

	// ConvTranspose1D performs a 1D transposed convolution with the given kernel and parameters.
	ConvTranspose1D(layout *Layout, kernel BackendStorage, kernelLayout *Layout, params *ParamsConvTranspose1D) (BackendStorage, error)

	// Conv2D performs a 2D convolution with the given kernel and parameters.
	Conv2D(layout *Layout, kernel BackendStorage, kernelLayout *Layout, params *ParamsConv2D) (BackendStorage, error)

	// ConvTranspose2D performs a 2D transposed convolution with the given kernel and parameters.
	ConvTranspose2D(layout *Layout, kernel BackendStorage, kernelLayout *Layout, params *ParamsConvTranspose2D) (BackendStorage, error)

	// AvgPool2D applies 2D average pooling with the given kernel size and strides.
	AvgPool2D(layout *Layout, kernelSize, strides [2]int) (BackendStorage, error)

	// MaxPool2D applies 2D max pooling with the given kernel size and strides.
	MaxPool2D(layout *Layout, kernelSize, strides [2]int) (BackendStorage, error)

	// UpsampleNearest1D performs 1D nearest-neighbor upsampling to the specified output size.
	UpsampleNearest1D(layout *Layout, outSize int) (BackendStorage, error)

	// UpsampleNearest2D performs 2D nearest-neighbor upsampling to the specified output size.
	UpsampleNearest2D(layout *Layout, outH, outW int) (BackendStorage, error)

	// Gather collects elements along the specified dimension using indices.
	Gather(layout *Layout, indices BackendStorage, indicesLayout *Layout, dim int) (BackendStorage, error)

	// ScatterSet scatters values into this storage at the specified indices along the given dimension.
	ScatterSet(layout *Layout, indices BackendStorage, indicesLayout *Layout, values BackendStorage, valuesLayout *Layout, dim int) error

	// ScatterAddSet adds values into this storage at the specified indices along the given dimension.
	ScatterAddSet(layout *Layout, indices BackendStorage, indicesLayout *Layout, values BackendStorage, valuesLayout *Layout, dim int) error

	// IndexSelect selects elements along the specified dimension using indices.
	IndexSelect(indices BackendStorage, indicesLayout, layout *Layout, dim int) (BackendStorage, error)

	// IndexAdd adds values into this storage at the specified indices along the given dimension.
	IndexAdd(layout *Layout, indices BackendStorage, indicesLayout *Layout, values BackendStorage, valuesLayout *Layout, dim int) (BackendStorage, error)

	// Matmul performs matrix multiplication with the given right-hand side and dimensions.
	Matmul(rhs BackendStorage, dims [4]int, lhsLayout, rhsLayout *Layout) (BackendStorage, error)

	// CopyStridedSrc copies data from this storage to a destination with strided source layout.
	CopyStridedSrc(dst BackendStorage, dstOffset int, srcLayout *Layout) error

	// Copy2D copies a 2D region from this storage to a destination with specified strides and offsets.
	Copy2D(dst BackendStorage, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset int) error

	// ConstSet sets all elements in the storage to the given scalar value.
	ConstSet(scalar interface{}, layout *Layout) error
}

// BackendDevice defines operations for device management.
type BackendDevice interface {
	// Location returns the device location (e.g., CPU, GPU, device ID).
	Location() DeviceLocation

	// SameDevice checks if two devices are the same.
	SameDevice(other BackendDevice) bool

	// NewStorage creates a new storage with zeros for the given shape and data type.
	NewStorage(shape *Shape, dtype DType) (BackendStorage, error)

	// AllocUninit allocates uninitialized storage for the given shape and data type.
	// The caller must ensure the storage is initialized after allocation.
	AllocUninit(shape *Shape, dtype DType) (BackendStorage, error)

	// StorageFromSlice creates storage from a slice of values.
	StorageFromSlice(data interface{}) (BackendStorage, error)

	// StorageFromCpuStorage creates storage from CPU-based storage.
	StorageFromCpuStorage(cpuStorage CpuStorage) (BackendStorage, error)

	// StorageFromCpuStorageOwned creates storage by taking ownership of CPU-based storage.
	StorageFromCpuStorageOwned(cpuStorage CpuStorage) (BackendStorage, error)

	// RandUniform creates storage with random values from a uniform distribution.
	RandUniform(shape *Shape, dtype DType, low, high float64) (BackendStorage, error)

	// RandNormal creates storage with random values from a normal distribution.
	RandNormal(shape *Shape, dtype DType, mean, std float64) (BackendStorage, error)

	// SetSeed sets the random seed for the device.
	SetSeed(seed uint64) error

	// Synchronize blocks until all operations on the device are complete.
	Synchronize() error
}

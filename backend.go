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

	// Sub performs element-wise subtraction between this and another storage.
	Sub(BackendStorage[T], *Layout, *Layout, *Layout) (BackendStorage[T], error)

	// Mul performs element-wise multiplication between this and another storage.
	Mul(BackendStorage[T], *Layout, *Layout, *Layout) (BackendStorage[T], error)

	// Div performs element-wise division between this and another storage.
	Div(BackendStorage[T], *Layout, *Layout, *Layout) (BackendStorage[T], error)

	// Sqrt performs element-wise square root.
	Sqrt(*Layout) (BackendStorage[T], error)
}

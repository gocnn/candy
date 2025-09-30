package spark

type Device int

const (
	CPU Device = iota
	CUDA
	Metal
)

// String returns the string representation of the device.
func (d Device) String() string {
	switch d {
	case CPU:
		return "cpu"
	case CUDA:
		return "cuda"
	case Metal:
		return "metal"
	default:
		return "unknown"
	}
}

// DeviceLocation represents the location of a device (e.g., CPU, GPU, device ID).
type DeviceLocation int

const (
	CpuLocation DeviceLocation = iota
	GpuLocation
	MetalLocation
)

// String implements fmt.Stringer for DeviceLocation.
func (dl DeviceLocation) String() string {
	switch dl {
	case CpuLocation:
		return "cpu"
	case GpuLocation:
		return "gpu"
	case MetalLocation:
		return "metal"
	default:
		return "unknown"
	}
}

// BackendDevice defines operations for device management.
type BackendDevice[T D] interface {
	// Location returns the device location (e.g., CPU, GPU, device ID).
	Location() DeviceLocation

	// SameDevice checks if two devices are the same.
	SameDevice(other BackendDevice[T]) bool

	// AllocUninit allocates uninitialized storage for the given shape and data type.
	// The caller must ensure the storage is initialized after allocation.
	AllocUninit(*Shape, DType) (BackendStorage[T], error)

	// StorageFromSlice creates storage from a slice of values.
	StorageFromSlice([]T) (BackendStorage[T], error)

	// // StorageFromCpuStorage creates storage from CPU-based storage.
	// StorageFromCpuStorage(cpu.CpuStorage[T]) (BackendStorage[T], error)

	// // StorageFromCpuStorageOwned creates storage by taking ownership of CPU-based storage.
	// StorageFromCpuStorageOwned(cpu.CpuStorage[T]) (BackendStorage[T], error)

	// RandUniform creates storage with random values from a uniform distribution.
	RandUniform(*Shape, DType, float64, float64) (BackendStorage[T], error)

	// RandNormal creates storage with random values from a normal distribution.
	RandNormal(*Shape, DType, float64, float64) (BackendStorage[T], error)

	// SetSeed sets the random seed for the device.
	SetSeed(uint64) error

	// Synchronize blocks until all operations on the device are complete.
	Synchronize() error
}

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

	// IsSame checks if two devices are the same.
	IsSame(BackendDevice[T]) bool

	// StorageFromSlice creates storage from a slice of values.
	StorageFromSlice([]T) (BackendStorage[T], error)

	// SetSeed sets the random seed for the device.
	SetSeed(uint64) error

	// RandUniform creates storage with random values from a uniform distribution.
	RandUniform(*Shape, DType, float64, float64) (BackendStorage[T], error)

	// RandNormal creates storage with random values from a normal distribution.
	RandNormal(*Shape, DType, float64, float64) (BackendStorage[T], error)

	// Alloc allocates a zero-initialized storage for the given shape.
	Alloc(*Shape, DType) (BackendStorage[T], error)

	// Zeros creates a storage filled with zeros.
	Zeros(*Shape, DType) (BackendStorage[T], error)

	// Ones creates a storage filled with ones.
	Ones(*Shape, DType) (BackendStorage[T], error)

	// Full creates a storage filled with a specific value.
	Full(*Shape, DType, T) (BackendStorage[T], error)

	// Synchronize blocks until all operations on the device are complete.
	Synchronize() error
}

package goml

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

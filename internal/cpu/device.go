package cpu

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/internal/cpu/kernels"
)

var _ candy.BackendDevice[float32] = (*CpuDevice[float32])(nil)
var _ candy.BackendDevice[float64] = (*CpuDevice[float64])(nil)
var _ candy.BackendDevice[uint8] = (*CpuDevice[uint8])(nil)
var _ candy.BackendDevice[uint32] = (*CpuDevice[uint32])(nil)
var _ candy.BackendDevice[int64] = (*CpuDevice[int64])(nil)

// CpuDevice is a CPU-based implementation of the BackendDevice interface.
type CpuDevice[T kernels.D] struct{}

// NewCpuDevice creates a new CPU device.
func NewCpuDevice[T kernels.D]() candy.BackendDevice[T] {
	return &CpuDevice[T]{}
}

// Location returns the CPU device location.
func (c *CpuDevice[T]) Location() candy.DeviceLocation {
	return candy.CpuLocation
}

// IsSame checks if another device is a CpuDevice.
func (c *CpuDevice[T]) IsSame(other candy.BackendDevice[T]) bool {
	_, ok := other.(*CpuDevice[T])
	return ok
}

// StorageFromSlice creates a CpuStorage from a slice of data.
func (c *CpuDevice[T]) StorageFromSlice(data []T) (candy.BackendStorage[T], error) {
	return &CpuStorage[T]{data: slices.Clone(data)}, nil
}

// StorageFromCpuStorage creates a copy of the given CpuStorage.
func (c *CpuDevice[T]) StorageFromCpuStorage(s *CpuStorage[T]) (candy.BackendStorage[T], error) {
	return s.Clone()
}

// SetSeed is unsupported for CPU RNG (no global seed in math/rand/v2).
func (c *CpuDevice[T]) SetSeed(seed uint64) error {
	return errors.New("CPU RNG seeding unsupported")
}

// RandUniform generates a storage with uniformly distributed random values.
func (c *CpuDevice[T]) RandUniform(shape *candy.Shape, dtype candy.DType, min, max float64) (candy.BackendStorage[T], error) {
	storage := New(make([]T, shape.Numel()))
	data := storage.data

	switch dtype {
	case candy.F32:
		for i := range data {
			data[i] = any(float32(min + rand.Float64()*(max-min))).(T)
		}
	case candy.F64:
		for i := range data {
			data[i] = any(min + rand.Float64()*(max-min)).(T)
		}
	case candy.U8:
		minU8, maxU8 := uint8(min), uint8(max)
		if min < 0 || max > math.MaxUint8 || max < min {
			return nil, errors.New("invalid range for uint8")
		}
		rangeSize := uint32(maxU8 - minU8 + 1)
		for i := range data {
			data[i] = any(minU8 + uint8(rand.Uint32N(rangeSize))).(T)
		}
	case candy.U32:
		minU32, maxU32 := uint32(min), uint32(max)
		if min < 0 || max > math.MaxUint32 || max < min {
			return nil, errors.New("invalid range for uint32")
		}
		rangeSize := maxU32 - minU32 + 1
		for i := range data {
			data[i] = any(minU32 + rand.Uint32N(rangeSize)).(T)
		}
	case candy.I64:
		minI64, maxI64 := int64(min), int64(max)
		if max < min {
			return nil, errors.New("invalid range for int64")
		}
		rangeSize := maxI64 - minI64 + 1
		for i := range data {
			data[i] = any(minI64 + rand.Int64N(rangeSize)).(T)
		}
	default:
		return nil, errors.New("unsupported dtype")
	}
	return storage, nil
}

// RandNormal generates a storage with normally distributed random values.
func (c *CpuDevice[T]) RandNormal(shape *candy.Shape, dtype candy.DType, mean, std float64) (candy.BackendStorage[T], error) {
	storage := New(make([]T, shape.Numel()))
	data := storage.data

	switch dtype {
	case candy.F32:
		for i := range data {
			val := mean + std*rand.NormFloat64()
			data[i] = any(float32(val)).(T)
		}
	case candy.F64:
		for i := range data {
			val := mean + std*rand.NormFloat64()
			data[i] = any(val).(T)
		}
	case candy.U8:
		for i := range data {
			val := mean + std*rand.NormFloat64()
			if val < 0 {
				val = 0 // Clamp for unsigned.
			} else if val > math.MaxUint8 {
				val = math.MaxUint8
			}
			data[i] = any(uint8(val)).(T)
		}
	case candy.U32:
		for i := range data {
			val := mean + std*rand.NormFloat64()
			if val < 0 {
				val = 0 // Clamp for unsigned.
			} else if val > math.MaxUint32 {
				val = math.MaxUint32
			}
			data[i] = any(uint32(val)).(T)
		}
	case candy.I64:
		for i := range data {
			val := mean + std*rand.NormFloat64()
			// Check for int64 overflow (rare but possible).
			if val < math.MinInt64 {
				val = math.MinInt64
			} else if val > math.MaxInt64 {
				val = math.MaxInt64
			}
			data[i] = any(int64(val)).(T)
		}
	default:
		return nil, errors.New("unsupported dtype")
	}
	return storage, nil
}

// Alloc allocates a zero-initialized storage for the given shape.
func (c *CpuDevice[T]) Alloc(shape *candy.Shape, dtype candy.DType) (candy.BackendStorage[T], error) {
	return New(make([]T, shape.Numel())), nil
}

// Zeros creates a storage filled with zeros.
func (c *CpuDevice[T]) Zeros(shape *candy.Shape, dtype candy.DType) (candy.BackendStorage[T], error) {
	return c.Alloc(shape, dtype)
}

// Ones creates a storage filled with ones.
func (c *CpuDevice[T]) Ones(shape *candy.Shape, dtype candy.DType) (candy.BackendStorage[T], error) {
	return c.Full(shape, dtype, 1.0)
}

// Full creates a storage filled with a specific value.
func (c *CpuDevice[T]) Full(shape *candy.Shape, dtype candy.DType, value float64) (candy.BackendStorage[T], error) {
	storage := New(make([]T, shape.Numel()))
	data := storage.data

	for i := range data {
		switch any(data[i]).(type) {
		case float32:
			data[i] = any(float32(value)).(T)
		case float64:
			data[i] = any(value).(T)
		default:
			data[i] = any(value).(T)
		}
	}

	return storage, nil
}

// Synchronize is a no-op for CPU operations.
func (c *CpuDevice[T]) Synchronize() error {
	return nil
}

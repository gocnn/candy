package cpu

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/internal/cpu/kernels"
)

var _ spark.BackendDevice[float32] = (*CpuDevice[float32])(nil)
var _ spark.BackendDevice[float64] = (*CpuDevice[float64])(nil)

// CpuDevice is a CPU-based implementation of the BackendDevice interface.
type CpuDevice[T kernels.D] struct{}

// NewCpuDevice creates a new CPU device.
func NewCpuDevice[T kernels.D]() spark.BackendDevice[T] {
	return &CpuDevice[T]{}
}

// Location returns the CPU device location.
func (c *CpuDevice[T]) Location() spark.DeviceLocation {
	return spark.CpuLocation
}

// IsSame checks if another device is a CpuDevice.
func (c *CpuDevice[T]) IsSame(other spark.BackendDevice[T]) bool {
	_, ok := other.(*CpuDevice[T])
	return ok
}

// StorageFromSlice creates a CpuStorage from a slice of data.
func (c *CpuDevice[T]) StorageFromSlice(data []T) (spark.BackendStorage[T], error) {
	return &CpuStorage[T]{data: slices.Clone(data)}, nil
}

// StorageFromCpuStorage creates a copy of the given CpuStorage.
func (c *CpuDevice[T]) StorageFromCpuStorage(s *CpuStorage[T]) (spark.BackendStorage[T], error) {
	return s.TryClone()
}

// SetSeed is unsupported for CPU RNG.
func (c *CpuDevice[T]) SetSeed(seed uint64) error {
	return errors.New("CPU RNG seeding unsupported")
}

// RandUniform generates a storage with uniformly distributed random values.
func (c *CpuDevice[T]) RandUniform(shape *spark.Shape, dtype spark.DType, min, max float64) (spark.BackendStorage[T], error) {
	storage := New(make([]T, shape.ElemCount()))
	data := storage.data

	switch dtype {
	case spark.F32:
		for i := range data {
			data[i] = any(float32(min + rand.Float64()*(max-min))).(T)
		}
	case spark.F64:
		for i := range data {
			data[i] = any(min + rand.Float64()*(max-min)).(T)
		}
	}
	return storage, nil
}

// RandNormal generates a storage with normally distributed random values.
func (c *CpuDevice[T]) RandNormal(shape *spark.Shape, dtype spark.DType, mean, std float64) (spark.BackendStorage[T], error) {
	storage := New(make([]T, shape.ElemCount()))
	data := storage.data

	switch dtype {
	case spark.F32:
		for i := 0; i < len(data); i += 2 {
			u1, u2 := rand.Float64(), rand.Float64()
			z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
			data[i] = any(float32(mean + std*z0)).(T)
			if i+1 < len(data) {
				z1 := math.Sqrt(-2*math.Log(u1)) * math.Sin(2*math.Pi*u2)
				data[i+1] = any(float32(mean + std*z1)).(T)
			}
		}
	case spark.F64:
		for i := 0; i < len(data); i += 2 {
			u1, u2 := rand.Float64(), rand.Float64()
			z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
			data[i] = any(mean + std*z0).(T)
			if i+1 < len(data) {
				z1 := math.Sqrt(-2*math.Log(u1)) * math.Sin(2*math.Pi*u2)
				data[i+1] = any(mean + std*z1).(T)
			}
		}
	}
	return storage, nil
}

// Alloc allocates a zero-initialized storage for the given shape.
func (c *CpuDevice[T]) Alloc(shape *spark.Shape, dtype spark.DType) (spark.BackendStorage[T], error) {
	return New(make([]T, shape.ElemCount())), nil
}

// Zeros creates a storage filled with zeros.
func (c *CpuDevice[T]) Zeros(shape *spark.Shape, dtype spark.DType) (spark.BackendStorage[T], error) {
	return c.Alloc(shape, dtype)
}

// Ones creates a storage filled with ones.
func (c *CpuDevice[T]) Ones(shape *spark.Shape, dtype spark.DType) (spark.BackendStorage[T], error) {
	storage := New(make([]T, shape.ElemCount()))
	data := storage.data
	switch dtype {
	case spark.F32:
		for i := range data {
			data[i] = any(float32(1.0)).(T)
		}
	case spark.F64:
		for i := range data {
			data[i] = any(1.0).(T)
		}
	}
	return storage, nil
}

// Full creates a storage filled with a specific value.
func (c *CpuDevice[T]) Full(shape *spark.Shape, dtype spark.DType, value float64) (spark.BackendStorage[T], error) {
	storage := New(make([]T, shape.ElemCount()))
	data := storage.data
	switch dtype {
	case spark.F32:
		for i := range data {
			data[i] = any(float32(value)).(T)
		}
	case spark.F64:
		for i := range data {
			data[i] = any(value).(T)
		}
	}
	return storage, nil
}

// Synchronize is a no-op for CPU operations.
func (c *CpuDevice[T]) Synchronize() error {
	return nil
}

package cpu

import (
	"github.com/gocnn/spark"
	"github.com/gocnn/spark/internal/cpu/kernels"
)

// CpuDevice is a CPU-based implementation of the BackendDevice interface.
type CpuDevice[T kernels.D] struct{}

func NewCpuDevice[T kernels.D]() spark.BackendDevice[T] {
	return &CpuDevice[T]{}
}

// SameDevice checks if another device is also a CpuDevice.
func (c *CpuDevice[T]) SameDevice() bool {
	return true
}

// // StorageFromSlice creates a CpuStorage from a slice of float32 or float64.
// func (c *CpuDevice[T]) StorageFromSlice(data []T) CpuStorage[T] {
// 	return CpuStorage[T]{data: slices.Clone(data)}
// }

// // StorageFromCpuStorage creates a copy of the given CpuStorage.
// func (c *CpuDevice[T]) StorageFromCpuStorage(s CpuStorage[T]) CpuStorage[T] {
// 	return s.TryClone()
// }

// // StorageFromCpuStorageOwned takes ownership of the given CpuStorage without copying.
// func (c *CpuDevice[T]) StorageFromCpuStorageOwned(s CpuStorage[T]) CpuStorage[T] {
// 	return s
// }

// // SetSeed returns an error as seeding is not supported for CPU RNG.
// func (c *CpuDevice[T]) SetSeed(seed uint64) error {
// 	return errors.New("cannot seed the CPU RNG with SetSeed")
// }

// // AllocUninit allocates an uninitialized CpuStorage for the given shape.
// func (c *CpuDevice[T]) AllocUninit(shape *spark.Shape, dtype spark.DType) (CpuStorage[T], error) {
// 	return CpuStorage[T]{data: make([]T, shape.ElemCount())}, nil
// }

// // Zeros creates a CpuStorage filled with zeros.
// func (c *CpuDevice[T]) Zeros(shape *spark.Shape, dtype spark.DType) (CpuStorage[T], error) {
// 	return c.AllocUninit(shape, dtype) // make zeros slices in Go.
// }

// // Synchronize is a no-op for CPU, as CPU operations are synchronous.
// func (c *CpuDevice[T]) Synchronize() error {
// 	return nil
// }

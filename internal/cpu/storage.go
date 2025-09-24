package cpu

import "github.com/qntx/spark"

type CpuStorage[T spark.D] struct {
	data   []T
	dtype  spark.DType
	device *CpuDevice
}

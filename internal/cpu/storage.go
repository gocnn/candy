package cpu

import "github.com/gocnn/spark"

type CpuStorage[T spark.D] struct {
	data   []T
	dtype  spark.DType
	device *CpuDevice
}

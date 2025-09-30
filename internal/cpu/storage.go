package cpu

import (
	"slices"

	"github.com/gocnn/spark"
)

type CpuStorage[T spark.D] struct {
	data []T
}

func (s *CpuStorage[T]) Clone() CpuStorage[T] {
	return CpuStorage[T]{data: slices.Clone(s.data)}
}

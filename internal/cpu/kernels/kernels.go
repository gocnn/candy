package kernels

import "github.com/gocnn/spark"

type D interface {
	spark.D

	~float32 | ~float64 | ~uint8 | ~uint32 | ~int64
}

type I interface {
	~int | ~int32 | ~int64 | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

// Helper functions (should be moved to kernels.go)
func IsContiguous(ndims int, dims, strides []int) bool {
	if ndims == 0 {
		return true
	}

	expectedStride := 1
	for i := ndims - 1; i >= 0; i-- {
		if strides[i] != expectedStride {
			return false
		}
		expectedStride *= dims[i]
	}
	return true
}

func GetStridedIndex(linearIdx int, ndims int, dims, strides []int) int {
	stridedIdx := 0
	for i := ndims - 1; i >= 0; i-- {
		dimIdx := linearIdx % dims[i]
		linearIdx /= dims[i]
		stridedIdx += dimIdx * strides[i]
	}
	return stridedIdx
}

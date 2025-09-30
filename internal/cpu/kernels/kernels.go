package kernels

// Helper functions (should be moved to kernels.go)
func IsContiguous(numDims int, dims, strides []int) bool {
	if numDims == 0 {
		return true
	}

	expectedStride := 1
	for i := numDims - 1; i >= 0; i-- {
		if strides[i] != expectedStride {
			return false
		}
		expectedStride *= dims[i]
	}
	return true
}

func GetStridedIndex(linearIdx int, numDims int, dims, strides []int) int {
	stridedIdx := 0
	for i := numDims - 1; i >= 0; i-- {
		dimIdx := linearIdx % dims[i]
		linearIdx /= dims[i]
		stridedIdx += dimIdx * strides[i]
	}
	return stridedIdx
}

package kernels

// Fill fills the destination array with a constant value
func Fill[T D](numel int, val T, dst []T) {
	for i := range numel {
		dst[i] = val
	}
}

// FillF32 fills the destination array with a constant float32 value
func FillF32(numel int, val float32, dst []float32) {
	for i := range numel {
		dst[i] = val
	}
}

// FillF64 fills the destination array with a constant float64 value
func FillF64(numel int, val float64, dst []float64) {
	for i := range numel {
		dst[i] = val
	}
}

// FillStrided fills the destination array with a constant value for strided memory
func FillStrided[T D](numel, numDims int, dims, strides []int, val T, dst []T) {
	if IsContiguous(numDims, dims, strides) {
		Fill(numel, val, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, strides)] = val
	}
}

// FillStridedF32 fills the destination array with a constant float32 value for strided memory
func FillStridedF32(numel, numDims int, dims, strides []int, val float32, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
		FillF32(numel, val, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, strides)] = val
	}
}

// FillStridedF64 fills the destination array with a constant float64 value for strided memory
func FillStridedF64(numel, numDims int, dims, strides []int, val float64, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
		FillF64(numel, val, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, strides)] = val
	}
}

// Copy2d copies a 2D region from src to dst
func Copy2d[T D](rows, cols, lda, ldc int, src, dst []T) {
	for i := range rows {
		for j := range cols {
			dst[i*ldc+j] = src[i*lda+j]
		}
	}
}

// Copy2dF32 copies a 2D region from src to dst for float32
func Copy2dF32(rows, cols, lda, ldc int, src, dst []float32) {
	for i := range rows {
		for j := range cols {
			dst[i*ldc+j] = src[i*lda+j]
		}
	}
}

// Copy2dF64 copies a 2D region from src to dst for float64
func Copy2dF64(rows, cols, lda, ldc int, src, dst []float64) {
	for i := range rows {
		for j := range cols {
			dst[i*ldc+j] = src[i*lda+j]
		}
	}
}

// ConstSet sets a constant value at specified indices
func ConstSet[T D](numel int, val T, ids []int, dst []T) {
	for i := range numel {
		dst[ids[i]] = val
	}
}

// ConstSetF32 sets a constant float32 value at specified indices
func ConstSetF32(numel int, val float32, ids []int, dst []float32) {
	for i := range numel {
		dst[ids[i]] = val
	}
}

// ConstSetF64 sets a constant float64 value at specified indices
func ConstSetF64(numel int, val float64, ids []int, dst []float64) {
	for i := range numel {
		dst[ids[i]] = val
	}
}

// ConstSetStrided sets a constant value at specified indices for strided memory
func ConstSetStrided[T D](numel, numDims int, dims, strides []int, val T, ids []int, dst []T) {
	if IsContiguous(numDims, dims, strides) {
		ConstSet(numel, val, ids, dst)
		return
	}
	for i := range numel {
		dst[ids[i]] = val
	}
}

// ConstSetStridedF32 sets a constant float32 value at specified indices for strided memory
func ConstSetStridedF32(numel, numDims int, dims, strides []int, val float32, ids []int, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
		ConstSetF32(numel, val, ids, dst)
		return
	}
	for i := range numel {
		dst[ids[i]] = val
	}
}

// ConstSetStridedF64 sets a constant float64 value at specified indices for strided memory
func ConstSetStridedF64(numel, numDims int, dims, strides []int, val float64, ids []int, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
		ConstSetF64(numel, val, ids, dst)
		return
	}
	for i := range numel {
		dst[ids[i]] = val
	}
}

package kernels

// AffineF32 performs y = a*x + b operation for float32
func AffineF32(numel int, a, b float32, x, y []float32) {
	for i := range numel {
		y[i] = a*x[i] + b
	}
}

// AffineF64 performs y = a*x + b operation for float64
func AffineF64(numel int, a, b float64, x, y []float64) {
	for i := range numel {
		y[i] = a*x[i] + b
	}
}

// AffineStridedF32 performs strided affine operation for float32
func AffineStridedF32(numel int, numDims int, dims, strides []int, a, b float32, x, y []float32) {
	if IsContiguous(numDims, dims, strides) {
		AffineF32(numel, a, b, x, y)
		return
	}
	for i := range numel {
		y[i] = a*x[GetStridedIndex(i, numDims, dims, strides)] + b
	}
}

// AffineStridedF64 performs strided affine operation for float64
func AffineStridedF64(numel int, numDims int, dims, strides []int, a, b float64, x, y []float64) {
	if IsContiguous(numDims, dims, strides) {
		AffineF64(numel, a, b, x, y)
		return
	}

	for i := range numel {
		y[i] = a*x[GetStridedIndex(i, numDims, dims, strides)] + b
	}
}

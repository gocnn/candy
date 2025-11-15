package kernels

// Affine performs y = a*x + b operation for any supported numeric type
func Affine[T D](numel int, a, b T, x, y []T) {
	for i := range numel {
		y[i] = a*x[i] + b
	}
}

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

// AffineU8 performs y = a*x + b operation for uint8
func AffineU8(numel int, a, b uint8, x, y []uint8) {
	for i := range numel {
		y[i] = a*x[i] + b
	}
}

// AffineU32 performs y = a*x + b operation for uint32
func AffineU32(numel int, a, b uint32, x, y []uint32) {
	for i := range numel {
		y[i] = a*x[i] + b
	}
}

// AffineI64 performs y = a*x + b operation for int64
func AffineI64(numel int, a, b int64, x, y []int64) {
	for i := range numel {
		y[i] = a*x[i] + b
	}
}

// AffineStrided performs strided affine operation for any supported numeric type
func AffineStrided[T D](numel int, ndims int, dims, strides []int, a, b T, x, y []T) {
	if IsContiguous(ndims, dims, strides) {
		Affine(numel, a, b, x, y)
		return
	}
	for i := range numel {
		y[i] = a*x[GetStridedIndex(i, ndims, dims, strides)] + b
	}
}

// AffineStridedF32 performs strided affine operation for float32
func AffineStridedF32(numel int, ndims int, dims, strides []int, a, b float32, x, y []float32) {
	if IsContiguous(ndims, dims, strides) {
		AffineF32(numel, a, b, x, y)
		return
	}
	for i := range numel {
		y[i] = a*x[GetStridedIndex(i, ndims, dims, strides)] + b
	}
}

// AffineStridedF64 performs strided affine operation for float64
func AffineStridedF64(numel int, ndims int, dims, strides []int, a, b float64, x, y []float64) {
	if IsContiguous(ndims, dims, strides) {
		AffineF64(numel, a, b, x, y)
		return
	}
	for i := range numel {
		y[i] = a*x[GetStridedIndex(i, ndims, dims, strides)] + b
	}
}

// AffineStridedU8 performs strided affine operation for uint8
func AffineStridedU8(numel int, ndims int, dims, strides []int, a, b uint8, x, y []uint8) {
	if IsContiguous(ndims, dims, strides) {
		AffineU8(numel, a, b, x, y)
		return
	}
	for i := range numel {
		y[i] = a*x[GetStridedIndex(i, ndims, dims, strides)] + b
	}
}

// AffineStridedU32 performs strided affine operation for uint32
func AffineStridedU32(numel int, ndims int, dims, strides []int, a, b uint32, x, y []uint32) {
	if IsContiguous(ndims, dims, strides) {
		AffineU32(numel, a, b, x, y)
		return
	}
	for i := range numel {
		y[i] = a*x[GetStridedIndex(i, ndims, dims, strides)] + b
	}
}

// AffineStridedI64 performs strided affine operation for int64
func AffineStridedI64(numel int, ndims int, dims, strides []int, a, b int64, x, y []int64) {
	if IsContiguous(ndims, dims, strides) {
		AffineI64(numel, a, b, x, y)
		return
	}
	for i := range numel {
		y[i] = a*x[GetStridedIndex(i, ndims, dims, strides)] + b
	}
}

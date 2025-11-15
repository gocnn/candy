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

// FillU8 fills the destination array with a constant uint8 value
func FillU8(numel int, val uint8, dst []uint8) {
	for i := range numel {
		dst[i] = val
	}
}

// FillU32 fills the destination array with a constant uint32 value
func FillU32(numel int, val uint32, dst []uint32) {
	for i := range numel {
		dst[i] = val
	}
}

// FillI64 fills the destination array with a constant int64 value
func FillI64(numel int, val int64, dst []int64) {
	for i := range numel {
		dst[i] = val
	}
}

// FillStrided fills the destination array with a constant value for strided memory
func FillStrided[T D](numel, ndims int, dims, strides []int, val T, dst []T) {
	if IsContiguous(ndims, dims, strides) {
		Fill(numel, val, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, strides)] = val
	}
}

// FillStridedF32 fills the destination array with a constant float32 value for strided memory
func FillStridedF32(numel, ndims int, dims, strides []int, val float32, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		FillF32(numel, val, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, strides)] = val
	}
}

// FillStridedF64 fills the destination array with a constant float64 value for strided memory
func FillStridedF64(numel, ndims int, dims, strides []int, val float64, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		FillF64(numel, val, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, strides)] = val
	}
}

// FillStridedU8 fills the destination array with a constant uint8 value for strided memory
func FillStridedU8(numel, ndims int, dims, strides []int, val uint8, dst []uint8) {
	if IsContiguous(ndims, dims, strides) {
		FillU8(numel, val, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, strides)] = val
	}
}

// FillStridedU32 fills the destination array with a constant uint32 value for strided memory
func FillStridedU32(numel, ndims int, dims, strides []int, val uint32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FillU32(numel, val, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, strides)] = val
	}
}

// FillStridedI64 fills the destination array with a constant int64 value for strided memory
func FillStridedI64(numel, ndims int, dims, strides []int, val int64, dst []int64) {
	if IsContiguous(ndims, dims, strides) {
		FillI64(numel, val, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, strides)] = val
	}
}

// Copy2d copies a 2D region from src to dst with offsets
func Copy2d[T D](rows, cols, lda, ldc, srcOffset, dstOffset int, src, dst []T) {
	if lda == cols && ldc == cols {
		copy(dst[dstOffset:dstOffset+rows*cols], src[srcOffset:srcOffset+rows*cols])
		return
	}
	for i := range rows {
		srcStart := srcOffset + i*lda
		dstStart := dstOffset + i*ldc
		copy(dst[dstStart:dstStart+cols], src[srcStart:srcStart+cols])
	}
}

// Copy2dF32 copies a 2D region from src to dst for float32 with offsets
func Copy2dF32(rows, cols, lda, ldc, srcOffset, dstOffset int, src, dst []float32) {
	if lda == cols && ldc == cols {
		copy(dst[dstOffset:dstOffset+rows*cols], src[srcOffset:srcOffset+rows*cols])
		return
	}
	for i := range rows {
		srcStart := srcOffset + i*lda
		dstStart := dstOffset + i*ldc
		copy(dst[dstStart:dstStart+cols], src[srcStart:srcStart+cols])
	}
}

// Copy2dF64 copies a 2D region from src to dst for float64 with offsets
func Copy2dF64(rows, cols, lda, ldc, srcOffset, dstOffset int, src, dst []float64) {
	if lda == cols && ldc == cols {
		copy(dst[dstOffset:dstOffset+rows*cols], src[srcOffset:srcOffset+rows*cols])
		return
	}
	for i := range rows {
		srcStart := srcOffset + i*lda
		dstStart := dstOffset + i*ldc
		copy(dst[dstStart:dstStart+cols], src[srcStart:srcStart+cols])
	}
}

// Copy2dU8 copies a 2D region from src to dst for uint8 with offsets
func Copy2dU8(rows, cols, lda, ldc, srcOffset, dstOffset int, src, dst []uint8) {
	if lda == cols && ldc == cols {
		copy(dst[dstOffset:dstOffset+rows*cols], src[srcOffset:srcOffset+rows*cols])
		return
	}
	for i := range rows {
		srcStart := srcOffset + i*lda
		dstStart := dstOffset + i*ldc
		copy(dst[dstStart:dstStart+cols], src[srcStart:srcStart+cols])
	}
}

// Copy2dU32 copies a 2D region from src to dst for uint32 with offsets
func Copy2dU32(rows, cols, lda, ldc, srcOffset, dstOffset int, src, dst []uint32) {
	if lda == cols && ldc == cols {
		copy(dst[dstOffset:dstOffset+rows*cols], src[srcOffset:srcOffset+rows*cols])
		return
	}
	for i := range rows {
		srcStart := srcOffset + i*lda
		dstStart := dstOffset + i*ldc
		copy(dst[dstStart:dstStart+cols], src[srcStart:srcStart+cols])
	}
}

// Copy2dI64 copies a 2D region from src to dst for int64 with offsets
func Copy2dI64(rows, cols, lda, ldc, srcOffset, dstOffset int, src, dst []int64) {
	if lda == cols && ldc == cols {
		copy(dst[dstOffset:dstOffset+rows*cols], src[srcOffset:srcOffset+rows*cols])
		return
	}
	for i := range rows {
		srcStart := srcOffset + i*lda
		dstStart := dstOffset + i*ldc
		copy(dst[dstStart:dstStart+cols], src[srcStart:srcStart+cols])
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

// ConstSetU8 sets a constant uint8 value at specified indices
func ConstSetU8(numel int, val uint8, ids []int, dst []uint8) {
	for i := range numel {
		dst[ids[i]] = val
	}
}

// ConstSetU32 sets a constant uint32 value at specified indices
func ConstSetU32(numel int, val uint32, ids []int, dst []uint32) {
	for i := range numel {
		dst[ids[i]] = val
	}
}

// ConstSetI64 sets a constant int64 value at specified indices
func ConstSetI64(numel int, val int64, ids []int, dst []int64) {
	for i := range numel {
		dst[ids[i]] = val
	}
}

// ConstSetStrided sets a constant value at specified indices for strided memory
func ConstSetStrided[T D](numel, ndims int, dims, strides []int, val T, ids []int, dst []T) {
	if IsContiguous(ndims, dims, strides) {
		ConstSet(numel, val, ids, dst)
		return
	}
	for i := range numel {
		idx := GetStridedIndex(ids[i], ndims, dims, strides)
		dst[idx] = val
	}
}

// ConstSetStridedF32 sets a constant float32 value at specified indices for strided memory
func ConstSetStridedF32(numel, ndims int, dims, strides []int, val float32, ids []int, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		ConstSetF32(numel, val, ids, dst)
		return
	}
	for i := range numel {
		idx := GetStridedIndex(ids[i], ndims, dims, strides)
		dst[idx] = val
	}
}

// ConstSetStridedF64 sets a constant float64 value at specified indices for strided memory
func ConstSetStridedF64(numel, ndims int, dims, strides []int, val float64, ids []int, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		ConstSetF64(numel, val, ids, dst)
		return
	}
	for i := range numel {
		idx := GetStridedIndex(ids[i], ndims, dims, strides)
		dst[idx] = val
	}
}

// ConstSetStridedU8 sets a constant uint8 value at specified indices for strided memory
func ConstSetStridedU8(numel, ndims int, dims, strides []int, val uint8, ids []int, dst []uint8) {
	if IsContiguous(ndims, dims, strides) {
		ConstSetU8(numel, val, ids, dst)
		return
	}
	for i := range numel {
		idx := GetStridedIndex(ids[i], ndims, dims, strides)
		dst[idx] = val
	}
}

// ConstSetStridedU32 sets a constant uint32 value at specified indices for strided memory
func ConstSetStridedU32(numel, ndims int, dims, strides []int, val uint32, ids []int, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ConstSetU32(numel, val, ids, dst)
		return
	}
	for i := range numel {
		idx := GetStridedIndex(ids[i], ndims, dims, strides)
		dst[idx] = val
	}
}

// ConstSetStridedI64 sets a constant int64 value at specified indices for strided memory
func ConstSetStridedI64(numel, ndims int, dims, strides []int, val int64, ids []int, dst []int64) {
	if IsContiguous(ndims, dims, strides) {
		ConstSetI64(numel, val, ids, dst)
		return
	}
	for i := range numel {
		idx := GetStridedIndex(ids[i], ndims, dims, strides)
		dst[idx] = val
	}
}

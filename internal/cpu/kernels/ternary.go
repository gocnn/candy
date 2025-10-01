package kernels

// WhereI64F32 selects elements from t or f based on int64 indices for float32
func WhereI64F32(numel int, ids []int64, t, f, dst []float32) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereI64F64 selects elements from t or f based on int64 indices for float64
func WhereI64F64(numel int, ids []int64, t, f, dst []float64) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereU32F32 selects elements from t or f based on uint32 indices for float32
func WhereU32F32(numel int, ids []uint32, t, f, dst []float32) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereU32F64 selects elements from t or f based on uint32 indices for float64
func WhereU32F64(numel int, ids []uint32, t, f, dst []float64) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereU8F32 selects elements from t or f based on uint8 indices for float32
func WhereU8F32(numel int, ids []uint8, t, f, dst []float32) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereU8F64 selects elements from t or f based on uint8 indices for float64
func WhereU8F64(numel int, ids []uint8, t, f, dst []float64) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereStridedI64F32 selects elements from t or f based on int64 indices for float32 with strided memory
func WhereStridedI64F32(numel, numDims int, dims, strides, stridesT, stridesF []int, ids []int64, t, f, dst []float32) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesT) && IsContiguous(numDims, dims, stridesF) {
		WhereI64F32(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, numDims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, numDims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, numDims, dims, stridesF)]
		}
	}
}

// WhereStridedI64F64 selects elements from t or f based on int64 indices for float64 with strided memory
func WhereStridedI64F64(numel, numDims int, dims, strides, stridesT, stridesF []int, ids []int64, t, f, dst []float64) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesT) && IsContiguous(numDims, dims, stridesF) {
		WhereI64F64(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, numDims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, numDims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, numDims, dims, stridesF)]
		}
	}
}

// WhereStridedU32F32 selects elements from t or f based on uint32 indices for float32 with strided memory
func WhereStridedU32F32(numel, numDims int, dims, strides, stridesT, stridesF []int, ids []uint32, t, f, dst []float32) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesT) && IsContiguous(numDims, dims, stridesF) {
		WhereU32F32(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, numDims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, numDims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, numDims, dims, stridesF)]
		}
	}
}

// WhereStridedU32F64 selects elements from t or f based on uint32 indices for float64 with strided memory
func WhereStridedU32F64(numel, numDims int, dims, strides, stridesT, stridesF []int, ids []uint32, t, f, dst []float64) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesT) && IsContiguous(numDims, dims, stridesF) {
		WhereU32F64(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, numDims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, numDims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, numDims, dims, stridesF)]
		}
	}
}

// WhereStridedU8F32 selects elements from t or f based on uint8 indices for float32 with strided memory
func WhereStridedU8F32(numel, numDims int, dims, strides, stridesT, stridesF []int, ids []uint8, t, f, dst []float32) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesT) && IsContiguous(numDims, dims, stridesF) {
		WhereU8F32(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, numDims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, numDims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, numDims, dims, stridesF)]
		}
	}
}

// WhereStridedU8F64 selects elements from t or f based on uint8 indices for float64 with strided memory
func WhereStridedU8F64(numel, numDims int, dims, strides, stridesT, stridesF []int, ids []uint8, t, f, dst []float64) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesT) && IsContiguous(numDims, dims, stridesF) {
		WhereU8F64(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, numDims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, numDims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, numDims, dims, stridesF)]
		}
	}
}

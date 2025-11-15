package kernels

// Where selects elements from t or f based on indices of type I for data of type T
func Where[U I, T D](numel int, ids []U, t, f, dst []T) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereF32 selects elements from t or f based on float32 condition for any data type T
func WhereF32[T D](numel int, cond []float32, t, f, dst []T) {
	for i := range numel {
		if cond[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereF64 selects elements from t or f based on float64 condition for any data type T
func WhereF64[T D](numel int, cond []float64, t, f, dst []T) {
	for i := range numel {
		if cond[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereU8 selects elements from t or f based on uint8 condition for any data type T
func WhereU8[T D](numel int, ids []uint8, t, f, dst []T) {
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

// WhereU8U8 selects elements from t or f based on uint8 indices for uint8
func WhereU8U8(numel int, ids []uint8, t, f, dst []uint8) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereU8U32 selects elements from t or f based on uint8 indices for uint32
func WhereU8U32(numel int, ids []uint8, t, f, dst []uint32) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereU8I64 selects elements from t or f based on uint8 indices for int64
func WhereU8I64(numel int, ids []uint8, t, f, dst []int64) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereU32 selects elements from t or f based on uint32 condition for any data type T
func WhereU32[T D](numel int, ids []uint32, t, f, dst []T) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereU32U8 selects elements from t or f based on uint32 indices for uint8
func WhereU32U8(numel int, ids []uint32, t, f, dst []uint8) {
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

// WhereU32U32 selects elements from t or f based on uint32 indices for uint32
func WhereU32U32(numel int, ids []uint32, t, f, dst []uint32) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereU32I64 selects elements from t or f based on uint32 indices for int64
func WhereU32I64(numel int, ids []uint32, t, f, dst []int64) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereI64 selects elements from t or f based on int64 condition for any data type T
func WhereI64[T D](numel int, ids []int64, t, f, dst []T) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

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

// WhereI64U8 selects elements from t or f based on int64 indices for uint8
func WhereI64U8(numel int, ids []int64, t, f, dst []uint8) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereI64U32 selects elements from t or f based on int64 indices for uint32
func WhereI64U32(numel int, ids []int64, t, f, dst []uint32) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereI64I64 selects elements from t or f based on int64 indices for int64
func WhereI64I64(numel int, ids []int64, t, f, dst []int64) {
	for i := range numel {
		if ids[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}

// WhereStrided selects elements from t or f based on indices of type I for data of type T with strided memory
func WhereStrided[U I, T D](numel, ndims int, dims, strides, stridesT, stridesF []int, ids []U, t, f, dst []T) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		Where(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedF32 selects elements from t or f based on float32 condition with strided memory
func WhereStridedF32[T D](numel, ndims int, dims, strides, stridesT, stridesF []int, cond []float32, t, f, dst []T) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereF32(numel, cond, t, f, dst)
		return
	}
	for i := range numel {
		if cond[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedF64 selects elements from t or f based on float64 condition with strided memory
func WhereStridedF64[T D](numel, ndims int, dims, strides, stridesT, stridesF []int, cond []float64, t, f, dst []T) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereF64(numel, cond, t, f, dst)
		return
	}
	for i := range numel {
		if cond[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU8 selects elements from t or f based on uint8 condition with strided memory
func WhereStridedU8[T D](numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint8, t, f, dst []T) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU8(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU8F32 selects elements from t or f based on uint8 indices for float32 with strided memory
func WhereStridedU8F32(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint8, t, f, dst []float32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU8F32(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU8F64 selects elements from t or f based on uint8 indices for float64 with strided memory
func WhereStridedU8F64(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint8, t, f, dst []float64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU8F64(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU8U8 selects elements from t or f based on uint8 indices for uint8 with strided memory
func WhereStridedU8U8(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint8, t, f, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU8U8(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU8U32 selects elements from t or f based on uint8 indices for uint32 with strided memory
func WhereStridedU8U32(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint8, t, f, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU8U32(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU8I64 selects elements from t or f based on uint8 indices for int64 with strided memory
func WhereStridedU8I64(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint8, t, f, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU8I64(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU32 selects elements from t or f based on uint32 condition with strided memory
func WhereStridedU32[T D](numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint32, t, f, dst []T) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU32(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU32F32 selects elements from t or f based on uint32 indices for float32 with strided memory
func WhereStridedU32F32(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint32, t, f, dst []float32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU32F32(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU32F64 selects elements from t or f based on uint32 indices for float64 with strided memory
func WhereStridedU32F64(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint32, t, f, dst []float64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU32F64(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU32U8 selects elements from t or f based on uint32 indices for uint8 with strided memory
func WhereStridedU32U8(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint32, t, f, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU32U8(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU32U32 selects elements from t or f based on uint32 indices for uint32 with strided memory
func WhereStridedU32U32(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint32, t, f, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU32U32(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedU32I64 selects elements from t or f based on uint32 indices for int64 with strided memory
func WhereStridedU32I64(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []uint32, t, f, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereU32I64(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedI64 selects elements from t or f based on int64 condition with strided memory
func WhereStridedI64[T D](numel, ndims int, dims, strides, stridesT, stridesF []int, ids []int64, t, f, dst []T) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereI64(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedI64F32 selects elements from t or f based on int64 indices for float32 with strided memory
func WhereStridedI64F32(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []int64, t, f, dst []float32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereI64F32(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedI64F64 selects elements from t or f based on int64 indices for float64 with strided memory
func WhereStridedI64F64(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []int64, t, f, dst []float64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereI64F64(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedI64U8 selects elements from t or f based on int64 indices for uint8 with strided memory
func WhereStridedI64U8(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []int64, t, f, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereI64U8(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedI64U32 selects elements from t or f based on int64 indices for uint32 with strided memory
func WhereStridedI64U32(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []int64, t, f, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereI64U32(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

// WhereStridedI64I64 selects elements from t or f based on int64 indices for int64 with strided memory
func WhereStridedI64I64(numel, ndims int, dims, strides, stridesT, stridesF []int, ids []int64, t, f, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesT) && IsContiguous(ndims, dims, stridesF) {
		WhereI64I64(numel, ids, t, f, dst)
		return
	}
	for i := range numel {
		if ids[GetStridedIndex(i, ndims, dims, strides)] != 0 {
			dst[i] = t[GetStridedIndex(i, ndims, dims, stridesT)]
		} else {
			dst[i] = f[GetStridedIndex(i, ndims, dims, stridesF)]
		}
	}
}

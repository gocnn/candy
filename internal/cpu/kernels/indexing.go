package kernels

// IndexSelect selects elements from src at indices of type I for data of type T
func IndexSelect[U I, T D](numel int, ids []U, src, dst []T) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectI64F32 selects elements from src at int64 indices for float32
func IndexSelectI64F32(numel int, ids []int64, src, dst []float32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectI64F64 selects elements from src at int64 indices for float64
func IndexSelectI64F64(numel int, ids []int64, src, dst []float64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectI64U8 selects elements from src at int64 indices for uint8
func IndexSelectI64U8(numel int, ids []int64, src, dst []uint8) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectI64U32 selects elements from src at int64 indices for uint32
func IndexSelectI64U32(numel int, ids []int64, src, dst []uint32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectI64I64 selects elements from src at int64 indices for int64
func IndexSelectI64I64(numel int, ids []int64, src, dst []int64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectU32F32 selects elements from src at uint32 indices for float32
func IndexSelectU32F32(numel int, ids []uint32, src, dst []float32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectU32F64 selects elements from src at uint32 indices for float64
func IndexSelectU32F64(numel int, ids []uint32, src, dst []float64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectU32U8 selects elements from src at uint32 indices for uint8
func IndexSelectU32U8(numel int, ids []uint32, src, dst []uint8) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectU32U32 selects elements from src at uint32 indices for uint32
func IndexSelectU32U32(numel int, ids []uint32, src, dst []uint32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectU32I64 selects elements from src at uint32 indices for int64
func IndexSelectU32I64(numel int, ids []uint32, src, dst []int64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectU8F32 selects elements from src at uint8 indices for float32
func IndexSelectU8F32(numel int, ids []uint8, src, dst []float32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectU8F64 selects elements from src at uint8 indices for float64
func IndexSelectU8F64(numel int, ids []uint8, src, dst []float64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectU8U8 selects elements from src at uint8 indices for uint8
func IndexSelectU8U8(numel int, ids []uint8, src, dst []uint8) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectU8U32 selects elements from src at uint8 indices for uint32
func IndexSelectU8U32(numel int, ids []uint8, src, dst []uint32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectU8I64 selects elements from src at uint8 indices for int64
func IndexSelectU8I64(numel int, ids []uint8, src, dst []int64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// IndexSelectStrided selects elements from src at indices of type I for data of type T with strided memory
func IndexSelectStrided[U I, T D](numel, ndims int, dims, stridesSrc, stridesDst []int, ids []U, src, dst []T) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelect(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedI64F32 selects elements from src at int64 indices for float32 with strided memory
func IndexSelectStridedI64F32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectI64F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedI64F64 selects elements from src at int64 indices for float64 with strided memory
func IndexSelectStridedI64F64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectI64F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedI64U8 selects elements from src at int64 indices for uint8 with strided memory
func IndexSelectStridedI64U8(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectI64U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedI64U32 selects elements from src at int64 indices for uint32 with strided memory
func IndexSelectStridedI64U32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectI64U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedI64I64 selects elements from src at int64 indices for int64 with strided memory
func IndexSelectStridedI64I64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectI64I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU32F32 selects elements from src at uint32 indices for float32 with strided memory
func IndexSelectStridedU32F32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectU32F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU32F64 selects elements from src at uint32 indices for float64 with strided memory
func IndexSelectStridedU32F64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectU32F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU32U8 selects elements from src at uint32 indices for uint8 with strided memory
func IndexSelectStridedU32U8(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectU32U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU32U32 selects elements from src at uint32 indices for uint32 with strided memory
func IndexSelectStridedU32U32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectU32U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU32I64 selects elements from src at uint32 indices for int64 with strided memory
func IndexSelectStridedU32I64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectU32I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU8F32 selects elements from src at uint8 indices for float32 with strided memory
func IndexSelectStridedU8F32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectU8F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU8F64 selects elements from src at uint8 indices for float64 with strided memory
func IndexSelectStridedU8F64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectU8F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU8U8 selects elements from src at uint8 indices for uint8 with strided memory
func IndexSelectStridedU8U8(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectU8U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU8U32 selects elements from src at uint8 indices for uint32 with strided memory
func IndexSelectStridedU8U32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectU8U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU8I64 selects elements from src at uint8 indices for int64 with strided memory
func IndexSelectStridedU8I64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexSelectU8I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), ndims, dims, stridesSrc)]
	}
}

// Gather gathers elements from src at indices of type I for data of type T
func Gather[U I, T D](numel int, ids []U, src, dst []T) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherI64F32 gathers elements from src at int64 indices for float32
func GatherI64F32(numel int, ids []int64, src, dst []float32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherI64F64 gathers elements from src at int64 indices for float64
func GatherI64F64(numel int, ids []int64, src, dst []float64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherI64U8 gathers elements from src at int64 indices for uint8
func GatherI64U8(numel int, ids []int64, src, dst []uint8) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherI64U32 gathers elements from src at int64 indices for uint32
func GatherI64U32(numel int, ids []int64, src, dst []uint32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherI64I64 gathers elements from src at int64 indices for int64
func GatherI64I64(numel int, ids []int64, src, dst []int64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherU32F32 gathers elements from src at uint32 indices for float32
func GatherU32F32(numel int, ids []uint32, src, dst []float32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherU32F64 gathers elements from src at uint32 indices for float64
func GatherU32F64(numel int, ids []uint32, src, dst []float64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherU32U8 gathers elements from src at uint32 indices for uint8
func GatherU32U8(numel int, ids []uint32, src, dst []uint8) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherU32U32 gathers elements from src at uint32 indices for uint32
func GatherU32U32(numel int, ids []uint32, src, dst []uint32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherU32I64 gathers elements from src at uint32 indices for int64
func GatherU32I64(numel int, ids []uint32, src, dst []int64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherU8F32 gathers elements from src at uint8 indices for float32
func GatherU8F32(numel int, ids []uint8, src, dst []float32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherU8F64 gathers elements from src at uint8 indices for float64
func GatherU8F64(numel int, ids []uint8, src, dst []float64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherU8U8 gathers elements from src at uint8 indices for uint8
func GatherU8U8(numel int, ids []uint8, src, dst []uint8) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherU8U32 gathers elements from src at uint8 indices for uint32
func GatherU8U32(numel int, ids []uint8, src, dst []uint32) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherU8I64 gathers elements from src at uint8 indices for int64
func GatherU8I64(numel int, ids []uint8, src, dst []int64) {
	for i := range numel {
		dst[i] = src[ids[i]]
	}
}

// GatherStrided gathers elements from src at indices of type I for data of type T with strided memory
func GatherStrided[U I, T D](numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []U, src, dst []T) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		Gather(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedI64F32 gathers elements from src at int64 indices for float32 with strided memory
func GatherStridedI64F32(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []int64, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherI64F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedI64F64 gathers elements from src at int64 indices for float64 with strided memory
func GatherStridedI64F64(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []int64, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherI64F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedI64U8 gathers elements from src at int64 indices for uint8 with strided memory
func GatherStridedI64U8(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []int64, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherI64U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedI64U32 gathers elements from src at int64 indices for uint32 with strided memory
func GatherStridedI64U32(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []int64, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherI64U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedI64I64 gathers elements from src at int64 indices for int64 with strided memory
func GatherStridedI64I64(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []int64, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherI64I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedU32F32 gathers elements from src at uint32 indices for float32 with strided memory
func GatherStridedU32F32(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint32, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherU32F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedU32F64 gathers elements from src at uint32 indices for float64 with strided memory
func GatherStridedU32F64(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint32, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherU32F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedU32U8 gathers elements from src at uint32 indices for uint8 with strided memory
func GatherStridedU32U8(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint32, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherU32U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedU32U32 gathers elements from src at uint32 indices for uint32 with strided memory
func GatherStridedU32U32(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint32, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherU32U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedU32I64 gathers elements from src at uint32 indices for int64 with strided memory
func GatherStridedU32I64(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint32, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherU32I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedU8F32 gathers elements from src at uint8 indices for float32 with strided memory
func GatherStridedU8F32(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint8, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherU8F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedU8F64 gathers elements from src at uint8 indices for float64 with strided memory
func GatherStridedU8F64(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint8, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherU8F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedU8U8 gathers elements from src at uint8 indices for uint8 with strided memory
func GatherStridedU8U8(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint8, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherU8U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedU8U32 gathers elements from src at uint8 indices for uint32 with strided memory
func GatherStridedU8U32(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint8, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherU8U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// GatherStridedU8I64 gathers elements from src at uint8 indices for int64 with strided memory
func GatherStridedU8I64(numel, ndims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint8, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) && IsContiguous(ndims, dims, stridesIds) {
		GatherU8I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, ndims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, ndims, dims, stridesIds)]), ndims, dims, stridesSrc)]
	}
}

// IndexAdd adds src to dst at indices of type I for data of type T
func IndexAdd[U I, T D](numel int, ids []U, src, dst []T) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddI64F32 adds src to dst at int64 indices for float32
func IndexAddI64F32(numel int, ids []int64, src, dst []float32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddI64F64 adds src to dst at int64 indices for float64
func IndexAddI64F64(numel int, ids []int64, src, dst []float64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddI64U8 adds src to dst at int64 indices for uint8
func IndexAddI64U8(numel int, ids []int64, src, dst []uint8) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddI64U32 adds src to dst at int64 indices for uint32
func IndexAddI64U32(numel int, ids []int64, src, dst []uint32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddI64I64 adds src to dst at int64 indices for int64
func IndexAddI64I64(numel int, ids []int64, src, dst []int64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddU32F32 adds src to dst at uint32 indices for float32
func IndexAddU32F32(numel int, ids []uint32, src, dst []float32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddU32F64 adds src to dst at uint32 indices for float64
func IndexAddU32F64(numel int, ids []uint32, src, dst []float64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddU32U8 adds src to dst at uint32 indices for uint8
func IndexAddU32U8(numel int, ids []uint32, src, dst []uint8) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddU32U32 adds src to dst at uint32 indices for uint32
func IndexAddU32U32(numel int, ids []uint32, src, dst []uint32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddU32I64 adds src to dst at uint32 indices for int64
func IndexAddU32I64(numel int, ids []uint32, src, dst []int64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddU8F32 adds src to dst at uint8 indices for float32
func IndexAddU8F32(numel int, ids []uint8, src, dst []float32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddU8F64 adds src to dst at uint8 indices for float64
func IndexAddU8F64(numel int, ids []uint8, src, dst []float64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddU8U8 adds src to dst at uint8 indices for uint8
func IndexAddU8U8(numel int, ids []uint8, src, dst []uint8) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddU8U32 adds src to dst at uint8 indices for uint32
func IndexAddU8U32(numel int, ids []uint8, src, dst []uint32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddU8I64 adds src to dst at uint8 indices for int64
func IndexAddU8I64(numel int, ids []uint8, src, dst []int64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// IndexAddStrided adds src to dst at indices of type I for data of type T with strided memory
func IndexAddStrided[U I, T D](numel, ndims int, dims, stridesSrc, stridesDst []int, ids []U, src, dst []T) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAdd(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedI64F32 adds src to dst at int64 indices for float32 with strided memory
func IndexAddStridedI64F32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddI64F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedI64F64 adds src to dst at int64 indices for float64 with strided memory
func IndexAddStridedI64F64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddI64F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedI64U8 adds src to dst at int64 indices for uint8 with strided memory
func IndexAddStridedI64U8(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddI64U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedI64U32 adds src to dst at int64 indices for uint32 with strided memory
func IndexAddStridedI64U32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddI64U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedI64I64 adds src to dst at int64 indices for int64 with strided memory
func IndexAddStridedI64I64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddI64I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedU32F32 adds src to dst at uint32 indices for float32 with strided memory
func IndexAddStridedU32F32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddU32F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedU32F64 adds src to dst at uint32 indices for float64 with strided memory
func IndexAddStridedU32F64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddU32F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedU32U8 adds src to dst at uint32 indices for uint8 with strided memory
func IndexAddStridedU32U8(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddU32U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedU32U32 adds src to dst at uint32 indices for uint32 with strided memory
func IndexAddStridedU32U32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddU32U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedU32I64 adds src to dst at uint32 indices for int64 with strided memory
func IndexAddStridedU32I64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddU32I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedU8F32 adds src to dst at uint8 indices for float32 with strided memory
func IndexAddStridedU8F32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddU8F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedU8F64 adds src to dst at uint8 indices for float64 with strided memory
func IndexAddStridedU8F64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddU8F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedU8U8 adds src to dst at uint8 indices for uint8 with strided memory
func IndexAddStridedU8U8(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddU8U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedU8U32 adds src to dst at uint8 indices for uint32 with strided memory
func IndexAddStridedU8U32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddU8U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// IndexAddStridedU8I64 adds src to dst at uint8 indices for int64 with strided memory
func IndexAddStridedU8I64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		IndexAddU8I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// Scatter scatters src elements to dst at indices of type I for data of type T
func Scatter[U I, T D](numel int, ids []U, src, dst []T) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterI64F32 scatters src elements to dst at int64 indices for float32
func ScatterI64F32(numel int, ids []int64, src, dst []float32) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterI64F64 scatters src elements to dst at int64 indices for float64
func ScatterI64F64(numel int, ids []int64, src, dst []float64) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterI64U8 scatters src elements to dst at int64 indices for uint8
func ScatterI64U8(numel int, ids []int64, src, dst []uint8) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterI64U32 scatters src elements to dst at int64 indices for uint32
func ScatterI64U32(numel int, ids []int64, src, dst []uint32) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterI64I64 scatters src elements to dst at int64 indices for int64
func ScatterI64I64(numel int, ids []int64, src, dst []int64) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterU32F32 scatters src elements to dst at uint32 indices for float32
func ScatterU32F32(numel int, ids []uint32, src, dst []float32) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterU32F64 scatters src elements to dst at uint32 indices for float64
func ScatterU32F64(numel int, ids []uint32, src, dst []float64) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterU32U8 scatters src elements to dst at uint32 indices for uint8
func ScatterU32U8(numel int, ids []uint32, src, dst []uint8) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterU32U32 scatters src elements to dst at uint32 indices for uint32
func ScatterU32U32(numel int, ids []uint32, src, dst []uint32) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterU32I64 scatters src elements to dst at uint32 indices for int64
func ScatterU32I64(numel int, ids []uint32, src, dst []int64) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterU8F32 scatters src elements to dst at uint8 indices for float32
func ScatterU8F32(numel int, ids []uint8, src, dst []float32) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterU8F64 scatters src elements to dst at uint8 indices for float64
func ScatterU8F64(numel int, ids []uint8, src, dst []float64) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterU8U8 scatters src elements to dst at uint8 indices for uint8
func ScatterU8U8(numel int, ids []uint8, src, dst []uint8) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterU8U32 scatters src elements to dst at uint8 indices for uint32
func ScatterU8U32(numel int, ids []uint8, src, dst []uint32) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterU8I64 scatters src elements to dst at uint8 indices for int64
func ScatterU8I64(numel int, ids []uint8, src, dst []int64) {
	for i := range numel {
		dst[ids[i]] = src[i]
	}
}

// ScatterStrided scatters src elements to dst at indices of type I for data of type T with strided memory
func ScatterStrided[U I, T D](numel, ndims int, dims, stridesSrc, stridesDst []int, ids []U, src, dst []T) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		Scatter(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedI64F32 scatters src elements to dst at int64 indices for float32 with strided memory
func ScatterStridedI64F32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterI64F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedI64F64 scatters src elements to dst at int64 indices for float64 with strided memory
func ScatterStridedI64F64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterI64F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedI64U8 scatters src elements to dst at int64 indices for uint8 with strided memory
func ScatterStridedI64U8(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterI64U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedI64U32 scatters src elements to dst at int64 indices for uint32 with strided memory
func ScatterStridedI64U32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterI64U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedI64I64 scatters src elements to dst at int64 indices for int64 with strided memory
func ScatterStridedI64I64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterI64I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedU32F32 scatters src elements to dst at uint32 indices for float32 with strided memory
func ScatterStridedU32F32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterU32F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedU32F64 scatters src elements to dst at uint32 indices for float64 with strided memory
func ScatterStridedU32F64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterU32F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedU32U8 scatters src elements to dst at uint32 indices for uint8 with strided memory
func ScatterStridedU32U8(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterU32U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedU32U32 scatters src elements to dst at uint32 indices for uint32 with strided memory
func ScatterStridedU32U32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterU32U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedU32I64 scatters src elements to dst at uint32 indices for int64 with strided memory
func ScatterStridedU32I64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterU32I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedU8F32 scatters src elements to dst at uint8 indices for float32 with strided memory
func ScatterStridedU8F32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterU8F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedU8F64 scatters src elements to dst at uint8 indices for float64 with strided memory
func ScatterStridedU8F64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterU8F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedU8U8 scatters src elements to dst at uint8 indices for uint8 with strided memory
func ScatterStridedU8U8(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterU8U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedU8U32 scatters src elements to dst at uint8 indices for uint32 with strided memory
func ScatterStridedU8U32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterU8U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterStridedU8I64 scatters src elements to dst at uint8 indices for int64 with strided memory
func ScatterStridedU8I64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterU8I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] = src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterAdd adds src elements to dst at indices of type I for data of type T
func ScatterAdd[U I, T D](numel int, ids []U, src, dst []T) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddI64F32 adds src elements to dst at int64 indices for float32
func ScatterAddI64F32(numel int, ids []int64, src, dst []float32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddI64F64 adds src elements to dst at int64 indices for float64
func ScatterAddI64F64(numel int, ids []int64, src, dst []float64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddI64U8 adds src elements to dst at int64 indices for uint8
func ScatterAddI64U8(numel int, ids []int64, src, dst []uint8) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddI64U32 adds src elements to dst at int64 indices for uint32
func ScatterAddI64U32(numel int, ids []int64, src, dst []uint32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddI64I64 adds src elements to dst at int64 indices for int64
func ScatterAddI64I64(numel int, ids []int64, src, dst []int64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddU32F32 adds src elements to dst at uint32 indices for float32
func ScatterAddU32F32(numel int, ids []uint32, src, dst []float32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddU32F64 adds src elements to dst at uint32 indices for float64
func ScatterAddU32F64(numel int, ids []uint32, src, dst []float64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddU32U8 adds src elements to dst at uint32 indices for uint8
func ScatterAddU32U8(numel int, ids []uint32, src, dst []uint8) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddU32U32 adds src elements to dst at uint32 indices for uint32
func ScatterAddU32U32(numel int, ids []uint32, src, dst []uint32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddU32I64 adds src elements to dst at uint32 indices for int64
func ScatterAddU32I64(numel int, ids []uint32, src, dst []int64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddU8F32 adds src elements to dst at uint8 indices for float32
func ScatterAddU8F32(numel int, ids []uint8, src, dst []float32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddU8F64 adds src elements to dst at uint8 indices for float64
func ScatterAddU8F64(numel int, ids []uint8, src, dst []float64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddU8U8 adds src elements to dst at uint8 indices for uint8
func ScatterAddU8U8(numel int, ids []uint8, src, dst []uint8) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddU8U32 adds src elements to dst at uint8 indices for uint32
func ScatterAddU8U32(numel int, ids []uint8, src, dst []uint32) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddU8I64 adds src elements to dst at uint8 indices for int64
func ScatterAddU8I64(numel int, ids []uint8, src, dst []int64) {
	for i := range numel {
		dst[ids[i]] += src[i]
	}
}

// ScatterAddStrided adds src elements to dst at indices of type I for data of type T with strided memory
func ScatterAddStrided[U I, T D](numel, ndims int, dims, stridesSrc, stridesDst []int, ids []U, src, dst []T) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterAdd(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterAddStridedI64F32 adds src elements to dst at int64 indices for float32 with strided memory
func ScatterAddStridedI64F32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterAddI64F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterAddStridedI64F64 adds src elements to dst at int64 indices for float64 with strided memory
func ScatterAddStridedI64F64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterAddI64F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterAddStridedI64U8 adds src elements to dst at int64 indices for uint8 with strided memory
func ScatterAddStridedI64U8(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []uint8) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterAddI64U8(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterAddStridedI64U32 adds src elements to dst at int64 indices for uint32 with strided memory
func ScatterAddStridedI64U32(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []uint32) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterAddI64U32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

// ScatterAddStridedI64I64 adds src elements to dst at int64 indices for int64 with strided memory
func ScatterAddStridedI64I64(numel, ndims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []int64) {
	if IsContiguous(ndims, dims, stridesSrc) && IsContiguous(ndims, dims, stridesDst) {
		ScatterAddI64I64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), ndims, dims, stridesDst)] += src[GetStridedIndex(i, ndims, dims, stridesSrc)]
	}
}

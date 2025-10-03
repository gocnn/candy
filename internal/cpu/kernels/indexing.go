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

// IndexSelectStrided selects elements from src at indices of type I for data of type T with strided memory
func IndexSelectStrided[U I, T D](numel, numDims int, dims, stridesSrc, stridesDst []int, ids []U, src, dst []T) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexSelect(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), numDims, dims, stridesSrc)]
	}
}

// IndexSelectStridedI64F32 selects elements from src at int64 indices for float32 with strided memory
func IndexSelectStridedI64F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexSelectI64F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), numDims, dims, stridesSrc)]
	}
}

// IndexSelectStridedI64F64 selects elements from src at int64 indices for float64 with strided memory
func IndexSelectStridedI64F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexSelectI64F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), numDims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU32F32 selects elements from src at uint32 indices for float32 with strided memory
func IndexSelectStridedU32F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexSelectU32F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), numDims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU32F64 selects elements from src at uint32 indices for float64 with strided memory
func IndexSelectStridedU32F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexSelectU32F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), numDims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU8F32 selects elements from src at uint8 indices for float32 with strided memory
func IndexSelectStridedU8F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexSelectU8F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), numDims, dims, stridesSrc)]
	}
}

// IndexSelectStridedU8F64 selects elements from src at uint8 indices for float64 with strided memory
func IndexSelectStridedU8F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexSelectU8F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[i]), numDims, dims, stridesSrc)]
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

// GatherStrided gathers elements from src at indices of type I for data of type T with strided memory
func GatherStrided[U I, T D](numel, numDims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []U, src, dst []T) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) && IsContiguous(numDims, dims, stridesIds) {
		Gather(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, numDims, dims, stridesIds)]), numDims, dims, stridesSrc)]
	}
}

// GatherStridedI64F32 gathers elements from src at int64 indices for float32 with strided memory
func GatherStridedI64F32(numel, numDims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []int64, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) && IsContiguous(numDims, dims, stridesIds) {
		GatherI64F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, numDims, dims, stridesIds)]), numDims, dims, stridesSrc)]
	}
}

// GatherStridedI64F64 gathers elements from src at int64 indices for float64 with strided memory
func GatherStridedI64F64(numel, numDims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []int64, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) && IsContiguous(numDims, dims, stridesIds) {
		GatherI64F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, numDims, dims, stridesIds)]), numDims, dims, stridesSrc)]
	}
}

// GatherStridedU32F32 gathers elements from src at uint32 indices for float32 with strided memory
func GatherStridedU32F32(numel, numDims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint32, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) && IsContiguous(numDims, dims, stridesIds) {
		GatherU32F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, numDims, dims, stridesIds)]), numDims, dims, stridesSrc)]
	}
}

// GatherStridedU32F64 gathers elements from src at uint32 indices for float64 with strided memory
func GatherStridedU32F64(numel, numDims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint32, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) && IsContiguous(numDims, dims, stridesIds) {
		GatherU32F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, numDims, dims, stridesIds)]), numDims, dims, stridesSrc)]
	}
}

// GatherStridedU8F32 gathers elements from src at uint8 indices for float32 with strided memory
func GatherStridedU8F32(numel, numDims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint8, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) && IsContiguous(numDims, dims, stridesIds) {
		GatherU8F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, numDims, dims, stridesIds)]), numDims, dims, stridesSrc)]
	}
}

// GatherStridedU8F64 gathers elements from src at uint8 indices for float64 with strided memory
func GatherStridedU8F64(numel, numDims int, dims, stridesSrc, stridesDst, stridesIds []int, ids []uint8, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) && IsContiguous(numDims, dims, stridesIds) {
		GatherU8F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(i, numDims, dims, stridesDst)] = src[GetStridedIndex(int(ids[GetStridedIndex(i, numDims, dims, stridesIds)]), numDims, dims, stridesSrc)]
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

// IndexAddStrided adds src to dst at indices of type I for data of type T with strided memory
func IndexAddStrided[U I, T D](numel, numDims int, dims, stridesSrc, stridesDst []int, ids []U, src, dst []T) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexAdd(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// IndexAddStridedI64F32 adds src to dst at int64 indices for float32 with strided memory
func IndexAddStridedI64F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexAddI64F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// IndexAddStridedI64F64 adds src to dst at int64 indices for float64 with strided memory
func IndexAddStridedI64F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexAddI64F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// IndexAddStridedU32F32 adds src to dst at uint32 indices for float32 with strided memory
func IndexAddStridedU32F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexAddU32F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// IndexAddStridedU32F64 adds src to dst at uint32 indices for float64 with strided memory
func IndexAddStridedU32F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexAddU32F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// IndexAddStridedU8F32 adds src to dst at uint8 indices for float32 with strided memory
func IndexAddStridedU8F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexAddU8F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// IndexAddStridedU8F64 adds src to dst at uint8 indices for float64 with strided memory
func IndexAddStridedU8F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		IndexAddU8F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
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

// ScatterStrided scatters src elements to dst at indices of type I for data of type T with strided memory
func ScatterStrided[U I, T D](numel, numDims int, dims, stridesSrc, stridesDst []int, ids []U, src, dst []T) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		Scatter(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] = src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterStridedI64F32 scatters src elements to dst at int64 indices for float32 with strided memory
func ScatterStridedI64F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterI64F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] = src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterStridedI64F64 scatters src elements to dst at int64 indices for float64 with strided memory
func ScatterStridedI64F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterI64F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] = src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterStridedU32F32 scatters src elements to dst at uint32 indices for float32 with strided memory
func ScatterStridedU32F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterU32F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] = src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterStridedU32F64 scatters src elements to dst at uint32 indices for float64 with strided memory
func ScatterStridedU32F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterU32F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] = src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterStridedU8F32 scatters src elements to dst at uint8 indices for float32 with strided memory
func ScatterStridedU8F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterU8F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] = src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterStridedU8F64 scatters src elements to dst at uint8 indices for float64 with strided memory
func ScatterStridedU8F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterU8F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] = src[GetStridedIndex(i, numDims, dims, stridesSrc)]
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

// ScatterAddStrided adds src elements to dst at indices of type I for data of type T with strided memory
func ScatterAddStrided[U I, T D](numel, numDims int, dims, stridesSrc, stridesDst []int, ids []U, src, dst []T) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterAdd(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterAddStridedI64F32 adds src elements to dst at int64 indices for float32 with strided memory
func ScatterAddStridedI64F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterAddI64F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterAddStridedI64F64 adds src elements to dst at int64 indices for float64 with strided memory
func ScatterAddStridedI64F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []int64, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterAddI64F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterAddStridedU32F32 adds src elements to dst at uint32 indices for float32 with strided memory
func ScatterAddStridedU32F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterAddU32F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterAddStridedU32F64 adds src elements to dst at uint32 indices for float64 with strided memory
func ScatterAddStridedU32F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint32, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterAddU32F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterAddStridedU8F32 adds src elements to dst at uint8 indices for float32 with strided memory
func ScatterAddStridedU8F32(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float32) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterAddU8F32(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

// ScatterAddStridedU8F64 adds src elements to dst at uint8 indices for float64 with strided memory
func ScatterAddStridedU8F64(numel, numDims int, dims, stridesSrc, stridesDst []int, ids []uint8, src, dst []float64) {
	if IsContiguous(numDims, dims, stridesSrc) && IsContiguous(numDims, dims, stridesDst) {
		ScatterAddU8F64(numel, ids, src, dst)
		return
	}
	for i := range numel {
		dst[GetStridedIndex(int(ids[i]), numDims, dims, stridesDst)] += src[GetStridedIndex(i, numDims, dims, stridesSrc)]
	}
}

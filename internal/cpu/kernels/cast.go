package kernels

// CastF32F32 converts float32 to float32
func CastF32F32(numel int, x []float32, y []float32) {
	for i := range numel {
		y[i] = x[i]
	}
}

// CastF32F64 converts float32 to float64
func CastF32F64(numel int, x []float32, y []float64) {
	for i := range numel {
		y[i] = float64(x[i])
	}
}

// CastF32U8 converts float32 to uint8
func CastF32U8(numel int, x []float32, y []uint8) {
	for i := range numel {
		y[i] = uint8(x[i])
	}
}

// CastF32U32 converts float32 to uint32
func CastF32U32(numel int, x []float32, y []uint32) {
	for i := range numel {
		y[i] = uint32(x[i])
	}
}

// CastF32I64 converts float32 to int64
func CastF32I64(numel int, x []float32, y []int64) {
	for i := range numel {
		y[i] = int64(x[i])
	}
}

// CastStridedF32F32 converts float32 to float32 with strided memory
func CastStridedF32F32(numel, ndims int, dims, stridesX, stridesY []int, x, y []float32) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastF32F32(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x[GetStridedIndex(i, ndims, dims, stridesX)]
	}
}

// CastStridedF32F64 converts float32 to float64 with strided memory
func CastStridedF32F64(numel, ndims int, dims, stridesX, stridesY []int, x []float32, y []float64) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastF32F64(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = float64(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedF32U8 converts float32 to uint8 with strided memory
func CastStridedF32U8(numel, ndims int, dims, stridesX, stridesY []int, x []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastF32U8(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = uint8(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedF32U32 converts float32 to uint32 with strided memory
func CastStridedF32U32(numel, ndims int, dims, stridesX, stridesY []int, x []float32, y []uint32) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastF32U32(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = uint32(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedF32I64 converts float32 to int64 with strided memory
func CastStridedF32I64(numel, ndims int, dims, stridesX, stridesY []int, x []float32, y []int64) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastF32I64(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = int64(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastF64F32 converts float64 to float32
func CastF64F32(numel int, x []float64, y []float32) {
	for i := range numel {
		y[i] = float32(x[i])
	}
}

// CastF64F64 converts float64 to float64
func CastF64F64(numel int, x []float64, y []float64) {
	for i := range numel {
		y[i] = x[i]
	}
}

// CastF64U8 converts float64 to uint8
func CastF64U8(numel int, x []float64, y []uint8) {
	for i := range numel {
		y[i] = uint8(x[i])
	}
}

// CastF64U32 converts float64 to uint32
func CastF64U32(numel int, x []float64, y []uint32) {
	for i := range numel {
		y[i] = uint32(x[i])
	}
}

// CastF64I64 converts float64 to int64
func CastF64I64(numel int, x []float64, y []int64) {
	for i := range numel {
		y[i] = int64(x[i])
	}
}

// CastStridedF64F32 converts float64 to float32 with strided memory
func CastStridedF64F32(numel, ndims int, dims, stridesX, stridesY []int, x []float64, y []float32) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastF64F32(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = float32(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedF64F64 converts float64 to float64 with strided memory
func CastStridedF64F64(numel, ndims int, dims, stridesX, stridesY []int, x, y []float64) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastF64F64(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x[GetStridedIndex(i, ndims, dims, stridesX)]
	}
}

// CastStridedF64U8 converts float64 to uint8 with strided memory
func CastStridedF64U8(numel, ndims int, dims, stridesX, stridesY []int, x []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastF64U8(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = uint8(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedF64U32 converts float64 to uint32 with strided memory
func CastStridedF64U32(numel, ndims int, dims, stridesX, stridesY []int, x []float64, y []uint32) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastF64U32(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = uint32(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedF64I64 converts float64 to int64 with strided memory
func CastStridedF64I64(numel, ndims int, dims, stridesX, stridesY []int, x []float64, y []int64) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastF64I64(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = int64(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastU8F32 converts uint8 to float32
func CastU8F32(numel int, x []uint8, y []float32) {
	for i := range numel {
		y[i] = float32(x[i])
	}
}

// CastU8F64 converts uint8 to float64
func CastU8F64(numel int, x []uint8, y []float64) {
	for i := range numel {
		y[i] = float64(x[i])
	}
}

// CastU8U8 converts uint8 to uint8
func CastU8U8(numel int, x []uint8, y []uint8) {
	for i := range numel {
		y[i] = x[i]
	}
}

// CastU8U32 converts uint8 to uint32
func CastU8U32(numel int, x []uint8, y []uint32) {
	for i := range numel {
		y[i] = uint32(x[i])
	}
}

// CastU8I64 converts uint8 to int64
func CastU8I64(numel int, x []uint8, y []int64) {
	for i := range numel {
		y[i] = int64(x[i])
	}
}

// CastStridedU8F32 converts uint8 to float32 with strided memory
func CastStridedU8F32(numel, ndims int, dims, stridesX, stridesY []int, x []uint8, y []float32) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastU8F32(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = float32(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedU8F64 converts uint8 to float64 with strided memory
func CastStridedU8F64(numel, ndims int, dims, stridesX, stridesY []int, x []uint8, y []float64) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastU8F64(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = float64(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedU8U8 converts uint8 to uint8 with strided memory
func CastStridedU8U8(numel, ndims int, dims, stridesX, stridesY []int, x []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastU8U8(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x[GetStridedIndex(i, ndims, dims, stridesX)]
	}
}

// CastStridedU8U32 converts uint8 to uint32 with strided memory
func CastStridedU8U32(numel, ndims int, dims, stridesX, stridesY []int, x []uint8, y []uint32) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastU8U32(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = uint32(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedU8I64 converts uint8 to int64 with strided memory
func CastStridedU8I64(numel, ndims int, dims, stridesX, stridesY []int, x []uint8, y []int64) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastU8I64(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = int64(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastU32F32 converts uint32 to float32
func CastU32F32(numel int, x []uint32, y []float32) {
	for i := range numel {
		y[i] = float32(x[i])
	}
}

// CastU32F64 converts uint32 to float64
func CastU32F64(numel int, x []uint32, y []float64) {
	for i := range numel {
		y[i] = float64(x[i])
	}
}

// CastU32U8 converts uint32 to uint8
func CastU32U8(numel int, x []uint32, y []uint8) {
	for i := range numel {
		y[i] = uint8(x[i])
	}
}

// CastU32U32 converts uint32 to uint32
func CastU32U32(numel int, x []uint32, y []uint32) {
	for i := range numel {
		y[i] = x[i]
	}
}

// CastU32I64 converts uint32 to int64
func CastU32I64(numel int, x []uint32, y []int64) {
	for i := range numel {
		y[i] = int64(x[i])
	}
}

// CastStridedU32F32 converts uint32 to float32 with strided memory
func CastStridedU32F32(numel, ndims int, dims, stridesX, stridesY []int, x []uint32, y []float32) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastU32F32(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = float32(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedU32F64 converts uint32 to float64 with strided memory
func CastStridedU32F64(numel, ndims int, dims, stridesX, stridesY []int, x []uint32, y []float64) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastU32F64(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = float64(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedU32U8 converts uint32 to uint8 with strided memory
func CastStridedU32U8(numel, ndims int, dims, stridesX, stridesY []int, x []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastU32U8(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = uint8(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedU32U32 converts uint32 to uint32 with strided memory
func CastStridedU32U32(numel, ndims int, dims, stridesX, stridesY []int, x []uint32, y []uint32) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastU32U32(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x[GetStridedIndex(i, ndims, dims, stridesX)]
	}
}

// CastStridedU32I64 converts uint32 to int64 with strided memory
func CastStridedU32I64(numel, ndims int, dims, stridesX, stridesY []int, x []uint32, y []int64) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastU32I64(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = int64(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastI64F32 converts int64 to float32
func CastI64F32(numel int, x []int64, y []float32) {
	for i := range numel {
		y[i] = float32(x[i])
	}
}

// CastI64F64 converts int64 to float64
func CastI64F64(numel int, x []int64, y []float64) {
	for i := range numel {
		y[i] = float64(x[i])
	}
}

// CastI64U8 converts int64 to uint8
func CastI64U8(numel int, x []int64, y []uint8) {
	for i := range numel {
		y[i] = uint8(x[i])
	}
}

// CastI64U32 converts int64 to uint32
func CastI64U32(numel int, x []int64, y []uint32) {
	for i := range numel {
		y[i] = uint32(x[i])
	}
}

// CastI64I64 converts int64 to int64
func CastI64I64(numel int, x []int64, y []int64) {
	for i := range numel {
		y[i] = x[i]
	}
}

// CastStridedI64F32 converts int64 to float32 with strided memory
func CastStridedI64F32(numel, ndims int, dims, stridesX, stridesY []int, x []int64, y []float32) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastI64F32(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = float32(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedI64F64 converts int64 to float64 with strided memory
func CastStridedI64F64(numel, ndims int, dims, stridesX, stridesY []int, x []int64, y []float64) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastI64F64(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = float64(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedI64U8 converts int64 to uint8 with strided memory
func CastStridedI64U8(numel, ndims int, dims, stridesX, stridesY []int, x []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastI64U8(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = uint8(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedI64U32 converts int64 to uint32 with strided memory
func CastStridedI64U32(numel, ndims int, dims, stridesX, stridesY []int, x []int64, y []uint32) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastI64U32(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = uint32(x[GetStridedIndex(i, ndims, dims, stridesX)])
	}
}

// CastStridedI64I64 converts int64 to int64 with strided memory
func CastStridedI64I64(numel, ndims int, dims, stridesX, stridesY []int, x []int64, y []int64) {
	if IsContiguous(ndims, dims, stridesX) && IsContiguous(ndims, dims, stridesY) {
		CastI64I64(numel, x, y)
		return
	}
	for i := range numel {
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x[GetStridedIndex(i, ndims, dims, stridesX)]
	}
}

package kernels

// BAdd performs y = x1 + x2 for any supported numeric type
func BAdd[T D](numel int, x1, x2, y []T) {
	for i := range numel {
		y[i] = x1[i] + x2[i]
	}
}

// BAddF32 performs y = x1 + x2 for float32
func BAddF32(numel int, x1, x2, y []float32) {
	for i := range numel {
		y[i] = x1[i] + x2[i]
	}
}

// BAddF64 performs y = x1 + x2 for float64
func BAddF64(numel int, x1, x2, y []float64) {
	for i := range numel {
		y[i] = x1[i] + x2[i]
	}
}

// BAddU8 performs y = x1 + x2 for uint8
func BAddU8(numel int, x1, x2, y []uint8) {
	for i := range numel {
		y[i] = x1[i] + x2[i]
	}
}

// BAddU32 performs y = x1 + x2 for uint32
func BAddU32(numel int, x1, x2, y []uint32) {
	for i := range numel {
		y[i] = x1[i] + x2[i]
	}
}

// BAddI64 performs y = x1 + x2 for int64
func BAddI64(numel int, x1, x2, y []int64) {
	for i := range numel {
		y[i] = x1[i] + x2[i]
	}
}

// BAddStrided performs y = x1 + x2 for any supported numeric type with strided memory
func BAddStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BAdd(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] + x2[idx2]
	}
}

// BAddStridedF32 performs y = x1 + x2 for float32 with strided memory
func BAddStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BAddF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] + x2[idx2]
	}
}

// BAddStridedF64 performs y = x1 + x2 for float64 with strided memory
func BAddStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BAddF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] + x2[idx2]
	}
}

// BAddStridedU8 performs y = x1 + x2 for uint8 with strided memory
func BAddStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BAddU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] + x2[idx2]
	}
}

// BAddStridedU32 performs y = x1 + x2 for uint32 with strided memory
func BAddStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BAddU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] + x2[idx2]
	}
}

// BAddStridedI64 performs y = x1 + x2 for int64 with strided memory
func BAddStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BAddI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] + x2[idx2]
	}
}

// BSub performs y = x1 - x2 for any supported numeric type
func BSub[T D](numel int, x1, x2, y []T) {
	for i := range numel {
		y[i] = x1[i] - x2[i]
	}
}

// BSubF32 performs y = x1 - x2 for float32
func BSubF32(numel int, x1, x2, y []float32) {
	for i := range numel {
		y[i] = x1[i] - x2[i]
	}
}

// BSubF64 performs y = x1 - x2 for float64
func BSubF64(numel int, x1, x2, y []float64) {
	for i := range numel {
		y[i] = x1[i] - x2[i]
	}
}

// BSubU8 performs y = x1 - x2 for uint8
func BSubU8(numel int, x1, x2, y []uint8) {
	for i := range numel {
		y[i] = x1[i] - x2[i]
	}
}

// BSubU32 performs y = x1 - x2 for uint32
func BSubU32(numel int, x1, x2, y []uint32) {
	for i := range numel {
		y[i] = x1[i] - x2[i]
	}
}

// BSubI64 performs y = x1 - x2 for int64
func BSubI64(numel int, x1, x2, y []int64) {
	for i := range numel {
		y[i] = x1[i] - x2[i]
	}
}

// BSubStrided performs y = x1 - x2 for any supported numeric type with strided memory
func BSubStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BSub(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] - x2[idx2]
	}
}

// BSubStridedF32 performs y = x1 - x2 for float32 with strided memory
func BSubStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BSubF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] - x2[idx2]
	}
}

// BSubStridedF64 performs y = x1 - x2 for float64 with strided memory
func BSubStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BSubF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] - x2[idx2]
	}
}

// BSubStridedU8 performs y = x1 - x2 for uint8 with strided memory
func BSubStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BSubU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] - x2[idx2]
	}
}

// BSubStridedU32 performs y = x1 - x2 for uint32 with strided memory
func BSubStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BSubU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] - x2[idx2]
	}
}

// BSubStridedI64 performs y = x1 - x2 for int64 with strided memory
func BSubStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BSubI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] - x2[idx2]
	}
}

// BMul performs y = x1 * x2 for any supported numeric type
func BMul[T D](numel int, x1, x2, y []T) {
	for i := range numel {
		y[i] = x1[i] * x2[i]
	}
}

// BMulF32 performs y = x1 * x2 for float32
func BMulF32(numel int, x1, x2, y []float32) {
	for i := range numel {
		y[i] = x1[i] * x2[i]
	}
}

// BMulF64 performs y = x1 * x2 for float64
func BMulF64(numel int, x1, x2, y []float64) {
	for i := range numel {
		y[i] = x1[i] * x2[i]
	}
}

// BMulU8 performs y = x1 * x2 for uint8
func BMulU8(numel int, x1, x2, y []uint8) {
	for i := range numel {
		y[i] = x1[i] * x2[i]
	}
}

// BMulU32 performs y = x1 * x2 for uint32
func BMulU32(numel int, x1, x2, y []uint32) {
	for i := range numel {
		y[i] = x1[i] * x2[i]
	}
}

// BMulI64 performs y = x1 * x2 for int64
func BMulI64(numel int, x1, x2, y []int64) {
	for i := range numel {
		y[i] = x1[i] * x2[i]
	}
}

// BMulStrided performs y = x1 * x2 for any supported numeric type with strided memory
func BMulStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMul(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] * x2[idx2]
	}
}

// BMulStridedF32 performs y = x1 * x2 for float32 with strided memory
func BMulStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMulF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] * x2[idx2]
	}
}

// BMulStridedF64 performs y = x1 * x2 for float64 with strided memory
func BMulStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMulF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] * x2[idx2]
	}
}

// BMulStridedU8 performs y = x1 * x2 for uint8 with strided memory
func BMulStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMulU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] * x2[idx2]
	}
}

// BMulStridedU32 performs y = x1 * x2 for uint32 with strided memory
func BMulStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMulU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] * x2[idx2]
	}
}

// BMulStridedI64 performs y = x1 * x2 for int64 with strided memory
func BMulStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMulI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] * x2[idx2]
	}
}

// BDiv performs y = x1 / x2 for any supported numeric type
func BDiv[T D](numel int, x1, x2, y []T) {
	for i := range numel {
		if x2[i] != 0 {
			y[i] = x1[i] / x2[i]
		} else {
			y[i] = 0 // Handle division by zero
		}
	}
}

// BDivF32 performs y = x1 / x2 for float32
func BDivF32(numel int, x1, x2, y []float32) {
	for i := range numel {
		if x2[i] != 0 {
			y[i] = x1[i] / x2[i]
		} else {
			y[i] = 0 // Handle division by zero
		}
	}
}

// BDivF64 performs y = x1 / x2 for float64
func BDivF64(numel int, x1, x2, y []float64) {
	for i := range numel {
		if x2[i] != 0 {
			y[i] = x1[i] / x2[i]
		} else {
			y[i] = 0 // Handle division by zero
		}
	}
}

// BDivU8 performs y = x1 / x2 for uint8
func BDivU8(numel int, x1, x2, y []uint8) {
	for i := range numel {
		if x2[i] != 0 {
			y[i] = x1[i] / x2[i]
		} else {
			y[i] = 0 // Handle division by zero
		}
	}
}

// BDivU32 performs y = x1 / x2 for uint32
func BDivU32(numel int, x1, x2, y []uint32) {
	for i := range numel {
		if x2[i] != 0 {
			y[i] = x1[i] / x2[i]
		} else {
			y[i] = 0 // Handle division by zero
		}
	}
}

// BDivI64 performs y = x1 / x2 for int64
func BDivI64(numel int, x1, x2, y []int64) {
	for i := range numel {
		if x2[i] != 0 {
			y[i] = x1[i] / x2[i]
		} else {
			y[i] = 0 // Handle division by zero
		}
	}
}

// BDivStrided performs y = x1 / x2 for any supported numeric type with strided memory
func BDivStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BDiv(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x2[idx2] != 0 {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] / x2[idx2]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0 // Handle division by zero
		}
	}
}

// BDivStridedF32 performs y = x1 / x2 for float32 with strided memory
func BDivStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BDivF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x2[idx2] != 0 {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] / x2[idx2]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0 // Handle division by zero
		}
	}
}

// BDivStridedF64 performs y = x1 / x2 for float64 with strided memory
func BDivStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BDivF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x2[idx2] != 0 {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] / x2[idx2]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0 // Handle division by zero
		}
	}
}

// BDivStridedU8 performs y = x1 / x2 for uint8 with strided memory
func BDivStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BDivU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x2[idx2] != 0 {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] / x2[idx2]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0 // Handle division by zero
		}
	}
}

// BDivStridedU32 performs y = x1 / x2 for uint32 with strided memory
func BDivStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BDivU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x2[idx2] != 0 {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] / x2[idx2]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0 // Handle division by zero
		}
	}
}

// BDivStridedI64 performs y = x1 / x2 for int64 with strided memory
func BDivStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BDivI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x2[idx2] != 0 {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1] / x2[idx2]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0 // Handle division by zero
		}
	}
}

// BMax performs y = max(x1, x2) for any supported numeric type
func BMax[T D](numel int, x1, x2, y []T) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = x1[i]
		} else {
			y[i] = x2[i]
		}
	}
}

// BMaximumF32 performs y = max(x1, x2) for float32
func BMaximumF32(numel int, x1, x2, y []float32) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = x1[i]
		} else {
			y[i] = x2[i]
		}
	}
}

// BMaximumF64 performs y = max(x1, x2) for float64
func BMaximumF64(numel int, x1, x2, y []float64) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = x1[i]
		} else {
			y[i] = x2[i]
		}
	}
}

// BMaximumU8 performs y = max(x1, x2) for uint8
func BMaximumU8(numel int, x1, x2, y []uint8) {
	for i := range numel {
		y[i] = max(x1[i], x2[i])
	}
}

// BMaximumU32 performs y = max(x1, x2) for uint32
func BMaximumU32(numel int, x1, x2, y []uint32) {
	for i := range numel {
		y[i] = max(x1[i], x2[i])
	}
}

// BMaximumI64 performs y = max(x1, x2) for int64
func BMaximumI64(numel int, x1, x2, y []int64) {
	for i := range numel {
		y[i] = max(x1[i], x2[i])
	}
}

// BMaxStrided performs y = max(x1, x2) for any supported numeric type with strided memory
func BMaxStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMax(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMaximumStridedF32 performs y = max(x1, x2) for float32 with strided memory
func BMaximumStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMaximumF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMaximumStridedF64 performs y = max(x1, x2) for float64 with strided memory
func BMaximumStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMaximumF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMaximumStridedU8 performs y = max(x1, x2) for uint8 with strided memory
func BMaximumStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMaximumU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMaximumStridedU32 performs y = max(x1, x2) for uint32 with strided memory
func BMaximumStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMaximumU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMaximumStridedI64 performs y = max(x1, x2) for int64 with strided memory
func BMaximumStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMaximumI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMin performs y = min(x1, x2) for any supported numeric type
func BMin[T D](numel int, x1, x2, y []T) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = x1[i]
		} else {
			y[i] = x2[i]
		}
	}
}

// BMinimumF32 performs y = min(x1, x2) for float32
func BMinimumF32(numel int, x1, x2, y []float32) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = x1[i]
		} else {
			y[i] = x2[i]
		}
	}
}

// BMinimumF64 performs y = min(x1, x2) for float64
func BMinimumF64(numel int, x1, x2, y []float64) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = x1[i]
		} else {
			y[i] = x2[i]
		}
	}
}

// BMinimumU8 performs y = min(x1, x2) for uint8
func BMinimumU8(numel int, x1, x2, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = x1[i]
		} else {
			y[i] = x2[i]
		}
	}
}

// BMinimumU32 performs y = min(x1, x2) for uint32
func BMinimumU32(numel int, x1, x2, y []uint32) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = x1[i]
		} else {
			y[i] = x2[i]
		}
	}
}

// BMinimumI64 performs y = min(x1, x2) for int64
func BMinimumI64(numel int, x1, x2, y []int64) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = x1[i]
		} else {
			y[i] = x2[i]
		}
	}
}

// BMinStrided performs y = min(x1, x2) for any supported numeric type with strided memory
func BMinStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMin(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMinimumStridedF32 performs y = min(x1, x2) for float32 with strided memory
func BMinimumStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMinimumF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMinimumStridedF64 performs y = min(x1, x2) for float64 with strided memory
func BMinimumStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMinimumF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMinimumStridedU8 performs y = min(x1, x2) for uint8 with strided memory
func BMinimumStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMinimumU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMinimumStridedU32 performs y = min(x1, x2) for uint32 with strided memory
func BMinimumStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMinimumU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMinimumStridedI64 performs y = min(x1, x2) for int64 with strided memory
func BMinimumStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMinimumI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = x2[idx2]
		}
	}
}

// Eq performs y = (x1 == x2) ? 1 : 0 for any supported numeric type
func Eq[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqF32 performs y = (x1 == x2) ? 1 : 0 for float32
func EqF32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqF64 performs y = (x1 == x2) ? 1 : 0 for float64
func EqF64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqU8 performs y = (x1 == x2) ? 1 : 0 for uint8
func EqU8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqU32 performs y = (x1 == x2) ? 1 : 0 for uint32
func EqU32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqI64 performs y = (x1 == x2) ? 1 : 0 for int64
func EqI64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqStrided performs y = (x1 == x2) ? 1 : 0 for any supported numeric type with strided memory
func EqStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		Eq(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] == x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// EqStridedF32 performs y = (x1 == x2) ? 1 : 0 for float32 with strided memory
func EqStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] == x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// EqStridedF64 performs y = (x1 == x2) ? 1 : 0 for float64 with strided memory
func EqStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] == x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// EqStridedU8 performs y = (x1 == x2) ? 1 : 0 for uint8 with strided memory
func EqStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] == x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// EqStridedU32 performs y = (x1 == x2) ? 1 : 0 for uint32 with strided memory
func EqStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] == x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// EqStridedI64 performs y = (x1 == x2) ? 1 : 0 for int64 with strided memory
func EqStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] == x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// Ne performs y = (x1 != x2) ? 1 : 0 for any supported numeric type
func Ne[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeF32 performs y = (x1 != x2) ? 1 : 0 for float32
func NeF32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeF64 performs y = (x1 != x2) ? 1 : 0 for float64
func NeF64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeU8 performs y = (x1 != x2) ? 1 : 0 for uint8
func NeU8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeU32 performs y = (x1 != x2) ? 1 : 0 for uint32
func NeU32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeI64 performs y = (x1 != x2) ? 1 : 0 for int64
func NeI64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeStrided performs y = (x1 != x2) ? 1 : 0 for any supported numeric type with strided memory
func NeStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		Ne(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] != x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// NeStridedF32 performs y = (x1 != x2) ? 1 : 0 for float32 with strided memory
func NeStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] != x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// NeStridedF64 performs y = (x1 != x2) ? 1 : 0 for float64 with strided memory
func NeStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] != x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// NeStridedU8 performs y = (x1 != x2) ? 1 : 0 for uint8 with strided memory
func NeStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] != x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// NeStridedU32 performs y = (x1 != x2) ? 1 : 0 for uint32 with strided memory
func NeStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] != x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// NeStridedI64 performs y = (x1 != x2) ? 1 : 0 for int64 with strided memory
func NeStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] != x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// Lt performs y = (x1 < x2) ? 1 : 0 for any supported numeric type
func Lt[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtF32 performs y = (x1 < x2) ? 1 : 0 for float32
func LtF32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtF64 performs y = (x1 < x2) ? 1 : 0 for float64
func LtF64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtU8 performs y = (x1 < x2) ? 1 : 0 for uint8
func LtU8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtU32 performs y = (x1 < x2) ? 1 : 0 for uint32
func LtU32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtI64 performs y = (x1 < x2) ? 1 : 0 for int64
func LtI64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtStrided performs y = (x1 < x2) ? 1 : 0 for any supported numeric type with strided memory
func LtStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		Lt(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// LtStridedF32 performs y = (x1 < x2) ? 1 : 0 for float32 with strided memory
func LtStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// LtStridedF64 performs y = (x1 < x2) ? 1 : 0 for float64 with strided memory
func LtStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// LtStridedU8 performs y = (x1 < x2) ? 1 : 0 for uint8 with strided memory
func LtStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// LtStridedU32 performs y = (x1 < x2) ? 1 : 0 for uint32 with strided memory
func LtStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// LtStridedI64 performs y = (x1 < x2) ? 1 : 0 for int64 with strided memory
func LtStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// Le performs y = (x1 <= x2) ? 1 : 0 for any supported numeric type
func Le[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeF32 performs y = (x1 <= x2) ? 1 : 0 for float32
func LeF32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeF64 performs y = (x1 <= x2) ? 1 : 0 for float64
func LeF64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeU8 performs y = (x1 <= x2) ? 1 : 0 for uint8
func LeU8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeU32 performs y = (x1 <= x2) ? 1 : 0 for uint32
func LeU32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeI64 performs y = (x1 <= x2) ? 1 : 0 for int64
func LeI64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeStrided performs y = (x1 <= x2) ? 1 : 0 for any supported numeric type with strided memory
func LeStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		Le(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] <= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// LeStridedF32 performs y = (x1 <= x2) ? 1 : 0 for float32 with strided memory
func LeStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] <= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// LeStridedF64 performs y = (x1 <= x2) ? 1 : 0 for float64 with strided memory
func LeStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] <= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// LeStridedU8 performs y = (x1 <= x2) ? 1 : 0 for uint8 with strided memory
func LeStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] <= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// LeStridedU32 performs y = (x1 <= x2) ? 1 : 0 for uint32 with strided memory
func LeStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] <= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// LeStridedI64 performs y = (x1 <= x2) ? 1 : 0 for int64 with strided memory
func LeStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] <= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// Gt performs y = (x1 > x2) ? 1 : 0 for any supported numeric type
func Gt[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtF32 performs y = (x1 > x2) ? 1 : 0 for float32
func GtF32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtF64 performs y = (x1 > x2) ? 1 : 0 for float64
func GtF64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtU8 performs y = (x1 > x2) ? 1 : 0 for uint8
func GtU8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtU32 performs y = (x1 > x2) ? 1 : 0 for uint32
func GtU32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtI64 performs y = (x1 > x2) ? 1 : 0 for int64
func GtI64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtStrided performs y = (x1 > x2) ? 1 : 0 for any supported numeric type with strided memory
func GtStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		Gt(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// GtStridedF32 performs y = (x1 > x2) ? 1 : 0 for float32 with strided memory
func GtStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// GtStridedF64 performs y = (x1 > x2) ? 1 : 0 for float64 with strided memory
func GtStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// GtStridedU8 performs y = (x1 > x2) ? 1 : 0 for uint8 with strided memory
func GtStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// GtStridedU32 performs y = (x1 > x2) ? 1 : 0 for uint32 with strided memory
func GtStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// GtStridedI64 performs y = (x1 > x2) ? 1 : 0 for int64 with strided memory
func GtStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// Ge performs y = (x1 >= x2) ? 1 : 0 for any supported numeric type
func Ge[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeF32 performs y = (x1 >= x2) ? 1 : 0 for float32
func GeF32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeF64 performs y = (x1 >= x2) ? 1 : 0 for float64
func GeF64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeU8 performs y = (x1 >= x2) ? 1 : 0 for uint8
func GeU8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeU32 performs y = (x1 >= x2) ? 1 : 0 for uint32
func GeU32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeI64 performs y = (x1 >= x2) ? 1 : 0 for int64
func GeI64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeStrided performs y = (x1 >= x2) ? 1 : 0 for any supported numeric type with strided memory
func GeStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		Ge(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] >= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// GeStridedF32 performs y = (x1 >= x2) ? 1 : 0 for float32 with strided memory
func GeStridedF32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] >= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// GeStridedF64 performs y = (x1 >= x2) ? 1 : 0 for float64 with strided memory
func GeStridedF64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] >= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// GeStridedU8 performs y = (x1 >= x2) ? 1 : 0 for uint8 with strided memory
func GeStridedU8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeU8(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] >= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// GeStridedU32 performs y = (x1 >= x2) ? 1 : 0 for uint32 with strided memory
func GeStridedU32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeU32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] >= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

// GeStridedI64 performs y = (x1 >= x2) ? 1 : 0 for int64 with strided memory
func GeStridedI64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeI64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		if x1[idx1] >= x2[idx2] {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, ndims, dims, stridesY)] = 0
		}
	}
}

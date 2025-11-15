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
func BMaximum[T D](numel int, x1, x2, y []T) {
	for i := range numel {
		y[i] = max(x1[i], x2[i])
	}
}

// BMaximumF32 performs y = max(x1, x2) for float32
func BMaximumF32(numel int, x1, x2, y []float32) {
	for i := range numel {
		y[i] = max(x1[i], x2[i])
	}
}

// BMaximumF64 performs y = max(x1, x2) for float64
func BMaximumF64(numel int, x1, x2, y []float64) {
	for i := range numel {
		y[i] = max(x1[i], x2[i])
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
func BMaximumStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMaximum(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = max(x1[idx1], x2[idx2])
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
		y[GetStridedIndex(i, ndims, dims, stridesY)] = max(x1[idx1], x2[idx2])
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
		y[GetStridedIndex(i, ndims, dims, stridesY)] = max(x1[idx1], x2[idx2])
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
		y[GetStridedIndex(i, ndims, dims, stridesY)] = max(x1[idx1], x2[idx2])
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
		y[GetStridedIndex(i, ndims, dims, stridesY)] = max(x1[idx1], x2[idx2])
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
		y[GetStridedIndex(i, ndims, dims, stridesY)] = max(x1[idx1], x2[idx2])
	}
}

// BMin performs y = min(x1, x2) for any supported numeric type
func BMinimum[T D](numel int, x1, x2, y []T) {
	for i := range numel {
		y[i] = min(x1[i], x2[i])
	}
}

// BMinimumF32 performs y = min(x1, x2) for float32
func BMinimumF32(numel int, x1, x2, y []float32) {
	for i := range numel {
		y[i] = min(x1[i], x2[i])
	}
}

// BMinimumF64 performs y = min(x1, x2) for float64
func BMinimumF64(numel int, x1, x2, y []float64) {
	for i := range numel {
		y[i] = min(x1[i], x2[i])
	}
}

// BMinimumU8 performs y = min(x1, x2) for uint8
func BMinimumU8(numel int, x1, x2, y []uint8) {
	for i := range numel {
		y[i] = min(x1[i], x2[i])
	}
}

// BMinimumU32 performs y = min(x1, x2) for uint32
func BMinimumU32(numel int, x1, x2, y []uint32) {
	for i := range numel {
		y[i] = min(x1[i], x2[i])
	}
}

// BMinimumI64 performs y = min(x1, x2) for int64
func BMinimumI64(numel int, x1, x2, y []int64) {
	for i := range numel {
		y[i] = min(x1[i], x2[i])
	}
}

// BMinStrided performs y = min(x1, x2) for any supported numeric type with strided memory
func BMinimumStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		BMinimum(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, ndims, dims, stridesX1)
		idx2 := GetStridedIndex(i, ndims, dims, stridesX2)
		y[GetStridedIndex(i, ndims, dims, stridesY)] = min(x1[idx1], x2[idx2])
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
		y[GetStridedIndex(i, ndims, dims, stridesY)] = min(x1[idx1], x2[idx2])
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
		y[GetStridedIndex(i, ndims, dims, stridesY)] = min(x1[idx1], x2[idx2])
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
		y[GetStridedIndex(i, ndims, dims, stridesY)] = min(x1[idx1], x2[idx2])
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
		y[GetStridedIndex(i, ndims, dims, stridesY)] = min(x1[idx1], x2[idx2])
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
		y[GetStridedIndex(i, ndims, dims, stridesY)] = min(x1[idx1], x2[idx2])
	}
}

// Eq performs y = (x1 == x2) ? 1 : 0 for any supported numeric type
func Eq[T D](numel int, x1, x2 []T, y []T) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqF32F32 performs y = (x1 == x2) ? 1 : 0 for float32
func EqF32F32(numel int, x1, x2 []float32, y []float32) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqF64F64 performs y = (x1 == x2) ? 1 : 0 for float64
func EqF64F64(numel int, x1, x2 []float64, y []float64) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqU32U32 performs y = (x1 == x2) ? 1 : 0 for uint32
func EqU32U32(numel int, x1, x2 []uint32, y []uint32) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqI64I64 performs y = (x1 == x2) ? 1 : 0 for int64
func EqI64I64(numel int, x1, x2 []int64, y []int64) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqU8 performs y = (x1 == x2) ? 1 : 0 for any supported numeric type with uint8 output
func EqU8[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqU8F32 performs y = (x1 == x2) ? 1 : 0 for float32
func EqU8F32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqU8F64 performs y = (x1 == x2) ? 1 : 0 for float64
func EqU8F64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqU8U8 performs y = (x1 == x2) ? 1 : 0 for uint8
func EqU8U8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqU8U32 performs y = (x1 == x2) ? 1 : 0 for uint32
func EqU8U32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqU8I64 performs y = (x1 == x2) ? 1 : 0 for int64
func EqU8I64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] == x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// EqStrided performs y = (x1 == x2) ? 1 : 0 for any supported numeric type with strided memory
func EqStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []T) {
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

// EqStridedF32F32 performs y = (x1 == x2) ? 1 : 0 for float32 with strided memory
func EqStridedF32F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqF32F32(numel, x1, x2, y)
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

// EqStridedF64F64 performs y = (x1 == x2) ? 1 : 0 for float64 with strided memory
func EqStridedF64F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqF64F64(numel, x1, x2, y)
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

// EqStridedU32U32 performs y = (x1 == x2) ? 1 : 0 for uint32 with strided memory
func EqStridedU32U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqU32U32(numel, x1, x2, y)
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

// EqStridedI64I64 performs y = (x1 == x2) ? 1 : 0 for int64 with strided memory
func EqStridedI64I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqI64I64(numel, x1, x2, y)
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

// EqStridedU8 performs y = (x1 == x2) ? 1 : 0 for any supported numeric type with uint8 output with strided memory
func EqStridedU8[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
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

// EqStridedU8F32 performs y = (x1 == x2) ? 1 : 0 for float32 with strided memory
func EqStridedU8F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqU8F32(numel, x1, x2, y)
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

// EqStridedU8F64 performs y = (x1 == x2) ? 1 : 0 for float64 with strided memory
func EqStridedU8F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqU8F64(numel, x1, x2, y)
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

// EqStridedU8U8 performs y = (x1 == x2) ? 1 : 0 for uint8 with strided memory
func EqStridedU8U8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqU8U8(numel, x1, x2, y)
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

// EqStridedU8U32 performs y = (x1 == x2) ? 1 : 0 for uint32 with strided memory
func EqStridedU8U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqU8U32(numel, x1, x2, y)
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

// EqStridedU8I64 performs y = (x1 == x2) ? 1 : 0 for int64 with strided memory
func EqStridedU8I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		EqU8I64(numel, x1, x2, y)
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

// Ne performs y = (x1 != x2) ? 1 : 0 for any supported numeric type with same-type output
func Ne[T D](numel int, x1, x2 []T, y []T) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeF32F32 performs y = (x1 != x2) ? 1 : 0 for float32 with float32 output
func NeF32F32(numel int, x1, x2 []float32, y []float32) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeF64F64 performs y = (x1 != x2) ? 1 : 0 for float64 with float64 output
func NeF64F64(numel int, x1, x2 []float64, y []float64) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeU32U32 performs y = (x1 != x2) ? 1 : 0 for uint32 with uint32 output
func NeU32U32(numel int, x1, x2 []uint32, y []uint32) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeI64I64 performs y = (x1 != x2) ? 1 : 0 for int64 with int64 output
func NeI64I64(numel int, x1, x2 []int64, y []int64) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeU8 performs y = (x1 != x2) ? 1 : 0 for any supported numeric type with uint8 output
func NeU8[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeU8F32 performs y = (x1 != x2) ? 1 : 0 for float32 with uint8 output
func NeU8F32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeU8F64 performs y = (x1 != x2) ? 1 : 0 for float64 with uint8 output
func NeU8F64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeU8U8 performs y = (x1 != x2) ? 1 : 0 for uint8 with uint8 output
func NeU8U8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeU8U32 performs y = (x1 != x2) ? 1 : 0 for uint32 with uint8 output
func NeU8U32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeU8I64 performs y = (x1 != x2) ? 1 : 0 for int64 with uint8 output
func NeU8I64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] != x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// NeStrided performs y = (x1 != x2) ? 1 : 0 for any supported numeric type with strided memory and same-type output
func NeStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []T) {
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

// NeStridedF32F32 performs y = (x1 != x2) ? 1 : 0 for float32 with strided memory and float32 output
func NeStridedF32F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeF32F32(numel, x1, x2, y)
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

// NeStridedF64F64 performs y = (x1 != x2) ? 1 : 0 for float64 with strided memory and float64 output
func NeStridedF64F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeF64F64(numel, x1, x2, y)
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

// NeStridedU32U32 performs y = (x1 != x2) ? 1 : 0 for uint32 with strided memory and uint32 output
func NeStridedU32U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeU32U32(numel, x1, x2, y)
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

// NeStridedI64I64 performs y = (x1 != x2) ? 1 : 0 for int64 with strided memory and int64 output
func NeStridedI64I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeI64I64(numel, x1, x2, y)
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

// NeStridedU8 performs y = (x1 != x2) ? 1 : 0 for any supported numeric type with strided memory and uint8 output
func NeStridedU8[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
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

// NeStridedU8F32 performs y = (x1 != x2) ? 1 : 0 for float32 with strided memory and uint8 output
func NeStridedU8F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeU8F32(numel, x1, x2, y)
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

// NeStridedU8F64 performs y = (x1 != x2) ? 1 : 0 for float64 with strided memory and uint8 output
func NeStridedU8F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeU8F64(numel, x1, x2, y)
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

// NeStridedU8U8 performs y = (x1 != x2) ? 1 : 0 for uint8 with strided memory and uint8 output
func NeStridedU8U8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeU8U8(numel, x1, x2, y)
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

// NeStridedU8U32 performs y = (x1 != x2) ? 1 : 0 for uint32 with strided memory and uint8 output
func NeStridedU8U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeU8U32(numel, x1, x2, y)
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

// NeStridedU8I64 performs y = (x1 != x2) ? 1 : 0 for int64 with strided memory and uint8 output
func NeStridedU8I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		NeU8I64(numel, x1, x2, y)
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

// Lt performs y = (x1 < x2) ? 1 : 0 for any supported numeric type with same-type output
func Lt[T D](numel int, x1, x2 []T, y []T) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtF32F32 performs y = (x1 < x2) ? 1 : 0 for float32 with float32 output
func LtF32F32(numel int, x1, x2 []float32, y []float32) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtF64F64 performs y = (x1 < x2) ? 1 : 0 for float64 with float64 output
func LtF64F64(numel int, x1, x2 []float64, y []float64) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtU32U32 performs y = (x1 < x2) ? 1 : 0 for uint32 with uint32 output
func LtU32U32(numel int, x1, x2 []uint32, y []uint32) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtI64I64 performs y = (x1 < x2) ? 1 : 0 for int64 with int64 output
func LtI64I64(numel int, x1, x2 []int64, y []int64) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtU8 performs y = (x1 < x2) ? 1 : 0 for any supported numeric type with uint8 output
func LtU8[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtU8F32 performs y = (x1 < x2) ? 1 : 0 for float32 with uint8 output
func LtU8F32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtU8F64 performs y = (x1 < x2) ? 1 : 0 for float64 with uint8 output
func LtU8F64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtU8U8 performs y = (x1 < x2) ? 1 : 0 for uint8 with uint8 output
func LtU8U8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtU8U32 performs y = (x1 < x2) ? 1 : 0 for uint32 with uint8 output
func LtU8U32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtU8I64 performs y = (x1 < x2) ? 1 : 0 for int64 with uint8 output
func LtU8I64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] < x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LtStrided performs y = (x1 < x2) ? 1 : 0 for any supported numeric type with strided memory and same-type output
func LtStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []T) {
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

// LtStridedF32F32 performs y = (x1 < x2) ? 1 : 0 for float32 with strided memory and float32 output
func LtStridedF32F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtF32F32(numel, x1, x2, y)
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

// LtStridedF64F64 performs y = (x1 < x2) ? 1 : 0 for float64 with strided memory and float64 output
func LtStridedF64F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtF64F64(numel, x1, x2, y)
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

// LtStridedU32U32 performs y = (x1 < x2) ? 1 : 0 for uint32 with strided memory and uint32 output
func LtStridedU32U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtU32U32(numel, x1, x2, y)
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

// LtStridedI64I64 performs y = (x1 < x2) ? 1 : 0 for int64 with strided memory and int64 output
func LtStridedI64I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtI64I64(numel, x1, x2, y)
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

// LtStridedU8 performs y = (x1 < x2) ? 1 : 0 for any supported numeric type with strided memory and uint8 output
func LtStridedU8[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
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

// LtStridedU8F32 performs y = (x1 < x2) ? 1 : 0 for float32 with strided memory and uint8 output
func LtStridedU8F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtU8F32(numel, x1, x2, y)
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

// LtStridedU8F64 performs y = (x1 < x2) ? 1 : 0 for float64 with strided memory and uint8 output
func LtStridedU8F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtU8F64(numel, x1, x2, y)
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

// LtStridedU8U8 performs y = (x1 < x2) ? 1 : 0 for uint8 with strided memory and uint8 output
func LtStridedU8U8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtU8U8(numel, x1, x2, y)
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

// LtStridedU8U32 performs y = (x1 < x2) ? 1 : 0 for uint32 with strided memory and uint8 output
func LtStridedU8U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtU8U32(numel, x1, x2, y)
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

// LtStridedU8I64 performs y = (x1 < x2) ? 1 : 0 for int64 with strided memory and uint8 output
func LtStridedU8I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LtU8I64(numel, x1, x2, y)
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

// Le performs y = (x1 <= x2) ? 1 : 0 for any supported numeric type with same-type output
func Le[T D](numel int, x1, x2 []T, y []T) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeF32F32 performs y = (x1 <= x2) ? 1 : 0 for float32 with float32 output
func LeF32F32(numel int, x1, x2 []float32, y []float32) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeF64F64 performs y = (x1 <= x2) ? 1 : 0 for float64 with float64 output
func LeF64F64(numel int, x1, x2 []float64, y []float64) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeU32U32 performs y = (x1 <= x2) ? 1 : 0 for uint32 with uint32 output
func LeU32U32(numel int, x1, x2 []uint32, y []uint32) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeI64I64 performs y = (x1 <= x2) ? 1 : 0 for int64 with int64 output
func LeI64I64(numel int, x1, x2 []int64, y []int64) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeU8 performs y = (x1 <= x2) ? 1 : 0 for any supported numeric type with uint8 output
func LeU8[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeU8F32 performs y = (x1 <= x2) ? 1 : 0 for float32 with uint8 output
func LeU8F32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeU8F64 performs y = (x1 <= x2) ? 1 : 0 for float64 with uint8 output
func LeU8F64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeU8U8 performs y = (x1 <= x2) ? 1 : 0 for uint8 with uint8 output
func LeU8U8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeU8U32 performs y = (x1 <= x2) ? 1 : 0 for uint32 with uint8 output
func LeU8U32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeU8I64 performs y = (x1 <= x2) ? 1 : 0 for int64 with uint8 output
func LeU8I64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] <= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// LeStrided performs y = (x1 <= x2) ? 1 : 0 for any supported numeric type with strided memory and same-type output
func LeStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []T) {
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

// LeStridedF32F32 performs y = (x1 <= x2) ? 1 : 0 for float32 with strided memory and float32 output
func LeStridedF32F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeF32F32(numel, x1, x2, y)
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

// LeStridedF64F64 performs y = (x1 <= x2) ? 1 : 0 for float64 with strided memory and float64 output
func LeStridedF64F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeF64F64(numel, x1, x2, y)
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

// LeStridedU32U32 performs y = (x1 <= x2) ? 1 : 0 for uint32 with strided memory and uint32 output
func LeStridedU32U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeU32U32(numel, x1, x2, y)
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

// LeStridedI64I64 performs y = (x1 <= x2) ? 1 : 0 for int64 with strided memory and int64 output
func LeStridedI64I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeI64I64(numel, x1, x2, y)
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

// LeStridedU8 performs y = (x1 <= x2) ? 1 : 0 for any supported numeric type with strided memory and uint8 output
func LeStridedU8[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
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

// LeStridedU8F32 performs y = (x1 <= x2) ? 1 : 0 for float32 with strided memory and uint8 output
func LeStridedU8F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeU8F32(numel, x1, x2, y)
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

// LeStridedU8F64 performs y = (x1 <= x2) ? 1 : 0 for float64 with strided memory and uint8 output
func LeStridedU8F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeU8F64(numel, x1, x2, y)
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

// LeStridedU8U8 performs y = (x1 <= x2) ? 1 : 0 for uint8 with strided memory and uint8 output
func LeStridedU8U8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeU8U8(numel, x1, x2, y)
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

// LeStridedU8U32 performs y = (x1 <= x2) ? 1 : 0 for uint32 with strided memory and uint8 output
func LeStridedU8U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeU8U32(numel, x1, x2, y)
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

// LeStridedU8I64 performs y = (x1 <= x2) ? 1 : 0 for int64 with strided memory and uint8 output
func LeStridedU8I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		LeU8I64(numel, x1, x2, y)
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

// Gt performs y = (x1 > x2) ? 1 : 0 for any supported numeric type with same-type output
func Gt[T D](numel int, x1, x2 []T, y []T) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtF32F32 performs y = (x1 > x2) ? 1 : 0 for float32 with float32 output
func GtF32F32(numel int, x1, x2 []float32, y []float32) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtF64F64 performs y = (x1 > x2) ? 1 : 0 for float64 with float64 output
func GtF64F64(numel int, x1, x2 []float64, y []float64) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtU32U32 performs y = (x1 > x2) ? 1 : 0 for uint32 with uint32 output
func GtU32U32(numel int, x1, x2 []uint32, y []uint32) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtI64I64 performs y = (x1 > x2) ? 1 : 0 for int64 with int64 output
func GtI64I64(numel int, x1, x2 []int64, y []int64) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtU8 performs y = (x1 > x2) ? 1 : 0 for any supported numeric type with uint8 output
func GtU8[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtU8F32 performs y = (x1 > x2) ? 1 : 0 for float32 with uint8 output
func GtU8F32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtU8F64 performs y = (x1 > x2) ? 1 : 0 for float64 with uint8 output
func GtU8F64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtU8U8 performs y = (x1 > x2) ? 1 : 0 for uint8 with uint8 output
func GtU8U8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtU8U32 performs y = (x1 > x2) ? 1 : 0 for uint32 with uint8 output
func GtU8U32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtU8I64 performs y = (x1 > x2) ? 1 : 0 for int64 with uint8 output
func GtU8I64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] > x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GtStrided performs y = (x1 > x2) ? 1 : 0 for any supported numeric type with strided memory and same-type output
func GtStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []T) {
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

// GtStridedF32F32 performs y = (x1 > x2) ? 1 : 0 for float32 with strided memory and float32 output
func GtStridedF32F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtF32F32(numel, x1, x2, y)
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

// GtStridedF64F64 performs y = (x1 > x2) ? 1 : 0 for float64 with strided memory and float64 output
func GtStridedF64F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtF64F64(numel, x1, x2, y)
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

// GtStridedU32U32 performs y = (x1 > x2) ? 1 : 0 for uint32 with strided memory and uint32 output
func GtStridedU32U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtU32U32(numel, x1, x2, y)
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

// GtStridedI64I64 performs y = (x1 > x2) ? 1 : 0 for int64 with strided memory and int64 output
func GtStridedI64I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtI64I64(numel, x1, x2, y)
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

// GtStridedU8 performs y = (x1 > x2) ? 1 : 0 for any supported numeric type with strided memory and uint8 output
func GtStridedU8[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
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

// GtStridedU8F32 performs y = (x1 > x2) ? 1 : 0 for float32 with strided memory and uint8 output
func GtStridedU8F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtU8F32(numel, x1, x2, y)
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

// GtStridedU8F64 performs y = (x1 > x2) ? 1 : 0 for float64 with strided memory and uint8 output
func GtStridedU8F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtU8F64(numel, x1, x2, y)
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

// GtStridedU8U8 performs y = (x1 > x2) ? 1 : 0 for uint8 with strided memory and uint8 output
func GtStridedU8U8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtU8U8(numel, x1, x2, y)
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

// GtStridedU8U32 performs y = (x1 > x2) ? 1 : 0 for uint32 with strided memory and uint8 output
func GtStridedU8U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtU8U32(numel, x1, x2, y)
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

// GtStridedU8I64 performs y = (x1 > x2) ? 1 : 0 for int64 with strided memory and uint8 output
func GtStridedU8I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GtU8I64(numel, x1, x2, y)
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

// Ge performs y = (x1 >= x2) ? 1 : 0 for any supported numeric type with same-type output
func Ge[T D](numel int, x1, x2 []T, y []T) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeF32F32 performs y = (x1 >= x2) ? 1 : 0 for float32 with float32 output
func GeF32F32(numel int, x1, x2 []float32, y []float32) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeF64F64 performs y = (x1 >= x2) ? 1 : 0 for float64 with float64 output
func GeF64F64(numel int, x1, x2 []float64, y []float64) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeU32U32 performs y = (x1 >= x2) ? 1 : 0 for uint32 with uint32 output
func GeU32U32(numel int, x1, x2 []uint32, y []uint32) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeI64I64 performs y = (x1 >= x2) ? 1 : 0 for int64 with int64 output
func GeI64I64(numel int, x1, x2 []int64, y []int64) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeU8 performs y = (x1 >= x2) ? 1 : 0 for any supported numeric type with uint8 output
func GeU8[T D](numel int, x1, x2 []T, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeU8F32 performs y = (x1 >= x2) ? 1 : 0 for float32 with uint8 output
func GeU8F32(numel int, x1, x2 []float32, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeU8F64 performs y = (x1 >= x2) ? 1 : 0 for float64 with uint8 output
func GeU8F64(numel int, x1, x2 []float64, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeU8U8 performs y = (x1 >= x2) ? 1 : 0 for uint8 with uint8 output
func GeU8U8(numel int, x1, x2 []uint8, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeU8U32 performs y = (x1 >= x2) ? 1 : 0 for uint32 with uint8 output
func GeU8U32(numel int, x1, x2 []uint32, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeU8I64 performs y = (x1 >= x2) ? 1 : 0 for int64 with uint8 output
func GeU8I64(numel int, x1, x2 []int64, y []uint8) {
	for i := range numel {
		if x1[i] >= x2[i] {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
}

// GeStrided performs y = (x1 >= x2) ? 1 : 0 for any supported numeric type with strided memory and same-type output
func GeStrided[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []T) {
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

// GeStridedF32F32 performs y = (x1 >= x2) ? 1 : 0 for float32 with strided memory and float32 output
func GeStridedF32F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []float32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeF32F32(numel, x1, x2, y)
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

// GeStridedF64F64 performs y = (x1 >= x2) ? 1 : 0 for float64 with strided memory and float64 output
func GeStridedF64F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []float64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeF64F64(numel, x1, x2, y)
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

// GeStridedU32U32 performs y = (x1 >= x2) ? 1 : 0 for uint32 with strided memory and uint32 output
func GeStridedU32U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint32) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeU32U32(numel, x1, x2, y)
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

// GeStridedI64I64 performs y = (x1 >= x2) ? 1 : 0 for int64 with strided memory and int64 output
func GeStridedI64I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []int64) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeI64I64(numel, x1, x2, y)
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

// GeStridedU8 performs y = (x1 >= x2) ? 1 : 0 for any supported numeric type with strided memory and uint8 output
func GeStridedU8[T D](numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
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

// GeStridedU8F32 performs y = (x1 >= x2) ? 1 : 0 for float32 with strided memory and uint8 output
func GeStridedU8F32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeU8F32(numel, x1, x2, y)
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

// GeStridedU8F64 performs y = (x1 >= x2) ? 1 : 0 for float64 with strided memory and uint8 output
func GeStridedU8F64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeU8F64(numel, x1, x2, y)
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

// GeStridedU8U8 performs y = (x1 >= x2) ? 1 : 0 for uint8 with strided memory and uint8 output
func GeStridedU8U8(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint8, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeU8U8(numel, x1, x2, y)
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

// GeStridedU8U32 performs y = (x1 >= x2) ? 1 : 0 for uint32 with strided memory and uint8 output
func GeStridedU8U32(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []uint32, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeU8U32(numel, x1, x2, y)
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

// GeStridedU8I64 performs y = (x1 >= x2) ? 1 : 0 for int64 with strided memory and uint8 output
func GeStridedU8I64(numel, ndims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []int64, y []uint8) {
	if IsContiguous(ndims, dims, stridesX1) && IsContiguous(ndims, dims, stridesX2) && IsContiguous(ndims, dims, stridesY) {
		GeU8I64(numel, x1, x2, y)
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

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

// BAddStrided performs y = x1 + x2 for any supported numeric type with strided memory
func BAddStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BAdd(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] + x2[idx2]
	}
}

// BAddStridedF32 performs y = x1 + x2 for float32 with strided memory
func BAddStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BAddF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] + x2[idx2]
	}
}

// BAddStridedF64 performs y = x1 + x2 for float64 with strided memory
func BAddStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BAddF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] + x2[idx2]
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

// BSubStrided performs y = x1 - x2 for any supported numeric type with strided memory
func BSubStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BSub(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] - x2[idx2]
	}
}

// BSubStridedF32 performs y = x1 - x2 for float32 with strided memory
func BSubStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BSubF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] - x2[idx2]
	}
}

// BSubStridedF64 performs y = x1 - x2 for float64 with strided memory
func BSubStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BSubF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] - x2[idx2]
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

// BMulStrided performs y = x1 * x2 for any supported numeric type with strided memory
func BMulStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BMul(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] * x2[idx2]
	}
}

// BMulStridedF32 performs y = x1 * x2 for float32 with strided memory
func BMulStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BMulF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] * x2[idx2]
	}
}

// BMulStridedF64 performs y = x1 * x2 for float64 with strided memory
func BMulStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BMulF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] * x2[idx2]
	}
}

// BDiv performs y = x1 / x2 for any supported numeric type
func BDiv[T D](numel int, x1, x2, y []T) {
	for i := range numel {
		y[i] = x1[i] / x2[i]
	}
}

// BDivF32 performs y = x1 / x2 for float32
func BDivF32(numel int, x1, x2, y []float32) {
	for i := range numel {
		y[i] = x1[i] / x2[i]
	}
}

// BDivF64 performs y = x1 / x2 for float64
func BDivF64(numel int, x1, x2, y []float64) {
	for i := range numel {
		y[i] = x1[i] / x2[i]
	}
}

// BDivStrided performs y = x1 / x2 for any supported numeric type with strided memory
func BDivStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BDiv(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] / x2[idx2]
	}
}

// BDivStridedF32 performs y = x1 / x2 for float32 with strided memory
func BDivStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BDivF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] / x2[idx2]
	}
}

// BDivStridedF64 performs y = x1 / x2 for float64 with strided memory
func BDivStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BDivF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1] / x2[idx2]
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

// BMaxStrided performs y = max(x1, x2) for any supported numeric type with strided memory
func BMaxStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BMax(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		val1 := x1[idx1]
		val2 := x2[idx2]
		if val1 > val2 {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = val1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = val2
		}
	}
}

// BMaximumStridedF32 performs y = max(x1, x2) for float32 with strided memory
func BMaximumStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BMaximumF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMaximumStridedF64 performs y = max(x1, x2) for float64 with strided memory
func BMaximumStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BMaximumF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = x2[idx2]
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

// BMinStrided performs y = min(x1, x2) for any supported numeric type with strided memory
func BMinStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []T) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BMin(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		val1 := x1[idx1]
		val2 := x2[idx2]
		if val1 < val2 {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = val1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = val2
		}
	}
}

// BMinimumStridedF32 performs y = min(x1, x2) for float32 with strided memory
func BMinimumStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float32) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BMinimumF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = x2[idx2]
		}
	}
}

// BMinimumStridedF64 performs y = min(x1, x2) for float64 with strided memory
func BMinimumStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2, y []float64) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		BMinimumF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = x1[idx1]
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = x2[idx2]
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

// EqStrided performs y = (x1 == x2) ? 1 : 0 for any supported numeric type with strided memory
func EqStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		Eq(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] == x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// EqStridedF32 performs y = (x1 == x2) ? 1 : 0 for float32 with strided memory
func EqStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		EqF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] == x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// EqStridedF64 performs y = (x1 == x2) ? 1 : 0 for float64 with strided memory
func EqStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		EqF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] == x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
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

// NeStrided performs y = (x1 != x2) ? 1 : 0 for any supported numeric type with strided memory
func NeStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		Ne(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] != x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// NeStridedF32 performs y = (x1 != x2) ? 1 : 0 for float32 with strided memory
func NeStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		NeF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] != x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// NeStridedF64 performs y = (x1 != x2) ? 1 : 0 for float64 with strided memory
func NeStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		NeF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] != x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
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

// LtStrided performs y = (x1 < x2) ? 1 : 0 for any supported numeric type with strided memory
func LtStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		Lt(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// LtStridedF32 performs y = (x1 < x2) ? 1 : 0 for float32 with strided memory
func LtStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		LtF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// LtStridedF64 performs y = (x1 < x2) ? 1 : 0 for float64 with strided memory
func LtStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		LtF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] < x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
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

// LeStrided performs y = (x1 <= x2) ? 1 : 0 for any supported numeric type with strided memory
func LeStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		Le(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] <= x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// LeStridedF32 performs y = (x1 <= x2) ? 1 : 0 for float32 with strided memory
func LeStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		LeF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] <= x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// LeStridedF64 performs y = (x1 <= x2) ? 1 : 0 for float64 with strided memory
func LeStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		LeF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] <= x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
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

// GtStrided performs y = (x1 > x2) ? 1 : 0 for any supported numeric type with strided memory
func GtStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		Gt(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// GtStridedF32 performs y = (x1 > x2) ? 1 : 0 for float32 with strided memory
func GtStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		GtF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// GtStridedF64 performs y = (x1 > x2) ? 1 : 0 for float64 with strided memory
func GtStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		GtF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] > x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
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

// GeStrided performs y = (x1 >= x2) ? 1 : 0 for any supported numeric type with strided memory
func GeStrided[T D](numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []T, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		Ge(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] >= x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// GeStridedF32 performs y = (x1 >= x2) ? 1 : 0 for float32 with strided memory
func GeStridedF32(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float32, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		GeF32(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] >= x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

// GeStridedF64 performs y = (x1 >= x2) ? 1 : 0 for float64 with strided memory
func GeStridedF64(numel, numDims int, dims, stridesX1, stridesX2, stridesY []int, x1, x2 []float64, y []uint8) {
	if IsContiguous(numDims, dims, stridesX1) && IsContiguous(numDims, dims, stridesX2) && IsContiguous(numDims, dims, stridesY) {
		GeF64(numel, x1, x2, y)
		return
	}
	for i := range numel {
		idx1 := GetStridedIndex(i, numDims, dims, stridesX1)
		idx2 := GetStridedIndex(i, numDims, dims, stridesX2)
		if x1[idx1] >= x2[idx2] {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 1
		} else {
			y[GetStridedIndex(i, numDims, dims, stridesY)] = 0
		}
	}
}

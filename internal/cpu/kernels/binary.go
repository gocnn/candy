package kernels

import "math"

// Generic binary operations
type BinaryOp[T any] func(a, b T) T
type CompareOp[T any] func(a, b T) bool

func ApplyBinary[T any](numel int, lhs, rhs, out []T, op BinaryOp[T]) {
	for i := range numel {
		out[i] = op(lhs[i], rhs[i])
	}
}

func ApplyCompare[T any](numel int, lhs, rhs []T, out []uint8, op CompareOp[T]) {
	for i := range numel {
		if op(lhs[i], rhs[i]) {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

// Arithmetic operations (contiguous)
func AddF32(numel int, lhs, rhs, out []float32) {
	for i := range numel {
		out[i] = lhs[i] + rhs[i]
	}
}

func AddF64(numel int, lhs, rhs, out []float64) {
	for i := range numel {
		out[i] = lhs[i] + rhs[i]
	}
}

func SubF32(numel int, lhs, rhs, out []float32) {
	for i := range numel {
		out[i] = lhs[i] - rhs[i]
	}
}

func SubF64(numel int, lhs, rhs, out []float64) {
	for i := range numel {
		out[i] = lhs[i] - rhs[i]
	}
}

func MulF32(numel int, lhs, rhs, out []float32) {
	for i := range numel {
		out[i] = lhs[i] * rhs[i]
	}
}

func MulF64(numel int, lhs, rhs, out []float64) {
	for i := range numel {
		out[i] = lhs[i] * rhs[i]
	}
}

func DivF32(numel int, lhs, rhs, out []float32) {
	for i := range numel {
		out[i] = lhs[i] / rhs[i]
	}
}

func DivF64(numel int, lhs, rhs, out []float64) {
	for i := range numel {
		out[i] = lhs[i] / rhs[i]
	}
}

func MaxF32(numel int, lhs, rhs, out []float32) {
	for i := range numel {
		out[i] = max(lhs[i], rhs[i])
	}
}

func MaxF64(numel int, lhs, rhs, out []float64) {
	for i := range numel {
		out[i] = math.Max(lhs[i], rhs[i])
	}
}

func MinF32(numel int, lhs, rhs, out []float32) {
	for i := range numel {
		out[i] = min(lhs[i], rhs[i])
	}
}

func MinF64(numel int, lhs, rhs, out []float64) {
	for i := range numel {
		out[i] = math.Min(lhs[i], rhs[i])
	}
}

// Comparison operations (contiguous, output uint8: 1=true, 0=false)
func EqF32(numel int, lhs, rhs []float32, out []uint8) {
	for i := range numel {
		if lhs[i] == rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func EqF64(numel int, lhs, rhs []float64, out []uint8) {
	for i := range numel {
		if lhs[i] == rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func NeF32(numel int, lhs, rhs []float32, out []uint8) {
	for i := range numel {
		if lhs[i] != rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func NeF64(numel int, lhs, rhs []float64, out []uint8) {
	for i := range numel {
		if lhs[i] != rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func LtF32(numel int, lhs, rhs []float32, out []uint8) {
	for i := range numel {
		if lhs[i] < rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func LtF64(numel int, lhs, rhs []float64, out []uint8) {
	for i := range numel {
		if lhs[i] < rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func LeF32(numel int, lhs, rhs []float32, out []uint8) {
	for i := range numel {
		if lhs[i] <= rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func LeF64(numel int, lhs, rhs []float64, out []uint8) {
	for i := range numel {
		if lhs[i] <= rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func GtF32(numel int, lhs, rhs []float32, out []uint8) {
	for i := range numel {
		if lhs[i] > rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func GtF64(numel int, lhs, rhs []float64, out []uint8) {
	for i := range numel {
		if lhs[i] > rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func GeF32(numel int, lhs, rhs []float32, out []uint8) {
	for i := range numel {
		if lhs[i] >= rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func GeF64(numel int, lhs, rhs []float64, out []uint8) {
	for i := range numel {
		if lhs[i] >= rhs[i] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

// Strided operations with broadcasting support
func AddStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float32) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		AddF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = lhs[lhsIdx] + rhs[rhsIdx]
	}
}

func AddStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float64) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		AddF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = lhs[lhsIdx] + rhs[rhsIdx]
	}
}

func SubStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float32) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		SubF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = lhs[lhsIdx] - rhs[rhsIdx]
	}
}

func SubStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float64) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		SubF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = lhs[lhsIdx] - rhs[rhsIdx]
	}
}

func MulStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float32) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		MulF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = lhs[lhsIdx] * rhs[rhsIdx]
	}
}

func MulStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float64) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		MulF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = lhs[lhsIdx] * rhs[rhsIdx]
	}
}

func DivStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float32) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		DivF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = lhs[lhsIdx] / rhs[rhsIdx]
	}
}

func DivStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float64) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		DivF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = lhs[lhsIdx] / rhs[rhsIdx]
	}
}

func MaxStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float32) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		MaxF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = max(lhs[lhsIdx], rhs[rhsIdx])
	}
}

func MaxStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float64) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		MaxF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = math.Max(lhs[lhsIdx], rhs[rhsIdx])
	}
}

func MinStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float32) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		MinF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = min(lhs[lhsIdx], rhs[rhsIdx])
	}
}

func MinStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs, out []float64) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		MinF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		out[i] = math.Min(lhs[lhsIdx], rhs[rhsIdx])
	}
}

func EqStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float32, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		EqF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] == rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func EqStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float64, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		EqF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] == rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func NeStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float32, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		NeF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] != rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func NeStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float64, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		NeF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] != rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func LtStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float32, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		LtF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] < rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func LtStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float64, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		LtF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] < rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func LeStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float32, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		LeF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] <= rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func LeStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float64, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		LeF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] <= rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func GtStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float32, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		GtF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] > rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func GtStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float64, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		GtF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] > rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func GeStridedF32(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float32, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		GeF32(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] >= rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

func GeStridedF64(numel, numDims int, dims, lhsStrides, rhsStrides []int, lhs, rhs []float64, out []uint8) {
	lhsCont := IsContiguous(numDims, dims, lhsStrides)
	rhsCont := IsContiguous(numDims, dims, rhsStrides)
	if lhsCont && rhsCont {
		GeF64(numel, lhs, rhs, out)
		return
	}
	for i := range numel {
		lhsIdx := i
		rhsIdx := i
		if !lhsCont {
			lhsIdx = GetStridedIndex(i, numDims, dims, lhsStrides)
		}
		if !rhsCont {
			rhsIdx = GetStridedIndex(i, numDims, dims, rhsStrides)
		}
		if lhs[lhsIdx] >= rhs[rhsIdx] {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

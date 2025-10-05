package kernels

import "math"

// UCopy performs element-wise copy for type T (contiguous memory)
func UCopy[T D](numel int, inp, out []T) {
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UCopyF32 performs element-wise copy for float32 (contiguous memory)
func UCopyF32(numel int, inp, out []float32) {
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UCopyF64 performs element-wise copy for float64 (contiguous memory)
func UCopyF64(numel int, inp, out []float64) {
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UCopyU8 performs element-wise copy for uint8 (contiguous memory)
func UCopyU8(numel int, inp, out []uint8) {
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UCopyU32 performs element-wise copy for uint32 (contiguous memory)
func UCopyU32(numel int, inp, out []uint32) {
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UCopyI64 performs element-wise copy for int64 (contiguous memory)
func UCopyI64(numel int, inp, out []int64) {
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UCopyStrided performs element-wise copy for type T (strided memory)
func UCopyStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UCopy(numel, inp, out)
		return
	}
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UCopyStridedF32 performs element-wise copy for float32 (strided memory)
func UCopyStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UCopyF32(numel, inp, out)
		return
	}
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UCopyStridedF64 performs element-wise copy for float64 (strided memory)
func UCopyStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UCopyF64(numel, inp, out)
		return
	}
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UCopyStridedU8 performs element-wise copy for uint8 (strided memory)
func UCopyStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UCopyU8(numel, inp, out)
		return
	}
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UCopyStridedU32 performs element-wise copy for uint32 (strided memory)
func UCopyStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UCopyU32(numel, inp, out)
		return
	}
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UCopyStridedI64 performs element-wise copy for int64 (strided memory)
func UCopyStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UCopyI64(numel, inp, out)
		return
	}
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UNeg performs element-wise negation for type T (contiguous memory)
func UNeg[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		out[i] = -inp[i]
	}
}

// UNegF32 performs element-wise negation for float32 (contiguous memory)
func UNegF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		out[i] = -inp[i]
	}
}

// UNegF64 performs element-wise negation for float64 (contiguous memory)
func UNegF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		out[i] = -inp[i]
	}
}

// UNegU8 performs element-wise negation for uint8 (contiguous memory)
func UNegU8(numel int, inp, out []uint8) {
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		out[i] = -inp[i]
	}
}

// UNegU32 performs element-wise negation for uint32 (contiguous memory)
func UNegU32(numel int, inp, out []uint32) {
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		out[i] = -inp[i]
	}
}

// UNegI64 performs element-wise negation for int64 (contiguous memory)
func UNegI64(numel int, inp, out []int64) {
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		out[i] = -inp[i]
	}
}

// UNegStrided performs element-wise negation for type T (strided memory)
func UNegStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UNeg(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = -inp[stridedI]
	}
}

// UNegStridedF32 performs element-wise negation for float32 (strided memory)
func UNegStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UNegF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = -inp[stridedI]
	}
}

// UNegStridedF64 performs element-wise negation for float64 (strided memory)
func UNegStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UNegF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = -inp[stridedI]
	}
}

// UNegStridedU8 performs element-wise negation for uint8 (strided memory)
func UNegStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UNegU8(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = -inp[stridedI]
	}
}

// UNegStridedU32 performs element-wise negation for uint32 (strided memory)
func UNegStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UNegU32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = -inp[stridedI]
	}
}

// UNegStridedI64 performs element-wise negation for int64 (strided memory)
func UNegStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UNegI64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = -inp[stridedI]
	}
}

// URecip performs element-wise reciprocal for type T (contiguous memory)
func URecip[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(1) / out[i]
		}
		return
	}
	for i := range numel {
		out[i] = T(1) / inp[i]
	}
}

// URecipF32 performs element-wise reciprocal for float32 (contiguous memory)
func URecipF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = 1.0 / out[i]
		}
		return
	}
	for i := range numel {
		out[i] = 1.0 / inp[i]
	}
}

// URecipF64 performs element-wise reciprocal for float64 (contiguous memory)
func URecipF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = 1.0 / out[i]
		}
		return
	}
	for i := range numel {
		out[i] = 1.0 / inp[i]
	}
}

// URecipU8 performs element-wise reciprocal for uint8 (contiguous memory)
func URecipU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// URecipU32 performs element-wise reciprocal for uint32 (contiguous memory)
func URecipU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// URecipI64 performs element-wise reciprocal for int64 (contiguous memory)
func URecipI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// URecipStrided performs element-wise reciprocal for type T (strided memory)
func URecipStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		URecip(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(1) / out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(1) / inp[stridedI]
	}
}

// URecipStridedF32 performs element-wise reciprocal for float32 (strided memory)
func URecipStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		URecipF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = 1.0 / out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = 1.0 / inp[stridedI]
	}
}

// URecipStridedF64 performs element-wise reciprocal for float64 (strided memory)
func URecipStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		URecipF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = 1.0 / out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = 1.0 / inp[stridedI]
	}
}

// URecipStridedU8 performs element-wise reciprocal for uint8 (strided memory)
func URecipStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		URecipU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// URecipStridedU32 performs element-wise reciprocal for uint32 (strided memory)
func URecipStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		URecipU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// URecipStridedI64 performs element-wise reciprocal for int64 (strided memory)
func URecipStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		URecipI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// UExp performs element-wise exponential for type T (contiguous memory)
func UExp[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Exp(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Exp(float64(inp[i])))
	}
}

// UExpF32 performs element-wise exponential for float32 (contiguous memory)
func UExpF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Exp(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Exp(float64(inp[i])))
	}
}

// UExpF64 performs element-wise exponential for float64 (contiguous memory)
func UExpF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Exp(out[i])
		}
		return
	}
	for i := range numel {
		out[i] = math.Exp(inp[i])
	}
}

// UExpU8 performs element-wise exponential for uint8 (contiguous memory)
func UExpU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// UExpU32 performs element-wise exponential for uint32 (contiguous memory)
func UExpU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// UExpI64 performs element-wise exponential for int64 (contiguous memory)
func UExpI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// UExpStrided performs element-wise exponential for type T (strided memory)
func UExpStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UExp(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Exp(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Exp(float64(inp[stridedI])))
	}
}

// UExpStridedF32 performs element-wise exponential for float32 (strided memory)
func UExpStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UExpF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Exp(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Exp(float64(inp[stridedI])))
	}
}

// UExpStridedF64 performs element-wise exponential for float64 (strided memory)
func UExpStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UExpF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Exp(out[i])
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Exp(inp[stridedI])
	}
}

// UExpStridedU8 performs element-wise exponential for uint8 (strided memory)
func UExpStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UExpU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// UExpStridedU32 performs element-wise exponential for uint32 (strided memory)
func UExpStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UExpU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// UExpStridedI64 performs element-wise exponential for int64 (strided memory)
func UExpStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UExpI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// ULog performs element-wise logarithm for type T (contiguous memory)
func ULog[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Log(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Log(float64(inp[i])))
	}
}

// ULogF32 performs element-wise logarithm for float32 (contiguous memory)
func ULogF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Log(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Log(float64(inp[i])))
	}
}

// ULogF64 performs element-wise logarithm for float64 (contiguous memory)
func ULogF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Log(out[i])
		}
		return
	}
	for i := range numel {
		out[i] = math.Log(inp[i])
	}
}

// ULogU8 performs element-wise logarithm for uint8 (contiguous memory)
func ULogU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// ULogU32 performs element-wise logarithm for uint32 (contiguous memory)
func ULogU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// ULogI64 performs element-wise logarithm for int64 (contiguous memory)
func ULogI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// ULogStrided performs element-wise logarithm for type T (strided memory)
func ULogStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		ULog(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Log(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Log(float64(inp[stridedI])))
	}
}

// ULogStridedF32 performs element-wise logarithm for float32 (strided memory)
func ULogStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		ULogF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Log(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Log(float64(inp[stridedI])))
	}
}

// ULogStridedF64 performs element-wise logarithm for float64 (strided memory)
func ULogStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		ULogF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Log(out[i])
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Log(inp[stridedI])
	}
}

// ULogStridedU8 performs element-wise logarithm for uint8 (strided memory)
func ULogStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		ULogU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// ULogStridedU32 performs element-wise logarithm for uint32 (strided memory)
func ULogStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ULogU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// ULogStridedI64 performs element-wise logarithm for int64 (strided memory)
func ULogStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		ULogI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// USin performs element-wise sine for type T (contiguous memory)
func USin[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Sin(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Sin(float64(inp[i])))
	}
}

// USinF32 performs element-wise sine for float32 (contiguous memory)
func USinF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Sin(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Sin(float64(inp[i])))
	}
}

// USinF64 performs element-wise sine for float64 (contiguous memory)
func USinF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Sin(out[i])
		}
		return
	}
	for i := range numel {
		out[i] = math.Sin(inp[i])
	}
}

// USinU8 performs element-wise sine for uint8 (contiguous memory)
func USinU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// USinU32 performs element-wise sine for uint32 (contiguous memory)
func USinU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// USinI64 performs element-wise sine for int64 (contiguous memory)
func USinI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// USinStrided performs element-wise sine for type T (strided memory)
func USinStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		USin(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Sin(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Sin(float64(inp[stridedI])))
	}
}

// USinStridedF32 performs element-wise sine for float32 (strided memory)
func USinStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		USinF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Sin(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Sin(float64(inp[stridedI])))
	}
}

// USinStridedF64 performs element-wise sine for float64 (strided memory)
func USinStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		USinF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Sin(out[i])
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Sin(inp[stridedI])
	}
}

// USinStridedU8 performs element-wise sine for uint8 (strided memory)
func USinStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		USinU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// USinStridedU32 performs element-wise sine for uint32 (strided memory)
func USinStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		USinU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// USinStridedI64 performs element-wise sine for int64 (strided memory)
func USinStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		USinI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// UCos performs element-wise cosine for type T (contiguous memory)
func UCos[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Cos(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Cos(float64(inp[i])))
	}
}

// UCosF32 performs element-wise cosine for float32 (contiguous memory)
func UCosF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Cos(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Cos(float64(inp[i])))
	}
}

// UCosF64 performs element-wise cosine for float64 (contiguous memory)
func UCosF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Cos(out[i])
		}
		return
	}
	for i := range numel {
		out[i] = math.Cos(inp[i])
	}
}

// UCosU8 performs element-wise cosine for uint8 (contiguous memory)
func UCosU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// UCosU32 performs element-wise cosine for uint32 (contiguous memory)
func UCosU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// UCosI64 performs element-wise cosine for int64 (contiguous memory)
func UCosI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// UCosStrided performs element-wise cosine for type T (strided memory)
func UCosStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UCos(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Cos(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Cos(float64(inp[stridedI])))
	}
}

// UCosStridedF32 performs element-wise cosine for float32 (strided memory)
func UCosStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UCosF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Cos(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Cos(float64(inp[stridedI])))
	}
}

// UCosStridedF64 performs element-wise cosine for float64 (strided memory)
func UCosStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UCosF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Cos(out[i])
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Cos(inp[stridedI])
	}
}

// UCosStridedU8 performs element-wise cosine for uint8 (strided memory)
func UCosStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UCosU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// UCosStridedU32 performs element-wise cosine for uint32 (strided memory)
func UCosStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UCosU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// UCosStridedI64 performs element-wise cosine for int64 (strided memory)
func UCosStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UCosI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// UTanh performs element-wise tanh for type T (contiguous memory)
func UTanh[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Tanh(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Tanh(float64(inp[i])))
	}
}

// UTanhF32 performs element-wise tanh for float32 (contiguous memory)
func UTanhF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Tanh(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Tanh(float64(inp[i])))
	}
}

// UTanhF64 performs element-wise tanh for float64 (contiguous memory)
func UTanhF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Tanh(out[i])
		}
		return
	}
	for i := range numel {
		out[i] = math.Tanh(inp[i])
	}
}

// UTanhU8 performs element-wise tanh for uint8 (contiguous memory)
func UTanhU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// UTanhU32 performs element-wise tanh for uint32 (contiguous memory)
func UTanhU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// UTanhI64 performs element-wise tanh for int64 (contiguous memory)
func UTanhI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// UTanhStrided performs element-wise tanh for type T (strided memory)
func UTanhStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UTanh(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Tanh(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Tanh(float64(inp[stridedI])))
	}
}

// UTanhStridedF32 performs element-wise tanh for float32 (strided memory)
func UTanhStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UTanhF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Tanh(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Tanh(float64(inp[stridedI])))
	}
}

// UTanhStridedF64 performs element-wise tanh for float64 (strided memory)
func UTanhStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UTanhF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Tanh(out[i])
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Tanh(inp[stridedI])
	}
}

// UTanhStridedU8 performs element-wise tanh for uint8 (strided memory)
func UTanhStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UTanhU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// UTanhStridedU32 performs element-wise tanh for uint32 (strided memory)
func UTanhStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UTanhU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// UTanhStridedI64 performs element-wise tanh for int64 (strided memory)
func UTanhStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UTanhI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// UErf performs element-wise erf for type T (contiguous memory)
func UErf[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Erf(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Erf(float64(inp[i])))
	}
}

// UErfF32 performs element-wise erf for float32 (contiguous memory)
func UErfF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Erf(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Erf(float64(inp[i])))
	}
}

// UErfF64 performs element-wise erf for float64 (contiguous memory)
func UErfF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Erf(out[i])
		}
		return
	}
	for i := range numel {
		out[i] = math.Erf(inp[i])
	}
}

// UErfU8 performs element-wise erf for uint8 (contiguous memory)
func UErfU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// UErfU32 performs element-wise erf for uint32 (contiguous memory)
func UErfU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// UErfI64 performs element-wise erf for int64 (contiguous memory)
func UErfI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// UErfStrided performs element-wise erf for type T (strided memory)
func UErfStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UErf(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Erf(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Erf(float64(inp[stridedI])))
	}
}

// UErfStridedF32 performs element-wise erf for float32 (strided memory)
func UErfStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UErfF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Erf(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Erf(float64(inp[stridedI])))
	}
}

// UErfStridedF64 performs element-wise erf for float64 (strided memory)
func UErfStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UErfF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Erf(out[i])
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Erf(inp[stridedI])
	}
}

// UErfStridedU8 performs element-wise erf for uint8 (strided memory)
func UErfStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UErfU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// UErfStridedU32 performs element-wise erf for uint32 (strided memory)
func UErfStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UErfU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// UErfStridedI64 performs element-wise erf for int64 (strided memory)
func UErfStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UErfI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// UCeil performs element-wise ceil for type T (contiguous memory)
func UCeil[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Ceil(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Ceil(float64(inp[i])))
	}
}

// UCeilF32 performs element-wise ceil for float32 (contiguous memory)
func UCeilF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Ceil(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Ceil(float64(inp[i])))
	}
}

// UCeilF64 performs element-wise ceil for float64 (contiguous memory)
func UCeilF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Ceil(out[i])
		}
		return
	}
	for i := range numel {
		out[i] = math.Ceil(inp[i])
	}
}

// UCeilU8 performs element-wise ceil for uint8 (contiguous memory)
func UCeilU8(numel int, inp, out []uint8) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UCeilU32 performs element-wise ceil for uint32 (contiguous memory)
func UCeilU32(numel int, inp, out []uint32) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UCeilI64 performs element-wise ceil for int64 (contiguous memory)
func UCeilI64(numel int, inp, out []int64) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UCeilStrided performs element-wise ceil for type T (strided memory)
func UCeilStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UCeil(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Ceil(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Ceil(float64(inp[stridedI])))
	}
}

// UCeilStridedF32 performs element-wise ceil for float32 (strided memory)
func UCeilStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UCeilF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Ceil(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Ceil(float64(inp[stridedI])))
	}
}

// UCeilStridedF64 performs element-wise ceil for float64 (strided memory)
func UCeilStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UCeilF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Ceil(out[i])
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Ceil(inp[stridedI])
	}
}

// UCeilStridedU8 performs element-wise ceil for uint8 (strided memory)
func UCeilStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UCeilU8(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UCeilStridedU32 performs element-wise ceil for uint32 (strided memory)
func UCeilStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UCeilU32(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UCeilStridedI64 performs element-wise ceil for int64 (strided memory)
func UCeilStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UCeilI64(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UFloor performs element-wise floor for type T (contiguous memory)
func UFloor[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Floor(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Floor(float64(inp[i])))
	}
}

// UFloorF32 performs element-wise floor for float32 (contiguous memory)
func UFloorF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Floor(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Floor(float64(inp[i])))
	}
}

// UFloorF64 performs element-wise floor for float64 (contiguous memory)
func UFloorF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Floor(out[i])
		}
		return
	}
	for i := range numel {
		out[i] = math.Floor(inp[i])
	}
}

// UFloorU8 performs element-wise floor for uint8 (contiguous memory)
func UFloorU8(numel int, inp, out []uint8) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UFloorU32 performs element-wise floor for uint32 (contiguous memory)
func UFloorU32(numel int, inp, out []uint32) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UFloorI64 performs element-wise floor for int64 (contiguous memory)
func UFloorI64(numel int, inp, out []int64) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UFloorStrided performs element-wise floor for type T (strided memory)
func UFloorStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UFloor(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Floor(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Floor(float64(inp[stridedI])))
	}
}

// UFloorStridedF32 performs element-wise floor for float32 (strided memory)
func UFloorStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UFloorF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Floor(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Floor(float64(inp[stridedI])))
	}
}

// UFloorStridedF64 performs element-wise floor for float64 (strided memory)
func UFloorStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UFloorF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Floor(out[i])
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Floor(inp[stridedI])
	}
}

// UFloorStridedU8 performs element-wise floor for uint8 (strided memory)
func UFloorStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UFloorU8(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UFloorStridedU32 performs element-wise floor for uint32 (strided memory)
func UFloorStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UFloorU32(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UFloorStridedI64 performs element-wise floor for int64 (strided memory)
func UFloorStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UFloorI64(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// URound performs element-wise round for type T (contiguous memory)
func URound[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Round(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Round(float64(inp[i])))
	}
}

// URoundF32 performs element-wise round for float32 (contiguous memory)
func URoundF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Round(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Round(float64(inp[i])))
	}
}

// URoundF64 performs element-wise round for float64 (contiguous memory)
func URoundF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Round(out[i])
		}
		return
	}
	for i := range numel {
		out[i] = math.Round(inp[i])
	}
}

// URoundU8 performs element-wise round for uint8 (contiguous memory)
func URoundU8(numel int, inp, out []uint8) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// URoundU32 performs element-wise round for uint32 (contiguous memory)
func URoundU32(numel int, inp, out []uint32) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// URoundI64 performs element-wise round for int64 (contiguous memory)
func URoundI64(numel int, inp, out []int64) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// URoundStrided performs element-wise round for type T (strided memory)
func URoundStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		URound(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Round(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Round(float64(inp[stridedI])))
	}
}

// URoundStridedF32 performs element-wise round for float32 (strided memory)
func URoundStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		URoundF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Round(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Round(float64(inp[stridedI])))
	}
}

// URoundStridedF64 performs element-wise round for float64 (strided memory)
func URoundStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		URoundF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Round(out[i])
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Round(inp[stridedI])
	}
}

// URoundStridedU8 performs element-wise round for uint8 (strided memory)
func URoundStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		URoundU8(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// URoundStridedU32 performs element-wise round for uint32 (strided memory)
func URoundStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		URoundU32(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// URoundStridedI64 performs element-wise round for int64 (strided memory)
func URoundStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		URoundI64(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UNormcdf performs element-wise normal CDF for type T (contiguous memory)
func UNormcdf[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = T(0.5 * (1 + math.Erf(x/math.Sqrt(2))))
		}
		return
	}
	for i := range numel {
		x := float64(inp[i])
		out[i] = T(0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UNormcdfF32 performs element-wise normal CDF for float32 (contiguous memory)
func UNormcdfF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = float32(0.5 * (1 + math.Erf(x/math.Sqrt(2))))
		}
		return
	}
	for i := range numel {
		x := float64(inp[i])
		out[i] = float32(0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UNormcdfF64 performs element-wise normal CDF for float64 (contiguous memory)
func UNormcdfF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = 0.5 * (1 + math.Erf(out[i]/math.Sqrt(2)))
		}
		return
	}
	for i := range numel {
		out[i] = 0.5 * (1 + math.Erf(inp[i]/math.Sqrt(2)))
	}
}

// UNormcdfU8 performs element-wise normal CDF for uint8 (contiguous memory)
func UNormcdfU8(numel int, inp, out []uint8) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UNormcdfU32 performs element-wise normal CDF for uint32 (contiguous memory)
func UNormcdfU32(numel int, inp, out []uint32) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UNormcdfI64 performs element-wise normal CDF for int64 (contiguous memory)
func UNormcdfI64(numel int, inp, out []int64) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UNormcdfStrided performs element-wise normal CDF for type T (strided memory)
func UNormcdfStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UNormcdf(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = T(0.5 * (1 + math.Erf(x/math.Sqrt(2))))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = T(0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UNormcdfStridedF32 performs element-wise normal CDF for float32 (strided memory)
func UNormcdfStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UNormcdfF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = float32(0.5 * (1 + math.Erf(x/math.Sqrt(2))))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = float32(0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UNormcdfStridedF64 performs element-wise normal CDF for float64 (strided memory)
func UNormcdfStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UNormcdfF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = 0.5 * (1 + math.Erf(out[i]/math.Sqrt(2)))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = 0.5 * (1 + math.Erf(inp[stridedI]/math.Sqrt(2)))
	}
}

// UNormcdfStridedU8 performs element-wise normal CDF for uint8 (strided memory)
func UNormcdfStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UNormcdfU8(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UNormcdfStridedU32 performs element-wise normal CDF for uint32 (strided memory)
func UNormcdfStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UNormcdfU32(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UNormcdfStridedI64 performs element-wise normal CDF for int64 (strided memory)
func UNormcdfStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UNormcdfI64(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UAbs performs element-wise absolute value for type T (contiguous memory)
func UAbs[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Abs(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Abs(float64(inp[i])))
	}
}

// UAbsF32 performs element-wise absolute value for float32 (contiguous memory)
func UAbsF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Abs(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Abs(float64(inp[i])))
	}
}

// UAbsF64 performs element-wise absolute value for float64 (contiguous memory)
func UAbsF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Abs(out[i])
		}
		return
	}
	for i := range numel {
		out[i] = math.Abs(inp[i])
	}
}

// UAbsU8 performs element-wise absolute value for uint8 (contiguous memory)
func UAbsU8(numel int, inp, out []uint8) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UAbsU32 performs element-wise absolute value for uint32 (contiguous memory)
func UAbsU32(numel int, inp, out []uint32) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UAbsI64 performs element-wise absolute value for int64 (contiguous memory)
func UAbsI64(numel int, inp, out []int64) {
	if inp == nil {
		for i := range numel {
			if out[i] < 0 {
				out[i] = -out[i]
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x < 0 {
			out[i] = -x
		} else {
			out[i] = x
		}
	}
}

// UAbsStrided performs element-wise absolute value for type T (strided memory)
func UAbsStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UAbs(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Abs(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Abs(float64(inp[stridedI])))
	}
}

// UAbsStridedF32 performs element-wise absolute value for float32 (strided memory)
func UAbsStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UAbsF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Abs(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Abs(float64(inp[stridedI])))
	}
}

// UAbsStridedF64 performs element-wise absolute value for float64 (strided memory)
func UAbsStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UAbsF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Abs(out[i])
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Abs(inp[stridedI])
	}
}

// UAbsStridedU8 performs element-wise absolute value for uint8 (strided memory)
func UAbsStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UAbsU8(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UAbsStridedU32 performs element-wise absolute value for uint32 (strided memory)
func UAbsStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UAbsU32(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UAbsStridedI64 performs element-wise absolute value for int64 (strided memory)
func UAbsStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UAbsI64(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x < 0 {
			out[i] = -x
		} else {
			out[i] = x
		}
	}
}

// USqr performs element-wise square for type T (contiguous memory)
func USqr[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		out[i] = inp[i] * inp[i]
	}
}

// USqrF32 performs element-wise square for float32 (contiguous memory)
func USqrF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		out[i] = inp[i] * inp[i]
	}
}

// USqrF64 performs element-wise square for float64 (contiguous memory)
func USqrF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		out[i] = inp[i] * inp[i]
	}
}

// USqrU8 performs element-wise square for uint8 (contiguous memory)
func USqrU8(numel int, inp, out []uint8) {
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		out[i] = inp[i] * inp[i]
	}
}

// USqrU32 performs element-wise square for uint32 (contiguous memory)
func USqrU32(numel int, inp, out []uint32) {
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		out[i] = inp[i] * inp[i]
	}
}

// USqrI64 performs element-wise square for int64 (contiguous memory)
func USqrI64(numel int, inp, out []int64) {
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		out[i] = inp[i] * inp[i]
	}
}

// USqrStrided performs element-wise square for type T (strided memory)
func USqrStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		USqr(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI] * inp[stridedI]
	}
}

// USqrStridedF32 performs element-wise square for float32 (strided memory)
func USqrStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		USqrF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI] * inp[stridedI]
	}
}

// USqrStridedF64 performs element-wise square for float64 (strided memory)
func USqrStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		USqrF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI] * inp[stridedI]
	}
}

// USqrStridedU8 performs element-wise square for uint8 (strided memory)
func USqrStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		USqrU8(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI] * inp[stridedI]
	}
}

// USqrStridedU32 performs element-wise square for uint32 (strided memory)
func USqrStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		USqrU32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI] * inp[stridedI]
	}
}

// USqrStridedI64 performs element-wise square for int64 (strided memory)
func USqrStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		USqrI64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI] * inp[stridedI]
	}
}

// USqrt performs element-wise square root for type T (contiguous memory)
func USqrt[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Sqrt(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Sqrt(float64(inp[i])))
	}
}

// USqrtF32 performs element-wise square root for float32 (contiguous memory)
func USqrtF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Sqrt(float64(out[i])))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Sqrt(float64(inp[i])))
	}
}

// USqrtF64 performs element-wise square root for float64 (contiguous memory)
func USqrtF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Sqrt(out[i])
		}
		return
	}
	for i := range numel {
		out[i] = math.Sqrt(inp[i])
	}
}

// USqrtU8 performs element-wise square root for uint8 (contiguous memory)
func USqrtU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// USqrtU32 performs element-wise square root for uint32 (contiguous memory)
func USqrtU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// USqrtI64 performs element-wise square root for int64 (contiguous memory)
func USqrtI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// USqrtStrided performs element-wise square root for type T (strided memory)
func USqrtStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		USqrt(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Sqrt(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Sqrt(float64(inp[stridedI])))
	}
}

// USqrtStridedF32 performs element-wise square root for float32 (strided memory)
func USqrtStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		USqrtF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Sqrt(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Sqrt(float64(inp[stridedI])))
	}
}

// USqrtStridedF64 performs element-wise square root for float64 (strided memory)
func USqrtStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		USqrtF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Sqrt(out[i])
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Sqrt(inp[stridedI])
	}
}

// USqrtStridedU8 performs element-wise square root for uint8 (strided memory)
func USqrtStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		USqrtU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// USqrtStridedU32 performs element-wise square root for uint32 (strided memory)
func USqrtStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		USqrtU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// USqrtStridedI64 performs element-wise square root for int64 (strided memory)
func USqrtStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		USqrtI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// UGelu performs element-wise GELU for type T (contiguous memory)
func UGelu[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			xSq := x * x
			xCube := xSq * x
			alpha := x + 0.044715*xCube
			out[i] = T(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha)))
		}
		return
	}
	for i := range numel {
		x := float64(inp[i])
		xSq := x * x
		xCube := xSq * x
		alpha := x + 0.044715*xCube
		out[i] = T(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha)))
	}
}

// UGeluF32 performs element-wise GELU for float32 (contiguous memory)
func UGeluF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			xSq := x * x
			xCube := xSq * x
			alpha := x + 0.044715*xCube
			out[i] = float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha)))
		}
		return
	}
	for i := range numel {
		x := float64(inp[i])
		xSq := x * x
		xCube := xSq * x
		alpha := x + 0.044715*xCube
		out[i] = float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha)))
	}
}

// UGeluF64 performs element-wise GELU for float64 (contiguous memory)
func UGeluF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			xSq := x * x
			xCube := xSq * x
			alpha := x + 0.044715*xCube
			out[i] = 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha))
		}
		return
	}
	for i := range numel {
		x := inp[i]
		xSq := x * x
		xCube := xSq * x
		alpha := x + 0.044715*xCube
		out[i] = 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha))
	}
}

// UGeluU8 performs element-wise GELU for uint8 (contiguous memory)
func UGeluU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// UGeluU32 performs element-wise GELU for uint32 (contiguous memory)
func UGeluU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// UGeluI64 performs element-wise GELU for int64 (contiguous memory)
func UGeluI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// UGeluStrided performs element-wise GELU for type T (strided memory)
func UGeluStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UGelu(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			xSq := x * x
			xCube := xSq * x
			alpha := x + 0.044715*xCube
			out[i] = T(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha)))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := float64(inp[stridedI])
		xSq := x * x
		xCube := xSq * x
		alpha := x + 0.044715*xCube
		out[i] = T(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha)))
	}
}

// UGeluStridedF32 performs element-wise GELU for float32 (strided memory)
func UGeluStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UGeluF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			xSq := x * x
			xCube := xSq * x
			alpha := x + 0.044715*xCube
			out[i] = float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha)))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := float64(inp[stridedI])
		xSq := x * x
		xCube := xSq * x
		alpha := x + 0.044715*xCube
		out[i] = float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha)))
	}
}

// UGeluStridedF64 performs element-wise GELU for float64 (strided memory)
func UGeluStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UGeluF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			xSq := x * x
			xCube := xSq * x
			alpha := x + 0.044715*xCube
			out[i] = 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		xSq := x * x
		xCube := xSq * x
		alpha := x + 0.044715*xCube
		out[i] = 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha))
	}
}

// UGeluStridedU8 performs element-wise GELU for uint8 (strided memory)
func UGeluStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UGeluU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// UGeluStridedU32 performs element-wise GELU for uint32 (strided memory)
func UGeluStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UGeluU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// UGeluStridedI64 performs element-wise GELU for int64 (strided memory)
func UGeluStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UGeluI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// UGeluErf performs element-wise GELU (ERF-based) for type T (contiguous memory)
func UGeluErf[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = T(x * 0.5 * (1 + math.Erf(x/math.Sqrt(2))))
		}
		return
	}
	for i := range numel {
		x := float64(inp[i])
		out[i] = T(x * 0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UGeluErfF32 performs element-wise GELU (ERF-based) for float32 (contiguous memory)
func UGeluErfF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = float32(x * 0.5 * (1 + math.Erf(x/math.Sqrt(2))))
		}
		return
	}
	for i := range numel {
		x := float64(inp[i])
		out[i] = float32(x * 0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UGeluErfF64 performs element-wise GELU (ERF-based) for float64 (contiguous memory)
func UGeluErfF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			out[i] = x * 0.5 * (1 + math.Erf(x/math.Sqrt(2)))
		}
		return
	}
	for i := range numel {
		x := inp[i]
		out[i] = x * 0.5 * (1 + math.Erf(x/math.Sqrt(2)))
	}
}

// UGeluErfU8 performs element-wise GELU (ERF-based) for uint8 (contiguous memory)
func UGeluErfU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// UGeluErfU32 performs element-wise GELU (ERF-based) for uint32 (contiguous memory)
func UGeluErfU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// UGeluErfI64 performs element-wise GELU (ERF-based) for int64 (contiguous memory)
func UGeluErfI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// UGeluErfStrided performs element-wise GELU (ERF-based) for type T (strided memory)
func UGeluErfStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UGeluErf(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = T(x * 0.5 * (1 + math.Erf(x/math.Sqrt(2))))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = T(x * 0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UGeluErfStridedF32 performs element-wise GELU (ERF-based) for float32 (strided memory)
func UGeluErfStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UGeluErfF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = float32(x * 0.5 * (1 + math.Erf(x/math.Sqrt(2))))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = float32(x * 0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UGeluErfStridedF64 performs element-wise GELU (ERF-based) for float64 (strided memory)
func UGeluErfStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UGeluErfF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			out[i] = x * 0.5 * (1 + math.Erf(x/math.Sqrt(2)))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		out[i] = x * 0.5 * (1 + math.Erf(x/math.Sqrt(2)))
	}
}

// UGeluErfStridedU8 performs element-wise GELU (ERF-based) for uint8 (strided memory)
func UGeluErfStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UGeluErfU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// UGeluErfStridedU32 performs element-wise GELU (ERF-based) for uint32 (strided memory)
func UGeluErfStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UGeluErfU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// UGeluErfStridedI64 performs element-wise GELU (ERF-based) for int64 (strided memory)
func UGeluErfStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UGeluErfI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// URelu performs element-wise ReLU for type T (contiguous memory)
func URelu[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			if out[i] < 0 {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x < 0 {
			x = 0
		}
		out[i] = x
	}
}

// UReluF32 performs element-wise ReLU for float32 (contiguous memory)
func UReluF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			if out[i] < 0 {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x < 0 {
			x = 0
		}
		out[i] = x
	}
}

// UReluF64 performs element-wise ReLU for float64 (contiguous memory)
func UReluF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			if out[i] < 0 {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x < 0 {
			x = 0
		}
		out[i] = x
	}
}

// UReluU8 performs element-wise ReLU for uint8 (contiguous memory)
func UReluU8(numel int, inp, out []uint8) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UReluU32 performs element-wise ReLU for uint32 (contiguous memory)
func UReluU32(numel int, inp, out []uint32) {
	if inp == nil {
		return
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UReluI64 performs element-wise ReLU for int64 (contiguous memory)
func UReluI64(numel int, inp, out []int64) {
	if inp == nil {
		for i := range numel {
			if out[i] < 0 {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		x := max(inp[i], 0)
		out[i] = x
	}
}

// UReluStrided performs element-wise ReLU for type T (strided memory)
func UReluStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		URelu(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			if out[i] < 0 {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x < 0 {
			x = 0
		}
		out[i] = x
	}
}

// UReluStridedF32 performs element-wise ReLU for float32 (strided memory)
func UReluStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UReluF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			if out[i] < 0 {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x < 0 {
			x = 0
		}
		out[i] = x
	}
}

// UReluStridedF64 performs element-wise ReLU for float64 (strided memory)
func UReluStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UReluF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			if out[i] < 0 {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x < 0 {
			x = 0
		}
		out[i] = x
	}
}

// UReluStridedU8 performs element-wise ReLU for uint8 (strided memory)
func UReluStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UReluU8(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UReluStridedU32 performs element-wise ReLU for uint32 (strided memory)
func UReluStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UReluU32(numel, inp, out)
		return
	}
	if inp == nil {
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UReluStridedI64 performs element-wise ReLU for int64 (strided memory)
func UReluStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UReluI64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			if out[i] < 0 {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := max(inp[stridedI], 0)
		out[i] = x
	}
}

// UElu performs element-wise ELU for type T with parameter alpha (contiguous memory)
func UElu[T D](alpha T, numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x <= 0 {
				out[i] = alpha * (T(math.Exp(float64(x))) - 1)
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x > 0 {
			out[i] = x
		} else {
			out[i] = alpha * (T(math.Exp(float64(x))) - 1)
		}
	}
}

// UEluF32 performs element-wise ELU for float32 with parameter alpha (contiguous memory)
func UEluF32(alpha float32, numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x <= 0 {
				out[i] = alpha * (float32(math.Exp(float64(x))) - 1)
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x > 0 {
			out[i] = x
		} else {
			out[i] = alpha * (float32(math.Exp(float64(x))) - 1)
		}
	}
}

// UEluF64 performs element-wise ELU for float64 with parameter alpha (contiguous memory)
func UEluF64(alpha float64, numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x <= 0 {
				out[i] = alpha * (math.Exp(x) - 1)
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x > 0 {
			out[i] = x
		} else {
			out[i] = alpha * (math.Exp(x) - 1)
		}
	}
}

// UEluU8 performs element-wise ELU for uint8 with parameter alpha (contiguous memory)
func UEluU8(alpha uint8, numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// UEluU32 performs element-wise ELU for uint32 with parameter alpha (contiguous memory)
func UEluU32(alpha uint32, numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// UEluI64 performs element-wise ELU for int64 with parameter alpha (contiguous memory)
func UEluI64(alpha int64, numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// UEluStrided performs element-wise ELU for type T with parameter alpha (strided memory)
func UEluStrided[T D](numel, ndims int, dims, strides []int, alpha T, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UElu(alpha, numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x <= 0 {
				out[i] = alpha * (T(math.Exp(float64(x))) - 1)
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = x
		} else {
			out[i] = alpha * (T(math.Exp(float64(x))) - 1)
		}
	}
}

// UEluStridedF32 performs element-wise ELU for float32 with parameter alpha (strided memory)
func UEluStridedF32(numel, ndims int, dims, strides []int, alpha float32, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UEluF32(alpha, numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x <= 0 {
				out[i] = alpha * (float32(math.Exp(float64(x))) - 1)
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = x
		} else {
			out[i] = alpha * (float32(math.Exp(float64(x))) - 1)
		}
	}
}

// UEluStridedF64 performs element-wise ELU for float64 with parameter alpha (strided memory)
func UEluStridedF64(numel, ndims int, dims, strides []int, alpha float64, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UEluF64(alpha, numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x <= 0 {
				out[i] = alpha * (math.Exp(x) - 1)
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = x
		} else {
			out[i] = alpha * (math.Exp(x) - 1)
		}
	}
}

// UEluStridedU8 performs element-wise ELU for uint8 with parameter alpha (strided memory)
func UEluStridedU8(numel, ndims int, dims, strides []int, alpha uint8, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UEluU8(alpha, numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// UEluStridedU32 performs element-wise ELU for uint32 with parameter alpha (strided memory)
func UEluStridedU32(numel, ndims int, dims, strides []int, alpha uint32, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UEluU32(alpha, numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// UEluStridedI64 performs element-wise ELU for int64 with parameter alpha (strided memory)
func UEluStridedI64(numel, ndims int, dims, strides []int, alpha int64, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UEluI64(alpha, numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// USilu performs element-wise SiLU for type T (contiguous memory)
func USilu[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = T(x / (1 + math.Exp(-x)))
		}
		return
	}
	for i := range numel {
		x := float64(inp[i])
		out[i] = T(x / (1 + math.Exp(-x)))
	}
}

// USiluF32 performs element-wise SiLU for float32 (contiguous memory)
func USiluF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = float32(x / (1 + math.Exp(-x)))
		}
		return
	}
	for i := range numel {
		x := float64(inp[i])
		out[i] = float32(x / (1 + math.Exp(-x)))
	}
}

// USiluF64 performs element-wise SiLU for float64 (contiguous memory)
func USiluF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			out[i] = x / (1 + math.Exp(-x))
		}
		return
	}
	for i := range numel {
		x := inp[i]
		out[i] = x / (1 + math.Exp(-x))
	}
}

// USiluU8 performs element-wise SiLU for uint8 (contiguous memory)
func USiluU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// USiluU32 performs element-wise SiLU for uint32 (contiguous memory)
func USiluU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// USiluI64 performs element-wise SiLU for int64 (contiguous memory)
func USiluI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// USiluStrided performs element-wise SiLU for type T (strided memory)
func USiluStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		USilu(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = T(x / (1 + math.Exp(-x)))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = T(x / (1 + math.Exp(-x)))
	}
}

// USiluStridedF32 performs element-wise SiLU for float32 (strided memory)
func USiluStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		USiluF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = float32(x / (1 + math.Exp(-x)))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = float32(x / (1 + math.Exp(-x)))
	}
}

// USiluStridedF64 performs element-wise SiLU for float64 (strided memory)
func USiluStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		USiluF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			out[i] = x / (1 + math.Exp(-x))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		out[i] = x / (1 + math.Exp(-x))
	}
}

// USiluStridedU8 performs element-wise SiLU for uint8 (strided memory)
func USiluStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		USiluU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// USiluStridedU32 performs element-wise SiLU for uint32 (strided memory)
func USiluStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		USiluU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// USiluStridedI64 performs element-wise SiLU for int64 (strided memory)
func USiluStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		USiluI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// UPowf performs element-wise power for type T with parameter param (contiguous memory)
func UPowf[T D](param T, numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Pow(float64(out[i]), float64(param)))
		}
		return
	}
	for i := range numel {
		out[i] = T(math.Pow(float64(inp[i]), float64(param)))
	}
}

// UPowfF32 performs element-wise power for float32 with parameter param (contiguous memory)
func UPowfF32(param float32, numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Pow(float64(out[i]), float64(param)))
		}
		return
	}
	for i := range numel {
		out[i] = float32(math.Pow(float64(inp[i]), float64(param)))
	}
}

// UPowfF64 performs element-wise power for float64 with parameter param (contiguous memory)
func UPowfF64(param float64, numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			out[i] = math.Pow(out[i], param)
		}
		return
	}
	for i := range numel {
		out[i] = math.Pow(inp[i], param)
	}
}

// UPowfU8 performs element-wise power for uint8 with parameter param (contiguous memory)
func UPowfU8(param uint8, numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// UPowfU32 performs element-wise power for uint32 with parameter param (contiguous memory)
func UPowfU32(param uint32, numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// UPowfI64 performs element-wise power for int64 with parameter param (contiguous memory)
func UPowfI64(param int64, numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// UPowfStrided performs element-wise power for type T with parameter param (strided memory)
func UPowfStrided[T D](numel, ndims int, dims, strides []int, param T, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		UPowf(param, numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Pow(float64(out[i]), float64(param)))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = T(math.Pow(float64(inp[stridedI]), float64(param)))
	}
}

// UPowfStridedF32 performs element-wise power for float32 with parameter param (strided memory)
func UPowfStridedF32(numel, ndims int, dims, strides []int, param float32, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		UPowfF32(param, numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = float32(math.Pow(float64(out[i]), float64(param)))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = float32(math.Pow(float64(inp[stridedI]), float64(param)))
	}
}

// UPowfStridedF64 performs element-wise power for float64 with parameter param (strided memory)
func UPowfStridedF64(numel, ndims int, dims, strides []int, param float64, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		UPowfF64(param, numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = math.Pow(out[i], param)
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		out[i] = math.Pow(inp[stridedI], param)
	}
}

// UPowfStridedU8 performs element-wise power for uint8 with parameter param (strided memory)
func UPowfStridedU8(numel, ndims int, dims, strides []int, param uint8, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		UPowfU8(param, numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// UPowfStridedU32 performs element-wise power for uint32 with parameter param (strided memory)
func UPowfStridedU32(numel, ndims int, dims, strides []int, param uint32, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		UPowfU32(param, numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// UPowfStridedI64 performs element-wise power for int64 with parameter param (strided memory)
func UPowfStridedI64(numel, ndims int, dims, strides []int, param int64, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		UPowfI64(param, numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

// USignF32 performs element-wise sign for float32 (contiguous memory)
func USignF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = 1
			} else if x < 0 {
				out[i] = -1
			} else {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x > 0 {
			out[i] = 1
		} else if x < 0 {
			out[i] = -1
		} else {
			out[i] = 0
		}
	}
}

// USignF64 performs element-wise sign for float64 (contiguous memory)
func USignF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = 1
			} else if x < 0 {
				out[i] = -1
			} else {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x > 0 {
			out[i] = 1
		} else if x < 0 {
			out[i] = -1
		} else {
			out[i] = 0
		}
	}
}

// USignU8 performs element-wise sign for uint8 (contiguous memory)
func USignU8(numel int, inp, out []uint8) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = 1
			} else {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x > 0 {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

// USignU32 performs element-wise sign for uint32 (contiguous memory)
func USignU32(numel int, inp, out []uint32) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = 1
			} else {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x > 0 {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

// USignI64 performs element-wise sign for int64 (contiguous memory)
func USignI64(numel int, inp, out []int64) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = 1
			} else if x < 0 {
				out[i] = -1
			} else {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x > 0 {
			out[i] = 1
		} else if x < 0 {
			out[i] = -1
		} else {
			out[i] = 0
		}
	}
}

// USignStridedF32 performs element-wise sign for float32 (strided memory)
func USignStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		USignF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = 1
			} else if x < 0 {
				out[i] = -1
			} else {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = 1
		} else if x < 0 {
			out[i] = -1
		} else {
			out[i] = 0
		}
	}
}

// USignStridedF64 performs element-wise sign for float64 (strided memory)
func USignStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		USignF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = 1
			} else if x < 0 {
				out[i] = -1
			} else {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = 1
		} else if x < 0 {
			out[i] = -1
		} else {
			out[i] = 0
		}
	}
}

// USignStridedU8 performs element-wise sign for uint8 (strided memory)
func USignStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		USignU8(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = 1
			} else {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

// USignStridedU32 performs element-wise sign for uint32 (strided memory)
func USignStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		USignU32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = 1
			} else {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
}

// USignStridedI64 performs element-wise sign for int64 (strided memory)
func USignStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		USignI64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = 1
			} else if x < 0 {
				out[i] = -1
			} else {
				out[i] = 0
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = 1
		} else if x < 0 {
			out[i] = -1
		} else {
			out[i] = 0
		}
	}
}

// USigmoid performs element-wise sigmoid for type T (contiguous memory)
func USigmoid[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = T(1 / (1 + math.Exp(-x)))
		}
		return
	}
	for i := range numel {
		x := float64(inp[i])
		out[i] = T(1 / (1 + math.Exp(-x)))
	}
}

// USigmoidF32 performs element-wise sigmoid for float32 (contiguous memory)
func USigmoidF32(numel int, inp, out []float32) {
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = float32(1 / (1 + math.Exp(-x)))
		}
		return
	}
	for i := range numel {
		x := float64(inp[i])
		out[i] = float32(1 / (1 + math.Exp(-x)))
	}
}

// USigmoidF64 performs element-wise sigmoid for float64 (contiguous memory)
func USigmoidF64(numel int, inp, out []float64) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			out[i] = 1 / (1 + math.Exp(-x))
		}
		return
	}
	for i := range numel {
		x := inp[i]
		out[i] = 1 / (1 + math.Exp(-x))
	}
}

// USigmoidU8 performs element-wise sigmoid for uint8 (contiguous memory)
func USigmoidU8(numel int, inp, out []uint8) {
	panic("no unary function for u8")
}

// USigmoidU32 performs element-wise sigmoid for uint32 (contiguous memory)
func USigmoidU32(numel int, inp, out []uint32) {
	panic("no unary function for u32")
}

// USigmoidI64 performs element-wise sigmoid for int64 (contiguous memory)
func USigmoidI64(numel int, inp, out []int64) {
	panic("no unary function for i64")
}

// USigmoidStrided performs element-wise sigmoid for type T (strided memory)
func USigmoidStrided[T D](numel, ndims int, dims, strides []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		USigmoid(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = T(1 / (1 + math.Exp(-x)))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = T(1 / (1 + math.Exp(-x)))
	}
}

// USigmoidStridedF32 performs element-wise sigmoid for float32 (strided memory)
func USigmoidStridedF32(numel, ndims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		USigmoidF32(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := float64(out[i])
			out[i] = float32(1 / (1 + math.Exp(-x)))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = float32(1 / (1 + math.Exp(-x)))
	}
}

// USigmoidStridedF64 performs element-wise sigmoid for float64 (strided memory)
func USigmoidStridedF64(numel, ndims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		USigmoidF64(numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			out[i] = 1 / (1 + math.Exp(-x))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		x := inp[stridedI]
		out[i] = 1 / (1 + math.Exp(-x))
	}
}

// USigmoidStridedU8 performs element-wise sigmoid for uint8 (strided memory)
func USigmoidStridedU8(numel, ndims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		USigmoidU8(numel, inp, out)
		return
	}
	panic("no unary function for u8")
}

// USigmoidStridedU32 performs element-wise sigmoid for uint32 (strided memory)
func USigmoidStridedU32(numel, ndims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		USigmoidU32(numel, inp, out)
		return
	}
	panic("no unary function for u32")
}

// USigmoidStridedI64 performs element-wise sigmoid for int64 (strided memory)
func USigmoidStridedI64(numel, ndims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		USigmoidI64(numel, inp, out)
		return
	}
	panic("no unary function for i64")
}

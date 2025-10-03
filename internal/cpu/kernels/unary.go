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

// UCopyStrided performs element-wise copy for type T (strided memory)
func UCopyStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UCopy[T](numel, inp, out)
		return
	}
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UCopyStridedF32 performs element-wise copy for float32 (strided memory)
func UCopyStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UCopyF32(numel, inp, out)
		return
	}
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = inp[stridedI]
	}
}

// UCopyStridedF64 performs element-wise copy for float64 (strided memory)
func UCopyStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UCopyF64(numel, inp, out)
		return
	}
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
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

// UNegStrided performs element-wise negation for type T (strided memory)
func UNegStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UNeg[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = -out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = -inp[stridedI]
	}
}

// UNegStridedF32 performs element-wise negation for float32 (strided memory)
func UNegStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = -inp[stridedI]
	}
}

// UNegStridedF64 performs element-wise negation for float64 (strided memory)
func UNegStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
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

// URecipStrided performs element-wise reciprocal for type T (strided memory)
func URecipStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		URecip[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(1) / out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(1) / inp[stridedI]
	}
}

// URecipStridedF32 performs element-wise reciprocal for float32 (strided memory)
func URecipStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = 1.0 / inp[stridedI]
	}
}

// URecipStridedF64 performs element-wise reciprocal for float64 (strided memory)
func URecipStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = 1.0 / inp[stridedI]
	}
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

// UExpStrided performs element-wise exponential for type T (strided memory)
func UExpStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UExp[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Exp(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Exp(float64(inp[stridedI])))
	}
}

// UExpStridedF32 performs element-wise exponential for float32 (strided memory)
func UExpStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Exp(float64(inp[stridedI])))
	}
}

// UExpStridedF64 performs element-wise exponential for float64 (strided memory)
func UExpStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Exp(inp[stridedI])
	}
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

// ULogStrided performs element-wise logarithm for type T (strided memory)
func ULogStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		ULog[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Log(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Log(float64(inp[stridedI])))
	}
}

// ULogStridedF32 performs element-wise logarithm for float32 (strided memory)
func ULogStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Log(float64(inp[stridedI])))
	}
}

// ULogStridedF64 performs element-wise logarithm for float64 (strided memory)
func ULogStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Log(inp[stridedI])
	}
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

// USinStrided performs element-wise sine for type T (strided memory)
func USinStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		USin[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Sin(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Sin(float64(inp[stridedI])))
	}
}

// USinStridedF32 performs element-wise sine for float32 (strided memory)
func USinStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Sin(float64(inp[stridedI])))
	}
}

// USinStridedF64 performs element-wise sine for float64 (strided memory)
func USinStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Sin(inp[stridedI])
	}
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

// UCosStrided performs element-wise cosine for type T (strided memory)
func UCosStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UCos[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Cos(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Cos(float64(inp[stridedI])))
	}
}

// UCosStridedF32 performs element-wise cosine for float32 (strided memory)
func UCosStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Cos(float64(inp[stridedI])))
	}
}

// UCosStridedF64 performs element-wise cosine for float64 (strided memory)
func UCosStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Cos(inp[stridedI])
	}
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

// UTanhStrided performs element-wise tanh for type T (strided memory)
func UTanhStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UTanh[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Tanh(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Tanh(float64(inp[stridedI])))
	}
}

// UTanhStridedF32 performs element-wise tanh for float32 (strided memory)
func UTanhStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Tanh(float64(inp[stridedI])))
	}
}

// UTanhStridedF64 performs element-wise tanh for float64 (strided memory)
func UTanhStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Tanh(inp[stridedI])
	}
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

// UErfStrided performs element-wise erf for type T (strided memory)
func UErfStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UErf[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Erf(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Erf(float64(inp[stridedI])))
	}
}

// UErfStridedF32 performs element-wise erf for float32 (strided memory)
func UErfStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Erf(float64(inp[stridedI])))
	}
}

// UErfStridedF64 performs element-wise erf for float64 (strided memory)
func UErfStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Erf(inp[stridedI])
	}
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

// UCeilStrided performs element-wise ceil for type T (strided memory)
func UCeilStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UCeil[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Ceil(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Ceil(float64(inp[stridedI])))
	}
}

// UCeilStridedF32 performs element-wise ceil for float32 (strided memory)
func UCeilStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Ceil(float64(inp[stridedI])))
	}
}

// UCeilStridedF64 performs element-wise ceil for float64 (strided memory)
func UCeilStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Ceil(inp[stridedI])
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

// UFloorStrided performs element-wise floor for type T (strided memory)
func UFloorStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UFloor[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Floor(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Floor(float64(inp[stridedI])))
	}
}

// UFloorStridedF32 performs element-wise floor for float32 (strided memory)
func UFloorStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Floor(float64(inp[stridedI])))
	}
}

// UFloorStridedF64 performs element-wise floor for float64 (strided memory)
func UFloorStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Floor(inp[stridedI])
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

// URoundStrided performs element-wise round for type T (strided memory)
func URoundStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		URound[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Round(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Round(float64(inp[stridedI])))
	}
}

// URoundStridedF32 performs element-wise round for float32 (strided memory)
func URoundStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Round(float64(inp[stridedI])))
	}
}

// URoundStridedF64 performs element-wise round for float64 (strided memory)
func URoundStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Round(inp[stridedI])
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

// UNormcdfStrided performs element-wise normal CDF for type T (strided memory)
func UNormcdfStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UNormcdf[T](numel, inp, out)
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = T(0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UNormcdfStridedF32 performs element-wise normal CDF for float32 (strided memory)
func UNormcdfStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = float32(0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UNormcdfStridedF64 performs element-wise normal CDF for float64 (strided memory)
func UNormcdfStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = 0.5 * (1 + math.Erf(inp[stridedI]/math.Sqrt(2)))
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

// UAbsStrided performs element-wise absolute value for type T (strided memory)
func UAbsStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UAbs[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Abs(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Abs(float64(inp[stridedI])))
	}
}

// UAbsStridedF32 performs element-wise absolute value for float32 (strided memory)
func UAbsStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Abs(float64(inp[stridedI])))
	}
}

// UAbsStridedF64 performs element-wise absolute value for float64 (strided memory)
func UAbsStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Abs(inp[stridedI])
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

// USqrStrided performs element-wise square for type T (strided memory)
func USqrStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		USqr[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = out[i] * out[i]
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = inp[stridedI] * inp[stridedI]
	}
}

// USqrStridedF32 performs element-wise square for float32 (strided memory)
func USqrStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = inp[stridedI] * inp[stridedI]
	}
}

// USqrStridedF64 performs element-wise square for float64 (strided memory)
func USqrStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
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

// USqrtStrided performs element-wise square root for type T (strided memory)
func USqrtStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		USqrt[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Sqrt(float64(out[i])))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Sqrt(float64(inp[stridedI])))
	}
}

// USqrtStridedF32 performs element-wise square root for float32 (strided memory)
func USqrtStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Sqrt(float64(inp[stridedI])))
	}
}

// USqrtStridedF64 performs element-wise square root for float64 (strided memory)
func USqrtStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Sqrt(inp[stridedI])
	}
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

// UGeluStrided performs element-wise GELU for type T (strided memory)
func UGeluStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UGelu[T](numel, inp, out)
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := float64(inp[stridedI])
		xSq := x * x
		xCube := xSq * x
		alpha := x + 0.044715*xCube
		out[i] = T(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha)))
	}
}

// UGeluStridedF32 performs element-wise GELU for float32 (strided memory)
func UGeluStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := float64(inp[stridedI])
		xSq := x * x
		xCube := xSq * x
		alpha := x + 0.044715*xCube
		out[i] = float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha)))
	}
}

// UGeluStridedF64 performs element-wise GELU for float64 (strided memory)
func UGeluStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := inp[stridedI]
		xSq := x * x
		xCube := xSq * x
		alpha := x + 0.044715*xCube
		out[i] = 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*alpha))
	}
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

// UGeluErfStrided performs element-wise GELU (ERF-based) for type T (strided memory)
func UGeluErfStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UGeluErf[T](numel, inp, out)
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = T(x * 0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UGeluErfStridedF32 performs element-wise GELU (ERF-based) for float32 (strided memory)
func UGeluErfStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = float32(x * 0.5 * (1 + math.Erf(x/math.Sqrt(2))))
	}
}

// UGeluErfStridedF64 performs element-wise GELU (ERF-based) for float64 (strided memory)
func UGeluErfStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := inp[stridedI]
		out[i] = x * 0.5 * (1 + math.Erf(x/math.Sqrt(2)))
	}
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

// UReluStrided performs element-wise ReLU for type T (strided memory)
func UReluStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		URelu[T](numel, inp, out)
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := inp[stridedI]
		if x < 0 {
			x = 0
		}
		out[i] = x
	}
}

// UReluStridedF32 performs element-wise ReLU for float32 (strided memory)
func UReluStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := inp[stridedI]
		if x < 0 {
			x = 0
		}
		out[i] = x
	}
}

// UReluStridedF64 performs element-wise ReLU for float64 (strided memory)
func UReluStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := inp[stridedI]
		if x < 0 {
			x = 0
		}
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

// UEluStrided performs element-wise ELU for type T with parameter alpha (strided memory)
func UEluStrided[T D](alpha T, numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UElu[T](alpha, numel, inp, out)
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = x
		} else {
			out[i] = alpha * (T(math.Exp(float64(x))) - 1)
		}
	}
}

// UEluStridedF32 performs element-wise ELU for float32 with parameter alpha (strided memory)
func UEluStridedF32(alpha float32, numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = x
		} else {
			out[i] = alpha * (float32(math.Exp(float64(x))) - 1)
		}
	}
}

// UEluStridedF64 performs element-wise ELU for float64 with parameter alpha (strided memory)
func UEluStridedF64(alpha float64, numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = x
		} else {
			out[i] = alpha * (math.Exp(x) - 1)
		}
	}
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

// USiluStrided performs element-wise SiLU for type T (strided memory)
func USiluStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		USilu[T](numel, inp, out)
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = T(x / (1 + math.Exp(-x)))
	}
}

// USiluStridedF32 performs element-wise SiLU for float32 (strided memory)
func USiluStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = float32(x / (1 + math.Exp(-x)))
	}
}

// USiluStridedF64 performs element-wise SiLU for float64 (strided memory)
func USiluStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := inp[stridedI]
		out[i] = x / (1 + math.Exp(-x))
	}
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

// UPowfStrided performs element-wise power for type T with parameter param (strided memory)
func UPowfStrided[T D](param T, numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		UPowf[T](param, numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			out[i] = T(math.Pow(float64(out[i]), float64(param)))
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = T(math.Pow(float64(inp[stridedI]), float64(param)))
	}
}

// UPowfStridedF32 performs element-wise power for float32 with parameter param (strided memory)
func UPowfStridedF32(param float32, numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = float32(math.Pow(float64(inp[stridedI]), float64(param)))
	}
}

// UPowfStridedF64 performs element-wise power for float64 with parameter param (strided memory)
func UPowfStridedF64(param float64, numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		out[i] = math.Pow(inp[stridedI], param)
	}
}

// USign performs element-wise sign for type T (contiguous memory)
func USign[T D](numel int, inp, out []T) {
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = T(1)
			} else if x < 0 {
				out[i] = T(-1)
			} else {
				out[i] = T(0)
			}
		}
		return
	}
	for i := range numel {
		x := inp[i]
		if x > 0 {
			out[i] = T(1)
		} else if x < 0 {
			out[i] = T(-1)
		} else {
			out[i] = T(0)
		}
	}
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

// USignStrided performs element-wise sign for type T (strided memory)
func USignStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		USign[T](numel, inp, out)
		return
	}
	if inp == nil {
		for i := range numel {
			x := out[i]
			if x > 0 {
				out[i] = T(1)
			} else if x < 0 {
				out[i] = T(-1)
			} else {
				out[i] = T(0)
			}
		}
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := inp[stridedI]
		if x > 0 {
			out[i] = T(1)
		} else if x < 0 {
			out[i] = T(-1)
		} else {
			out[i] = T(0)
		}
	}
}

// USignStridedF32 performs element-wise sign for float32 (strided memory)
func USignStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
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
func USignStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
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

// USigmoidStrided performs element-wise sigmoid for type T (strided memory)
func USigmoidStrided[T D](numel, numDims int, dims, strides []int, inp, out []T) {
	if IsContiguous(numDims, dims, strides) {
		USigmoid[T](numel, inp, out)
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = T(1 / (1 + math.Exp(-x)))
	}
}

// USigmoidStridedF32 performs element-wise sigmoid for float32 (strided memory)
func USigmoidStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := float64(inp[stridedI])
		out[i] = float32(1 / (1 + math.Exp(-x)))
	}
}

// USigmoidStridedF64 performs element-wise sigmoid for float64 (strided memory)
func USigmoidStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		x := inp[stridedI]
		out[i] = 1 / (1 + math.Exp(-x))
	}
}

package kernels

import "math"

// UnaryCopyF32 performs element-wise copy for float32 (contiguous memory)
func UnaryCopyF32(numel int, inp, out []float32) {
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UnaryCopyStridedF32 performs element-wise copy for float32 (strided memory)
func UnaryCopyStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryCopyF32(numel, inp, out)
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

// UnaryCopyF64 performs element-wise copy for float64 (contiguous memory)
func UnaryCopyF64(numel int, inp, out []float64) {
	if inp == nil {
		return // No-op for in-place
	}
	for i := range numel {
		out[i] = inp[i]
	}
}

// UnaryCopyStridedF64 performs element-wise copy for float64 (strided memory)
func UnaryCopyStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryCopyF64(numel, inp, out)
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

// UnaryNegF32 performs element-wise negation for float32 (contiguous memory)
func UnaryNegF32(numel int, inp, out []float32) {
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

// UnaryNegStridedF32 performs element-wise negation for float32 (strided memory)
func UnaryNegStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryNegF32(numel, inp, out)
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

// UnaryNegF64 performs element-wise negation for float64 (contiguous memory)
func UnaryNegF64(numel int, inp, out []float64) {
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

// UnaryNegStridedF64 performs element-wise negation for float64 (strided memory)
func UnaryNegStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryNegF64(numel, inp, out)
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

// UnaryRecipF32 performs element-wise reciprocal for float32 (contiguous memory)
func UnaryRecipF32(numel int, inp, out []float32) {
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

// UnaryRecipStridedF32 performs element-wise reciprocal for float32 (strided memory)
func UnaryRecipStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryRecipF32(numel, inp, out)
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

// UnaryRecipF64 performs element-wise reciprocal for float64 (contiguous memory)
func UnaryRecipF64(numel int, inp, out []float64) {
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

// UnaryRecipStridedF64 performs element-wise reciprocal for float64 (strided memory)
func UnaryRecipStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryRecipF64(numel, inp, out)
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

// UnaryExpF32 performs element-wise exponential for float32 (contiguous memory)
func UnaryExpF32(numel int, inp, out []float32) {
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

// UnaryExpStridedF32 performs element-wise exponential for float32 (strided memory)
func UnaryExpStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryExpF32(numel, inp, out)
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

// UnaryExpF64 performs element-wise exponential for float64 (contiguous memory)
func UnaryExpF64(numel int, inp, out []float64) {
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

// UnaryExpStridedF64 performs element-wise exponential for float64 (strided memory)
func UnaryExpStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryExpF64(numel, inp, out)
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

// UnaryLogF32 performs element-wise logarithm for float32 (contiguous memory)
func UnaryLogF32(numel int, inp, out []float32) {
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

// UnaryLogStridedF32 performs element-wise logarithm for float32 (strided memory)
func UnaryLogStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryLogF32(numel, inp, out)
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

// UnaryLogF64 performs element-wise logarithm for float64 (contiguous memory)
func UnaryLogF64(numel int, inp, out []float64) {
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

// UnaryLogStridedF64 performs element-wise logarithm for float64 (strided memory)
func UnaryLogStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryLogF64(numel, inp, out)
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

// UnarySinF32 performs element-wise sine for float32 (contiguous memory)
func UnarySinF32(numel int, inp, out []float32) {
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

// UnarySinStridedF32 performs element-wise sine for float32 (strided memory)
func UnarySinStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnarySinF32(numel, inp, out)
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

// UnarySinF64 performs element-wise sine for float64 (contiguous memory)
func UnarySinF64(numel int, inp, out []float64) {
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

// UnarySinStridedF64 performs element-wise sine for float64 (strided memory)
func UnarySinStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnarySinF64(numel, inp, out)
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

// UnaryCosF32 performs element-wise cosine for float32 (contiguous memory)
func UnaryCosF32(numel int, inp, out []float32) {
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

// UnaryCosStridedF32 performs element-wise cosine for float32 (strided memory)
func UnaryCosStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryCosF32(numel, inp, out)
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

// UnaryCosF64 performs element-wise cosine for float64 (contiguous memory)
func UnaryCosF64(numel int, inp, out []float64) {
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

// UnaryCosStridedF64 performs element-wise cosine for float64 (strided memory)
func UnaryCosStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryCosF64(numel, inp, out)
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

// UnaryTanhF32 performs element-wise tanh for float32 (contiguous memory)
func UnaryTanhF32(numel int, inp, out []float32) {
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

// UnaryTanhStridedF32 performs element-wise tanh for float32 (strided memory)
func UnaryTanhStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryTanhF32(numel, inp, out)
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

// UnaryTanhF64 performs element-wise tanh for float64 (contiguous memory)
func UnaryTanhF64(numel int, inp, out []float64) {
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

// UnaryTanhStridedF64 performs element-wise tanh for float64 (strided memory)
func UnaryTanhStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryTanhF64(numel, inp, out)
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

// UnaryErfF32 performs element-wise erf for float32 (contiguous memory)
func UnaryErfF32(numel int, inp, out []float32) {
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

// UnaryErfStridedF32 performs element-wise erf for float32 (strided memory)
func UnaryErfStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryErfF32(numel, inp, out)
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

// UnaryErfF64 performs element-wise erf for float64 (contiguous memory)
func UnaryErfF64(numel int, inp, out []float64) {
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

// UnaryErfStridedF64 performs element-wise erf for float64 (strided memory)
func UnaryErfStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryErfF64(numel, inp, out)
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

// UnaryCeilF32 performs element-wise ceil for float32 (contiguous memory)
func UnaryCeilF32(numel int, inp, out []float32) {
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

// UnaryCeilStridedF32 performs element-wise ceil for float32 (strided memory)
func UnaryCeilStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryCeilF32(numel, inp, out)
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

// UnaryCeilF64 performs element-wise ceil for float64 (contiguous memory)
func UnaryCeilF64(numel int, inp, out []float64) {
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

// UnaryCeilStridedF64 performs element-wise ceil for float64 (strided memory)
func UnaryCeilStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryCeilF64(numel, inp, out)
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

// UnaryFloorF32 performs element-wise floor for float32 (contiguous memory)
func UnaryFloorF32(numel int, inp, out []float32) {
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

// UnaryFloorStridedF32 performs element-wise floor for float32 (strided memory)
func UnaryFloorStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryFloorF32(numel, inp, out)
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

// UnaryFloorF64 performs element-wise floor for float64 (contiguous memory)
func UnaryFloorF64(numel int, inp, out []float64) {
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

// UnaryFloorStridedF64 performs element-wise floor for float64 (strided memory)
func UnaryFloorStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryFloorF64(numel, inp, out)
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

// UnaryRoundF32 performs element-wise round for float32 (contiguous memory)
func UnaryRoundF32(numel int, inp, out []float32) {
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

// UnaryRoundStridedF32 performs element-wise round for float32 (strided memory)
func UnaryRoundStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryRoundF32(numel, inp, out)
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

// UnaryRoundF64 performs element-wise round for float64 (contiguous memory)
func UnaryRoundF64(numel int, inp, out []float64) {
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

// UnaryRoundStridedF64 performs element-wise round for float64 (strided memory)
func UnaryRoundStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryRoundF64(numel, inp, out)
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

// UnaryNormcdfF32 performs element-wise normal CDF for float32 (contiguous memory)
func UnaryNormcdfF32(numel int, inp, out []float32) {
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

// UnaryNormcdfStridedF32 performs element-wise normal CDF for float32 (strided memory)
func UnaryNormcdfStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryNormcdfF32(numel, inp, out)
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

// UnaryNormcdfF64 performs element-wise normal CDF for float64 (contiguous memory)
func UnaryNormcdfF64(numel int, inp, out []float64) {
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

// UnaryNormcdfStridedF64 performs element-wise normal CDF for float64 (strided memory)
func UnaryNormcdfStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryNormcdfF64(numel, inp, out)
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

// UnaryAbsF32 performs element-wise absolute value for float32 (contiguous memory)
func UnaryAbsF32(numel int, inp, out []float32) {
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

// UnaryAbsStridedF32 performs element-wise absolute value for float32 (strided memory)
func UnaryAbsStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryAbsF32(numel, inp, out)
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

// UnaryAbsF64 performs element-wise absolute value for float64 (contiguous memory)
func UnaryAbsF64(numel int, inp, out []float64) {
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

// UnaryAbsStridedF64 performs element-wise absolute value for float64 (strided memory)
func UnaryAbsStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryAbsF64(numel, inp, out)
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

// UnarySqrF32 performs element-wise square for float32 (contiguous memory)
func UnarySqrF32(numel int, inp, out []float32) {
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

// UnarySqrStridedF32 performs element-wise square for float32 (strided memory)
func UnarySqrStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnarySqrF32(numel, inp, out)
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

// UnarySqrF64 performs element-wise square for float64 (contiguous memory)
func UnarySqrF64(numel int, inp, out []float64) {
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

// UnarySqrStridedF64 performs element-wise square for float64 (strided memory)
func UnarySqrStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnarySqrF64(numel, inp, out)
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

// UnarySqrtF32 performs element-wise square root for float32 (contiguous memory)
func UnarySqrtF32(numel int, inp, out []float32) {
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

// UnarySqrtStridedF32 performs element-wise square root for float32 (strided memory)
func UnarySqrtStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnarySqrtF32(numel, inp, out)
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

// UnarySqrtF64 performs element-wise square root for float64 (contiguous memory)
func UnarySqrtF64(numel int, inp, out []float64) {
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

// UnarySqrtStridedF64 performs element-wise square root for float64 (strided memory)
func UnarySqrtStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnarySqrtF64(numel, inp, out)
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

// UnaryGeluF32 performs element-wise GELU for float32 (contiguous memory)
func UnaryGeluF32(numel int, inp, out []float32) {
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

// UnaryGeluStridedF32 performs element-wise GELU for float32 (strided memory)
func UnaryGeluStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryGeluF32(numel, inp, out)
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

// UnaryGeluF64 performs element-wise GELU for float64 (contiguous memory)
func UnaryGeluF64(numel int, inp, out []float64) {
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

// UnaryGeluStridedF64 performs element-wise GELU for float64 (strided memory)
func UnaryGeluStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryGeluF64(numel, inp, out)
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

// UnaryGeluErfF32 performs element-wise GELU (ERF-based) for float32 (contiguous memory)
func UnaryGeluErfF32(numel int, inp, out []float32) {
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

// UnaryGeluErfStridedF32 performs element-wise GELU (ERF-based) for float32 (strided memory)
func UnaryGeluErfStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryGeluErfF32(numel, inp, out)
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

// UnaryGeluErfF64 performs element-wise GELU (ERF-based) for float64 (contiguous memory)
func UnaryGeluErfF64(numel int, inp, out []float64) {
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

// UnaryGeluErfStridedF64 performs element-wise GELU (ERF-based) for float64 (strided memory)
func UnaryGeluErfStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryGeluErfF64(numel, inp, out)
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

// UnaryReluF32 performs element-wise ReLU for float32 (contiguous memory)
func UnaryReluF32(numel int, inp, out []float32) {
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

// UnaryReluStridedF32 performs element-wise ReLU for float32 (strided memory)
func UnaryReluStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryReluF32(numel, inp, out)
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

// UnaryReluF64 performs element-wise ReLU for float64 (contiguous memory)
func UnaryReluF64(numel int, inp, out []float64) {
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

// UnaryReluStridedF64 performs element-wise ReLU for float64 (strided memory)
func UnaryReluStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryReluF64(numel, inp, out)
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

// UnaryEluF32 performs element-wise ELU for float32 with parameter alpha (contiguous memory)
func UnaryEluF32(alpha float32, numel int, inp, out []float32) {
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

// UnaryEluStridedF32 performs element-wise ELU for float32 with parameter alpha (strided memory)
func UnaryEluStridedF32(alpha float32, numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryEluF32(alpha, numel, inp, out)
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

// UnaryEluF64 performs element-wise ELU for float64 with parameter alpha (contiguous memory)
func UnaryEluF64(alpha float64, numel int, inp, out []float64) {
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

// UnaryEluStridedF64 performs element-wise ELU for float64 with parameter alpha (strided memory)
func UnaryEluStridedF64(alpha float64, numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryEluF64(alpha, numel, inp, out)
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

// UnarySiluF32 performs element-wise SiLU for float32 (contiguous memory)
func UnarySiluF32(numel int, inp, out []float32) {
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

// UnarySiluStridedF32 performs element-wise SiLU for float32 (strided memory)
func UnarySiluStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnarySiluF32(numel, inp, out)
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

// UnarySiluF64 performs element-wise SiLU for float64 (contiguous memory)
func UnarySiluF64(numel int, inp, out []float64) {
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

// UnarySiluStridedF64 performs element-wise SiLU for float64 (strided memory)
func UnarySiluStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnarySiluF64(numel, inp, out)
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

// UnaryPowfF32 performs element-wise power for float32 with parameter param (contiguous memory)
func UnaryPowfF32(param float32, numel int, inp, out []float32) {
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

// UnaryPowfStridedF32 performs element-wise power for float32 with parameter param (strided memory)
func UnaryPowfStridedF32(param float32, numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnaryPowfF32(param, numel, inp, out)
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

// UnaryPowfF64 performs element-wise power for float64 with parameter param (contiguous memory)
func UnaryPowfF64(param float64, numel int, inp, out []float64) {
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

// UnaryPowfStridedF64 performs element-wise power for float64 with parameter param (strided memory)
func UnaryPowfStridedF64(param float64, numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnaryPowfF64(param, numel, inp, out)
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

// UnarySignF32 performs element-wise sign for float32 (contiguous memory)
func UnarySignF32(numel int, inp, out []float32) {
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

// UnarySignStridedF32 performs element-wise sign for float32 (strided memory)
func UnarySignStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnarySignF32(numel, inp, out)
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

// UnarySignF64 performs element-wise sign for float64 (contiguous memory)
func UnarySignF64(numel int, inp, out []float64) {
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

// UnarySignStridedF64 performs element-wise sign for float64 (strided memory)
func UnarySignStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnarySignF64(numel, inp, out)
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

// UnarySigmoidF32 performs element-wise sigmoid for float32 (contiguous memory)
func UnarySigmoidF32(numel int, inp, out []float32) {
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

// UnarySigmoidStridedF32 performs element-wise sigmoid for float32 (strided memory)
func UnarySigmoidStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		UnarySigmoidF32(numel, inp, out)
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

// UnarySigmoidF64 performs element-wise sigmoid for float64 (contiguous memory)
func UnarySigmoidF64(numel int, inp, out []float64) {
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

// UnarySigmoidStridedF64 performs element-wise sigmoid for float64 (strided memory)
func UnarySigmoidStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		UnarySigmoidF64(numel, inp, out)
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

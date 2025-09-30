package kernels

// Common converters
var (
	// F32 converters
	F32ToF32 = func(x float32) float32 { return x }
	F32ToF64 = func(x float32) float64 { return float64(x) }
	F32ToU8  = func(x float32) uint8 { return uint8(x) }
	F32ToU32 = func(x float32) uint32 { return uint32(x) }
	F32ToI64 = func(x float32) int64 { return int64(x) }

	// F64 converters
	F64ToF32 = func(x float64) float32 { return float32(x) }
	F64ToF64 = func(x float64) float64 { return x }
	F64ToU8  = func(x float64) uint8 { return uint8(x) }
	F64ToU32 = func(x float64) uint32 { return uint32(x) }
	F64ToI64 = func(x float64) int64 { return int64(x) }

	// U8 converters
	U8ToU8  = func(x uint8) uint8 { return x }
	U8ToU32 = func(x uint8) uint32 { return uint32(x) }
	U8ToF32 = func(x uint8) float32 { return float32(x) }
	U8ToF64 = func(x uint8) float64 { return float64(x) }
	U8ToI64 = func(x uint8) int64 { return int64(x) }

	// U32 converters
	U32ToU8  = func(x uint32) uint8 { return uint8(x) }
	U32ToU32 = func(x uint32) uint32 { return x }
	U32ToF32 = func(x uint32) float32 { return float32(x) }
	U32ToF64 = func(x uint32) float64 { return float64(x) }
	U32ToI64 = func(x uint32) int64 { return int64(x) }

	// I64 converters
	I64ToU8  = func(x int64) uint8 { return uint8(x) }
	I64ToU32 = func(x int64) uint32 { return uint32(x) }
	I64ToI64 = func(x int64) int64 { return x }
	I64ToF32 = func(x int64) float32 { return float32(x) }
	I64ToF64 = func(x int64) float64 { return float64(x) }
)

// Generic casting with Go generics
func Cast[S, T any](numel int, inp []S, out []T, converter func(S) T) {
	for i := range numel {
		out[i] = converter(inp[i])
	}
}

func CastStrided[S, T any](numel, numDims int, dims, strides []int, inp []S, out []T, converter func(S) T) {
	if IsContiguous(numDims, dims, strides) {
		Cast(numel, inp, out, converter)
		return
	}
	for i := range numel {
		out[i] = converter(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

// F32 casting operations (contiguous)
func CastF32F32(numel int, inp, out []float32) {
	for i := range numel {
		out[i] = inp[i]
	}
}

func CastF32F64(numel int, inp []float32, out []float64) {
	for i := range numel {
		out[i] = float64(inp[i])
	}
}

func CastF32U8(numel int, inp []float32, out []uint8) {
	for i := range numel {
		out[i] = uint8(inp[i])
	}
}

func CastF32U32(numel int, inp []float32, out []uint32) {
	for i := range numel {
		out[i] = uint32(inp[i])
	}
}

func CastF32I64(numel int, inp []float32, out []int64) {
	for i := range numel {
		out[i] = int64(inp[i])
	}
}

// F64 casting operations (contiguous)
func CastF64F32(numel int, inp []float64, out []float32) {
	for i := range numel {
		out[i] = float32(inp[i])
	}
}

func CastF64F64(numel int, inp, out []float64) {
	for i := range numel {
		out[i] = inp[i]
	}
}

func CastF64U8(numel int, inp []float64, out []uint8) {
	for i := range numel {
		out[i] = uint8(inp[i])
	}
}

func CastF64U32(numel int, inp []float64, out []uint32) {
	for i := range numel {
		out[i] = uint32(inp[i])
	}
}

func CastF64I64(numel int, inp []float64, out []int64) {
	for i := range numel {
		out[i] = int64(inp[i])
	}
}

// U8 casting operations (contiguous)
func CastU8U8(numel int, inp, out []uint8) {
	for i := range numel {
		out[i] = inp[i]
	}
}

func CastU8U32(numel int, inp []uint8, out []uint32) {
	for i := range numel {
		out[i] = uint32(inp[i])
	}
}

func CastU8I64(numel int, inp []uint8, out []int64) {
	for i := range numel {
		out[i] = int64(inp[i])
	}
}

func CastU8F32(numel int, inp []uint8, out []float32) {
	for i := range numel {
		out[i] = float32(inp[i])
	}
}

func CastU8F64(numel int, inp []uint8, out []float64) {
	for i := range numel {
		out[i] = float64(inp[i])
	}
}

// U32 casting operations (contiguous)
func CastU32U8(numel int, inp []uint32, out []uint8) {
	for i := range numel {
		out[i] = uint8(inp[i])
	}
}

func CastU32U32(numel int, inp, out []uint32) {
	for i := range numel {
		out[i] = inp[i]
	}
}

func CastU32I64(numel int, inp []uint32, out []int64) {
	for i := range numel {
		out[i] = int64(inp[i])
	}
}

func CastU32F32(numel int, inp []uint32, out []float32) {
	for i := range numel {
		out[i] = float32(inp[i])
	}
}

func CastU32F64(numel int, inp []uint32, out []float64) {
	for i := range numel {
		out[i] = float64(inp[i])
	}
}

// I64 casting operations (contiguous)
func CastI64U8(numel int, inp []int64, out []uint8) {
	for i := range numel {
		out[i] = uint8(inp[i])
	}
}

func CastI64U32(numel int, inp []int64, out []uint32) {
	for i := range numel {
		out[i] = uint32(inp[i])
	}
}

func CastI64I64(numel int, inp, out []int64) {
	for i := range numel {
		out[i] = inp[i]
	}
}

func CastI64F32(numel int, inp []int64, out []float32) {
	for i := range numel {
		out[i] = float32(inp[i])
	}
}

func CastI64F64(numel int, inp []int64, out []float64) {
	for i := range numel {
		out[i] = float64(inp[i])
	}
}

// Strided casting operations
func CastStridedF32F32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		CastF32F32(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = inp[GetStridedIndex(i, numDims, dims, strides)]
	}
}

func CastStridedF32F64(numel, numDims int, dims, strides []int, inp []float32, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		CastF32F64(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = float64(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedF32U8(numel, numDims int, dims, strides []int, inp []float32, out []uint8) {
	if IsContiguous(numDims, dims, strides) {
		CastF32U8(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = uint8(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedF32U32(numel, numDims int, dims, strides []int, inp []float32, out []uint32) {
	if IsContiguous(numDims, dims, strides) {
		CastF32U32(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = uint32(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedF32I64(numel, numDims int, dims, strides []int, inp []float32, out []int64) {
	if IsContiguous(numDims, dims, strides) {
		CastF32I64(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = int64(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedF64F32(numel, numDims int, dims, strides []int, inp []float64, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		CastF64F32(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = float32(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedF64F64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		CastF64F64(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = inp[GetStridedIndex(i, numDims, dims, strides)]
	}
}

func CastStridedF64U8(numel, numDims int, dims, strides []int, inp []float64, out []uint8) {
	if IsContiguous(numDims, dims, strides) {
		CastF64U8(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = uint8(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedF64U32(numel, numDims int, dims, strides []int, inp []float64, out []uint32) {
	if IsContiguous(numDims, dims, strides) {
		CastF64U32(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = uint32(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedF64I64(numel, numDims int, dims, strides []int, inp []float64, out []int64) {
	if IsContiguous(numDims, dims, strides) {
		CastF64I64(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = int64(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedU8U8(numel, numDims int, dims, strides []int, inp, out []uint8) {
	if IsContiguous(numDims, dims, strides) {
		CastU8U8(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = inp[GetStridedIndex(i, numDims, dims, strides)]
	}
}

func CastStridedU8U32(numel, numDims int, dims, strides []int, inp []uint8, out []uint32) {
	if IsContiguous(numDims, dims, strides) {
		CastU8U32(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = uint32(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedU8I64(numel, numDims int, dims, strides []int, inp []uint8, out []int64) {
	if IsContiguous(numDims, dims, strides) {
		CastU8I64(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = int64(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedU8F32(numel, numDims int, dims, strides []int, inp []uint8, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		CastU8F32(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = float32(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedU8F64(numel, numDims int, dims, strides []int, inp []uint8, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		CastU8F64(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = float64(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedU32U8(numel, numDims int, dims, strides []int, inp []uint32, out []uint8) {
	if IsContiguous(numDims, dims, strides) {
		CastU32U8(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = uint8(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedU32U32(numel, numDims int, dims, strides []int, inp, out []uint32) {
	if IsContiguous(numDims, dims, strides) {
		CastU32U32(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = inp[GetStridedIndex(i, numDims, dims, strides)]
	}
}

func CastStridedU32I64(numel, numDims int, dims, strides []int, inp []uint32, out []int64) {
	if IsContiguous(numDims, dims, strides) {
		CastU32I64(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = int64(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedU32F32(numel, numDims int, dims, strides []int, inp []uint32, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		CastU32F32(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = float32(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedU32F64(numel, numDims int, dims, strides []int, inp []uint32, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		CastU32F64(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = float64(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedI64U8(numel, numDims int, dims, strides []int, inp []int64, out []uint8) {
	if IsContiguous(numDims, dims, strides) {
		CastI64U8(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = uint8(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedI64U32(numel, numDims int, dims, strides []int, inp []int64, out []uint32) {
	if IsContiguous(numDims, dims, strides) {
		CastI64U32(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = uint32(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedI64I64(numel, numDims int, dims, strides []int, inp, out []int64) {
	if IsContiguous(numDims, dims, strides) {
		CastI64I64(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = inp[GetStridedIndex(i, numDims, dims, strides)]
	}
}

func CastStridedI64F32(numel, numDims int, dims, strides []int, inp []int64, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		CastI64F32(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = float32(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

func CastStridedI64F64(numel, numDims int, dims, strides []int, inp []int64, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		CastI64F64(numel, inp, out)
		return
	}
	for i := range numel {
		out[i] = float64(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
}

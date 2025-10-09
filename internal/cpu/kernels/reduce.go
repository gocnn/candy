package kernels

import (
	"math"
	"slices"
)

// FastSum computes the sum over the last dimension for type T
func FastSum[T D](numel, ndims int, dims []int, src, dst []T) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		var sum T
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[j]
		}
		dst[i] = sum
	}
}

// FastSumF32 computes the sum over the last dimension for float32
func FastSumF32(numel, ndims int, dims []int, src, dst []float32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		sum := float32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[j]
		}
		dst[i] = sum
	}
}

// FastSumF64 computes the sum over the last dimension for float64
func FastSumF64(numel, ndims int, dims []int, src, dst []float64) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		sum := float64(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[j]
		}
		dst[i] = sum
	}
}

// FastSumU8 computes the sum over the last dimension for uint8
func FastSumU8(numel, ndims int, dims []int, src, dst []uint8) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		sum := uint8(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[j]
		}
		dst[i] = sum
	}
}

// FastSumU32 computes the sum over the last dimension for uint32
func FastSumU32(numel, ndims int, dims []int, src, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		sum := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[j]
		}
		dst[i] = sum
	}
}

// FastSumI64 computes the sum over the last dimension for int64
func FastSumI64(numel, ndims int, dims []int, src, dst []int64) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		sum := int64(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[j]
		}
		dst[i] = sum
	}
}

// FastSumStrided computes the sum over the last dimension for type T with strided memory
func FastSumStrided[T D](numel, ndims int, dims, strides []int, src, dst []T) {
	if IsContiguous(ndims, dims, strides) {
		FastSum(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		var sum T
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[GetStridedIndex(j, ndims, dims, strides)]
		}
		dst[i] = sum
	}
}

// FastSumStridedF32 computes the sum over the last dimension for float32 with strided memory
func FastSumStridedF32(numel, ndims int, dims, strides []int, src, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		FastSumF32(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		sum := float32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[GetStridedIndex(j, ndims, dims, strides)]
		}
		dst[i] = sum
	}
}

// FastSumStridedF64 computes the sum over the last dimension for float64 with strided memory
func FastSumStridedF64(numel, ndims int, dims, strides []int, src, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		FastSumF64(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		sum := float64(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[GetStridedIndex(j, ndims, dims, strides)]
		}
		dst[i] = sum
	}
}

// FastSumStridedU8 computes the sum over the last dimension for uint8 with strided memory
func FastSumStridedU8(numel, ndims int, dims, strides []int, src, dst []uint8) {
	if IsContiguous(ndims, dims, strides) {
		FastSumU8(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		sum := uint8(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[GetStridedIndex(j, ndims, dims, strides)]
		}
		dst[i] = sum
	}
}

// FastSumStridedU32 computes the sum over the last dimension for uint32 with strided memory
func FastSumStridedU32(numel, ndims int, dims, strides []int, src, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastSumU32(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		sum := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[GetStridedIndex(j, ndims, dims, strides)]
		}
		dst[i] = sum
	}
}

// FastSumStridedI64 computes the sum over the last dimension for int64 with strided memory
func FastSumStridedI64(numel, ndims int, dims, strides []int, src, dst []int64) {
	if IsContiguous(ndims, dims, strides) {
		FastSumI64(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		sum := int64(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[GetStridedIndex(j, ndims, dims, strides)]
		}
		dst[i] = sum
	}
}

// FastMin computes the minimum over the last dimension for type T
func FastMin[T D](numel, ndims int, dims []int, src, dst []T) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		startIdx := i * dims[ndims-1]
		minVal := src[startIdx] // init minVal to first element
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx + 1; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
			}
		}
		dst[i] = minVal
	}
}

// FastMinF32 computes the minimum over the last dimension for float32
func FastMinF32(numel, ndims int, dims []int, src, dst []float32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := float32(math.MaxFloat32)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
			}
		}
		dst[i] = minVal
	}
}

// FastMinF64 computes the minimum over the last dimension for float64
func FastMinF64(numel, ndims int, dims []int, src, dst []float64) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := float64(math.MaxFloat64)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
			}
		}
		dst[i] = minVal
	}
}

// FastMinU8 computes the minimum over the last dimension for uint8
func FastMinU8(numel, ndims int, dims []int, src, dst []uint8) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := uint8(math.MaxUint8)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
			}
		}
		dst[i] = minVal
	}
}

// FastMinU32 computes the minimum over the last dimension for uint32
func FastMinU32(numel, ndims int, dims []int, src, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := uint32(math.MaxUint32)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
			}
		}
		dst[i] = minVal
	}
}

// FastMinI64 computes the minimum over the last dimension for int64
func FastMinI64(numel, ndims int, dims []int, src, dst []int64) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := int64(math.MaxInt64)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
			}
		}
		dst[i] = minVal
	}
}

// FastMinStrided computes the minimum over the last dimension for type T with strided memory
func FastMinStrided[T D](numel, ndims int, dims, strides []int, src, dst []T) {
	if IsContiguous(ndims, dims, strides) {
		FastMin(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		startIdx := i * dims[ndims-1]
		minVal := src[GetStridedIndex(startIdx, ndims, dims, strides)]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx + 1; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// FastMinStridedF32 computes the minimum over the last dimension for float32 with strided memory
func FastMinStridedF32(numel, ndims int, dims, strides []int, src, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		FastMinF32(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := float32(math.MaxFloat32)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// FastMinStridedF64 computes the minimum over the last dimension for float64 with strided memory
func FastMinStridedF64(numel, ndims int, dims, strides []int, src, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		FastMinF64(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := float64(math.MaxFloat64)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// FastMinStridedU8 computes the minimum over the last dimension for uint8 with strided memory
func FastMinStridedU8(numel, ndims int, dims, strides []int, src, dst []uint8) {
	if IsContiguous(ndims, dims, strides) {
		FastMinU8(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := uint8(math.MaxUint8)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// FastMinStridedU32 computes the minimum over the last dimension for uint32 with strided memory
func FastMinStridedU32(numel, ndims int, dims, strides []int, src, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastMinU32(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := uint32(math.MaxUint32)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// FastMinStridedI64 computes the minimum over the last dimension for int64 with strided memory
func FastMinStridedI64(numel, ndims int, dims, strides []int, src, dst []int64) {
	if IsContiguous(ndims, dims, strides) {
		FastMinI64(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := int64(math.MaxInt64)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// FastMax computes the maximum over the last dimension for type T
func FastMax[T D](numel, ndims int, dims []int, src, dst []T) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		startIdx := i * dims[ndims-1]
		maxVal := src[startIdx]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx + 1; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
			}
		}
		dst[i] = maxVal
	}
}

// FastMaxF32 computes the maximum over the last dimension for float32
func FastMaxF32(numel, ndims int, dims []int, src, dst []float32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := float32(-math.MaxFloat32)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
			}
		}
		dst[i] = maxVal
	}
}

// FastMaxF64 computes the maximum over the last dimension for float64
func FastMaxF64(numel, ndims int, dims []int, src, dst []float64) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := float64(-math.MaxFloat64)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
			}
		}
		dst[i] = maxVal
	}
}

// FastMaxU8 computes the maximum over the last dimension for uint8
func FastMaxU8(numel, ndims int, dims []int, src, dst []uint8) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := uint8(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
			}
		}
		dst[i] = maxVal
	}
}

// FastMaxU32 computes the maximum over the last dimension for uint32
func FastMaxU32(numel, ndims int, dims []int, src, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
			}
		}
		dst[i] = maxVal
	}
}

// FastMaxI64 computes the maximum over the last dimension for int64
func FastMaxI64(numel, ndims int, dims []int, src, dst []int64) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := int64(math.MinInt64)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
			}
		}
		dst[i] = maxVal
	}
}

// FastMaxStrided computes the maximum over the last dimension for type T with strided memory
func FastMaxStrided[T D](numel, ndims int, dims, strides []int, src, dst []T) {
	if IsContiguous(ndims, dims, strides) {
		FastMax(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		startIdx := i * dims[ndims-1]
		maxVal := src[GetStridedIndex(startIdx, ndims, dims, strides)]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx + 1; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val > maxVal {
				maxVal = val
			}
		}
		dst[i] = maxVal
	}
}

// FastMaxStridedF32 computes the maximum over the last dimension for float32 with strided memory
func FastMaxStridedF32(numel, ndims int, dims, strides []int, src, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		FastMaxF32(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := float32(-math.MaxFloat32)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val > maxVal {
				maxVal = val
			}
		}
		dst[i] = maxVal
	}
}

// FastMaxStridedF64 computes the maximum over the last dimension for float64 with strided memory
func FastMaxStridedF64(numel, ndims int, dims, strides []int, src, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		FastMaxF64(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := float64(-math.MaxFloat64)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val > maxVal {
				maxVal = val
			}
		}
		dst[i] = maxVal
	}
}

// FastArgmin computes the index of the minimum over the last dimension for type T
func FastArgmin[T D](numel, ndims int, dims []int, src []T, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		minVal := src[startIdx]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx + 1; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminF32 computes the index of the minimum over the last dimension for float32
func FastArgminF32(numel, ndims int, dims []int, src []float32, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := float32(math.MaxFloat32)
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminF64 computes the index of the minimum over the last dimension for float64
func FastArgminF64(numel, ndims int, dims []int, src []float64, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := float64(math.MaxFloat64)
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminU8 computes the index of the minimum over the last dimension for uint8
func FastArgminU8(numel, ndims int, dims []int, src []uint8, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := uint8(math.MaxUint8)
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminU32 computes the index of the minimum over the last dimension for uint32
func FastArgminU32(numel, ndims int, dims []int, src []uint32, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := uint32(math.MaxUint32)
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminI64 computes the index of the minimum over the last dimension for int64
func FastArgminI64(numel, ndims int, dims []int, src []int64, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := int64(math.MaxInt64)
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminStrided computes the index of the minimum over the last dimension for type T with strided memory
func FastArgminStrided[T D](numel, ndims int, dims, strides []int, src []T, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgmin(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		minVal := src[GetStridedIndex(startIdx, ndims, dims, strides)]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx + 1; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminStridedF32 computes the index of the minimum over the last dimension for float32 with strided memory
func FastArgminStridedF32(numel, ndims int, dims, strides []int, src []float32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgminF32(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := float32(math.MaxFloat32)
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminStridedF64 computes the index of the minimum over the last dimension for float64 with strided memory
func FastArgminStridedF64(numel, ndims int, dims, strides []int, src []float64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgminF64(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := float64(math.MaxFloat64)
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminStridedU8 computes the index of the minimum over the last dimension for uint8 with strided memory
func FastArgminStridedU8(numel, ndims int, dims, strides []int, src []uint8, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgminU8(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := uint8(math.MaxUint8)
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminStridedU32 computes the index of the minimum over the last dimension for uint32 with strided memory
func FastArgminStridedU32(numel, ndims int, dims, strides []int, src []uint32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgminU32(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := uint32(math.MaxUint32)
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminStridedI64 computes the index of the minimum over the last dimension for int64 with strided memory
func FastArgminStridedI64(numel, ndims int, dims, strides []int, src []int64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgminI64(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		minVal := int64(math.MaxInt64)
		minIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val < minVal {
				minVal = val
				minIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgmax computes the index of the maximum over the last dimension for type T
func FastArgmax[T D](numel, ndims int, dims []int, src []T, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		maxVal := src[startIdx]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx + 1; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxF32 computes the index of the maximum over the last dimension for float32
func FastArgmaxF32(numel, ndims int, dims []int, src []float32, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := float32(-math.MaxFloat32)
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxF64 computes the index of the maximum over the last dimension for float64
func FastArgmaxF64(numel, ndims int, dims []int, src []float64, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := float64(-math.MaxFloat64)
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxU8 computes the index of the maximum over the last dimension for uint8
func FastArgmaxU8(numel, ndims int, dims []int, src []uint8, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := uint8(0)
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxU32 computes the index of the maximum over the last dimension for uint32
func FastArgmaxU32(numel, ndims int, dims []int, src []uint32, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := uint32(0)
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxI64 computes the index of the maximum over the last dimension for int64
func FastArgmaxI64(numel, ndims int, dims []int, src []int64, dst []uint32) {
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := int64(math.MinInt64)
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxStrided computes the index of the maximum over the last dimension for type T with strided memory
func FastArgmaxStrided[T D](numel, ndims int, dims, strides []int, src []T, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgmax(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		maxVal := src[GetStridedIndex(startIdx, ndims, dims, strides)]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx + 1; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxStridedF32 computes the index of the maximum over the last dimension for float32 with strided memory
func FastArgmaxStridedF32(numel, ndims int, dims, strides []int, src []float32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgmaxF32(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := float32(-math.MaxFloat32)
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxStridedF64 computes the index of the maximum over the last dimension for float64 with strided memory
func FastArgmaxStridedF64(numel, ndims int, dims, strides []int, src []float64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgmaxF64(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := float64(-math.MaxFloat64)
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxStridedU8 computes the index of the maximum over the last dimension for uint8 with strided memory
func FastArgmaxStridedU8(numel, ndims int, dims, strides []int, src []uint8, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgmaxU8(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := uint8(0)
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxStridedU32 computes the index of the maximum over the last dimension for uint32 with strided memory
func FastArgmaxStridedU32(numel, ndims int, dims, strides []int, src []uint32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgmaxU32(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := uint32(0)
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxStridedI64 computes the index of the maximum over the last dimension for int64 with strided memory
func FastArgmaxStridedI64(numel, ndims int, dims, strides []int, src []int64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastArgmaxI64(numel, ndims, dims, src, dst)
		return
	}
	dstSize := numel / dims[ndims-1]
	for i := range dstSize {
		maxVal := int64(math.MinInt64)
		maxIdx := uint32(0)
		startIdx := i * dims[ndims-1]
		stopIdx := min(startIdx+dims[ndims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, ndims, dims, strides)]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j % dims[ndims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// Sum performs sum reduction over specified dimension indices for type T
func Sum[T D](numel, ndims int, dims, sumDims []int, inp, out []T) {
	for i := range numel {
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[i]
	}
}

// SumF32 performs sum reduction over specified dimension indices for float32
func SumF32(numel, ndims int, dims, sumDims []int, inp, out []float32) {
	for i := range numel {
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[i]
	}
}

// SumF64 performs sum reduction over specified dimension indices for float64
func SumF64(numel, ndims int, dims, sumDims []int, inp, out []float64) {
	for i := range numel {
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[i]
	}
}

// SumU8 performs sum reduction over specified dimension indices for uint8
func SumU8(numel, ndims int, dims, sumDims []int, inp, out []uint8) {
	for i := range numel {
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[i]
	}
}

// SumU32 performs sum reduction over specified dimension indices for uint32
func SumU32(numel, ndims int, dims, sumDims []int, inp, out []uint32) {
	for i := range numel {
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[i]
	}
}

// SumI64 performs sum reduction over specified dimension indices for int64
func SumI64(numel, ndims int, dims, sumDims []int, inp, out []int64) {
	for i := range numel {
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[i]
	}
}

// SumStrided performs strided sum reduction over specified dimension indices for type T
func SumStrided[T D](numel, ndims int, dims, strides, sumDims []int, inp, out []T) {
	if IsContiguous(ndims, dims, strides) {
		Sum(numel, ndims, dims, sumDims, inp, out)
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[stridedI]
	}
}

// SumStridedF32 performs strided sum reduction over specified dimension indices for float32
func SumStridedF32(numel, ndims int, dims, strides, sumDims []int, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		SumF32(numel, ndims, dims, sumDims, inp, out)
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[stridedI]
	}
}

// SumStridedF64 performs strided sum reduction over specified dimension indices for float64
func SumStridedF64(numel, ndims int, dims, strides, sumDims []int, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		SumF64(numel, ndims, dims, sumDims, inp, out)
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[stridedI]
	}
}

// SumStridedU8 performs strided sum reduction over specified dimension indices for uint8
func SumStridedU8(numel, ndims int, dims, strides, sumDims []int, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		SumU8(numel, ndims, dims, sumDims, inp, out)
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[stridedI]
	}
}

// SumStridedU32 performs strided sum reduction over specified dimension indices for uint32
func SumStridedU32(numel, ndims int, dims, strides, sumDims []int, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		SumU32(numel, ndims, dims, sumDims, inp, out)
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[stridedI]
	}
}

// SumStridedI64 performs strided sum reduction over specified dimension indices for int64
func SumStridedI64(numel, ndims int, dims, strides, sumDims []int, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		SumI64(numel, ndims, dims, sumDims, inp, out)
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, ndims, dims, strides)
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := ndims - 1; d >= 0; d-- {
			coord := coords % dims[d]
			coords /= dims[d]
			isSum := slices.Contains(sumDims, d)
			if !isSum {
				dstIndex += coord * currentStride
				currentStride *= dims[d]
			}
		}
		out[dstIndex] += inp[stridedI]
	}
}

// Min computes the minimum over the specified dimension for type T
func Min[T D](numel, ndims int, dims []int, dim int, src, dst []T) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := src[baseIdx] // init to first element
		for j := 1; j < reduceSize; j++ {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
			}
		}
		dst[i] = minVal
	}
}

// MinF32 computes the minimum over the specified dimension for float32
func MinF32(numel, ndims int, dims []int, dim int, src, dst []float32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := float32(math.MaxFloat32)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
			}
		}
		dst[i] = minVal
	}
}

// MinF64 computes the minimum over the specified dimension for float64
func MinF64(numel, ndims int, dims []int, dim int, src, dst []float64) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := math.MaxFloat64
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
			}
		}
		dst[i] = minVal
	}
}

// MinU8 computes the minimum over the specified dimension for uint8
func MinU8(numel, ndims int, dims []int, dim int, src, dst []uint8) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := uint8(math.MaxUint8)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
			}
		}
		dst[i] = minVal
	}
}

// MinU32 computes the minimum over the specified dimension for uint32
func MinU32(numel, ndims int, dims []int, dim int, src, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := uint32(math.MaxUint32)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
			}
		}
		dst[i] = minVal
	}
}

// MinI64 computes the minimum over the specified dimension for int64
func MinI64(numel, ndims int, dims []int, dim int, src, dst []int64) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := int64(math.MaxInt64)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
			}
		}
		dst[i] = minVal
	}
}

// MinStrided computes the minimum over the specified dimension for type T with strided memory
func MinStrided[T D](numel, ndims int, dims, strides []int, dim int, src, dst []T) {
	if IsContiguous(ndims, dims, strides) {
		Min(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		stridedFirst := GetStridedIndex(baseIdx, ndims, dims, strides)
		minVal := src[stridedFirst] // init to first element
		for j := 1; j < reduceSize; j++ {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			if src[stridedI] < minVal {
				minVal = src[stridedI]
			}
		}
		dst[i] = minVal
	}
}

// MinStridedF32 computes the minimum over the specified dimension for float32 with strided memory
func MinStridedF32(numel, ndims int, dims, strides []int, dim int, src, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		MinF32(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := float32(math.MaxFloat32)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// MinStridedF64 computes the minimum over the specified dimension for float64 with strided memory
func MinStridedF64(numel, ndims int, dims, strides []int, dim int, src, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		MinF64(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := math.MaxFloat64
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// MinStridedU8 computes the minimum over the specified dimension for uint8 with strided memory
func MinStridedU8(numel, ndims int, dims, strides []int, dim int, src, dst []uint8) {
	if IsContiguous(ndims, dims, strides) {
		MinU8(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := uint8(math.MaxUint8)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// MinStridedU32 computes the minimum over the specified dimension for uint32 with strided memory
func MinStridedU32(numel, ndims int, dims, strides []int, dim int, src, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		MinU32(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := uint32(math.MaxUint32)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// MinStridedI64 computes the minimum over the specified dimension for int64 with strided memory
func MinStridedI64(numel, ndims int, dims, strides []int, dim int, src, dst []int64) {
	if IsContiguous(ndims, dims, strides) {
		MinI64(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := int64(math.MaxInt64)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// Max computes the maximum over the specified dimension for type T
func Max[T D](numel, ndims int, dims []int, dim int, src, dst []T) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := src[baseIdx] // init to first element
		for j := 1; j < reduceSize; j++ {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
			}
		}
		dst[i] = maxVal
	}
}

// MaxF32 computes the maximum over the specified dimension for float32
func MaxF32(numel, ndims int, dims []int, dim int, src, dst []float32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := float32(-math.MaxFloat32)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
			}
		}
		dst[i] = maxVal
	}
}

// MaxF64 computes the maximum over the specified dimension for float64
func MaxF64(numel, ndims int, dims []int, dim int, src, dst []float64) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := -math.MaxFloat64
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
			}
		}
		dst[i] = maxVal
	}
}

// MaxU8 computes the maximum over the specified dimension for uint8
func MaxU8(numel, ndims int, dims []int, dim int, src, dst []uint8) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := uint8(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
			}
		}
		dst[i] = maxVal
	}
}

// MaxU32 computes the maximum over the specified dimension for uint32
func MaxU32(numel, ndims int, dims []int, dim int, src, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
			}
		}
		dst[i] = maxVal
	}
}

// MaxI64 computes the maximum over the specified dimension for int64
func MaxI64(numel, ndims int, dims []int, dim int, src, dst []int64) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := int64(math.MinInt64)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
			}
		}
		dst[i] = maxVal
	}
}

// MaxStrided computes the maximum over the specified dimension for type T with strided memory
func MaxStrided[T D](numel, ndims int, dims, strides []int, dim int, src, dst []T) {
	if IsContiguous(ndims, dims, strides) {
		Max(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		stridedFirst := GetStridedIndex(baseIdx, ndims, dims, strides)
		maxVal := src[stridedFirst] // init to first element
		for j := 1; j < reduceSize; j++ {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			if src[stridedI] > maxVal {
				maxVal = src[stridedI]
			}
		}
		dst[i] = maxVal
	}
}

// MaxStridedF32 computes the maximum over the specified dimension for float32 with strided memory
func MaxStridedF32(numel, ndims int, dims, strides []int, dim int, src, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		MaxF32(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := float32(-math.MaxFloat32)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val > maxVal {
				maxVal = val
			}
		}
		dst[i] = maxVal
	}
}

// MaxStridedF64 computes the maximum over the specified dimension for float64 with strided memory
func MaxStridedF64(numel, ndims int, dims, strides []int, dim int, src, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		MaxF64(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := -math.MaxFloat64
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val > maxVal {
				maxVal = val
			}
		}
		dst[i] = maxVal
	}
}

// MaxStridedU8 computes the maximum over the specified dimension for uint8 with strided memory
func MaxStridedU8(numel, ndims int, dims, strides []int, dim int, src, dst []uint8) {
	if IsContiguous(ndims, dims, strides) {
		MaxU8(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := uint8(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val > maxVal {
				maxVal = val
			}
		}
		dst[i] = maxVal
	}
}

// MaxStridedU32 computes the maximum over the specified dimension for uint32 with strided memory
func MaxStridedU32(numel, ndims int, dims, strides []int, dim int, src, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		MaxU32(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val > maxVal {
				maxVal = val
			}
		}
		dst[i] = maxVal
	}
}

// MaxStridedI64 computes the maximum over the specified dimension for int64 with strided memory
func MaxStridedI64(numel, ndims int, dims, strides []int, dim int, src, dst []int64) {
	if IsContiguous(ndims, dims, strides) {
		MaxI64(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := int64(math.MinInt64)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val > maxVal {
				maxVal = val
			}
		}
		dst[i] = maxVal
	}
}

// Argmin computes the index of the minimum over the specified dimension for type T
func Argmin[T D](numel, ndims int, dims []int, dim int, src []T, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minIdx := uint32(0)
		minVal := src[baseIdx]
		for j := 1; j < reduceSize; j++ {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// ArgminF32 computes the index of the minimum over the specified dimension for float32
func ArgminF32(numel, ndims int, dims []int, dim int, src []float32, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := float32(math.MaxFloat32)
		minIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// ArgminF64 computes the index of the minimum over the specified dimension for float64
func ArgminF64(numel, ndims int, dims []int, dim int, src []float64, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := math.MaxFloat64
		minIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// ArgminU8 computes the index of the minimum over the specified dimension for uint8
func ArgminU8(numel, ndims int, dims []int, dim int, src []uint8, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := uint8(math.MaxUint8)
		minIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// ArgminU32 computes the index of the minimum over the specified dimension for uint32
func ArgminU32(numel, ndims int, dims []int, dim int, src []uint32, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := uint32(math.MaxUint32)
		minIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// ArgminI64 computes the index of the minimum over the specified dimension for int64
func ArgminI64(numel, ndims int, dims []int, dim int, src []int64, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := int64(math.MaxInt64)
		minIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] < minVal {
				minVal = src[idx]
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// ArgminStrided computes the index of the minimum over the specified dimension for type T with strided memory
func ArgminStrided[T D](numel, ndims int, dims, strides []int, dim int, src []T, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		Argmin(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		stridedFirst := GetStridedIndex(baseIdx, ndims, dims, strides)
		minVal := src[stridedFirst]
		minIdx := uint32(0)
		for j := 1; j < reduceSize; j++ {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val < minVal {
				minVal = val
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// ArgminStridedF32 computes the index of the minimum over the specified dimension for float32 with strided memory
func ArgminStridedF32(numel, ndims int, dims, strides []int, dim int, src []float32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ArgminF32(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := float32(math.MaxFloat32)
		minIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val < minVal {
				minVal = val
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// ArgminStridedF64 computes the index of the minimum over the specified dimension for float64 with strided memory
func ArgminStridedF64(numel, ndims int, dims, strides []int, dim int, src []float64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ArgminF64(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := math.MaxFloat64
		minIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val < minVal {
				minVal = val
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// ArgminStridedU8 computes the index of the minimum over the specified dimension for uint8 with strided memory
func ArgminStridedU8(numel, ndims int, dims, strides []int, dim int, src []uint8, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ArgminU8(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := uint8(math.MaxUint8)
		minIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val < minVal {
				minVal = val
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// ArgminStridedU32 computes the index of the minimum over the specified dimension for uint32 with strided memory
func ArgminStridedU32(numel, ndims int, dims, strides []int, dim int, src []uint32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ArgminU32(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := uint32(math.MaxUint32)
		minIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val < minVal {
				minVal = val
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// ArgminStridedI64 computes the index of the minimum over the specified dimension for int64 with strided memory
func ArgminStridedI64(numel, ndims int, dims, strides []int, dim int, src []int64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ArgminI64(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		minVal := int64(math.MaxInt64)
		minIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val < minVal {
				minVal = val
				minIdx = uint32(j)
			}
		}
		dst[i] = minIdx
	}
}

// Argmax computes the index of the maximum over the specified dimension for type T
func Argmax[T D](numel, ndims int, dims []int, dim int, src []T, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxIdx := uint32(0)
		maxVal := src[baseIdx]
		for j := 1; j < reduceSize; j++ {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// ArgmaxF32 computes the index of the maximum over the specified dimension for float32
func ArgmaxF32(numel, ndims int, dims []int, dim int, src []float32, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := float32(-math.MaxFloat32)
		maxIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// ArgmaxF64 computes the index of the maximum over the specified dimension for float64
func ArgmaxF64(numel, ndims int, dims []int, dim int, src []float64, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := -math.MaxFloat64
		maxIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// ArgmaxU8 computes the index of the maximum over the specified dimension for uint8
func ArgmaxU8(numel, ndims int, dims []int, dim int, src []uint8, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := uint8(0)
		maxIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// ArgmaxU32 computes the index of the maximum over the specified dimension for uint32
func ArgmaxU32(numel, ndims int, dims []int, dim int, src []uint32, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := uint32(0)
		maxIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// ArgmaxI64 computes the index of the maximum over the specified dimension for int64
func ArgmaxI64(numel, ndims int, dims []int, dim int, src []int64, dst []uint32) {
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := int64(math.MinInt64)
		maxIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			if src[idx] > maxVal {
				maxVal = src[idx]
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// ArgmaxStrided computes the index of the maximum over the specified dimension for type T with strided memory
func ArgmaxStrided[T D](numel, ndims int, dims, strides []int, dim int, src []T, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		Argmax(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		stridedFirst := GetStridedIndex(baseIdx, ndims, dims, strides)
		maxVal := src[stridedFirst]
		maxIdx := uint32(0)
		for j := 1; j < reduceSize; j++ {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// ArgmaxStridedF32 computes the index of the maximum over the specified dimension for float32 with strided memory
func ArgmaxStridedF32(numel, ndims int, dims, strides []int, dim int, src []float32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ArgmaxF32(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := float32(-math.MaxFloat32)
		maxIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// ArgmaxStridedF64 computes the index of the maximum over the specified dimension for float64 with strided memory
func ArgmaxStridedF64(numel, ndims int, dims, strides []int, dim int, src []float64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ArgmaxF64(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := -math.MaxFloat64
		maxIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// ArgmaxStridedU8 computes the index of the maximum over the specified dimension for uint8 with strided memory
func ArgmaxStridedU8(numel, ndims int, dims, strides []int, dim int, src []uint8, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ArgmaxU8(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := uint8(0)
		maxIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// ArgmaxStridedU32 computes the index of the maximum over the last dimension for uint32 with strided memory
func ArgmaxStridedU32(numel, ndims int, dims, strides []int, dim int, src []uint32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ArgmaxU32(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := uint32(0)
		maxIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// ArgmaxStridedI64 computes the index of the maximum over the specified dimension for int64 with strided memory
func ArgmaxStridedI64(numel, ndims int, dims, strides []int, dim int, src []int64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ArgmaxI64(numel, ndims, dims, dim, src, dst)
		return
	}
	prefix := 1
	for i := range dim {
		prefix *= dims[i]
	}
	reduceSize := dims[dim]
	suffix := 1
	for i := dim + 1; i < ndims; i++ {
		suffix *= dims[i]
	}
	dstSize := numel / reduceSize
	for i := range dstSize {
		outer := i / suffix
		inner := i % suffix
		baseIdx := outer*(reduceSize*suffix) + inner
		maxVal := int64(math.MinInt64)
		maxIdx := uint32(0)
		for j := range reduceSize {
			idx := baseIdx + j*suffix
			stridedI := GetStridedIndex(idx, ndims, dims, strides)
			val := src[stridedI]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j)
			}
		}
		dst[i] = maxIdx
	}
}

// FastSoftmax performs softmax along the last dimension for type T (contiguous memory)
func FastSoftmax[T D](numel, ndims int, dims []int, src, dst []T) {
	var zero T
	switch any(zero).(type) {
	case float32, float64:
	default:
		panic("softmax: unsupported type")
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		maxVal := src[row*ncols]
		for col := range ncols {
			i := row*ncols + col
			if src[i] > maxVal {
				maxVal = src[i]
			}
		}
		var sum T
		for col := range ncols {
			i := row*ncols + col
			val := T(math.Exp(float64(src[i] - maxVal)))
			dst[i] = val
			sum += val
		}
		invSum := T(1) / sum
		for col := range ncols {
			i := row*ncols + col
			dst[i] *= invSum
		}
	}
}

// FastSoftmaxF32 performs softmax along the last dimension for float32 (contiguous memory)
func FastSoftmaxF32(numel int, ndims int, dims []int, src, dst []float32) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		maxVal := float32(-math.MaxFloat32)
		for col := range ncols {
			i := row*ncols + col
			if src[i] > maxVal {
				maxVal = src[i]
			}
		}
		sum := float32(0)
		for col := range ncols {
			i := row*ncols + col
			val := float32(math.Exp(float64(src[i] - maxVal)))
			dst[i] = val
			sum += val
		}
		invSum := 1 / sum
		for col := range ncols {
			i := row*ncols + col
			dst[i] *= invSum
		}
	}
}

// FastSoftmaxF64 performs softmax along the last dimension for float64 (contiguous memory)
func FastSoftmaxF64(numel int, ndims int, dims []int, src, dst []float64) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		maxVal := -math.MaxFloat64
		for col := range ncols {
			i := row*ncols + col
			if src[i] > maxVal {
				maxVal = src[i]
			}
		}
		sum := 0.0
		for col := range ncols {
			i := row*ncols + col
			val := math.Exp(src[i] - maxVal)
			dst[i] = val
			sum += val
		}
		invSum := 1 / sum
		for col := range ncols {
			i := row*ncols + col
			dst[i] *= invSum
		}
	}
}

// FastSoftmaxU8 performs softmax along the last dimension for uint8 (contiguous memory)
func FastSoftmaxU8(numel, ndims int, dims []int, src, dst []uint8) {
	panic("uint8 softmax not implemented")
}

// FastSoftmaxU32 performs softmax along the last dimension for uint32 (contiguous memory)
func FastSoftmaxU32(numel, ndims int, dims []int, src, dst []uint32) {
	panic("uint32 softmax not implemented")
}

// FastSoftmaxI64 performs softmax along the last dimension for int64 (contiguous memory)
func FastSoftmaxI64(numel, ndims int, dims []int, src, dst []int64) {
	panic("int64 softmax not implemented")
}

// FastSoftmaxStrided performs strided softmax along the last dimension for type T
func FastSoftmaxStrided[T D](numel, ndims int, dims, strides []int, src, dst []T) {
	var zero T
	switch any(zero).(type) {
	case float32, float64:
	default:
		panic("softmax_strided: unsupported type")
	}
	if IsContiguous(ndims, dims, strides) {
		FastSoftmax(numel, ndims, dims, src, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		maxVal := src[GetStridedIndex(row*ncols, ndims, dims, strides)]
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, ndims, dims, strides)
			if src[stridedI] > maxVal {
				maxVal = src[stridedI]
			}
		}
		var sum T
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, ndims, dims, strides)
			val := T(math.Exp(float64(src[stridedI] - maxVal)))
			dst[stridedI] = val
			sum += val
		}
		invSum := T(1) / sum
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, ndims, dims, strides)
			dst[stridedI] *= invSum
		}
	}
}

// FastSoftmaxStridedF32 performs strided softmax along the last dimension for float32
func FastSoftmaxStridedF32(numel int, ndims int, dims, strides []int, src, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		FastSoftmaxF32(numel, ndims, dims, src, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		maxVal := float32(-math.MaxFloat32)
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, ndims, dims, strides)
			if src[stridedI] > maxVal {
				maxVal = src[stridedI]
			}
		}
		sum := float32(0)
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, ndims, dims, strides)
			val := float32(math.Exp(float64(src[stridedI] - maxVal)))
			dst[stridedI] = val
			sum += val
		}
		invSum := 1 / sum
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, ndims, dims, strides)
			dst[stridedI] *= invSum
		}
	}
}

// FastSoftmaxStridedF64 performs strided softmax along the last dimension for float64
func FastSoftmaxStridedF64(numel int, ndims int, dims, strides []int, src, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		FastSoftmaxF64(numel, ndims, dims, src, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		maxVal := -math.MaxFloat64
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, ndims, dims, strides)
			if src[stridedI] > maxVal {
				maxVal = src[stridedI]
			}
		}
		sum := 0.0
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, ndims, dims, strides)
			val := math.Exp(src[stridedI] - maxVal)
			dst[stridedI] = val
			sum += val
		}
		invSum := 1 / sum
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, ndims, dims, strides)
			dst[stridedI] *= invSum
		}
	}
}

// FastSoftmaxStridedU8 performs strided softmax along the last dimension for uint8
func FastSoftmaxStridedU8(numel, ndims int, dims, strides []int, src, dst []uint8) {
	panic("uint8 softmax not implemented")
}

// FastSoftmaxStridedU32 performs strided softmax along the last dimension for uint32
func FastSoftmaxStridedU32(numel, ndims int, dims, strides []int, src, dst []uint32) {
	panic("uint32 softmax not implemented")
}

// FastSoftmaxStridedI64 performs strided softmax along the last dimension for int64
func FastSoftmaxStridedI64(numel, ndims int, dims, strides []int, src, dst []int64) {
	panic("int64 softmax not implemented")
}

// FastRmsNorm performs RMS normalization along the last dimension for type T (contiguous memory)
func FastRmsNorm[T D](numel, ndims int, dims []int, eps T, alpha, x, dst []T) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum T
		for col := range ncols {
			idx := row*ncols + col
			xi := x[idx]
			sum += xi * xi
		}
		mean := sum / T(ncols)
		scale := T(1) / T(math.Sqrt(float64(mean+eps)))
		for col := range ncols {
			idx := row*ncols + col
			if alpha != nil {
				dst[idx] = scale * x[idx] * alpha[col]
			} else {
				dst[idx] = scale * x[idx]
			}
		}
	}
}

// FastRmsNormF32 performs RMS normalization along the last dimension for float32 (contiguous memory)
func FastRmsNormF32(numel int, ndims int, dims []int, eps float32, alpha, x, dst []float32) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float32
		for col := range ncols {
			idx := row*ncols + col
			xi := x[idx]
			sum += xi * xi
		}
		mean := sum / float32(ncols)
		scale := 1 / float32(math.Sqrt(float64(mean+eps)))
		for col := range ncols {
			idx := row*ncols + col
			if alpha != nil {
				dst[idx] = scale * x[idx] * alpha[col]
			} else {
				dst[idx] = scale * x[idx]
			}
		}
	}
}

// FastRmsNormF64 performs RMS normalization along the last dimension for float64 (contiguous memory)
func FastRmsNormF64(numel int, ndims int, dims []int, eps float64, alpha, x, dst []float64) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float64
		for col := range ncols {
			idx := row*ncols + col
			xi := x[idx]
			sum += xi * xi
		}
		mean := sum / float64(ncols)
		scale := 1 / math.Sqrt(mean+eps)
		for col := range ncols {
			idx := row*ncols + col
			if alpha != nil {
				dst[idx] = scale * x[idx] * alpha[col]
			} else {
				dst[idx] = scale * x[idx]
			}
		}
	}
}

// FastRmsNormU8 performs RMS normalization along the last dimension for uint8 (contiguous memory)
func FastRmsNormU8(numel, ndims int, dims []int, eps float64, alpha, x, dst []uint8) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float64
		for col := range ncols {
			idx := row*ncols + col
			xi := float64(x[idx])
			sum += xi * xi
		}
		mean := sum / float64(ncols)
		scale := 1 / math.Sqrt(mean+eps)
		for col := range ncols {
			idx := row*ncols + col
			var result float64
			if alpha != nil {
				result = scale * float64(x[idx]) * float64(alpha[col])
			} else {
				result = scale * float64(x[idx])
			}
			dst[idx] = uint8(result * 255) // Scale to uint8 range
		}
	}
}

// FastRmsNormU32 performs RMS normalization along the last dimension for uint32 (contiguous memory)
func FastRmsNormU32(numel, ndims int, dims []int, eps float64, alpha, x, dst []uint32) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float64
		for col := range ncols {
			idx := row*ncols + col
			xi := float64(x[idx])
			sum += xi * xi
		}
		mean := sum / float64(ncols)
		scale := 1 / math.Sqrt(mean+eps)
		for col := range ncols {
			idx := row*ncols + col
			var result float64
			if alpha != nil {
				result = scale * float64(x[idx]) * float64(alpha[col])
			} else {
				result = scale * float64(x[idx])
			}
			dst[idx] = uint32(result * float64(math.MaxUint32)) // Scale to uint32 range
		}
	}
}

// FastRmsNormI64 performs RMS normalization along the last dimension for int64 (contiguous memory)
func FastRmsNormI64(numel, ndims int, dims []int, eps float64, alpha, x, dst []int64) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float64
		for col := range ncols {
			idx := row*ncols + col
			xi := float64(x[idx])
			sum += xi * xi
		}
		mean := sum / float64(ncols)
		scale := 1 / math.Sqrt(mean+eps)
		for col := range ncols {
			idx := row*ncols + col
			var result float64
			if alpha != nil {
				result = scale * float64(x[idx]) * float64(alpha[col])
			} else {
				result = scale * float64(x[idx])
			}
			dst[idx] = int64(result * float64(math.MaxInt64)) // Scale to int64 range
		}
	}
}

// FastRmsNormStrided performs strided RMS normalization along the last dimension for type T
func FastRmsNormStrided[T D](numel, ndims int, dims, strides []int, eps T, alpha, x, dst []T) {
	if IsContiguous(ndims, dims, strides) {
		FastRmsNorm(numel, ndims, dims, eps, alpha, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum T
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := x[stridedI]
			sum += xi * xi
		}
		mean := sum / T(ncols)
		scale := T(1) / T(math.Sqrt(float64(mean+eps)))
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			if alpha != nil {
				dst[stridedI] = scale * x[stridedI] * alpha[col]
			} else {
				dst[stridedI] = scale * x[stridedI]
			}
		}
	}
}

// FastRmsNormStridedF32 performs strided RMS normalization along the last dimension for float32
func FastRmsNormStridedF32(numel int, ndims int, dims, strides []int, eps float32, alpha, x, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		FastRmsNormF32(numel, ndims, dims, eps, alpha, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float32
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := x[stridedI]
			sum += xi * xi
		}
		mean := sum / float32(ncols)
		scale := 1 / float32(math.Sqrt(float64(mean+eps)))
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			if alpha != nil {
				dst[stridedI] = scale * x[stridedI] * alpha[col]
			} else {
				dst[stridedI] = scale * x[stridedI]
			}
		}
	}
}

// FastRmsNormStridedF64 performs strided RMS normalization along the last dimension for float64
func FastRmsNormStridedF64(numel int, ndims int, dims, strides []int, eps float64, alpha, x, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		FastRmsNormF64(numel, ndims, dims, eps, alpha, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float64
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := x[stridedI]
			sum += xi * xi
		}
		mean := sum / float64(ncols)
		scale := 1 / math.Sqrt(mean+eps)
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			if alpha != nil {
				dst[stridedI] = scale * x[stridedI] * alpha[col]
			} else {
				dst[stridedI] = scale * x[stridedI]
			}
		}
	}
}

// FastRmsNormStridedU8 performs strided RMS normalization along the last dimension for uint8
func FastRmsNormStridedU8(numel, ndims int, dims, strides []int, eps float64, alpha, x, dst []uint8) {
	if IsContiguous(ndims, dims, strides) {
		FastRmsNormU8(numel, ndims, dims, eps, alpha, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float64
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := float64(x[stridedI])
			sum += xi * xi
		}
		mean := sum / float64(ncols)
		scale := 1 / math.Sqrt(mean+eps)
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			var result float64
			if alpha != nil {
				result = scale * float64(x[stridedI]) * float64(alpha[col])
			} else {
				result = scale * float64(x[stridedI])
			}
			dst[stridedI] = uint8(result * 255) // Scale to uint8 range
		}
	}
}

// FastRmsNormStridedU32 performs strided RMS normalization along the last dimension for uint32
func FastRmsNormStridedU32(numel, ndims int, dims, strides []int, eps float64, alpha, x, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastRmsNormU32(numel, ndims, dims, eps, alpha, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float64
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := float64(x[stridedI])
			sum += xi * xi
		}
		mean := sum / float64(ncols)
		scale := 1 / math.Sqrt(mean+eps)
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			var result float64
			if alpha != nil {
				result = scale * float64(x[stridedI]) * float64(alpha[col])
			} else {
				result = scale * float64(x[stridedI])
			}
			dst[stridedI] = uint32(result * float64(math.MaxUint32)) // Scale to uint32 range
		}
	}
}

// FastRmsNormStridedI64 performs strided RMS normalization along the last dimension for int64
func FastRmsNormStridedI64(numel, ndims int, dims, strides []int, eps float64, alpha, x, dst []int64) {
	if IsContiguous(ndims, dims, strides) {
		FastRmsNormI64(numel, ndims, dims, eps, alpha, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float64
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := float64(x[stridedI])
			sum += xi * xi
		}
		mean := sum / float64(ncols)
		scale := 1 / math.Sqrt(mean+eps)
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			var result float64
			if alpha != nil {
				result = scale * float64(x[stridedI]) * float64(alpha[col])
			} else {
				result = scale * float64(x[stridedI])
			}
			dst[stridedI] = int64(result * float64(math.MaxInt64)) // Scale to int64 range
		}
	}
}

// FastLayerNorm performs Layer normalization along the last dimension for type T (contiguous memory)
func FastLayerNorm[T D](numel, ndims int, dims []int, eps T, alpha, beta, x, dst []T) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq T
		for col := range ncols {
			idx := row*ncols + col
			xi := x[idx]
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / T(ncols)
		variance := sumSq/T(ncols) - mean*mean
		scale := T(1) / T(math.Sqrt(float64(variance+eps)))
		for col := range ncols {
			idx := row*ncols + col
			lhs := (x[idx] - mean) * scale
			if alpha != nil && beta != nil {
				dst[idx] = lhs*alpha[col] + beta[col]
			} else if alpha != nil {
				dst[idx] = lhs * alpha[col]
			} else if beta != nil {
				dst[idx] = lhs + beta[col]
			} else {
				dst[idx] = lhs
			}
		}
	}
}

// FastLayerNormF32 performs Layer normalization along the last dimension for float32 (contiguous memory)
func FastLayerNormF32(numel int, ndims int, dims []int, eps float32, alpha, beta, x, dst []float32) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float32
		for col := range ncols {
			idx := row*ncols + col
			xi := x[idx]
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float32(ncols)
		variance := sumSq/float32(ncols) - mean*mean
		scale := 1 / float32(math.Sqrt(float64(variance+eps)))
		for col := range ncols {
			idx := row*ncols + col
			lhs := (x[idx] - mean) * scale
			if alpha != nil && beta != nil {
				dst[idx] = lhs*alpha[col] + beta[col]
			} else if alpha != nil {
				dst[idx] = lhs * alpha[col]
			} else if beta != nil {
				dst[idx] = lhs + beta[col]
			} else {
				dst[idx] = lhs
			}
		}
	}
}

// FastLayerNormF64 performs Layer normalization along the last dimension for float64 (contiguous memory)
func FastLayerNormF64(numel int, ndims int, dims []int, eps float64, alpha, beta, x, dst []float64) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float64
		for col := range ncols {
			idx := row*ncols + col
			xi := x[idx]
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float64(ncols)
		variance := sumSq/float64(ncols) - mean*mean
		scale := 1 / math.Sqrt(variance+eps)
		for col := range ncols {
			idx := row*ncols + col
			lhs := (x[idx] - mean) * scale
			if alpha != nil && beta != nil {
				dst[idx] = lhs*alpha[col] + beta[col]
			} else if alpha != nil {
				dst[idx] = lhs * alpha[col]
			} else if beta != nil {
				dst[idx] = lhs + beta[col]
			} else {
				dst[idx] = lhs
			}
		}
	}
}

// FastLayerNormU8 performs Layer normalization along the last dimension for uint8 (contiguous memory)
func FastLayerNormU8(numel, ndims int, dims []int, eps float64, alpha, beta, x, dst []uint8) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float64
		for col := range ncols {
			idx := row*ncols + col
			xi := float64(x[idx])
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float64(ncols)
		variance := sumSq/float64(ncols) - mean*mean
		scale := 1 / math.Sqrt(variance+eps)
		for col := range ncols {
			idx := row*ncols + col
			lhs := (float64(x[idx]) - mean) * scale
			var result float64
			if alpha != nil && beta != nil {
				result = lhs*float64(alpha[col]) + float64(beta[col])
			} else if alpha != nil {
				result = lhs * float64(alpha[col])
			} else if beta != nil {
				result = lhs + float64(beta[col])
			} else {
				result = lhs
			}
			dst[idx] = uint8(result * 255) // Scale to uint8 range
		}
	}
}

// FastLayerNormU32 performs Layer normalization along the last dimension for uint32 (contiguous memory)
func FastLayerNormU32(numel, ndims int, dims []int, eps float64, alpha, beta, x, dst []uint32) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float64
		for col := range ncols {
			idx := row*ncols + col
			xi := float64(x[idx])
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float64(ncols)
		variance := sumSq/float64(ncols) - mean*mean
		scale := 1 / math.Sqrt(variance+eps)
		for col := range ncols {
			idx := row*ncols + col
			lhs := (float64(x[idx]) - mean) * scale
			var result float64
			if alpha != nil && beta != nil {
				result = lhs*float64(alpha[col]) + float64(beta[col])
			} else if alpha != nil {
				result = lhs * float64(alpha[col])
			} else if beta != nil {
				result = lhs + float64(beta[col])
			} else {
				result = lhs
			}
			dst[idx] = uint32(result * float64(math.MaxUint32)) // Scale to uint32 range
		}
	}
}

// FastLayerNormI64 performs Layer normalization along the last dimension for int64 (contiguous memory)
func FastLayerNormI64(numel, ndims int, dims []int, eps float64, alpha, beta, x, dst []int64) {
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float64
		for col := range ncols {
			idx := row*ncols + col
			xi := float64(x[idx])
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float64(ncols)
		variance := sumSq/float64(ncols) - mean*mean
		scale := 1 / math.Sqrt(variance+eps)
		for col := range ncols {
			idx := row*ncols + col
			lhs := (float64(x[idx]) - mean) * scale
			var result float64
			if alpha != nil && beta != nil {
				result = lhs*float64(alpha[col]) + float64(beta[col])
			} else if alpha != nil {
				result = lhs * float64(alpha[col])
			} else if beta != nil {
				result = lhs + float64(beta[col])
			} else {
				result = lhs
			}
			dst[idx] = int64(result * float64(math.MaxInt64)) // Scale to int64 range
		}
	}
}

// FastLayerNormStrided performs strided Layer normalization along the last dimension for type T
func FastLayerNormStrided[T D](numel, ndims int, dims, strides []int, eps T, alpha, beta, x, dst []T) {
	if IsContiguous(ndims, dims, strides) {
		FastLayerNorm(numel, ndims, dims, eps, alpha, beta, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq T
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := x[stridedI]
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / T(ncols)
		variance := sumSq/T(ncols) - mean*mean
		scale := T(1) / T(math.Sqrt(float64(variance+eps)))
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			lhs := (x[stridedI] - mean) * scale
			if alpha != nil && beta != nil {
				dst[stridedI] = lhs*alpha[col] + beta[col]
			} else if alpha != nil {
				dst[stridedI] = lhs * alpha[col]
			} else if beta != nil {
				dst[stridedI] = lhs + beta[col]
			} else {
				dst[stridedI] = lhs
			}
		}
	}
}

// FastLayerNormStridedF32 performs strided Layer normalization along the last dimension for float32
func FastLayerNormStridedF32(numel int, ndims int, dims, strides []int, eps float32, alpha, beta, x, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		FastLayerNormF32(numel, ndims, dims, eps, alpha, beta, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float32
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := x[stridedI]
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float32(ncols)
		variance := sumSq/float32(ncols) - mean*mean
		scale := 1 / float32(math.Sqrt(float64(variance+eps)))
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			lhs := (x[stridedI] - mean) * scale
			if alpha != nil && beta != nil {
				dst[stridedI] = lhs*alpha[col] + beta[col]
			} else if alpha != nil {
				dst[stridedI] = lhs * alpha[col]
			} else if beta != nil {
				dst[stridedI] = lhs + beta[col]
			} else {
				dst[stridedI] = lhs
			}
		}
	}
}

// FastLayerNormStridedF64 performs strided Layer normalization along the last dimension for float64
func FastLayerNormStridedF64(numel int, ndims int, dims, strides []int, eps float64, alpha, beta, x, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		FastLayerNormF64(numel, ndims, dims, eps, alpha, beta, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float64
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := x[stridedI]
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float64(ncols)
		variance := sumSq/float64(ncols) - mean*mean
		scale := 1 / math.Sqrt(variance+eps)
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			lhs := (x[stridedI] - mean) * scale
			if alpha != nil && beta != nil {
				dst[stridedI] = lhs*alpha[col] + beta[col]
			} else if alpha != nil {
				dst[stridedI] = lhs * alpha[col]
			} else if beta != nil {
				dst[stridedI] = lhs + beta[col]
			} else {
				dst[stridedI] = lhs
			}
		}
	}
}

// FastLayerNormStridedU8 performs strided Layer normalization along the last dimension for uint8
func FastLayerNormStridedU8(numel, ndims int, dims, strides []int, eps float64, alpha, beta, x, dst []uint8) {
	if IsContiguous(ndims, dims, strides) {
		FastLayerNormU8(numel, ndims, dims, eps, alpha, beta, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float64
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := float64(x[stridedI])
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float64(ncols)
		variance := sumSq/float64(ncols) - mean*mean
		scale := 1 / math.Sqrt(variance+eps)
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			lhs := (float64(x[stridedI]) - mean) * scale
			var result float64
			if alpha != nil && beta != nil {
				result = lhs*float64(alpha[col]) + float64(beta[col])
			} else if alpha != nil {
				result = lhs * float64(alpha[col])
			} else if beta != nil {
				result = lhs + float64(beta[col])
			} else {
				result = lhs
			}
			dst[stridedI] = uint8(result * 255) // Scale to uint8 range
		}
	}
}

// FastLayerNormStridedU32 performs strided Layer normalization along the last dimension for uint32
func FastLayerNormStridedU32(numel, ndims int, dims, strides []int, eps float64, alpha, beta, x, dst []uint32) {
	if IsContiguous(ndims, dims, strides) {
		FastLayerNormU32(numel, ndims, dims, eps, alpha, beta, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float64
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := float64(x[stridedI])
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float64(ncols)
		variance := sumSq/float64(ncols) - mean*mean
		scale := 1 / math.Sqrt(variance+eps)
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			lhs := (float64(x[stridedI]) - mean) * scale
			var result float64
			if alpha != nil && beta != nil {
				result = lhs*float64(alpha[col]) + float64(beta[col])
			} else if alpha != nil {
				result = lhs * float64(alpha[col])
			} else if beta != nil {
				result = lhs + float64(beta[col])
			} else {
				result = lhs
			}
			dst[stridedI] = uint32(result * float64(math.MaxUint32)) // Scale to uint32 range
		}
	}
}

// FastLayerNormStridedI64 performs strided Layer normalization along the last dimension for int64
func FastLayerNormStridedI64(numel, ndims int, dims, strides []int, eps float64, alpha, beta, x, dst []int64) {
	if IsContiguous(ndims, dims, strides) {
		FastLayerNormI64(numel, ndims, dims, eps, alpha, beta, x, dst)
		return
	}
	ncols := dims[ndims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float64
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			xi := float64(x[stridedI])
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float64(ncols)
		variance := sumSq/float64(ncols) - mean*mean
		scale := 1 / math.Sqrt(variance+eps)
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, ndims, dims, strides)
			lhs := (float64(x[stridedI]) - mean) * scale
			var result float64
			if alpha != nil && beta != nil {
				result = lhs*float64(alpha[col]) + float64(beta[col])
			} else if alpha != nil {
				result = lhs * float64(alpha[col])
			} else if beta != nil {
				result = lhs + float64(beta[col])
			} else {
				result = lhs
			}
			dst[stridedI] = int64(result * float64(math.MaxInt64)) // Scale to int64 range
		}
	}
}

// RopeI performs rotary position embedding (rope_i variant) for type T (contiguous memory)
func RopeI[T D](bh, td, strideB int, src, cos, sin, dst []T) {
	var zero T
	switch any(zero).(type) {
	case float32, float64:
	default:
		panic("rope_i: unsupported type")
	}
	numPairs := bh * td / 2
	for idx := range numPairs {
		ropeIdx := idx % (td / 2)
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			ropeIdx += bIdx * (td / 2)
		}
		c := cos[ropeIdx]
		s := sin[ropeIdx]
		dst[2*idx] = src[2*idx]*c - src[2*idx+1]*s
		dst[2*idx+1] = src[2*idx]*s + src[2*idx+1]*c
	}
}

// RopeIF32 performs rotary position embedding (rope_i variant) for float32 (contiguous memory)
func RopeIF32(bh int, td int, strideB int, src, cos, sin, dst []float32) {
	numPairs := bh * td / 2
	for idx := range numPairs {
		ropeIdx := idx % (td / 2)
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			ropeIdx += bIdx * (td / 2)
		}
		c := cos[ropeIdx]
		s := sin[ropeIdx]
		dst[2*idx] = src[2*idx]*c - src[2*idx+1]*s
		dst[2*idx+1] = src[2*idx]*s + src[2*idx+1]*c
	}
}

// RopeIF64 performs rotary position embedding (rope_i variant) for float64 (contiguous memory)
func RopeIF64(bh int, td int, strideB int, src, cos, sin, dst []float64) {
	numPairs := bh * td / 2
	for idx := range numPairs {
		ropeIdx := idx % (td / 2)
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			ropeIdx += bIdx * (td / 2)
		}
		c := cos[ropeIdx]
		s := sin[ropeIdx]
		dst[2*idx] = src[2*idx]*c - src[2*idx+1]*s
		dst[2*idx+1] = src[2*idx]*s + src[2*idx+1]*c
	}
}

// RopeIU8 performs rotary position embedding (rope_i variant) for uint8 (contiguous memory)
func RopeIU8(bh int, td int, strideB int, src, cos, sin, dst []uint8) {
	panic("uint8 rope not implemented")
}

// RopeIU32 performs rotary position embedding (rope_i variant) for uint32 (contiguous memory)
func RopeIU32(bh int, td int, strideB int, src, cos, sin, dst []uint32) {
	panic("uint32 rope not implemented")
}

// RopeII64 performs rotary position embedding (rope_i variant) for int64 (contiguous memory)
func RopeII64(bh int, td int, strideB int, src, cos, sin, dst []int64) {
	panic("int64 rope not implemented")
}

// RopeIStrided performs strided rotary position embedding (rope_i variant) for type T
func RopeIStrided[T D](ndims int, dims, strides []int, bh, td, strideB int, src, cos, sin, dst []T) {
	var zero T
	switch any(zero).(type) {
	case float32, float64:
	default:
		panic("rope_is: unsupported type")
	}
	if IsContiguous(ndims, dims, strides) {
		RopeI(bh, td, strideB, src, cos, sin, dst)
		return
	}
	numPairs := bh * td / 2
	for idx := range numPairs {
		ropeIdx := idx % (td / 2)
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			ropeIdx += bIdx * (td / 2)
		}
		c := cos[ropeIdx]
		s := sin[ropeIdx]
		strided2Idx := GetStridedIndex(2*idx, ndims, dims, strides)
		strided2IdxPlus1 := GetStridedIndex(2*idx+1, ndims, dims, strides)
		dst[strided2Idx] = src[strided2Idx]*c - src[strided2IdxPlus1]*s
		dst[strided2IdxPlus1] = src[strided2Idx]*s + src[strided2IdxPlus1]*c
	}
}

// RopeIStridedF32 performs strided rotary position embedding (rope_i variant) for float32
func RopeIStridedF32(ndims int, dims, strides []int, bh int, td int, strideB int, src, cos, sin, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		RopeIF32(bh, td, strideB, src, cos, sin, dst)
		return
	}
	numPairs := bh * td / 2
	for idx := range numPairs {
		ropeIdx := idx % (td / 2)
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			ropeIdx += bIdx * (td / 2)
		}
		c := cos[ropeIdx]
		s := sin[ropeIdx]
		strided2Idx := GetStridedIndex(2*idx, ndims, dims, strides)
		strided2IdxPlus1 := GetStridedIndex(2*idx+1, ndims, dims, strides)
		dst[strided2Idx] = src[strided2Idx]*c - src[strided2IdxPlus1]*s
		dst[strided2IdxPlus1] = src[strided2Idx]*s + src[strided2IdxPlus1]*c
	}
}

// RopeIStridedF64 performs strided rotary position embedding (rope_i variant) for float64
func RopeIStridedF64(ndims int, dims, strides []int, bh int, td int, strideB int, src, cos, sin, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		RopeIF64(bh, td, strideB, src, cos, sin, dst)
		return
	}
	numPairs := bh * td / 2
	for idx := range numPairs {
		ropeIdx := idx % (td / 2)
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			ropeIdx += bIdx * (td / 2)
		}
		c := cos[ropeIdx]
		s := sin[ropeIdx]
		strided2Idx := GetStridedIndex(2*idx, ndims, dims, strides)
		strided2IdxPlus1 := GetStridedIndex(2*idx+1, ndims, dims, strides)
		dst[strided2Idx] = src[strided2Idx]*c - src[strided2IdxPlus1]*s
		dst[strided2IdxPlus1] = src[strided2Idx]*s + src[strided2IdxPlus1]*c
	}
}

// RopeIStridedU8 performs strided rotary position embedding (rope_i variant) for uint8
func RopeIStridedU8(ndims int, dims, strides []int, bh int, td int, strideB int, src, cos, sin, dst []uint8) {
	panic("uint8 rope not implemented")
}

// RopeIStridedU32 performs strided rotary position embedding (rope_i variant) for uint32
func RopeIStridedU32(ndims int, dims, strides []int, bh int, td int, strideB int, src, cos, sin, dst []uint32) {
	panic("uint32 rope not implemented")
}

// RopeIStridedI64 performs strided rotary position embedding (rope_i variant) for int64
func RopeIStridedI64(ndims int, dims, strides []int, bh int, td int, strideB int, src, cos, sin, dst []int64) {
	panic("int64 rope not implemented")
}

// Rope performs rotary position embedding (rope variant) for type T (contiguous memory)
func Rope[T D](bh, td, d, strideB int, src, cos, sin, dst []T) {
	var zero T
	switch any(zero).(type) {
	case float32, float64:
	default:
		panic("rope: unsupported type")
	}
	numPairs := bh * td / 2
	for idx := range numPairs {
		iBh := idx / (td / 2)
		iTd := idx - (td/2)*iBh
		iT := iTd / (d / 2)
		iD := iTd - (d/2)*iT
		i1 := iBh*td + iT*d + iD
		i2 := i1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * (td / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		dst[i1] = src[i1]*c - src[i2]*s
		dst[i2] = src[i1]*s + src[i2]*c
	}
}

// RopeF32 performs rotary position embedding (rope variant) for float32 (contiguous memory)
func RopeF32(bh int, td int, d int, strideB int, src, cos, sin, dst []float32) {
	numPairs := bh * td / 2
	for idx := range numPairs {
		iBh := idx / (td / 2)
		iTd := idx - (td/2)*iBh
		iT := iTd / (d / 2)
		iD := iTd - (d/2)*iT
		i1 := iBh*td + iT*d + iD
		i2 := i1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * (td / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		dst[i1] = src[i1]*c - src[i2]*s
		dst[i2] = src[i1]*s + src[i2]*c
	}
}

// RopeF64 performs rotary position embedding (rope variant) for float64 (contiguous memory)
func RopeF64(bh int, td int, d int, strideB int, src, cos, sin, dst []float64) {
	numPairs := bh * td / 2
	for idx := range numPairs {
		iBh := idx / (td / 2)
		iTd := idx - (td/2)*iBh
		iT := iTd / (d / 2)
		iD := iTd - (d/2)*iT
		i1 := iBh*td + iT*d + iD
		i2 := i1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * (td / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		dst[i1] = src[i1]*c - src[i2]*s
		dst[i2] = src[i1]*s + src[i2]*c
	}
}

// RopeU8 performs rotary position embedding (rope variant) for uint8 (contiguous memory)
func RopeU8(bh, td, d, strideB int, src, cos, sin, dst []uint8) {
	panic("uint8 rope not implemented")
}

// RopeU32 performs rotary position embedding (rope variant) for uint32 (contiguous memory)
func RopeU32(bh, td, d, strideB int, src, cos, sin, dst []uint32) {
	panic("uint32 rope not implemented")
}

// RopeI64 performs rotary position embedding (rope variant) for int64 (contiguous memory)
func RopeI64(bh, td, d, strideB int, src, cos, sin, dst []int64) {
	panic("int64 rope not implemented")
}

// RopeStrided performs strided rotary position embedding (rope variant) for type T
func RopeStrided[T D](ndims int, dims, strides []int, bh, td, d, strideB int, src, cos, sin, dst []T) {
	var zero T
	switch any(zero).(type) {
	case float32, float64:
	default:
		panic("rope_strided: unsupported type")
	}
	if IsContiguous(ndims, dims, strides) {
		Rope(bh, td, d, strideB, src, cos, sin, dst)
		return
	}
	numPairs := bh * td / 2
	for idx := range numPairs {
		iBh := idx / (td / 2)
		iTd := idx - (td/2)*iBh
		iT := iTd / (d / 2)
		iD := iTd - (d/2)*iT
		logicalI1 := iBh*td + iT*d + iD
		logicalI2 := logicalI1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * (td / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		stridedI1 := GetStridedIndex(logicalI1, ndims, dims, strides)
		stridedI2 := GetStridedIndex(logicalI2, ndims, dims, strides)
		dst[stridedI1] = src[stridedI1]*c - src[stridedI2]*s
		dst[stridedI2] = src[stridedI1]*s + src[stridedI2]*c
	}
}

// RopeStridedF32 performs strided rotary position embedding (rope variant) for float32
func RopeStridedF32(ndims int, dims, strides []int, bh, td, d, strideB int, src, cos, sin, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		RopeF32(bh, td, d, strideB, src, cos, sin, dst)
		return
	}
	numPairs := bh * td / 2
	for idx := range numPairs {
		iBh := idx / (td / 2)
		iTd := idx - (td/2)*iBh
		iT := iTd / (d / 2)
		iD := iTd - (d/2)*iT
		logicalI1 := iBh*td + iT*d + iD
		logicalI2 := logicalI1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * (td / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		stridedI1 := GetStridedIndex(logicalI1, ndims, dims, strides)
		stridedI2 := GetStridedIndex(logicalI2, ndims, dims, strides)
		dst[stridedI1] = src[stridedI1]*c - src[stridedI2]*s
		dst[stridedI2] = src[stridedI1]*s + src[stridedI2]*c
	}
}

// RopeStridedF64 performs strided rotary position embedding (rope variant) for float64
func RopeStridedF64(ndims int, dims, strides []int, bh, td, d, strideB int, src, cos, sin, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		RopeF64(bh, td, d, strideB, src, cos, sin, dst)
		return
	}
	numPairs := bh * td / 2
	for idx := range numPairs {
		iBh := idx / (td / 2)
		iTd := idx - (td/2)*iBh
		iT := iTd / (d / 2)
		iD := iTd - (d/2)*iT
		logicalI1 := iBh*td + iT*d + iD
		logicalI2 := logicalI1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * (td / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		stridedI1 := GetStridedIndex(logicalI1, ndims, dims, strides)
		stridedI2 := GetStridedIndex(logicalI2, ndims, dims, strides)
		dst[stridedI1] = src[stridedI1]*c - src[stridedI2]*s
		dst[stridedI2] = src[stridedI1]*s + src[stridedI2]*c
	}
}

// RopeStridedU8 performs strided rotary position embedding (rope variant) for uint8
func RopeStridedU8(ndims int, dims, strides []int, bh, td, d, strideB int, src, cos, sin, dst []uint8) {
	panic("uint8 rope not implemented")
}

// RopeStridedU32 performs strided rotary position embedding (rope variant) for uint32
func RopeStridedU32(ndims int, dims, strides []int, bh, td, d, strideB int, src, cos, sin, dst []uint32) {
	panic("uint32 rope not implemented")
}

// RopeStridedI64 performs strided rotary position embedding (rope variant) for int64
func RopeStridedI64(ndims int, dims, strides []int, bh, td, d, strideB int, src, cos, sin, dst []int64) {
	panic("int64 rope not implemented")
}

// RopeThd performs rotary position embedding (rope_thd variant) for float32 (contiguous memory)
func RopeThd[T D](b int, t int, h int, d int, strideB int, src, cos, sin, dst []T) {
	var zero T
	switch any(zero).(type) {
	case float32, float64:
	default:
		panic("rope_thd: unsupported type")
	}
	numPairs := b * t * h * d / 2
	for idx := range numPairs {
		iBth := idx / (d / 2)
		iD := idx - (d/2)*iBth
		iT := (iBth / h) % t
		i1 := iBth*d + iD
		i2 := i1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * ((t * d) / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		dst[i1] = src[i1]*c - src[i2]*s
		dst[i2] = src[i1]*s + src[i2]*c
	}
}

// RopeThdF32 performs rotary position embedding (rope_thd variant) for float32 (contiguous memory)
func RopeThdF32(b int, t int, h int, d int, strideB int, src, cos, sin, dst []float32) {
	numPairs := b * t * h * d / 2
	for idx := range numPairs {
		iBth := idx / (d / 2)
		iD := idx - (d/2)*iBth
		iT := (iBth / h) % t
		i1 := iBth*d + iD
		i2 := i1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * ((t * d) / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		dst[i1] = src[i1]*c - src[i2]*s
		dst[i2] = src[i1]*s + src[i2]*c
	}
}

// RopeThdF64 performs rotary position embedding (rope_thd variant) for float64 (contiguous memory)
func RopeThdF64(b int, t int, h int, d int, strideB int, src, cos, sin, dst []float64) {
	numPairs := b * t * h * d / 2
	for idx := range numPairs {
		iBth := idx / (d / 2)
		iD := idx - (d/2)*iBth
		iT := (iBth / h) % t
		i1 := iBth*d + iD
		i2 := i1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * ((t * d) / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		dst[i1] = src[i1]*c - src[i2]*s
		dst[i2] = src[i1]*s + src[i2]*c
	}
}

// RopeThdU8 performs rotary position embedding (rope_thd variant) for uint8 (contiguous memory)
func RopeThdU8(b int, t int, h int, d int, strideB int, src, cos, sin, dst []uint8) {
	panic("uint8 rope not implemented")
}

// RopeThdU32 performs rotary position embedding (rope_thd variant) for uint32 (contiguous memory)
func RopeThdU32(b int, t int, h int, d int, strideB int, src, cos, sin, dst []uint32) {
	panic("uint32 rope not implemented")
}

// RopeThdI64 performs rotary position embedding (rope_thd variant) for int64 (contiguous memory)
func RopeThdI64(b int, t int, h int, d int, strideB int, src, cos, sin, dst []int64) {
	panic("int64 rope not implemented")
}

// RopeThdStrided performs strided rotary position embedding (rope_thd variant) for type T
func RopeThdStrided[T D](ndims int, dims, strides []int, b int, t int, h int, d int, strideB int, src, cos, sin, dst []T) {
	var zero T
	switch any(zero).(type) {
	case float32, float64:
	default:
		panic("rope_thd: unsupported type")
	}

	if IsContiguous(ndims, dims, strides) {
		RopeThd(b, t, h, d, strideB, src, cos, sin, dst)
		return
	}
	numPairs := b * t * h * d / 2
	for idx := range numPairs {
		iBth := idx / (d / 2)
		iD := idx - (d/2)*iBth
		iT := (iBth / h) % t
		logicalI1 := iBth*d + iD
		logicalI2 := logicalI1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * ((t * d) / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		stridedI1 := GetStridedIndex(logicalI1, ndims, dims, strides)
		stridedI2 := GetStridedIndex(logicalI2, ndims, dims, strides)
		dst[stridedI1] = src[stridedI1]*c - src[stridedI2]*s
		dst[stridedI2] = src[stridedI1]*s + src[stridedI2]*c
	}
}

// RopeThdStridedF32 performs strided rotary position embedding (rope_thd variant) for float32
func RopeThdStridedF32(ndims int, dims, strides []int, b int, t int, h int, d int, strideB int, src, cos, sin, dst []float32) {
	if IsContiguous(ndims, dims, strides) {
		RopeThdF32(b, t, h, d, strideB, src, cos, sin, dst)
		return
	}
	numPairs := b * t * h * d / 2
	for idx := range numPairs {
		iBth := idx / (d / 2)
		iD := idx - (d/2)*iBth
		iT := (iBth / h) % t
		logicalI1 := iBth*d + iD
		logicalI2 := logicalI1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * ((t * d) / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		stridedI1 := GetStridedIndex(logicalI1, ndims, dims, strides)
		stridedI2 := GetStridedIndex(logicalI2, ndims, dims, strides)
		dst[stridedI1] = src[stridedI1]*c - src[stridedI2]*s
		dst[stridedI2] = src[stridedI1]*s + src[stridedI2]*c
	}
}

// RopeThdStridedF64 performs strided rotary position embedding (rope_thd variant) for float64
func RopeThdStridedF64(ndims int, dims, strides []int, b int, t int, h int, d int, strideB int, src, cos, sin, dst []float64) {
	if IsContiguous(ndims, dims, strides) {
		RopeThdF64(b, t, h, d, strideB, src, cos, sin, dst)
		return
	}
	numPairs := b * t * h * d / 2
	for idx := range numPairs {
		iBth := idx / (d / 2)
		iD := idx - (d/2)*iBth
		iT := (iBth / h) % t
		logicalI1 := iBth*d + iD
		logicalI2 := logicalI1 + d/2
		iCs := iT*(d/2) + iD
		if strideB > 0 {
			bIdx := (2 * idx) / strideB
			iCs += bIdx * ((t * d) / 2)
		}
		c := cos[iCs]
		s := sin[iCs]
		stridedI1 := GetStridedIndex(logicalI1, ndims, dims, strides)
		stridedI2 := GetStridedIndex(logicalI2, ndims, dims, strides)
		dst[stridedI1] = src[stridedI1]*c - src[stridedI2]*s
		dst[stridedI2] = src[stridedI1]*s + src[stridedI2]*c
	}
}

// RopeThdStridedU8 performs strided rotary position embedding (rope_thd variant) for uint8
func RopeThdStridedU8(ndims int, dims, strides []int, b int, t int, h int, d int, strideB int, src, cos, sin, dst []uint8) {
	panic("uint8 rope not implemented")
}

// RopeThdStridedU32 performs strided rotary position embedding (rope_thd variant) for uint32
func RopeThdStridedU32(ndims int, dims, strides []int, b int, t int, h int, d int, strideB int, src, cos, sin, dst []uint32) {
	panic("uint32 rope not implemented")
}

// RopeThdStridedI64 performs strided rotary position embedding (rope_thd variant) for int64
func RopeThdStridedI64(ndims int, dims, strides []int, b int, t int, h int, d int, strideB int, src, cos, sin, dst []int64) {
	panic("int64 rope not implemented")
}

package kernels

import (
	"math"
	"slices"
)

// FastSumF32 computes the sum over the last dimension for float32
func FastSumF32(numel, numDims int, dims []int, src, dst []float32) {
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		sum := float32(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[j]
		}
		dst[i] = sum
	}
}

// FastSumF64 computes the sum over the last dimension for float64
func FastSumF64(numel, numDims int, dims []int, src, dst []float64) {
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		sum := float64(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[j]
		}
		dst[i] = sum
	}
}

// FastSumStridedF32 computes the sum over the last dimension for float32 with strided memory
func FastSumStridedF32(numel, numDims int, dims, strides []int, src, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
		FastSumF32(numel, numDims, dims, src, dst)
		return
	}
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		sum := float32(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[GetStridedIndex(j, numDims, dims, strides)]
		}
		dst[i] = sum
	}
}

// FastSumStridedF64 computes the sum over the last dimension for float64 with strided memory
func FastSumStridedF64(numel, numDims int, dims, strides []int, src, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
		FastSumF64(numel, numDims, dims, src, dst)
		return
	}
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		sum := float64(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			sum += src[GetStridedIndex(j, numDims, dims, strides)]
		}
		dst[i] = sum
	}
}

// FastMinF32 computes the minimum over the last dimension for float32
func FastMinF32(numel, numDims int, dims []int, src, dst []float32) {
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		minVal := float32(math.MaxFloat32)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
			}
		}
		dst[i] = minVal
	}
}

// FastMinF64 computes the minimum over the last dimension for float64
func FastMinF64(numel, numDims int, dims []int, src, dst []float64) {
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		minVal := float64(math.MaxFloat64)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
			}
		}
		dst[i] = minVal
	}
}

// FastMinStridedF32 computes the minimum over the last dimension for float32 with strided memory
func FastMinStridedF32(numel, numDims int, dims, strides []int, src, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
		FastMinF32(numel, numDims, dims, src, dst)
		return
	}
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		minVal := float32(math.MaxFloat32)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, numDims, dims, strides)]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// FastMinStridedF64 computes the minimum over the last dimension for float64 with strided memory
func FastMinStridedF64(numel, numDims int, dims, strides []int, src, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
		FastMinF64(numel, numDims, dims, src, dst)
		return
	}
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		minVal := float64(math.MaxFloat64)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, numDims, dims, strides)]
			if val < minVal {
				minVal = val
			}
		}
		dst[i] = minVal
	}
}

// FastMaxF32 computes the maximum over the last dimension for float32
func FastMaxF32(numel, numDims int, dims []int, src, dst []float32) {
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		maxVal := float32(-math.MaxFloat32)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
			}
		}
		dst[i] = maxVal
	}
}

// FastMaxF64 computes the maximum over the last dimension for float64
func FastMaxF64(numel, numDims int, dims []int, src, dst []float64) {
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		maxVal := float64(-math.MaxFloat64)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
			}
		}
		dst[i] = maxVal
	}
}

// FastMaxStridedF32 computes the maximum over the last dimension for float32 with strided memory
func FastMaxStridedF32(numel, numDims int, dims, strides []int, src, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
		FastMaxF32(numel, numDims, dims, src, dst)
		return
	}
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		maxVal := float32(-math.MaxFloat32)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, numDims, dims, strides)]
			if val > maxVal {
				maxVal = val
			}
		}
		dst[i] = maxVal
	}
}

// FastMaxStridedF64 computes the maximum over the last dimension for float64 with strided memory
func FastMaxStridedF64(numel, numDims int, dims, strides []int, src, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
		FastMaxF64(numel, numDims, dims, src, dst)
		return
	}
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		maxVal := float64(-math.MaxFloat64)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, numDims, dims, strides)]
			if val > maxVal {
				maxVal = val
			}
		}
		dst[i] = maxVal
	}
}

// FastArgminF32 computes the index of the minimum over the last dimension for float32
func FastArgminF32(numel, numDims int, dims []int, src []float32, dst []uint32) {
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		minVal := float32(math.MaxFloat32)
		minIdx := uint32(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
				minIdx = uint32(j % dims[numDims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminF64 computes the index of the minimum over the last dimension for float64
func FastArgminF64(numel, numDims int, dims []int, src []float64, dst []uint32) {
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		minVal := float64(math.MaxFloat64)
		minIdx := uint32(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] < minVal {
				minVal = src[j]
				minIdx = uint32(j % dims[numDims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminStridedF32 computes the index of the minimum over the last dimension for float32 with strided memory
func FastArgminStridedF32(numel, numDims int, dims, strides []int, src []float32, dst []uint32) {
	if IsContiguous(numDims, dims, strides) {
		FastArgminF32(numel, numDims, dims, src, dst)
		return
	}
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		minVal := float32(math.MaxFloat32)
		minIdx := uint32(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, numDims, dims, strides)]
			if val < minVal {
				minVal = val
				minIdx = uint32(j % dims[numDims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgminStridedF64 computes the index of the minimum over the last dimension for float64 with strided memory
func FastArgminStridedF64(numel, numDims int, dims, strides []int, src []float64, dst []uint32) {
	if IsContiguous(numDims, dims, strides) {
		FastArgminF64(numel, numDims, dims, src, dst)
		return
	}
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		minVal := float64(math.MaxFloat64)
		minIdx := uint32(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, numDims, dims, strides)]
			if val < minVal {
				minVal = val
				minIdx = uint32(j % dims[numDims-1])
			}
		}
		dst[i] = minIdx
	}
}

// FastArgmaxF32 computes the index of the maximum over the last dimension for float32
func FastArgmaxF32(numel, numDims int, dims []int, src []float32, dst []uint32) {
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		maxVal := float32(-math.MaxFloat32)
		maxIdx := uint32(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
				maxIdx = uint32(j % dims[numDims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxF64 computes the index of the maximum over the last dimension for float64
func FastArgmaxF64(numel, numDims int, dims []int, src []float64, dst []uint32) {
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		maxVal := float64(-math.MaxFloat64)
		maxIdx := uint32(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			if src[j] > maxVal {
				maxVal = src[j]
				maxIdx = uint32(j % dims[numDims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxStridedF32 computes the index of the maximum over the last dimension for float32 with strided memory
func FastArgmaxStridedF32(numel, numDims int, dims, strides []int, src []float32, dst []uint32) {
	if IsContiguous(numDims, dims, strides) {
		FastArgmaxF32(numel, numDims, dims, src, dst)
		return
	}
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		maxVal := float32(-math.MaxFloat32)
		maxIdx := uint32(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, numDims, dims, strides)]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j % dims[numDims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// FastArgmaxStridedF64 computes the index of the maximum over the last dimension for float64 with strided memory
func FastArgmaxStridedF64(numel, numDims int, dims, strides []int, src []float64, dst []uint32) {
	if IsContiguous(numDims, dims, strides) {
		FastArgmaxF64(numel, numDims, dims, src, dst)
		return
	}
	dstSize := numel / dims[numDims-1]
	for i := range dstSize {
		maxVal := float64(-math.MaxFloat64)
		maxIdx := uint32(0)
		startIdx := i * dims[numDims-1]
		stopIdx := min(startIdx+dims[numDims-1], numel)
		for j := startIdx; j < stopIdx; j++ {
			val := src[GetStridedIndex(j, numDims, dims, strides)]
			if val > maxVal {
				maxVal = val
				maxIdx = uint32(j % dims[numDims-1])
			}
		}
		dst[i] = maxIdx
	}
}

// SumF32 performs sum reduction over specified dimension indices for float32
func SumF32(numel int, numDims int, dims []int, sumDims []int, inp, out []float32) {
	for i := range numel {
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := numDims - 1; d >= 0; d-- {
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
func SumF64(numel int, numDims int, dims []int, sumDims []int, inp, out []float64) {
	for i := range numel {
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := numDims - 1; d >= 0; d-- {
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

// SumStridedF32 performs strided sum reduction over specified dimension indices for float32
func SumStridedF32(numel int, numDims int, dims, strides []int, sumDims []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		SumF32(numel, numDims, dims, sumDims, inp, out)
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := numDims - 1; d >= 0; d-- {
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
func SumStridedF64(numel int, numDims int, dims, strides []int, sumDims []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		SumF64(numel, numDims, dims, sumDims, inp, out)
		return
	}
	for i := range numel {
		stridedI := GetStridedIndex(i, numDims, dims, strides)
		dstIndex := 0
		currentStride := 1
		coords := i
		for d := numDims - 1; d >= 0; d-- {
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

// SoftmaxF32 performs softmax along the last dimension for float32 (contiguous memory)
func SoftmaxF32(numel int, numDims int, dims []int, src, dst []float32) {
	ncols := dims[numDims-1]
	rows := numel / ncols
	for row := 0; row < rows; row++ {
		maxVal := float32(-math.MaxFloat32)
		for col := 0; col < ncols; col++ {
			i := row*ncols + col
			if src[i] > maxVal {
				maxVal = src[i]
			}
		}
		sum := float32(0)
		for col := 0; col < ncols; col++ {
			i := row*ncols + col
			val := float32(math.Exp(float64(src[i] - maxVal)))
			dst[i] = val
			sum += val
		}
		invSum := 1 / sum
		for col := 0; col < ncols; col++ {
			i := row*ncols + col
			dst[i] *= invSum
		}
	}
}

// SoftmaxF64 performs softmax along the last dimension for float64 (contiguous memory)
func SoftmaxF64(numel int, numDims int, dims []int, src, dst []float64) {
	ncols := dims[numDims-1]
	rows := numel / ncols
	for row := 0; row < rows; row++ {
		maxVal := -math.MaxFloat64
		for col := 0; col < ncols; col++ {
			i := row*ncols + col
			if src[i] > maxVal {
				maxVal = src[i]
			}
		}
		sum := 0.0
		for col := 0; col < ncols; col++ {
			i := row*ncols + col
			val := math.Exp(src[i] - maxVal)
			dst[i] = val
			sum += val
		}
		invSum := 1 / sum
		for col := 0; col < ncols; col++ {
			i := row*ncols + col
			dst[i] *= invSum
		}
	}
}

// SoftmaxStridedF32 performs strided softmax along the last dimension for float32
func SoftmaxStridedF32(numel int, numDims int, dims, strides []int, src, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
		SoftmaxF32(numel, numDims, dims, src, dst)
		return
	}
	ncols := dims[numDims-1]
	rows := numel / ncols
	for row := range rows {
		maxVal := float32(-math.MaxFloat32)
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, numDims, dims, strides)
			if src[stridedI] > maxVal {
				maxVal = src[stridedI]
			}
		}
		sum := float32(0)
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, numDims, dims, strides)
			val := float32(math.Exp(float64(src[stridedI] - maxVal)))
			dst[stridedI] = val
			sum += val
		}
		invSum := 1 / sum
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, numDims, dims, strides)
			dst[stridedI] *= invSum
		}
	}
}

// SoftmaxStridedF64 performs strided softmax along the last dimension for float64
func SoftmaxStridedF64(numel int, numDims int, dims, strides []int, src, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
		SoftmaxF64(numel, numDims, dims, src, dst)
		return
	}
	ncols := dims[numDims-1]
	rows := numel / ncols
	for row := range rows {
		maxVal := -math.MaxFloat64
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, numDims, dims, strides)
			if src[stridedI] > maxVal {
				maxVal = src[stridedI]
			}
		}
		sum := 0.0
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, numDims, dims, strides)
			val := math.Exp(src[stridedI] - maxVal)
			dst[stridedI] = val
			sum += val
		}
		invSum := 1 / sum
		for col := range ncols {
			logicalI := row*ncols + col
			stridedI := GetStridedIndex(logicalI, numDims, dims, strides)
			dst[stridedI] *= invSum
		}
	}
}

// RmsNormF32 performs RMS normalization along the last dimension for float32 (contiguous memory)
func RmsNormF32(numel int, numDims int, dims []int, eps float32, alpha, x, dst []float32) {
	ncols := dims[numDims-1]
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

// RmsNormF64 performs RMS normalization along the last dimension for float64 (contiguous memory)
func RmsNormF64(numel int, numDims int, dims []int, eps float64, alpha, x, dst []float64) {
	ncols := dims[numDims-1]
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

// RmsNormStridedF32 performs strided RMS normalization along the last dimension for float32
func RmsNormStridedF32(numel int, numDims int, dims, strides []int, eps float32, alpha, x, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
		RmsNormF32(numel, numDims, dims, eps, alpha, x, dst)
		return
	}
	ncols := dims[numDims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float32
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, numDims, dims, strides)
			xi := x[stridedI]
			sum += xi * xi
		}
		mean := sum / float32(ncols)
		scale := 1 / float32(math.Sqrt(float64(mean+eps)))
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, numDims, dims, strides)
			if alpha != nil {
				dst[stridedI] = scale * x[stridedI] * alpha[col]
			} else {
				dst[stridedI] = scale * x[stridedI]
			}
		}
	}
}

// RmsNormStridedF64 performs strided RMS normalization along the last dimension for float64
func RmsNormStridedF64(numel int, numDims int, dims, strides []int, eps float64, alpha, x, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
		RmsNormF64(numel, numDims, dims, eps, alpha, x, dst)
		return
	}
	ncols := dims[numDims-1]
	rows := numel / ncols
	for row := range rows {
		var sum float64
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, numDims, dims, strides)
			xi := x[stridedI]
			sum += xi * xi
		}
		mean := sum / float64(ncols)
		scale := 1 / math.Sqrt(mean+eps)
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, numDims, dims, strides)
			if alpha != nil {
				dst[stridedI] = scale * x[stridedI] * alpha[col]
			} else {
				dst[stridedI] = scale * x[stridedI]
			}
		}
	}
}

// LayerNormF32 performs Layer normalization along the last dimension for float32 (contiguous memory)
func LayerNormF32(numel int, numDims int, dims []int, eps float32, alpha, beta, x, dst []float32) {
	ncols := dims[numDims-1]
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

// LayerNormF64 performs Layer normalization along the last dimension for float64 (contiguous memory)
func LayerNormF64(numel int, numDims int, dims []int, eps float64, alpha, beta, x, dst []float64) {
	ncols := dims[numDims-1]
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

// LayerNormStridedF32 performs strided Layer normalization along the last dimension for float32
func LayerNormStridedF32(numel int, numDims int, dims, strides []int, eps float32, alpha, beta, x, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
		LayerNormF32(numel, numDims, dims, eps, alpha, beta, x, dst)
		return
	}
	ncols := dims[numDims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float32
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, numDims, dims, strides)
			xi := x[stridedI]
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float32(ncols)
		variance := sumSq/float32(ncols) - mean*mean
		scale := 1 / float32(math.Sqrt(float64(variance+eps)))
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, numDims, dims, strides)
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

// LayerNormStridedF64 performs strided Layer normalization along the last dimension for float64
func LayerNormStridedF64(numel int, numDims int, dims, strides []int, eps float64, alpha, beta, x, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
		LayerNormF64(numel, numDims, dims, eps, alpha, beta, x, dst)
		return
	}
	ncols := dims[numDims-1]
	rows := numel / ncols
	for row := range rows {
		var sum, sumSq float64
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, numDims, dims, strides)
			xi := x[stridedI]
			sum += xi
			sumSq += xi * xi
		}
		mean := sum / float64(ncols)
		variance := sumSq/float64(ncols) - mean*mean
		scale := 1 / math.Sqrt(variance+eps)
		for col := range ncols {
			i := row*ncols + col
			stridedI := GetStridedIndex(i, numDims, dims, strides)
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

// RopeIStridedF32 performs strided rotary position embedding (rope_i variant) for float32
func RopeIStridedF32(numDims int, dims, strides []int, bh int, td int, strideB int, src, cos, sin, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		strided2Idx := GetStridedIndex(2*idx, numDims, dims, strides)
		strided2IdxPlus1 := GetStridedIndex(2*idx+1, numDims, dims, strides)
		dst[strided2Idx] = src[strided2Idx]*c - src[strided2IdxPlus1]*s
		dst[strided2IdxPlus1] = src[strided2Idx]*s + src[strided2IdxPlus1]*c
	}
}

// RopeIStridedF64 performs strided rotary position embedding (rope_i variant) for float64
func RopeIStridedF64(numDims int, dims, strides []int, bh int, td int, strideB int, src, cos, sin, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		strided2Idx := GetStridedIndex(2*idx, numDims, dims, strides)
		strided2IdxPlus1 := GetStridedIndex(2*idx+1, numDims, dims, strides)
		dst[strided2Idx] = src[strided2Idx]*c - src[strided2IdxPlus1]*s
		dst[strided2IdxPlus1] = src[strided2Idx]*s + src[strided2IdxPlus1]*c
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

// RopeStridedF32 performs strided rotary position embedding (rope variant) for float32
func RopeStridedF32(numDims int, dims, strides []int, bh int, td int, d int, strideB int, src, cos, sin, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI1 := GetStridedIndex(logicalI1, numDims, dims, strides)
		stridedI2 := GetStridedIndex(logicalI2, numDims, dims, strides)
		dst[stridedI1] = src[stridedI1]*c - src[stridedI2]*s
		dst[stridedI2] = src[stridedI1]*s + src[stridedI2]*c
	}
}

// RopeStridedF64 performs strided rotary position embedding (rope variant) for float64
func RopeStridedF64(numDims int, dims, strides []int, bh int, td int, d int, strideB int, src, cos, sin, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI1 := GetStridedIndex(logicalI1, numDims, dims, strides)
		stridedI2 := GetStridedIndex(logicalI2, numDims, dims, strides)
		dst[stridedI1] = src[stridedI1]*c - src[stridedI2]*s
		dst[stridedI2] = src[stridedI1]*s + src[stridedI2]*c
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

// RopeThdStridedF32 performs strided rotary position embedding (rope_thd variant) for float32
func RopeThdStridedF32(numDims int, dims, strides []int, b int, t int, h int, d int, strideB int, src, cos, sin, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI1 := GetStridedIndex(logicalI1, numDims, dims, strides)
		stridedI2 := GetStridedIndex(logicalI2, numDims, dims, strides)
		dst[stridedI1] = src[stridedI1]*c - src[stridedI2]*s
		dst[stridedI2] = src[stridedI1]*s + src[stridedI2]*c
	}
}

// RopeThdStridedF64 performs strided rotary position embedding (rope_thd variant) for float64
func RopeThdStridedF64(numDims int, dims, strides []int, b int, t int, h int, d int, strideB int, src, cos, sin, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
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
		stridedI1 := GetStridedIndex(logicalI1, numDims, dims, strides)
		stridedI2 := GetStridedIndex(logicalI2, numDims, dims, strides)
		dst[stridedI1] = src[stridedI1]*c - src[stridedI2]*s
		dst[stridedI2] = src[stridedI1]*s + src[stridedI2]*c
	}
}

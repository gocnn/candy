package kernels

import "math"

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

// SumF32 computes the sum over specified dimensions for float32
func SumF32(numel, numDims, numSumDims int, dims, sumDims []int, src, dst []float32) {
	for i := range numel {
		dstIdx := i
		for nd := range numSumDims {
			stride := dims[numDims-1]
			pre := dstIdx / stride
			post := dstIdx % stride
			dstIdx = (pre/sumDims[nd])*stride + post
		}
		dst[dstIdx] += src[i]
	}
}

// SumF64 computes the sum over specified dimensions for float64
func SumF64(numel, numDims, numSumDims int, dims, sumDims []int, src, dst []float64) {
	for i := range numel {
		dstIdx := i
		for nd := range numSumDims {
			stride := dims[numDims-1]
			pre := dstIdx / stride
			post := dstIdx % stride
			dstIdx = (pre/sumDims[nd])*stride + post
		}
		dst[dstIdx] += src[i]
	}
}

// SumStridedF32 computes the sum over specified dimensions for float32 with strided memory
func SumStridedF32(numel, numDims, numSumDims int, dims, strides, sumDims, sumStrides []int, src, dst []float32) {
	if IsContiguous(numDims, dims, strides) {
		SumF32(numel, numDims, numSumDims, dims, sumDims, src, dst)
		return
	}
	for i := range numel {
		dstIdx := i
		for nd := range numSumDims {
			stride := sumStrides[nd]
			pre := dstIdx / stride
			post := dstIdx % stride
			dstIdx = (pre/sumDims[nd])*stride + post
		}
		dst[dstIdx] += src[GetStridedIndex(i, numDims, dims, strides)]
	}
}

// SumStridedF64 computes the sum over specified dimensions for float64 with strided memory
func SumStridedF64(numel, numDims, numSumDims int, dims, strides, sumDims, sumStrides []int, src, dst []float64) {
	if IsContiguous(numDims, dims, strides) {
		SumF64(numel, numDims, numSumDims, dims, sumDims, src, dst)
		return
	}
	for i := range numel {
		dstIdx := i
		for nd := range numSumDims {
			stride := sumStrides[nd]
			pre := dstIdx / stride
			post := dstIdx % stride
			dstIdx = (pre/sumDims[nd])*stride + post
		}
		dst[dstIdx] += src[GetStridedIndex(i, numDims, dims, strides)]
	}
}

// SoftmaxF32 applies softmax along the last dimension for float32
func SoftmaxF32(ncols int, src, dst []float32) {
	rows := len(src) / ncols
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
			dst[row*ncols+col] *= invSum
		}
	}
}

// SoftmaxF64 applies softmax along the last dimension for float64
func SoftmaxF64(ncols int, src, dst []float64) {
	rows := len(src) / ncols
	for row := range rows {
		maxVal := float64(-math.MaxFloat64)
		for col := range ncols {
			i := row*ncols + col
			if src[i] > maxVal {
				maxVal = src[i]
			}
		}
		sum := float64(0)
		for col := range ncols {
			i := row*ncols + col
			val := math.Exp(src[i] - maxVal)
			dst[i] = val
			sum += val
		}
		invSum := 1 / sum
		for col := range ncols {
			dst[row*ncols+col] *= invSum
		}
	}
}

// RmsnormF32 applies RMS normalization for float32
func RmsnormF32(ncols, blockSize int, eps float32, src, dst, alpha []float32) {
	rows := len(src) / ncols
	for row := range rows {
		sum := float32(0)
		for col := 0; col < ncols; col += blockSize {
			if col < ncols {
				xi := src[row*ncols+col]
				sum += xi * xi
			}
		}
		mean := sum / float32(ncols)
		scale := 1 / float32(math.Sqrt(float64(mean+eps)))
		if alpha == nil {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = src[row*ncols+col] * scale
				}
			}
		} else {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = src[row*ncols+col] * scale * alpha[col]
				}
			}
		}
	}
}

// RmsnormF64 applies RMS normalization for float64
func RmsnormF64(ncols, blockSize int, eps float64, src, dst, alpha []float64) {
	rows := len(src) / ncols
	for row := range rows {
		sum := float64(0)
		for col := 0; col < ncols; col += blockSize {
			if col < ncols {
				xi := src[row*ncols+col]
				sum += xi * xi
			}
		}
		mean := sum / float64(ncols)
		scale := 1 / math.Sqrt(mean+eps)
		if alpha == nil {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = src[row*ncols+col] * scale
				}
			}
		} else {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = src[row*ncols+col] * scale * alpha[col]
				}
			}
		}
	}
}

// LayernormF32 applies layer normalization for float32
func LayernormF32(ncols, blockSize int, eps float32, src, dst, alpha, beta []float32) {
	rows := len(src) / ncols
	for row := range rows {
		meanVar := [2]float32{0, 0}
		for col := 0; col < ncols; col += blockSize {
			if col < ncols {
				xi := src[row*ncols+col]
				meanVar[0] += xi
				meanVar[1] += xi * xi
			}
		}
		mean := meanVar[0] / float32(ncols)
		varVal := meanVar[1]/float32(ncols) - mean*mean
		invStd := 1 / float32(math.Sqrt(float64(varVal+eps)))
		if alpha == nil && beta == nil {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = (src[row*ncols+col] - mean) * invStd
				}
			}
		} else if alpha == nil {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = (src[row*ncols+col]-mean)*invStd + beta[col]
				}
			}
		} else if beta == nil {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = (src[row*ncols+col] - mean) * invStd * alpha[col]
				}
			}
		} else {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = (src[row*ncols+col]-mean)*invStd*alpha[col] + beta[col]
				}
			}
		}
	}
}

// LayernormF64 applies layer normalization for float64
func LayernormF64(ncols, blockSize int, eps float64, src, dst, alpha, beta []float64) {
	rows := len(src) / ncols
	for row := range rows {
		meanVar := [2]float64{0, 0}
		for col := 0; col < ncols; col += blockSize {
			if col < ncols {
				xi := src[row*ncols+col]
				meanVar[0] += xi
				meanVar[1] += xi * xi
			}
		}
		mean := meanVar[0] / float64(ncols)
		varVal := meanVar[1]/float64(ncols) - mean*mean
		invStd := 1 / math.Sqrt(varVal+eps)
		if alpha == nil && beta == nil {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = (src[row*ncols+col] - mean) * invStd
				}
			}
		} else if alpha == nil {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = (src[row*ncols+col]-mean)*invStd + beta[col]
				}
			}
		} else if beta == nil {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = (src[row*ncols+col] - mean) * invStd * alpha[col]
				}
			}
		} else {
			for col := 0; col < ncols; col += blockSize {
				if col < ncols {
					dst[row*ncols+col] = (src[row*ncols+col]-mean)*invStd*alpha[col] + beta[col]
				}
			}
		}
	}
}

// RopeF32 applies rotary position embedding for float32
func RopeF32(bh, td, d, strideB int, src, cos, sin, dst []float32) {
	for idx := 0; idx < bh*td/2; idx++ {
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

// RopeF64 applies rotary position embedding for float64
func RopeF64(bh, td, d, strideB int, src, cos, sin, dst []float64) {
	for idx := 0; idx < bh*td/2; idx++ {
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

// RopeIF32 applies simplified rotary position embedding for float32
func RopeIF32(bh, td, strideB int, src, cos, sin, dst []float32) {
	for idx := 0; idx < bh*td/2; idx++ {
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

// RopeIF64 applies simplified rotary position embedding for float64
func RopeIF64(bh, td, strideB int, src, cos, sin, dst []float64) {
	for idx := 0; idx < bh*td/2; idx++ {
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

// RopeThdF32 applies rotary position embedding with thread dimensions for float32
func RopeThdF32(b, t, h, d, strideB int, src, cos, sin, dst []float32) {
	for idx := 0; idx < b*t*h*d/2; idx++ {
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

// RopeThdF64 applies rotary position embedding with thread dimensions for float64
func RopeThdF64(b, t, h, d, strideB int, src, cos, sin, dst []float64) {
	for idx := 0; idx < b*t*h*d/2; idx++ {
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

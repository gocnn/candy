package kernels

import "math"

// Sum reduction operations
func SumF32(numel int, inp, out []float32) {
	var sum float32
	for i := range numel {
		sum += inp[i]
	}
	out[0] = sum
}

func SumF64(numel int, inp, out []float64) {
	var sum float64
	for i := range numel {
		sum += inp[i]
	}
	out[0] = sum
}

func SumStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		SumF32(numel, inp, out)
		return
	}
	var sum float32
	for i := range numel {
		sum += inp[GetStridedIndex(i, numDims, dims, strides)]
	}
	out[0] = sum
}

func SumStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		SumF64(numel, inp, out)
		return
	}
	var sum float64
	for i := range numel {
		sum += inp[GetStridedIndex(i, numDims, dims, strides)]
	}
	out[0] = sum
}

// ArgMin reduction operations
func ArgMinF32(numel int, inp []float32, out []int64) {
	if numel == 0 {
		return
	}
	min := inp[0]
	minIdx := int64(0)
	for i := 1; i < numel; i++ {
		if inp[i] < min {
			min = inp[i]
			minIdx = int64(i)
		}
	}
	out[0] = minIdx
}

func ArgMinF64(numel int, inp []float64, out []int64) {
	if numel == 0 {
		return
	}
	min := inp[0]
	minIdx := int64(0)
	for i := 1; i < numel; i++ {
		if inp[i] < min {
			min = inp[i]
			minIdx = int64(i)
		}
	}
	out[0] = minIdx
}

func ArgMinStridedF32(numel, numDims int, dims, strides []int, inp []float32, out []int64) {
	if IsContiguous(numDims, dims, strides) {
		ArgMinF32(numel, inp, out)
		return
	}
	if numel == 0 {
		return
	}
	min := inp[GetStridedIndex(0, numDims, dims, strides)]
	minIdx := int64(0)
	for i := 1; i < numel; i++ {
		if val := inp[GetStridedIndex(i, numDims, dims, strides)]; val < min {
			min = val
			minIdx = int64(i)
		}
	}
	out[0] = minIdx
}

func ArgMinStridedF64(numel, numDims int, dims, strides []int, inp []float64, out []int64) {
	if IsContiguous(numDims, dims, strides) {
		ArgMinF64(numel, inp, out)
		return
	}
	if numel == 0 {
		return
	}
	min := inp[GetStridedIndex(0, numDims, dims, strides)]
	minIdx := int64(0)
	for i := 1; i < numel; i++ {
		if val := inp[GetStridedIndex(i, numDims, dims, strides)]; val < min {
			min = val
			minIdx = int64(i)
		}
	}
	out[0] = minIdx
}

// ArgMax reduction operations
func ArgMaxF32(numel int, inp []float32, out []int64) {
	if numel == 0 {
		return
	}
	max := inp[0]
	maxIdx := int64(0)
	for i := 1; i < numel; i++ {
		if inp[i] > max {
			max = inp[i]
			maxIdx = int64(i)
		}
	}
	out[0] = maxIdx
}

func ArgMaxF64(numel int, inp []float64, out []int64) {
	if numel == 0 {
		return
	}
	max := inp[0]
	maxIdx := int64(0)
	for i := 1; i < numel; i++ {
		if inp[i] > max {
			max = inp[i]
			maxIdx = int64(i)
		}
	}
	out[0] = maxIdx
}

func ArgMaxStridedF32(numel, numDims int, dims, strides []int, inp []float32, out []int64) {
	if IsContiguous(numDims, dims, strides) {
		ArgMaxF32(numel, inp, out)
		return
	}
	if numel == 0 {
		return
	}
	max := inp[GetStridedIndex(0, numDims, dims, strides)]
	maxIdx := int64(0)
	for i := 1; i < numel; i++ {
		if val := inp[GetStridedIndex(i, numDims, dims, strides)]; val > max {
			max = val
			maxIdx = int64(i)
		}
	}
	out[0] = maxIdx
}

func ArgMaxStridedF64(numel, numDims int, dims, strides []int, inp []float64, out []int64) {
	if IsContiguous(numDims, dims, strides) {
		ArgMaxF64(numel, inp, out)
		return
	}
	if numel == 0 {
		return
	}
	max := inp[GetStridedIndex(0, numDims, dims, strides)]
	maxIdx := int64(0)
	for i := 1; i < numel; i++ {
		if val := inp[GetStridedIndex(i, numDims, dims, strides)]; val > max {
			max = val
			maxIdx = int64(i)
		}
	}
	out[0] = maxIdx
}

// Softmax operations
func SoftmaxF32(numel int, inp, out []float32) {
	if numel == 0 {
		return
	}
	maxVal := inp[0]
	for i := 1; i < numel; i++ {
		if inp[i] > maxVal {
			maxVal = inp[i]
		}
	}
	var sum float32
	for i := range numel {
		out[i] = float32(math.Exp(float64(inp[i] - maxVal)))
		sum += out[i]
	}
	for i := range numel {
		out[i] /= sum
	}
}

func SoftmaxF64(numel int, inp, out []float64) {
	if numel == 0 {
		return
	}
	maxVal := inp[0]
	for i := 1; i < numel; i++ {
		if inp[i] > maxVal {
			maxVal = inp[i]
		}
	}
	var sum float64
	for i := range numel {
		out[i] = math.Exp(inp[i] - maxVal)
		sum += out[i]
	}
	for i := range numel {
		out[i] /= sum
	}
}

func SoftmaxStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		SoftmaxF32(numel, inp, out)
		return
	}
	if numel == 0 {
		return
	}
	maxVal := inp[GetStridedIndex(0, numDims, dims, strides)]
	for i := 1; i < numel; i++ {
		if val := inp[GetStridedIndex(i, numDims, dims, strides)]; val > maxVal {
			maxVal = val
		}
	}
	var sum float32
	for i := range numel {
		out[i] = float32(math.Exp(float64(inp[GetStridedIndex(i, numDims, dims, strides)] - maxVal)))
		sum += out[i]
	}
	for i := range numel {
		out[i] /= sum
	}
}

func SoftmaxStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		SoftmaxF64(numel, inp, out)
		return
	}
	if numel == 0 {
		return
	}
	maxVal := inp[GetStridedIndex(0, numDims, dims, strides)]
	for i := 1; i < numel; i++ {
		if val := inp[GetStridedIndex(i, numDims, dims, strides)]; val > maxVal {
			maxVal = val
		}
	}
	var sum float64
	for i := range numel {
		out[i] = math.Exp(inp[GetStridedIndex(i, numDims, dims, strides)] - maxVal)
		sum += out[i]
	}
	for i := range numel {
		out[i] /= sum
	}
}

// LogSoftmax operations
func LogSoftmaxF32(numel int, inp, out []float32) {
	if numel == 0 {
		return
	}
	maxVal := inp[0]
	for i := 1; i < numel; i++ {
		if inp[i] > maxVal {
			maxVal = inp[i]
		}
	}
	var sum float32
	for i := range numel {
		sum += float32(math.Exp(float64(inp[i] - maxVal)))
	}
	logSum := maxVal + float32(math.Log(float64(sum)))
	for i := range numel {
		out[i] = inp[i] - logSum
	}
}

func LogSoftmaxF64(numel int, inp, out []float64) {
	if numel == 0 {
		return
	}
	maxVal := inp[0]
	for i := 1; i < numel; i++ {
		if inp[i] > maxVal {
			maxVal = inp[i]
		}
	}
	var sum float64
	for i := range numel {
		sum += math.Exp(inp[i] - maxVal)
	}
	logSum := maxVal + math.Log(sum)
	for i := range numel {
		out[i] = inp[i] - logSum
	}
}

func LogSoftmaxStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		LogSoftmaxF32(numel, inp, out)
		return
	}
	if numel == 0 {
		return
	}
	maxVal := inp[GetStridedIndex(0, numDims, dims, strides)]
	for i := 1; i < numel; i++ {
		if val := inp[GetStridedIndex(i, numDims, dims, strides)]; val > maxVal {
			maxVal = val
		}
	}
	var sum float32
	for i := range numel {
		sum += float32(math.Exp(float64(inp[GetStridedIndex(i, numDims, dims, strides)] - maxVal)))
	}
	logSum := maxVal + float32(math.Log(float64(sum)))
	for i := range numel {
		out[i] = inp[GetStridedIndex(i, numDims, dims, strides)] - logSum
	}
}

func LogSoftmaxStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		LogSoftmaxF64(numel, inp, out)
		return
	}
	if numel == 0 {
		return
	}
	maxVal := inp[GetStridedIndex(0, numDims, dims, strides)]
	for i := 1; i < numel; i++ {
		if val := inp[GetStridedIndex(i, numDims, dims, strides)]; val > maxVal {
			maxVal = val
		}
	}
	var sum float64
	for i := range numel {
		sum += math.Exp(inp[GetStridedIndex(i, numDims, dims, strides)] - maxVal)
	}
	logSum := maxVal + math.Log(sum)
	for i := range numel {
		out[i] = inp[GetStridedIndex(i, numDims, dims, strides)] - logSum
	}
}

// LayerNorm operations
func LayerNormF32(numel int, inp, gamma, beta, out []float32, eps float32) {
	var sum, sumSq float32
	for i := range numel {
		x := inp[i]
		sum += x
		sumSq += x * x
	}
	mean := sum / float32(numel)
	var2 := sumSq/float32(numel) - mean*mean
	invStd := float32(1.0 / math.Sqrt(float64(var2+eps)))
	for i := range numel {
		out[i] = (inp[i]-mean)*invStd*gamma[i] + beta[i]
	}
}

func LayerNormF64(numel int, inp, gamma, beta, out []float64, eps float64) {
	var sum, sumSq float64
	for i := range numel {
		x := inp[i]
		sum += x
		sumSq += x * x
	}
	mean := sum / float64(numel)
	var2 := sumSq/float64(numel) - mean*mean
	invStd := 1.0 / math.Sqrt(var2+eps)
	for i := range numel {
		out[i] = (inp[i]-mean)*invStd*gamma[i] + beta[i]
	}
}

func LayerNormStridedF32(numel, numDims int, dims, strides []int, inp, gamma, beta, out []float32, eps float32) {
	if IsContiguous(numDims, dims, strides) {
		LayerNormF32(numel, inp, gamma, beta, out, eps)
		return
	}
	var sum, sumSq float32
	for i := range numel {
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		sum += x
		sumSq += x * x
	}
	mean := sum / float32(numel)
	var2 := sumSq/float32(numel) - mean*mean
	invStd := float32(1.0 / math.Sqrt(float64(var2+eps)))
	for i := range numel {
		out[i] = (inp[GetStridedIndex(i, numDims, dims, strides)]-mean)*invStd*gamma[i] + beta[i]
	}
}

func LayerNormStridedF64(numel, numDims int, dims, strides []int, inp, gamma, beta, out []float64, eps float64) {
	if IsContiguous(numDims, dims, strides) {
		LayerNormF64(numel, inp, gamma, beta, out, eps)
		return
	}
	var sum, sumSq float64
	for i := range numel {
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		sum += x
		sumSq += x * x
	}
	mean := sum / float64(numel)
	var2 := sumSq/float64(numel) - mean*mean
	invStd := 1.0 / math.Sqrt(var2+eps)
	for i := range numel {
		out[i] = (inp[GetStridedIndex(i, numDims, dims, strides)]-mean)*invStd*gamma[i] + beta[i]
	}
}

// RMSNorm operations
func RMSNormF32(numel int, inp, gamma, out []float32, eps float32) {
	var sumSq float32
	for i := range numel {
		x := inp[i]
		sumSq += x * x
	}
	rms := float32(math.Sqrt(float64(sumSq/float32(numel) + eps)))
	for i := range numel {
		out[i] = inp[i] / rms * gamma[i]
	}
}

func RMSNormF64(numel int, inp, gamma, out []float64, eps float64) {
	var sumSq float64
	for i := range numel {
		x := inp[i]
		sumSq += x * x
	}
	rms := math.Sqrt(sumSq/float64(numel) + eps)
	for i := range numel {
		out[i] = inp[i] / rms * gamma[i]
	}
}

func RMSNormStridedF32(numel, numDims int, dims, strides []int, inp, gamma, out []float32, eps float32) {
	if IsContiguous(numDims, dims, strides) {
		RMSNormF32(numel, inp, gamma, out, eps)
		return
	}
	var sumSq float32
	for i := range numel {
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		sumSq += x * x
	}
	rms := float32(math.Sqrt(float64(sumSq/float32(numel) + eps)))
	for i := range numel {
		out[i] = inp[GetStridedIndex(i, numDims, dims, strides)] / rms * gamma[i]
	}
}

func RMSNormStridedF64(numel, numDims int, dims, strides []int, inp, gamma, out []float64, eps float64) {
	if IsContiguous(numDims, dims, strides) {
		RMSNormF64(numel, inp, gamma, out, eps)
		return
	}
	var sumSq float64
	for i := range numel {
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		sumSq += x * x
	}
	rms := math.Sqrt(sumSq/float64(numel) + eps)
	for i := range numel {
		out[i] = inp[GetStridedIndex(i, numDims, dims, strides)] / rms * gamma[i]
	}
}

// GroupNorm operations
func GroupNormF32(numel, numGroups int, inp, gamma, beta, out []float32, eps float32) {
	groupSize := numel / numGroups
	for g := 0; g < numGroups; g++ {
		var sum, sumSq float32
		for i := 0; i < groupSize; i++ {
			idx := g*groupSize + i
			x := inp[idx]
			sum += x
			sumSq += x * x
		}
		mean := sum / float32(groupSize)
		var2 := sumSq/float32(groupSize) - mean*mean
		invStd := float32(1.0 / math.Sqrt(float64(var2+eps)))
		for i := 0; i < groupSize; i++ {
			idx := g*groupSize + i
			out[idx] = (inp[idx]-mean)*invStd*gamma[i] + beta[i]
		}
	}
}

func GroupNormF64(numel, numGroups int, inp, gamma, beta, out []float64, eps float64) {
	groupSize := numel / numGroups
	for g := 0; g < numGroups; g++ {
		var sum, sumSq float64
		for i := 0; i < groupSize; i++ {
			idx := g*groupSize + i
			x := inp[idx]
			sum += x
			sumSq += x * x
		}
		mean := sum / float64(groupSize)
		var2 := sumSq/float64(groupSize) - mean*mean
		invStd := 1.0 / math.Sqrt(var2+eps)
		for i := 0; i < groupSize; i++ {
			idx := g*groupSize + i
			out[idx] = (inp[idx]-mean)*invStd*gamma[i] + beta[i]
		}
	}
}

func GroupNormStridedF32(numel, numGroups, numDims int, dims, strides []int, inp, gamma, beta, out []float32, eps float32) {
	if IsContiguous(numDims, dims, strides) {
		GroupNormF32(numel, numGroups, inp, gamma, beta, out, eps)
		return
	}
	groupSize := numel / numGroups
	for g := 0; g < numGroups; g++ {
		var sum, sumSq float32
		for i := 0; i < groupSize; i++ {
			idx := g*groupSize + i
			x := inp[GetStridedIndex(idx, numDims, dims, strides)]
			sum += x
			sumSq += x * x
		}
		mean := sum / float32(groupSize)
		var2 := sumSq/float32(groupSize) - mean*mean
		invStd := float32(1.0 / math.Sqrt(float64(var2+eps)))
		for i := 0; i < groupSize; i++ {
			idx := g*groupSize + i
			out[idx] = (inp[GetStridedIndex(idx, numDims, dims, strides)]-mean)*invStd*gamma[i] + beta[i]
		}
	}
}

func GroupNormStridedF64(numel, numGroups, numDims int, dims, strides []int, inp, gamma, beta, out []float64, eps float64) {
	if IsContiguous(numDims, dims, strides) {
		GroupNormF64(numel, numGroups, inp, gamma, beta, out, eps)
		return
	}
	groupSize := numel / numGroups
	for g := 0; g < numGroups; g++ {
		var sum, sumSq float64
		for i := 0; i < groupSize; i++ {
			idx := g*groupSize + i
			x := inp[GetStridedIndex(idx, numDims, dims, strides)]
			sum += x
			sumSq += x * x
		}
		mean := sum / float64(groupSize)
		var2 := sumSq/float64(groupSize) - mean*mean
		invStd := 1.0 / math.Sqrt(var2+eps)
		for i := 0; i < groupSize; i++ {
			idx := g*groupSize + i
			out[idx] = (inp[GetStridedIndex(idx, numDims, dims, strides)]-mean)*invStd*gamma[i] + beta[i]
		}
	}
}

// Variance reduction operations (population variance)
func VarF32(numel int, inp, out []float32) {
	var sum, sumSq float32
	for i := range numel {
		x := inp[i]
		sum += x
		sumSq += x * x
	}
	mean := sum / float32(numel)
	out[0] = sumSq/float32(numel) - mean*mean
}

func VarF64(numel int, inp, out []float64) {
	var sum, sumSq float64
	for i := range numel {
		x := inp[i]
		sum += x
		sumSq += x * x
	}
	mean := sum / float64(numel)
	out[0] = sumSq/float64(numel) - mean*mean
}

func VarStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		VarF32(numel, inp, out)
		return
	}
	var sum, sumSq float32
	for i := range numel {
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		sum += x
		sumSq += x * x
	}
	mean := sum / float32(numel)
	out[0] = sumSq/float32(numel) - mean*mean
}

func VarStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		VarF64(numel, inp, out)
		return
	}
	var sum, sumSq float64
	for i := range numel {
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		sum += x
		sumSq += x * x
	}
	mean := sum / float64(numel)
	out[0] = sumSq/float64(numel) - mean*mean
}

// Standard deviation operations
func StdF32(numel int, inp, out []float32) {
	var sum, sumSq float32
	for i := range numel {
		x := inp[i]
		sum += x
		sumSq += x * x
	}
	mean := sum / float32(numel)
	out[0] = float32(math.Sqrt(float64(sumSq/float32(numel) - mean*mean)))
}

func StdF64(numel int, inp, out []float64) {
	var sum, sumSq float64
	for i := range numel {
		x := inp[i]
		sum += x
		sumSq += x * x
	}
	mean := sum / float64(numel)
	out[0] = math.Sqrt(sumSq/float64(numel) - mean*mean)
}

func StdStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		StdF32(numel, inp, out)
		return
	}
	var sum, sumSq float32
	for i := range numel {
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		sum += x
		sumSq += x * x
	}
	mean := sum / float32(numel)
	out[0] = float32(math.Sqrt(float64(sumSq/float32(numel) - mean*mean)))
}

func StdStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		StdF64(numel, inp, out)
		return
	}
	var sum, sumSq float64
	for i := range numel {
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		sum += x
		sumSq += x * x
	}
	mean := sum / float64(numel)
	out[0] = math.Sqrt(sumSq/float64(numel) - mean*mean)
}

// Mean reduction operations
func MeanF32(numel int, inp, out []float32) {
	var sum float32
	for i := range numel {
		sum += inp[i]
	}
	out[0] = sum / float32(numel)
}

func MeanF64(numel int, inp, out []float64) {
	var sum float64
	for i := range numel {
		sum += inp[i]
	}
	out[0] = sum / float64(numel)
}

func MeanStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		MeanF32(numel, inp, out)
		return
	}
	var sum float32
	for i := range numel {
		sum += inp[GetStridedIndex(i, numDims, dims, strides)]
	}
	out[0] = sum / float32(numel)
}

func MeanStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		MeanF64(numel, inp, out)
		return
	}
	var sum float64
	for i := range numel {
		sum += inp[GetStridedIndex(i, numDims, dims, strides)]
	}
	out[0] = sum / float64(numel)
}

// L1 norm operations
func L1NormF32(numel int, inp, out []float32) {
	var sum float32
	for i := range numel {
		sum += float32(math.Abs(float64(inp[i])))
	}
	out[0] = sum
}

func L1NormF64(numel int, inp, out []float64) {
	var sum float64
	for i := range numel {
		sum += math.Abs(inp[i])
	}
	out[0] = sum
}

func L1NormStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		L1NormF32(numel, inp, out)
		return
	}
	var sum float32
	for i := range numel {
		sum += float32(math.Abs(float64(inp[GetStridedIndex(i, numDims, dims, strides)])))
	}
	out[0] = sum
}

func L1NormStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		L1NormF64(numel, inp, out)
		return
	}
	var sum float64
	for i := range numel {
		sum += math.Abs(inp[GetStridedIndex(i, numDims, dims, strides)])
	}
	out[0] = sum
}

// L2 norm operations
func L2NormF32(numel int, inp, out []float32) {
	var sumSq float32
	for i := range numel {
		x := inp[i]
		sumSq += x * x
	}
	out[0] = float32(math.Sqrt(float64(sumSq)))
}

func L2NormF64(numel int, inp, out []float64) {
	var sumSq float64
	for i := range numel {
		x := inp[i]
		sumSq += x * x
	}
	out[0] = math.Sqrt(sumSq)
}

func L2NormStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		L2NormF32(numel, inp, out)
		return
	}
	var sumSq float32
	for i := range numel {
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		sumSq += x * x
	}
	out[0] = float32(math.Sqrt(float64(sumSq)))
}

func L2NormStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		L2NormF64(numel, inp, out)
		return
	}
	var sumSq float64
	for i := range numel {
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		sumSq += x * x
	}
	out[0] = math.Sqrt(sumSq)
}

// LogSumExp operations
func LogSumExpF32(numel int, inp, out []float32) {
	if numel == 0 {
		return
	}
	maxVal := inp[0]
	for i := 1; i < numel; i++ {
		if inp[i] > maxVal {
			maxVal = inp[i]
		}
	}
	var sum float32
	for i := range numel {
		sum += float32(math.Exp(float64(inp[i] - maxVal)))
	}
	out[0] = maxVal + float32(math.Log(float64(sum)))
}

func LogSumExpF64(numel int, inp, out []float64) {
	if numel == 0 {
		return
	}
	maxVal := inp[0]
	for i := 1; i < numel; i++ {
		if inp[i] > maxVal {
			maxVal = inp[i]
		}
	}
	var sum float64
	for i := range numel {
		sum += math.Exp(inp[i] - maxVal)
	}
	out[0] = maxVal + math.Log(sum)
}

func LogSumExpStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		LogSumExpF32(numel, inp, out)
		return
	}
	if numel == 0 {
		return
	}
	maxVal := inp[GetStridedIndex(0, numDims, dims, strides)]
	for i := 1; i < numel; i++ {
		if val := inp[GetStridedIndex(i, numDims, dims, strides)]; val > maxVal {
			maxVal = val
		}
	}
	var sum float32
	for i := range numel {
		sum += float32(math.Exp(float64(inp[GetStridedIndex(i, numDims, dims, strides)] - maxVal)))
	}
	out[0] = maxVal + float32(math.Log(float64(sum)))
}

func LogSumExpStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		LogSumExpF64(numel, inp, out)
		return
	}
	if numel == 0 {
		return
	}
	maxVal := inp[GetStridedIndex(0, numDims, dims, strides)]
	for i := 1; i < numel; i++ {
		if val := inp[GetStridedIndex(i, numDims, dims, strides)]; val > maxVal {
			maxVal = val
		}
	}
	var sum float64
	for i := range numel {
		sum += math.Exp(inp[GetStridedIndex(i, numDims, dims, strides)] - maxVal)
	}
	out[0] = maxVal + math.Log(sum)
}

// Welford variance operations (numerically stable)
func WelfordVarF32(numel int, inp, out []float32) {
	var mean, m2 float32
	count := float32(0)
	for i := range numel {
		count++
		x := inp[i]
		delta := x - mean
		mean += delta / count
		delta2 := x - mean
		m2 += delta * delta2
	}
	if count > 1 {
		out[0] = m2 / (count - 1) // Sample variance
	} else {
		out[0] = 0
	}
}

func WelfordVarF64(numel int, inp, out []float64) {
	var mean, m2 float64
	count := float64(0)
	for i := range numel {
		count++
		x := inp[i]
		delta := x - mean
		mean += delta / count
		delta2 := x - mean
		m2 += delta * delta2
	}
	if count > 1 {
		out[0] = m2 / (count - 1) // Sample variance
	} else {
		out[0] = 0
	}
}

func WelfordVarStridedF32(numel, numDims int, dims, strides []int, inp, out []float32) {
	if IsContiguous(numDims, dims, strides) {
		WelfordVarF32(numel, inp, out)
		return
	}
	var mean, m2 float32
	count := float32(0)
	for i := range numel {
		count++
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		delta := x - mean
		mean += delta / count
		delta2 := x - mean
		m2 += delta * delta2
	}
	if count > 1 {
		out[0] = m2 / (count - 1)
	} else {
		out[0] = 0
	}
}

func WelfordVarStridedF64(numel, numDims int, dims, strides []int, inp, out []float64) {
	if IsContiguous(numDims, dims, strides) {
		WelfordVarF64(numel, inp, out)
		return
	}
	var mean, m2 float64
	count := float64(0)
	for i := range numel {
		count++
		x := inp[GetStridedIndex(i, numDims, dims, strides)]
		delta := x - mean
		mean += delta / count
		delta2 := x - mean
		m2 += delta * delta2
	}
	if count > 1 {
		out[0] = m2 / (count - 1)
	} else {
		out[0] = 0
	}
}

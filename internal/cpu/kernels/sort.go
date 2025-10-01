package kernels

import "sort"

// AsortAscF32 performs ascending argsort along the last dimension for float32
func AsortAscF32(ncols int, src, dst []float32) []uint32 {
	rows := len(src) / ncols
	indices := make([]uint32, len(src))
	for i := range src {
		indices[i] = uint32(i % ncols)
	}
	for row := range rows {
		start := row * ncols
		end := start + ncols
		rowIndices := indices[start:end]
		rowSrc := src[start:end]
		// Sort indices based on src values in ascending order
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] < rowSrc[rowIndices[j]]
		})
		// Copy sorted indices to dst
		for col := range ncols {
			dst[start+col] = float32(rowIndices[col])
		}
	}
	return indices
}

// AsortAscF64 performs ascending argsort along the last dimension for float64
func AsortAscF64(ncols int, src, dst []float64) []uint32 {
	rows := len(src) / ncols
	indices := make([]uint32, len(src))
	for i := range src {
		indices[i] = uint32(i % ncols)
	}
	for row := range rows {
		start := row * ncols
		end := start + ncols
		rowIndices := indices[start:end]
		rowSrc := src[start:end]
		// Sort indices based on src values in ascending order
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] < rowSrc[rowIndices[j]]
		})
		// Copy sorted indices to dst
		for col := range ncols {
			dst[start+col] = float64(rowIndices[col])
		}
	}
	return indices
}

// AsortDescF32 performs descending argsort along the last dimension for float32
func AsortDescF32(ncols int, src, dst []float32) []uint32 {
	rows := len(src) / ncols
	indices := make([]uint32, len(src))
	for i := range src {
		indices[i] = uint32(i % ncols)
	}
	for row := range rows {
		start := row * ncols
		end := start + ncols
		rowIndices := indices[start:end]
		rowSrc := src[start:end]
		// Sort indices based on src values in descending order
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] > rowSrc[rowIndices[j]]
		})
		// Copy sorted indices to dst
		for col := range ncols {
			dst[start+col] = float32(rowIndices[col])
		}
	}
	return indices
}

// AsortDescF64 performs descending argsort along the last dimension for float64
func AsortDescF64(ncols int, src, dst []float64) []uint32 {
	rows := len(src) / ncols
	indices := make([]uint32, len(src))
	for i := range src {
		indices[i] = uint32(i % ncols)
	}
	for row := range rows {
		start := row * ncols
		end := start + ncols
		rowIndices := indices[start:end]
		rowSrc := src[start:end]
		// Sort indices based on src values in descending order
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] > rowSrc[rowIndices[j]]
		})
		// Copy sorted indices to dst
		for col := range ncols {
			dst[start+col] = float64(rowIndices[col])
		}
	}
	return indices
}

// AsortAscStridedF32 performs ascending argsort along the last dimension for float32 with strided memory
func AsortAscStridedF32(ncols, numDims int, dims, strides []int, src, dst []float32) []uint32 {
	if IsContiguous(numDims, dims, strides) {
		return AsortAscF32(ncols, src, dst)
	}
	rows := len(src) / ncols
	indices := make([]uint32, len(src))
	for i := range src {
		indices[i] = uint32(i % ncols)
	}
	for row := range rows {
		start := row * ncols
		end := start + ncols
		rowIndices := indices[start:end]
		// Sort indices based on src values in ascending order
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		// Copy sorted indices to dst
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, strides)] = float32(rowIndices[col])
		}
	}
	return indices
}

// AsortAscStridedF64 performs ascending argsort along the last dimension for float64 with strided memory
func AsortAscStridedF64(ncols, numDims int, dims, strides []int, src, dst []float64) []uint32 {
	if IsContiguous(numDims, dims, strides) {
		return AsortAscF64(ncols, src, dst)
	}
	rows := len(src) / ncols
	indices := make([]uint32, len(src))
	for i := range src {
		indices[i] = uint32(i % ncols)
	}
	for row := range rows {
		start := row * ncols
		end := start + ncols
		rowIndices := indices[start:end]
		// Sort indices based on src values in ascending order
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		// Copy sorted indices to dst
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, strides)] = float64(rowIndices[col])
		}
	}
	return indices
}

// AsortDescStridedF32 performs descending argsort along the last dimension for float32 with strided memory
func AsortDescStridedF32(ncols, numDims int, dims, strides []int, src, dst []float32) []uint32 {
	if IsContiguous(numDims, dims, strides) {
		return AsortDescF32(ncols, src, dst)
	}
	rows := len(src) / ncols
	indices := make([]uint32, len(src))
	for i := range src {
		indices[i] = uint32(i % ncols)
	}
	for row := range rows {
		start := row * ncols
		end := start + ncols
		rowIndices := indices[start:end]
		// Sort indices based on src values in descending order
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		// Copy sorted indices to dst
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, strides)] = float32(rowIndices[col])
		}
	}
	return indices
}

// AsortDescStridedF64 performs descending argsort along the last dimension for float64 with strided memory
func AsortDescStridedF64(ncols, numDims int, dims, strides []int, src, dst []float64) []uint32 {
	if IsContiguous(numDims, dims, strides) {
		return AsortDescF64(ncols, src, dst)
	}
	rows := len(src) / ncols
	indices := make([]uint32, len(src))
	for i := range src {
		indices[i] = uint32(i % ncols)
	}
	for row := range rows {
		start := row * ncols
		end := start + ncols
		rowIndices := indices[start:end]
		// Sort indices based on src values in descending order
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		// Copy sorted indices to dst
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, strides)] = float64(rowIndices[col])
		}
	}
	return indices
}

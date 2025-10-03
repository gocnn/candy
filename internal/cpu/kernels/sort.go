package kernels

import (
	"sort"
)

// AsortAsc performs ascending argsort along the last dimension for type T (contiguous memory) with indices of type U
func AsortAsc[U I, T D](ncols int, src []T, dst []U) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]U, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = U(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] < rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortAscI64F32 performs ascending argsort along the last dimension for float32 with int64 indices (contiguous memory)
func AsortAscI64F32(ncols int, src []float32, dst []int64) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]int64, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = int64(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] < rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortAscI64F64 performs ascending argsort along the last dimension for float64 with int64 indices (contiguous memory)
func AsortAscI64F64(ncols int, src []float64, dst []int64) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]int64, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = int64(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] < rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortAscU32F32 performs ascending argsort along the last dimension for float32 with uint32 indices (contiguous memory)
func AsortAscU32F32(ncols int, src []float32, dst []uint32) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]uint32, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint32(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] < rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortAscU32F64 performs ascending argsort along the last dimension for float64 with uint32 indices (contiguous memory)
func AsortAscU32F64(ncols int, src []float64, dst []uint32) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]uint32, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint32(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] < rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortAscU8F32 performs ascending argsort along the last dimension for float32 with uint8 indices (contiguous memory)
func AsortAscU8F32(ncols int, src []float32, dst []uint8) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]uint8, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint8(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] < rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortAscU8F64 performs ascending argsort along the last dimension for float64 with uint8 indices (contiguous memory)
func AsortAscU8F64(ncols int, src []float64, dst []uint8) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]uint8, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint8(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] < rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortAscStrided performs ascending argsort along the last dimension for type T (strided memory) with indices of type U
func AsortAscStrided[U I, T D](numDims int, dims, strides, stridesDst []int, ncols int, src []T, dst []U) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortAsc(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]U, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = U(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedI64F32 performs ascending argsort along the last dimension for float32 with int64 indices (strided memory)
func AsortAscStridedI64F32(numDims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []int64) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortAscI64F32(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]int64, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = int64(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedI64F64 performs ascending argsort along the last dimension for float64 with int64 indices (strided memory)
func AsortAscStridedI64F64(numDims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []int64) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortAscI64F64(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]int64, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = int64(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU32F32 performs ascending argsort along the last dimension for float32 with uint32 indices (strided memory)
func AsortAscStridedU32F32(numDims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []uint32) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortAscU32F32(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]uint32, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint32(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU32F64 performs ascending argsort along the last dimension for float64 with uint32 indices (strided memory)
func AsortAscStridedU32F64(numDims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []uint32) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortAscU32F64(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]uint32, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint32(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU8F32 performs ascending argsort along the last dimension for float32 with uint8 indices (strided memory)
func AsortAscStridedU8F32(numDims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []uint8) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortAscU8F32(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]uint8, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint8(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU8F64 performs ascending argsort along the last dimension for float64 with uint8 indices (strided memory)
func AsortAscStridedU8F64(numDims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []uint8) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortAscU8F64(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]uint8, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint8(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDesc performs descending argsort along the last dimension for type T (contiguous memory) with indices of type U
func AsortDesc[U I, T D](ncols int, src []T, dst []U) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]U, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = U(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] > rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortDescI64F32 performs descending argsort along the last dimension for float32 with int64 indices (contiguous memory)
func AsortDescI64F32(ncols int, src []float32, dst []int64) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]int64, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = int64(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] > rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortDescI64F64 performs descending argsort along the last dimension for float64 with int64 indices (contiguous memory)
func AsortDescI64F64(ncols int, src []float64, dst []int64) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]int64, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = int64(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] > rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortDescU32F32 performs descending argsort along the last dimension for float32 with uint32 indices (contiguous memory)
func AsortDescU32F32(ncols int, src []float32, dst []uint32) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]uint32, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint32(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] > rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortDescU32F64 performs descending argsort along the last dimension for float64 with uint32 indices (contiguous memory)
func AsortDescU32F64(ncols int, src []float64, dst []uint32) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]uint32, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint32(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] > rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortDescU8F32 performs descending argsort along the last dimension for float32 with uint8 indices (contiguous memory)
func AsortDescU8F32(ncols int, src []float32, dst []uint8) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]uint8, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint8(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] > rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortDescU8F64 performs descending argsort along the last dimension for float64 with uint8 indices (contiguous memory)
func AsortDescU8F64(ncols int, src []float64, dst []uint8) {
	if ncols == 0 || len(src) == 0 {
		return
	}
	rows := len(src) / ncols
	indices := make([]uint8, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint8(col)
		}
		start := row * ncols
		end := start + ncols
		rowIndices := indices[:ncols]
		rowSrc := src[start:end]
		sort.Slice(rowIndices, func(i, j int) bool {
			return rowSrc[rowIndices[i]] > rowSrc[rowIndices[j]]
		})
		for col := range ncols {
			dst[start+col] = rowIndices[col]
		}
	}
}

// AsortDescStrided performs descending argsort along the last dimension for type T (strided memory) with indices of type U
func AsortDescStrided[U I, T D](numDims int, dims, strides, stridesDst []int, ncols int, src []T, dst []U) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortDesc(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]U, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = U(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedI64F32 performs descending argsort along the last dimension for float32 with int64 indices (strided memory)
func AsortDescStridedI64F32(numDims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []int64) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortDescI64F32(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]int64, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = int64(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedI64F64 performs descending argsort along the last dimension for float64 with int64 indices (strided memory)
func AsortDescStridedI64F64(numDims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []int64) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortDescI64F64(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]int64, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = int64(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU32F32 performs descending argsort along the last dimension for float32 with uint32 indices (strided memory)
func AsortDescStridedU32F32(numDims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []uint32) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortDescU32F32(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]uint32, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint32(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU32F64 performs descending argsort along the last dimension for float64 with uint32 indices (strided memory)
func AsortDescStridedU32F64(numDims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []uint32) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortDescU32F64(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]uint32, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint32(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU8F32 performs descending argsort along the last dimension for float32 with uint8 indices (strided memory)
func AsortDescStridedU8F32(numDims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []uint8) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortDescU8F32(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]uint8, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint8(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU8F64 performs descending argsort along the last dimension for float64 with uint8 indices (strided memory)
func AsortDescStridedU8F64(numDims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []uint8) {
	if IsContiguous(numDims, dims, strides) && IsContiguous(numDims, dims, stridesDst) {
		AsortDescU8F64(ncols, src, dst)
		return
	}
	rows := len(src) / ncols
	indices := make([]uint8, ncols)
	for row := range rows {
		for col := range ncols {
			indices[col] = uint8(col)
		}
		start := row * ncols
		rowIndices := indices[:ncols]
		sort.Slice(rowIndices, func(i, j int) bool {
			return src[GetStridedIndex(start+int(rowIndices[i]), numDims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), numDims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, numDims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

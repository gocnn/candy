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
func AsortAscStrided[U I, T D](ndims int, dims, strides, stridesDst []int, ncols int, src []T, dst []U) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedI64F32 performs ascending argsort along the last dimension for float32 with int64 indices (strided memory)
func AsortAscStridedI64F32(ndims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedI64F64 performs ascending argsort along the last dimension for float64 with int64 indices (strided memory)
func AsortAscStridedI64F64(ndims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU32F32 performs ascending argsort along the last dimension for float32 with uint32 indices (strided memory)
func AsortAscStridedU32F32(ndims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU32F64 performs ascending argsort along the last dimension for float64 with uint32 indices (strided memory)
func AsortAscStridedU32F64(ndims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU8F32 performs ascending argsort along the last dimension for float32 with uint8 indices (strided memory)
func AsortAscStridedU8F32(ndims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU8F64 performs ascending argsort along the last dimension for float64 with uint8 indices (strided memory)
func AsortAscStridedU8F64(ndims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscI64U8 performs ascending argsort along the last dimension for uint8 with int64 indices (contiguous memory)
func AsortAscI64U8(ncols int, src []uint8, dst []int64) {
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

// AsortAscI64U32 performs ascending argsort along the last dimension for uint32 with int64 indices (contiguous memory)
func AsortAscI64U32(ncols int, src []uint32, dst []int64) {
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

// AsortAscI64I64 performs ascending argsort along the last dimension for int64 with int64 indices (contiguous memory)
func AsortAscI64I64(ncols int, src []int64, dst []int64) {
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

// AsortAscU32U8 performs ascending argsort along the last dimension for uint8 with uint32 indices (contiguous memory)
func AsortAscU32U8(ncols int, src []uint8, dst []uint32) {
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

// AsortAscU32U32 performs ascending argsort along the last dimension for uint32 with uint32 indices (contiguous memory)
func AsortAscU32U32(ncols int, src []uint32, dst []uint32) {
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

// AsortAscU32I64 performs ascending argsort along the last dimension for int64 with uint32 indices (contiguous memory)
func AsortAscU32I64(ncols int, src []int64, dst []uint32) {
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

// AsortAscU8U8 performs ascending argsort along the last dimension for uint8 with uint8 indices (contiguous memory)
func AsortAscU8U8(ncols int, src []uint8, dst []uint8) {
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

// AsortAscU8U32 performs ascending argsort along the last dimension for uint32 with uint8 indices (contiguous memory)
func AsortAscU8U32(ncols int, src []uint32, dst []uint8) {
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

// AsortAscU8I64 performs ascending argsort along the last dimension for int64 with uint8 indices (contiguous memory)
func AsortAscU8I64(ncols int, src []int64, dst []uint8) {
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

// AsortAscStridedI64U8 performs ascending argsort along the last dimension for uint8 with int64 indices (strided memory)
func AsortAscStridedI64U8(ndims int, dims, strides, stridesDst []int, ncols int, src []uint8, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortAscI64U8(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedI64U32 performs ascending argsort along the last dimension for uint32 with int64 indices (strided memory)
func AsortAscStridedI64U32(ndims int, dims, strides, stridesDst []int, ncols int, src []uint32, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortAscI64U32(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedI64I64 performs ascending argsort along the last dimension for int64 with int64 indices (strided memory)
func AsortAscStridedI64I64(ndims int, dims, strides, stridesDst []int, ncols int, src []int64, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortAscI64I64(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU32U8 performs ascending argsort along the last dimension for uint8 with uint32 indices (strided memory)
func AsortAscStridedU32U8(ndims int, dims, strides, stridesDst []int, ncols int, src []uint8, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortAscU32U8(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU32U32 performs ascending argsort along the last dimension for uint32 with uint32 indices (strided memory)
func AsortAscStridedU32U32(ndims int, dims, strides, stridesDst []int, ncols int, src []uint32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortAscU32U32(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU32I64 performs ascending argsort along the last dimension for int64 with uint32 indices (strided memory)
func AsortAscStridedU32I64(ndims int, dims, strides, stridesDst []int, ncols int, src []int64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortAscU32I64(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU8U8 performs ascending argsort along the last dimension for uint8 with uint8 indices (strided memory)
func AsortAscStridedU8U8(ndims int, dims, strides, stridesDst []int, ncols int, src []uint8, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortAscU8U8(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU8U32 performs ascending argsort along the last dimension for uint32 with uint8 indices (strided memory)
func AsortAscStridedU8U32(ndims int, dims, strides, stridesDst []int, ncols int, src []uint32, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortAscU8U32(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortAscStridedU8I64 performs ascending argsort along the last dimension for int64 with uint8 indices (strided memory)
func AsortAscStridedU8I64(ndims int, dims, strides, stridesDst []int, ncols int, src []int64, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortAscU8I64(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] <
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
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

// AsortDescStridedI64F32 performs descending argsort along the last dimension for float32 with int64 indices (strided memory)
func AsortDescStridedI64F32(ndims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedI64F64 performs descending argsort along the last dimension for float64 with int64 indices (strided memory)
func AsortDescStridedI64F64(ndims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU32F32 performs descending argsort along the last dimension for float32 with uint32 indices (strided memory)
func AsortDescStridedU32F32(ndims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU32F64 performs descending argsort along the last dimension for float64 with uint32 indices (strided memory)
func AsortDescStridedU32F64(ndims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU8F32 performs descending argsort along the last dimension for float32 with uint8 indices (strided memory)
func AsortDescStridedU8F32(ndims int, dims, strides, stridesDst []int, ncols int, src []float32, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU8F64 performs descending argsort along the last dimension for float64 with uint8 indices (strided memory)
func AsortDescStridedU8F64(ndims int, dims, strides, stridesDst []int, ncols int, src []float64, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescI64U8 performs descending argsort along the last dimension for uint8 with int64 indices (contiguous memory)
func AsortDescI64U8(ncols int, src []uint8, dst []int64) {
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

// AsortDescI64U32 performs descending argsort along the last dimension for uint32 with int64 indices (contiguous memory)
func AsortDescI64U32(ncols int, src []uint32, dst []int64) {
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

// AsortDescI64I64 performs descending argsort along the last dimension for int64 with int64 indices (contiguous memory)
func AsortDescI64I64(ncols int, src []int64, dst []int64) {
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

// AsortDescU32U8 performs descending argsort along the last dimension for uint8 with uint32 indices (contiguous memory)
func AsortDescU32U8(ncols int, src []uint8, dst []uint32) {
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

// AsortDescU32U32 performs descending argsort along the last dimension for uint32 with uint32 indices (contiguous memory)
func AsortDescU32U32(ncols int, src []uint32, dst []uint32) {
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

// AsortDescU32I64 performs descending argsort along the last dimension for int64 with uint32 indices (contiguous memory)
func AsortDescU32I64(ncols int, src []int64, dst []uint32) {
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

// AsortDescU8U8 performs descending argsort along the last dimension for uint8 with uint8 indices (contiguous memory)
func AsortDescU8U8(ncols int, src []uint8, dst []uint8) {
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

// AsortDescU8U32 performs descending argsort along the last dimension for uint32 with uint8 indices (contiguous memory)
func AsortDescU8U32(ncols int, src []uint32, dst []uint8) {
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

// AsortDescU8I64 performs descending argsort along the last dimension for int64 with uint8 indices (contiguous memory)
func AsortDescU8I64(ncols int, src []int64, dst []uint8) {
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
func AsortDescStrided[U I, T D](ndims int, dims, strides, stridesDst []int, ncols int, src []T, dst []U) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedI64U8 performs descending argsort along the last dimension for uint8 with int64 indices (strided memory)
func AsortDescStridedI64U8(ndims int, dims, strides, stridesDst []int, ncols int, src []uint8, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortDescI64U8(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedI64U32 performs descending argsort along the last dimension for uint32 with int64 indices (strided memory)
func AsortDescStridedI64U32(ndims int, dims, strides, stridesDst []int, ncols int, src []uint32, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortDescI64U32(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedI64I64 performs descending argsort along the last dimension for int64 with int64 indices (strided memory)
func AsortDescStridedI64I64(ndims int, dims, strides, stridesDst []int, ncols int, src []int64, dst []int64) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortDescI64I64(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU32U8 performs descending argsort along the last dimension for uint8 with uint32 indices (strided memory)
func AsortDescStridedU32U8(ndims int, dims, strides, stridesDst []int, ncols int, src []uint8, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortDescU32U8(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU32U32 performs descending argsort along the last dimension for uint32 with uint32 indices (strided memory)
func AsortDescStridedU32U32(ndims int, dims, strides, stridesDst []int, ncols int, src []uint32, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortDescU32U32(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU32I64 performs descending argsort along the last dimension for int64 with uint32 indices (strided memory)
func AsortDescStridedU32I64(ndims int, dims, strides, stridesDst []int, ncols int, src []int64, dst []uint32) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortDescU32I64(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU8U8 performs descending argsort along the last dimension for uint8 with uint8 indices (strided memory)
func AsortDescStridedU8U8(ndims int, dims, strides, stridesDst []int, ncols int, src []uint8, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortDescU8U8(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU8U32 performs descending argsort along the last dimension for uint32 with uint8 indices (strided memory)
func AsortDescStridedU8U32(ndims int, dims, strides, stridesDst []int, ncols int, src []uint32, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortDescU8U32(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

// AsortDescStridedU8I64 performs descending argsort along the last dimension for int64 with uint8 indices (strided memory)
func AsortDescStridedU8I64(ndims int, dims, strides, stridesDst []int, ncols int, src []int64, dst []uint8) {
	if IsContiguous(ndims, dims, strides) && IsContiguous(ndims, dims, stridesDst) {
		AsortDescU8I64(ncols, src, dst)
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
			return src[GetStridedIndex(start+int(rowIndices[i]), ndims, dims, strides)] >
				src[GetStridedIndex(start+int(rowIndices[j]), ndims, dims, strides)]
		})
		for col := range ncols {
			dst[GetStridedIndex(start+col, ndims, dims, stridesDst)] = rowIndices[col]
		}
	}
}

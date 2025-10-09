package kernels

import (
	"math"
)

// IndexSelectI64F32 performs indexselect along a specified dimension for int64 indices and float32 data
func IndexSelectI64F32(numel int, ids []int64, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := int64(math.MaxInt64)
	var zero float32
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedI64F32 performs indexselect along a specified dimension for int64 indices and float32 data with strided memory
func IndexSelectStridedI64F32(numel, ndims int, dims, strides []int, ids []int64, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectI64F32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := int64(math.MaxInt64)
	var zero float32
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectI64F64 performs indexselect along a specified dimension for int64 indices and float64 data
func IndexSelectI64F64(numel int, ids []int64, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := int64(math.MaxInt64)
	var zero float64
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedI64F64 performs indexselect along a specified dimension for int64 indices and float64 data with strided memory
func IndexSelectStridedI64F64(numel, ndims int, dims, strides []int, ids []int64, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectI64F64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := int64(math.MaxInt64)
	var zero float64
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectI64U8 performs indexselect along a specified dimension for int64 indices and uint8 data
func IndexSelectI64U8(numel int, ids []int64, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := int64(math.MaxInt64)
	var zero uint8
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedI64U8 performs indexselect along a specified dimension for int64 indices and uint8 data with strided memory
func IndexSelectStridedI64U8(numel, ndims int, dims, strides []int, ids []int64, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectI64U8(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := int64(math.MaxInt64)
	var zero uint8
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectI64U32 performs indexselect along a specified dimension for int64 indices and uint32 data
func IndexSelectI64U32(numel int, ids []int64, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := int64(math.MaxInt64)
	var zero uint32
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedI64U32 performs indexselect along a specified dimension for int64 indices and uint32 data with strided memory
func IndexSelectStridedI64U32(numel, ndims int, dims, strides []int, ids []int64, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectI64U32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := int64(math.MaxInt64)
	var zero uint32
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectI64I64 performs indexselect along a specified dimension for int64 indices and int64 data
func IndexSelectI64I64(numel int, ids []int64, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := int64(math.MaxInt64)
	var zero int64
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedI64I64 performs indexselect along a specified dimension for int64 indices and int64 data with strided memory
func IndexSelectStridedI64I64(numel, ndims int, dims, strides []int, ids []int64, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectI64I64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := int64(math.MaxInt64)
	var zero int64
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectU32F32 performs indexselect along a specified dimension for uint32 indices and float32 data
func IndexSelectU32F32(numel int, ids []uint32, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint32(math.MaxUint32)
	var zero float32
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedU32F32 performs indexselect along a specified dimension for uint32 indices and float32 data with strided memory
func IndexSelectStridedU32F32(numel, ndims int, dims, strides []int, ids []uint32, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectU32F32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := uint32(math.MaxUint32)
	var zero float32
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectU32F64 performs indexselect along a specified dimension for uint32 indices and float64 data
func IndexSelectU32F64(numel int, ids []uint32, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint32(math.MaxUint32)
	var zero float64
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedU32F64 performs indexselect along a specified dimension for uint32 indices and float64 data with strided memory
func IndexSelectStridedU32F64(numel, ndims int, dims, strides []int, ids []uint32, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectU32F64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := uint32(math.MaxUint32)
	var zero float64
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectU32U8 performs indexselect along a specified dimension for uint32 indices and uint8 data
func IndexSelectU32U8(numel int, ids []uint32, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint32(math.MaxUint32)
	var zero uint8
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedU32U8 performs indexselect along a specified dimension for uint32 indices and uint8 data with strided memory
func IndexSelectStridedU32U8(numel, ndims int, dims, strides []int, ids []uint32, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectU32U8(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := uint32(math.MaxUint32)
	var zero uint8
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectU32U32 performs indexselect along a specified dimension for uint32 indices and uint32 data
func IndexSelectU32U32(numel int, ids []uint32, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint32(math.MaxUint32)
	var zero uint32
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedU32U32 performs indexselect along a specified dimension for uint32 indices and uint32 data with strided memory
func IndexSelectStridedU32U32(numel, ndims int, dims, strides []int, ids []uint32, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectU32U32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := uint32(math.MaxUint32)
	var zero uint32
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectU32I64 performs indexselect along a specified dimension for uint32 indices and int64 data
func IndexSelectU32I64(numel int, ids []uint32, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint32(math.MaxUint32)
	var zero int64
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedU32I64 performs indexselect along a specified dimension for uint32 indices and int64 data with strided memory
func IndexSelectStridedU32I64(numel, ndims int, dims, strides []int, ids []uint32, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectU32I64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := uint32(math.MaxUint32)
	var zero int64
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectU8F32 performs indexselect along a specified dimension for uint8 indices and float32 data
func IndexSelectU8F32(numel int, ids []uint8, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint8(math.MaxUint8)
	var zero float32
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedU8F32 performs indexselect along a specified dimension for uint8 indices and float32 data with strided memory
func IndexSelectStridedU8F32(numel, ndims int, dims, strides []int, ids []uint8, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectU8F32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := uint8(math.MaxUint8)
	var zero float32
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectU8F64 performs indexselect along a specified dimension for uint8 indices and float64 data
func IndexSelectU8F64(numel int, ids []uint8, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint8(math.MaxUint8)
	var zero float64
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedU8F64 performs indexselect along a specified dimension for uint8 indices and float64 data with strided memory
func IndexSelectStridedU8F64(numel, ndims int, dims, strides []int, ids []uint8, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectU8F64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := uint8(math.MaxUint8)
	var zero float64
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectU8U8 performs indexselect along a specified dimension for uint8 indices and uint8 data
func IndexSelectU8U8(numel int, ids []uint8, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint8(math.MaxUint8)
	var zero uint8
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedU8U8 performs indexselect along a specified dimension for uint8 indices and uint8 data with strided memory
func IndexSelectStridedU8U8(numel, ndims int, dims, strides []int, ids []uint8, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectU8U8(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := uint8(math.MaxUint8)
	var zero uint8
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectU8U32 performs indexselect along a specified dimension for uint8 indices and uint32 data
func IndexSelectU8U32(numel int, ids []uint8, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint8(math.MaxUint8)
	var zero uint32
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedU8U32 performs indexselect along a specified dimension for uint8 indices and uint32 data with strided memory
func IndexSelectStridedU8U32(numel, ndims int, dims, strides []int, ids []uint8, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectU8U32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := uint8(math.MaxUint8)
	var zero uint32
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// IndexSelectU8I64 performs indexselect along a specified dimension for uint8 indices and int64 data
func IndexSelectU8I64(numel int, ids []uint8, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint8(math.MaxUint8)
	var zero int64
	for dstI := range numel {

		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize

		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		out[dstI] = inp[srcI]
	}
}

// IndexSelectStridedU8I64 performs indexselect along a specified dimension for uint8 indices and int64 data with strided memory
func IndexSelectStridedU8I64(numel, ndims int, dims, strides []int, ids []uint8, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		IndexSelectU8I64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	maxU := uint8(math.MaxUint8)
	var zero int64
	for dstI := range numel {
		leftI := dstI / (idsDimSize * rightSize)
		idI := (dstI / rightSize) % idsDimSize
		rightI := dstI % rightSize
		idx := ids[idI]
		if idx == maxU {
			out[dstI] = zero
			continue
		}
		srcI := leftI*srcDimSize*rightSize + int(idx)*rightSize + rightI
		stridedI := GetStridedIndex(srcI, ndims, dims, strides)
		out[dstI] = inp[stridedI]
	}
}

// GatherI64F32 performs gather along a specified dimension for int64 indices and float32 data
func GatherI64F32(numel int, ids []int64, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := int64(math.MaxInt64)
	var zero float32
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedI64F32 performs gather along a specified dimension for int64 indices and float32 data with strided memory
func GatherStridedI64F32(numel, ndims int, dims, strides []int, ids []int64, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherI64F32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherI64F64 performs gather along a specified dimension for int64 indices and float64 data
func GatherI64F64(numel int, ids []int64, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := int64(math.MaxInt64)
	var zero float64
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedI64F64 performs gather along a specified dimension for int64 indices and float64 data with strided memory
func GatherStridedI64F64(numel, ndims int, dims, strides []int, ids []int64, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherI64F64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherI64U8 performs gather along a specified dimension for int64 indices and uint8 data
func GatherI64U8(numel int, ids []int64, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := int64(math.MaxInt64)
	var zero uint8
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedI64U8 performs gather along a specified dimension for int64 indices and uint8 data with strided memory
func GatherStridedI64U8(numel, ndims int, dims, strides []int, ids []int64, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherI64U8(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherI64U32 performs gather along a specified dimension for int64 indices and uint32 data
func GatherI64U32(numel int, ids []int64, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := int64(math.MaxInt64)
	var zero uint32
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedI64U32 performs gather along a specified dimension for int64 indices and uint32 data with strided memory
func GatherStridedI64U32(numel, ndims int, dims, strides []int, ids []int64, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherI64U32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherI64I64 performs gather along a specified dimension for int64 indices and int64 data
func GatherI64I64(numel int, ids []int64, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := int64(math.MaxInt64)
	var zero int64
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedI64I64 performs gather along a specified dimension for int64 indices and int64 data with strided memory
func GatherStridedI64I64(numel, ndims int, dims, strides []int, ids []int64, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherI64I64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherU32F32 performs gather along a specified dimension for uint32 indices and float32 data
func GatherU32F32(numel int, ids []uint32, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint32(math.MaxUint32)
	var zero float32
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedU32F32 performs gather along a specified dimension for uint32 indices and float32 data with strided memory
func GatherStridedU32F32(numel, ndims int, dims, strides []int, ids []uint32, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherU32F32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherU32F64 performs gather along a specified dimension for uint32 indices and float64 data
func GatherU32F64(numel int, ids []uint32, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint32(math.MaxUint32)
	var zero float64
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedU32F64 performs gather along a specified dimension for uint32 indices and float64 data with strided memory
func GatherStridedU32F64(numel, ndims int, dims, strides []int, ids []uint32, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherU32F64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherU32U8 performs gather along a specified dimension for uint32 indices and uint8 data
func GatherU32U8(numel int, ids []uint32, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint32(math.MaxUint32)
	var zero uint8
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedU32U8 performs gather along a specified dimension for uint32 indices and uint8 data with strided memory
func GatherStridedU32U8(numel, ndims int, dims, strides []int, ids []uint32, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherU32U8(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherU32U32 performs gather along a specified dimension for uint32 indices and uint32 data
func GatherU32U32(numel int, ids []uint32, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint32(math.MaxUint32)
	var zero uint32
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedU32U32 performs gather along a specified dimension for uint32 indices and uint32 data with strided memory
func GatherStridedU32U32(numel, ndims int, dims, strides []int, ids []uint32, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherU32U32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherU32I64 performs gather along a specified dimension for uint32 indices and int64 data
func GatherU32I64(numel int, ids []uint32, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint32(math.MaxUint32)
	var zero int64
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedU32I64 performs gather along a specified dimension for uint32 indices and int64 data with strided memory
func GatherStridedU32I64(numel, ndims int, dims, strides []int, ids []uint32, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherU32I64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherU8F32 performs gather along a specified dimension for uint8 indices and float32 data
func GatherU8F32(numel int, ids []uint8, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint8(math.MaxUint8)
	var zero float32
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedU8F32 performs gather along a specified dimension for uint8 indices and float32 data with strided memory
func GatherStridedU8F32(numel, ndims int, dims, strides []int, ids []uint8, inp, out []float32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherU8F32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherU8F64 performs gather along a specified dimension for uint8 indices and float64 data
func GatherU8F64(numel int, ids []uint8, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint8(math.MaxUint8)
	var zero float64
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedU8F64 performs gather along a specified dimension for uint8 indices and float64 data with strided memory
func GatherStridedU8F64(numel, ndims int, dims, strides []int, ids []uint8, inp, out []float64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherU8F64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherU8U8 performs gather along a specified dimension for uint8 indices and uint8 data
func GatherU8U8(numel int, ids []uint8, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint8(math.MaxUint8)
	var zero uint8
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedU8U8 performs gather along a specified dimension for uint8 indices and uint8 data with strided memory
func GatherStridedU8U8(numel, ndims int, dims, strides []int, ids []uint8, inp, out []uint8, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherU8U8(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherU8U32 performs gather along a specified dimension for uint8 indices and uint32 data
func GatherU8U32(numel int, ids []uint8, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint8(math.MaxUint8)
	var zero uint32
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedU8U32 performs gather along a specified dimension for uint8 indices and uint32 data with strided memory
func GatherStridedU8U32(numel, ndims int, dims, strides []int, ids []uint8, inp, out []uint32, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherU8U32(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// GatherU8I64 performs gather along a specified dimension for uint8 indices and int64 data
func GatherU8I64(numel int, ids []uint8, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	maxU := uint8(math.MaxUint8)
	var zero int64
	for i := range numel {

		post := i % rightSize
		pre := i / (rightSize * idsDimSize)

		idx := ids[i]
		if idx == maxU {
			out[i] = zero
			continue
		}
		srcI := (pre*srcDimSize+int(idx))*rightSize + post
		out[i] = inp[srcI]
	}
}

// GatherStridedU8I64 performs gather along a specified dimension for uint8 indices and int64 data with strided memory
func GatherStridedU8I64(numel, ndims int, dims, strides []int, ids []uint8, inp, out []int64, leftSize, srcDimSize, idsDimSize, rightSize int) {
	if IsContiguous(ndims, dims, strides) {
		GatherU8I64(numel, ids, inp, out, leftSize, srcDimSize, idsDimSize, rightSize)
		return
	}
	panic("strided not supported for Gather")
}

// IndexAddI64F32 performs indexadd along a specified dimension for int64 indices and float32 data
func IndexAddI64F32(leftSize int, idsDimSize int, inp, out []float32, dstDimSize int, rightSize int, ids []int64) {
	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedI64F32 performs indexadd along a specified dimension for int64 indices and float32 data with strided memory
func IndexAddStridedI64F32(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []float32, dstDimSize int, rightSize int, ids []int64) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddI64F32(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddI64F64 performs indexadd along a specified dimension for int64 indices and float64 data
func IndexAddI64F64(leftSize int, idsDimSize int, inp, out []float64, dstDimSize int, rightSize int, ids []int64) {
	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedI64F64 performs indexadd along a specified dimension for int64 indices and float64 data with strided memory
func IndexAddStridedI64F64(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []float64, dstDimSize int, rightSize int, ids []int64) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddI64F64(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddI64U8 performs indexadd along a specified dimension for int64 indices and uint8 data
func IndexAddI64U8(leftSize int, idsDimSize int, inp, out []uint8, dstDimSize int, rightSize int, ids []int64) {
	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedI64U8 performs indexadd along a specified dimension for int64 indices and uint8 data with strided memory
func IndexAddStridedI64U8(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []uint8, dstDimSize int, rightSize int, ids []int64) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddI64U8(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddI64U32 performs indexadd along a specified dimension for int64 indices and uint32 data
func IndexAddI64U32(leftSize int, idsDimSize int, inp, out []uint32, dstDimSize int, rightSize int, ids []int64) {
	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedI64U32 performs indexadd along a specified dimension for int64 indices and uint32 data with strided memory
func IndexAddStridedI64U32(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []uint32, dstDimSize int, rightSize int, ids []int64) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddI64U32(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddI64I64 performs indexadd along a specified dimension for int64 indices and int64 data
func IndexAddI64I64(leftSize int, idsDimSize int, inp, out []int64, dstDimSize int, rightSize int, ids []int64) {
	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedI64I64 performs indexadd along a specified dimension for int64 indices and int64 data with strided memory
func IndexAddStridedI64I64(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []int64, dstDimSize int, rightSize int, ids []int64) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddI64I64(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddU32F32 performs indexadd along a specified dimension for uint32 indices and float32 data
func IndexAddU32F32(leftSize int, idsDimSize int, inp, out []float32, dstDimSize int, rightSize int, ids []uint32) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedU32F32 performs indexadd along a specified dimension for uint32 indices and float32 data with strided memory
func IndexAddStridedU32F32(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []float32, dstDimSize int, rightSize int, ids []uint32) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddU32F32(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddU32F64 performs indexadd along a specified dimension for uint32 indices and float64 data
func IndexAddU32F64(leftSize int, idsDimSize int, inp, out []float64, dstDimSize int, rightSize int, ids []uint32) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedU32F64 performs indexadd along a specified dimension for uint32 indices and float64 data with strided memory
func IndexAddStridedU32F64(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []float64, dstDimSize int, rightSize int, ids []uint32) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddU32F64(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddU32U8 performs indexadd along a specified dimension for uint32 indices and uint8 data
func IndexAddU32U8(leftSize int, idsDimSize int, inp, out []uint8, dstDimSize int, rightSize int, ids []uint32) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedU32U8 performs indexadd along a specified dimension for uint32 indices and uint8 data with strided memory
func IndexAddStridedU32U8(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []uint8, dstDimSize int, rightSize int, ids []uint32) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddU32U8(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddU32U32 performs indexadd along a specified dimension for uint32 indices and uint32 data
func IndexAddU32U32(leftSize int, idsDimSize int, inp, out []uint32, dstDimSize int, rightSize int, ids []uint32) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedU32U32 performs indexadd along a specified dimension for uint32 indices and uint32 data with strided memory
func IndexAddStridedU32U32(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []uint32, dstDimSize int, rightSize int, ids []uint32) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddU32U32(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddU32I64 performs indexadd along a specified dimension for uint32 indices and int64 data
func IndexAddU32I64(leftSize int, idsDimSize int, inp, out []int64, dstDimSize int, rightSize int, ids []uint32) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedU32I64 performs indexadd along a specified dimension for uint32 indices and int64 data with strided memory
func IndexAddStridedU32I64(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []int64, dstDimSize int, rightSize int, ids []uint32) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddU32I64(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddU8F32 performs indexadd along a specified dimension for uint8 indices and float32 data
func IndexAddU8F32(leftSize int, idsDimSize int, inp, out []float32, dstDimSize int, rightSize int, ids []uint8) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedU8F32 performs indexadd along a specified dimension for uint8 indices and float32 data with strided memory
func IndexAddStridedU8F32(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []float32, dstDimSize int, rightSize int, ids []uint8) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddU8F32(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddU8F64 performs indexadd along a specified dimension for uint8 indices and float64 data
func IndexAddU8F64(leftSize int, idsDimSize int, inp, out []float64, dstDimSize int, rightSize int, ids []uint8) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedU8F64 performs indexadd along a specified dimension for uint8 indices and float64 data with strided memory
func IndexAddStridedU8F64(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []float64, dstDimSize int, rightSize int, ids []uint8) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddU8F64(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddU8U8 performs indexadd along a specified dimension for uint8 indices and uint8 data
func IndexAddU8U8(leftSize int, idsDimSize int, inp, out []uint8, dstDimSize int, rightSize int, ids []uint8) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedU8U8 performs indexadd along a specified dimension for uint8 indices and uint8 data with strided memory
func IndexAddStridedU8U8(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []uint8, dstDimSize int, rightSize int, ids []uint8) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddU8U8(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddU8U32 performs indexadd along a specified dimension for uint8 indices and uint32 data
func IndexAddU8U32(leftSize int, idsDimSize int, inp, out []uint32, dstDimSize int, rightSize int, ids []uint8) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedU8U32 performs indexadd along a specified dimension for uint8 indices and uint32 data with strided memory
func IndexAddStridedU8U32(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []uint32, dstDimSize int, rightSize int, ids []uint8) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddU8U32(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// IndexAddU8I64 performs indexadd along a specified dimension for uint8 indices and int64 data
func IndexAddU8I64(leftSize int, idsDimSize int, inp, out []int64, dstDimSize int, rightSize int, ids []uint8) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range idsDimSize {
			idx := ids[j]
			if idx == maxU {
				continue
			}
			srcI := (pre*idsDimSize+j)*rightSize + post
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// IndexAddStridedU8I64 performs indexadd along a specified dimension for uint8 indices and int64 data with strided memory
func IndexAddStridedU8I64(leftSize, ndims int, dims, strides []int, idsDimSize int, inp, out []int64, dstDimSize int, rightSize int, ids []uint8) {
	if IsContiguous(ndims, dims, strides) {
		IndexAddU8I64(leftSize, idsDimSize, inp, out, dstDimSize, rightSize, ids)
		return
	}
	panic("strided not supported for IndexAdd")
}

// ScatterI64F32 performs scatter along a specified dimension for int64 indices and float32 data
func ScatterI64F32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []float32) {

	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedI64F32 performs scatter along a specified dimension for int64 indices and float32 data with strided memory
func ScatterStridedI64F32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterI64F32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterI64F64 performs scatter along a specified dimension for int64 indices and float64 data
func ScatterI64F64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []float64) {

	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedI64F64 performs scatter along a specified dimension for int64 indices and float64 data with strided memory
func ScatterStridedI64F64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterI64F64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterI64U8 performs scatter along a specified dimension for int64 indices and uint8 data
func ScatterI64U8(leftSize, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []uint8) {

	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedI64U8 performs scatter along a specified dimension for int64 indices and uint8 data with strided memory
func ScatterStridedI64U8(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		ScatterI64U8(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterI64U32 performs scatter along a specified dimension for int64 indices and uint32 data
func ScatterI64U32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []uint32) {

	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedI64U32 performs scatter along a specified dimension for int64 indices and uint32 data with strided memory
func ScatterStridedI64U32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterI64U32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterI64I64 performs scatter along a specified dimension for int64 indices and int64 data
func ScatterI64I64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []int64) {

	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedI64I64 performs scatter along a specified dimension for int64 indices and int64 data with strided memory
func ScatterStridedI64I64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterI64I64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterU32F32 performs scatter along a specified dimension for uint32 indices and float32 data
func ScatterU32F32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []float32) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedU32F32 performs scatter along a specified dimension for uint32 indices and float32 data with strided memory
func ScatterStridedU32F32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterU32F32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterU32F64 performs scatter along a specified dimension for uint32 indices and float64 data
func ScatterU32F64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []float64) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedU32F64 performs scatter along a specified dimension for uint32 indices and float64 data with strided memory
func ScatterStridedU32F64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterU32F64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterU32U8 performs scatter along a specified dimension for uint32 indices and uint8 data
func ScatterU32U8(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []uint8) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedU32U8 performs scatter along a specified dimension for uint32 indices and uint8 data with strided memory
func ScatterStridedU32U8(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		ScatterU32U8(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterU32U32 performs scatter along a specified dimension for uint32 indices and uint32 data
func ScatterU32U32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []uint32) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedU32U32 performs scatter along a specified dimension for uint32 indices and uint32 data with strided memory
func ScatterStridedU32U32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterU32U32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterU32I64 performs scatter along a specified dimension for uint32 indices and int64 data
func ScatterU32I64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []int64) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedU32I64 performs scatter along a specified dimension for uint32 indices and int64 data with strided memory
func ScatterStridedU32I64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterU32I64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterU8F32 performs scatter along a specified dimension for uint8 indices and float32 data
func ScatterU8F32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []float32) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedU8F32 performs scatter along a specified dimension for uint8 indices and float32 data with strided memory
func ScatterStridedU8F32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterU8F32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterU8F64 performs scatter along a specified dimension for uint8 indices and float64 data
func ScatterU8F64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []float64) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedU8F64 performs scatter along a specified dimension for uint8 indices and float64 data with strided memory
func ScatterStridedU8F64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterU8F64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterU8U8 performs scatter along a specified dimension for uint8 indices and uint8 data
func ScatterU8U8(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []uint8) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedU8U8 performs scatter along a specified dimension for uint8 indices and uint8 data with strided memory
func ScatterStridedU8U8(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		ScatterU8U8(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterU8U32 performs scatter along a specified dimension for uint8 indices and uint32 data
func ScatterU8U32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []uint32) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedU8U32 performs scatter along a specified dimension for uint8 indices and uint32 data with strided memory
func ScatterStridedU8U32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterU8U32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterU8I64 performs scatter along a specified dimension for uint8 indices and int64 data
func ScatterU8I64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []int64) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] = inp[srcI]
		}
	}
}

// ScatterStridedU8I64 performs scatter along a specified dimension for uint8 indices and int64 data with strided memory
func ScatterStridedU8I64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterU8I64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for Scatter")
}

// ScatterAddI64F32 performs scatteradd along a specified dimension for int64 indices and float32 data
func ScatterAddI64F32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []float32) {

	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedI64F32 performs scatteradd along a specified dimension for int64 indices and float32 data with strided memory
func ScatterAddStridedI64F32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddI64F32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddI64F64 performs scatteradd along a specified dimension for int64 indices and float64 data
func ScatterAddI64F64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []float64) {

	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedI64F64 performs scatteradd along a specified dimension for int64 indices and float64 data with strided memory
func ScatterAddStridedI64F64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddI64F64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddI64U8 performs scatteradd along a specified dimension for int64 indices and uint8 data
func ScatterAddI64U8(leftSize, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []uint8) {

	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedI64U8 performs scatteradd along a specified dimension for int64 indices and uint8 data with strided memory
func ScatterAddStridedI64U8(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddI64U8(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddI64U32 performs scatteradd along a specified dimension for int64 indices and uint32 data
func ScatterAddI64U32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []uint32) {

	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedI64U32 performs scatteradd along a specified dimension for int64 indices and uint32 data with strided memory
func ScatterAddStridedI64U32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddI64U32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddI64I64 performs scatteradd along a specified dimension for int64 indices and int64 data
func ScatterAddI64I64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []int64) {

	maxU := int64(math.MaxInt64)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedI64I64 performs scatteradd along a specified dimension for int64 indices and int64 data with strided memory
func ScatterAddStridedI64I64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []int64, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddI64I64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddU32F32 performs scatteradd along a specified dimension for uint32 indices and float32 data
func ScatterAddU32F32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []float32) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedU32F32 performs scatteradd along a specified dimension for uint32 indices and float32 data with strided memory
func ScatterAddStridedU32F32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddU32F32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddU32F64 performs scatteradd along a specified dimension for uint32 indices and float64 data
func ScatterAddU32F64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []float64) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedU32F64 performs scatteradd along a specified dimension for uint32 indices and float64 data with strided memory
func ScatterAddStridedU32F64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddU32F64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddU32U8 performs scatteradd along a specified dimension for uint32 indices and uint8 data
func ScatterAddU32U8(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []uint8) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedU32U8 performs scatteradd along a specified dimension for uint32 indices and uint8 data with strided memory
func ScatterAddStridedU32U8(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddU32U8(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddU32U32 performs scatteradd along a specified dimension for uint32 indices and uint32 data
func ScatterAddU32U32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []uint32) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedU32U32 performs scatteradd along a specified dimension for uint32 indices and uint32 data with strided memory
func ScatterAddStridedU32U32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddU32U32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddU32I64 performs scatteradd along a specified dimension for uint32 indices and int64 data
func ScatterAddU32I64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []int64) {
	maxU := uint32(math.MaxUint32)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedU32I64 performs scatteradd along a specified dimension for uint32 indices and int64 data with strided memory
func ScatterAddStridedU32I64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint32, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddU32I64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddU8F32 performs scatteradd along a specified dimension for uint8 indices and float32 data
func ScatterAddU8F32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []float32) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedU8F32 performs scatteradd along a specified dimension for uint8 indices and float32 data with strided memory
func ScatterAddStridedU8F32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []float32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddU8F32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddU8F64 performs scatteradd along a specified dimension for uint8 indices and float64 data
func ScatterAddU8F64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []float64) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedU8F64 performs scatteradd along a specified dimension for uint8 indices and float64 data with strided memory
func ScatterAddStridedU8F64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []float64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddU8F64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddU8U8 performs scatteradd along a specified dimension for uint8 indices and uint8 data
func ScatterAddU8U8(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []uint8) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedU8U8 performs scatteradd along a specified dimension for uint8 indices and uint8 data with strided memory
func ScatterAddStridedU8U8(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []uint8) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddU8U8(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddU8U32 performs scatteradd along a specified dimension for uint8 indices and uint32 data
func ScatterAddU8U32(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []uint32) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedU8U32 performs scatteradd along a specified dimension for uint8 indices and uint32 data with strided memory
func ScatterAddStridedU8U32(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []uint32) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddU8U32(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

// ScatterAddU8I64 performs scatteradd along a specified dimension for uint8 indices and int64 data
func ScatterAddU8I64(leftSize, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []int64) {
	maxU := uint8(math.MaxUint8)
	numel := leftSize * rightSize
	for i := range numel {
		pre := i / rightSize
		post := i % rightSize
		for j := range srcDimSize {
			srcI := (pre*srcDimSize+j)*rightSize + post
			idx := ids[srcI]
			if idx == maxU {
				continue
			}
			dstI := (pre*dstDimSize+int(idx))*rightSize + post
			out[dstI] += inp[srcI]
		}
	}
}

// ScatterAddStridedU8I64 performs scatteradd along a specified dimension for uint8 indices and int64 data with strided memory
func ScatterAddStridedU8I64(leftSize, ndims int, dims, strides []int, srcDimSize, dstDimSize, rightSize int, ids []uint8, inp, out []int64) {
	if IsContiguous(ndims, dims, strides) {
		ScatterAddU8I64(leftSize, srcDimSize, dstDimSize, rightSize, ids, inp, out)
		return
	}
	panic("strided not supported for ScatterAdd")
}

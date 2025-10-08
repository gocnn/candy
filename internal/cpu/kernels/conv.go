package kernels

import (
	"math"

	"github.com/gocnn/gomat/blas"
	"github.com/gocnn/gomat/blas/blas32"
	"github.com/gocnn/gomat/blas/blas64"
)

// MatMulF32 performs simple matrix multiplication for float32: C = A*B
func MatMulF32(m, n, k int, a, b, c []float32) {
	blas32.Gemm(blas.NoTrans, blas.NoTrans, m, n, k, 1.0, a, k, b, n, 0.0, c, n)
}

// MatMulF64 performs simple matrix multiplication for float64: C = A*B
func MatMulF64(m, n, k int, a, b, c []float64) {
	blas64.Gemm(blas.NoTrans, blas.NoTrans, m, n, k, 1.0, a, k, b, n, 0.0, c, n)
}

// MatMulBatchedF32 performs batched matrix multiplication for float32: C[i] = A[i] * B[i]
func MatMulBatchedF32(b, m, n, k int, a, bMat, c []float32) {
	for i := range b {
		aOffset := i * m * k
		bOffset := i * k * n
		cOffset := i * m * n
		blas32.Gemm(blas.NoTrans, blas.NoTrans, m, n, k, 1.0,
			a[aOffset:], k, bMat[bOffset:], n, 0.0, c[cOffset:], n)
	}
}

// MatMulBatchedF64 performs batched matrix multiplication for float64: C[i] = A[i] * B[i]
func MatMulBatchedF64(b, m, n, k int, a, bMat, c []float64) {
	for i := range b {
		aOffset := i * m * k
		bOffset := i * k * n
		cOffset := i * m * n
		blas64.Gemm(blas.NoTrans, blas.NoTrans, m, n, k, 1.0,
			a[aOffset:], k, bMat[bOffset:], n, 0.0, c[cOffset:], n)
	}
}

// NaiveMatMul performs matrix multiplication for any numeric type using direct loops
func NaiveMatMul[T D](m, n, k int, a, b, c []T) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum T
			for l := range k {
				sum += a[i*k+l] * b[l*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// NaiveMatMulF32 performs matrix multiplication for float32 using direct loops
func NaiveMatMulF32(m, n, k int, a, b, c []float32) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum float32
			for l := range k {
				sum += a[i*k+l] * b[l*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// NaiveMatMulF64 performs matrix multiplication for float64 using direct loops
func NaiveMatMulF64(m, n, k int, a, b, c []float64) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum float64
			for l := range k {
				sum += a[i*k+l] * b[l*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// NaiveMatMulU8 performs matrix multiplication for uint8 using direct loops
func NaiveMatMulU8(m, n, k int, a, b, c []uint8) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum uint8
			for l := range k {
				sum += a[i*k+l] * b[l*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// NaiveMatMulU32 performs matrix multiplication for uint32 using direct loops
func NaiveMatMulU32(m, n, k int, a, b, c []uint32) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum uint32
			for l := range k {
				sum += a[i*k+l] * b[l*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// NaiveMatMulI64 performs matrix multiplication for int64 using direct loops
func NaiveMatMulI64(m, n, k int, a, b, c []int64) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum int64
			for l := range k {
				sum += a[i*k+l] * b[l*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// NaiveMatMulStrided performs matrix multiplication for any numeric type using direct loops with support for non-contiguous memory
func NaiveMatMulStrided[T D](m, n, k int, a, b, c []T, aStrides, bStrides, cStrides []int) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum T
			for l := range k {
				aIdx := i*aStrides[0] + l*aStrides[1]
				bIdx := l*bStrides[0] + j*bStrides[1]
				sum += a[aIdx] * b[bIdx]
			}
			cIdx := i*cStrides[0] + j*cStrides[1]
			c[cIdx] = sum
		}
	}
}

// NaiveMatMulStridedF32 performs matrix multiplication for float32 using direct loops with support for non-contiguous memory
func NaiveMatMulStridedF32(m, n, k int, a, b, c []float32, aStrides, bStrides, cStrides []int) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum float32
			for l := range k {
				aIdx := i*aStrides[0] + l*aStrides[1]
				bIdx := l*bStrides[0] + j*bStrides[1]
				sum += a[aIdx] * b[bIdx]
			}
			cIdx := i*cStrides[0] + j*cStrides[1]
			c[cIdx] = sum
		}
	}
}

// NaiveMatMulStridedF64 performs matrix multiplication for float64 using direct loops with support for non-contiguous memory
func NaiveMatMulStridedF64(m, n, k int, a, b, c []float64, aStrides, bStrides, cStrides []int) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum float64
			for l := range k {
				aIdx := i*aStrides[0] + l*aStrides[1]
				bIdx := l*bStrides[0] + j*bStrides[1]
				sum += a[aIdx] * b[bIdx]
			}
			cIdx := i*cStrides[0] + j*cStrides[1]
			c[cIdx] = sum
		}
	}
}

// NaiveMatMulStridedU8 performs matrix multiplication for uint8 using direct loops with support for non-contiguous memory
func NaiveMatMulStridedU8(m, n, k int, a, b, c []uint8, aStrides, bStrides, cStrides []int) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum uint8
			for l := range k {
				aIdx := i*aStrides[0] + l*aStrides[1]
				bIdx := l*bStrides[0] + j*bStrides[1]
				sum += a[aIdx] * b[bIdx]
			}
			cIdx := i*cStrides[0] + j*cStrides[1]
			c[cIdx] = sum
		}
	}
}

// NaiveMatMulStridedU32 performs matrix multiplication for uint32 using direct loops with support for non-contiguous memory
func NaiveMatMulStridedU32(m, n, k int, a, b, c []uint32, aStrides, bStrides, cStrides []int) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum uint32
			for l := range k {
				aIdx := i*aStrides[0] + l*aStrides[1]
				bIdx := l*bStrides[0] + j*bStrides[1]
				sum += a[aIdx] * b[bIdx]
			}
			cIdx := i*cStrides[0] + j*cStrides[1]
			c[cIdx] = sum
		}
	}
}

// NaiveMatMulStridedI64 performs matrix multiplication for int64 using direct loops with support for non-contiguous memory
func NaiveMatMulStridedI64(m, n, k int, a, b, c []int64, aStrides, bStrides, cStrides []int) {
	// C[i,j] = sum(A[i,l] * B[l,j]) for l in [0,k)
	for i := range m {
		for j := range n {
			var sum int64
			for l := range k {
				aIdx := i*aStrides[0] + l*aStrides[1]
				bIdx := l*bStrides[0] + j*bStrides[1]
				sum += a[aIdx] * b[bIdx]
			}
			cIdx := i*cStrides[0] + j*cStrides[1]
			c[cIdx] = sum
		}
	}
}

// NaiveBatchedMatMul performs batched matrix multiplication for any numeric type using direct loops
// Assumes both A and B are batched with contiguous memory layout: A[bSize*m*k], B[bSize*k*n], C[bSize*m*n]
func NaiveBatchedMatMul[T D](bSize, m, n, k int, a, b, c []T) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		aBase := bb * m * k
		bBase := bb * k * n
		cBase := bb * m * n
		for i := range m {
			for j := range n {
				var sum T
				for l := range k {
					sum += a[aBase+i*k+l] * b[bBase+l*n+j]
				}
				c[cBase+i*n+j] = sum
			}
		}
	}
}

// NaiveBatchedMatMulF32 performs batched matrix multiplication for float32 using direct loops
// Assumes both A and B are batched with contiguous memory layout: A[bSize*m*k], B[bSize*k*n], C[bSize*m*n]
func NaiveBatchedMatMulF32(bSize, m, n, k int, a, b, c []float32) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		aBase := bb * m * k
		bBase := bb * k * n
		cBase := bb * m * n
		for i := range m {
			for j := range n {
				var sum float32
				for l := range k {
					sum += a[aBase+i*k+l] * b[bBase+l*n+j]
				}
				c[cBase+i*n+j] = sum
			}
		}
	}
}

// NaiveBatchedMatMulF64 performs batched matrix multiplication for float64 using direct loops
// Assumes both A and B are batched with contiguous memory layout: A[bSize*m*k], B[bSize*k*n], C[bSize*m*n]
func NaiveBatchedMatMulF64(bSize, m, n, k int, a, b, c []float64) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		aBase := bb * m * k
		bBase := bb * k * n
		cBase := bb * m * n
		for i := range m {
			for j := range n {
				var sum float64
				for l := range k {
					sum += a[aBase+i*k+l] * b[bBase+l*n+j]
				}
				c[cBase+i*n+j] = sum
			}
		}
	}
}

// NaiveBatchedMatMulU8 performs batched matrix multiplication for uint8 using direct loops
// Assumes both A and B are batched with contiguous memory layout: A[bSize*m*k], B[bSize*k*n], C[bSize*m*n]
func NaiveBatchedMatMulU8(bSize, m, n, k int, a, b, c []uint8) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		aBase := bb * m * k
		bBase := bb * k * n
		cBase := bb * m * n
		for i := range m {
			for j := range n {
				var sum uint8
				for l := range k {
					sum += a[aBase+i*k+l] * b[bBase+l*n+j]
				}
				c[cBase+i*n+j] = sum
			}
		}
	}
}

// NaiveBatchedMatMulU32 performs batched matrix multiplication for uint32 using direct loops
// Assumes both A and B are batched with contiguous memory layout: A[bSize*m*k], B[bSize*k*n], C[bSize*m*n]
func NaiveBatchedMatMulU32(bSize, m, n, k int, a, b, c []uint32) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		aBase := bb * m * k
		bBase := bb * k * n
		cBase := bb * m * n
		for i := range m {
			for j := range n {
				var sum uint32
				for l := range k {
					sum += a[aBase+i*k+l] * b[bBase+l*n+j]
				}
				c[cBase+i*n+j] = sum
			}
		}
	}
}

// NaiveBatchedMatMulI64 performs batched matrix multiplication for int64 using direct loops
// Assumes both A and B are batched with contiguous memory layout: A[bSize*m*k], B[bSize*k*n], C[bSize*m*n]
func NaiveBatchedMatMulI64(bSize, m, n, k int, a, b, c []int64) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		aBase := bb * m * k
		bBase := bb * k * n
		cBase := bb * m * n
		for i := range m {
			for j := range n {
				var sum int64
				for l := range k {
					sum += a[aBase+i*k+l] * b[bBase+l*n+j]
				}
				c[cBase+i*n+j] = sum
			}
		}
	}
}

// NaiveBatchedMatMulStrided performs batched matrix multiplication for any numeric type using direct loops with support for non-contiguous memory
// Strides are [batch_stride, row_stride, col_stride] for each tensor.
// Broadcasting over batch is supported if the batch_stride (strides[0]) is 0 for A or B.
func NaiveBatchedMatMulStrided[T D](bSize, m, n, k int, a, b, c []T, aStrides, bStrides, cStrides []int) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		for i := range m {
			for j := range n {
				var sum T
				for l := range k {
					aIdx := bb*aStrides[0] + i*aStrides[1] + l*aStrides[2]
					bIdx := bb*bStrides[0] + l*bStrides[1] + j*bStrides[2]
					sum += a[aIdx] * b[bIdx]
				}
				cIdx := bb*cStrides[0] + i*cStrides[1] + j*cStrides[2]
				c[cIdx] = sum
			}
		}
	}
}

// NaiveBatchedMatMulStridedF32 performs batched matrix multiplication for float32 using direct loops with support for non-contiguous memory
// Strides are [batch_stride, row_stride, col_stride] for each tensor.
// Broadcasting over batch is supported if the batch_stride (strides[0]) is 0 for A or B.
func NaiveBatchedMatMulStridedF32(bSize, m, n, k int, a, b, c []float32, aStrides, bStrides, cStrides []int) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		for i := range m {
			for j := range n {
				var sum float32
				for l := range k {
					aIdx := bb*aStrides[0] + i*aStrides[1] + l*aStrides[2]
					bIdx := bb*bStrides[0] + l*bStrides[1] + j*bStrides[2]
					sum += a[aIdx] * b[bIdx]
				}
				cIdx := bb*cStrides[0] + i*cStrides[1] + j*cStrides[2]
				c[cIdx] = sum
			}
		}
	}
}

// NaiveBatchedMatMulStridedF64 performs batched matrix multiplication for float64 using direct loops with support for non-contiguous memory
// Strides are [batch_stride, row_stride, col_stride] for each tensor.
// Broadcasting over batch is supported if the batch_stride (strides[0]) is 0 for A or B.
func NaiveBatchedMatMulStridedF64(bSize, m, n, k int, a, b, c []float64, aStrides, bStrides, cStrides []int) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		for i := range m {
			for j := range n {
				var sum float64
				for l := range k {
					aIdx := bb*aStrides[0] + i*aStrides[1] + l*aStrides[2]
					bIdx := bb*bStrides[0] + l*bStrides[1] + j*bStrides[2]
					sum += a[aIdx] * b[bIdx]
				}
				cIdx := bb*cStrides[0] + i*cStrides[1] + j*cStrides[2]
				c[cIdx] = sum
			}
		}
	}
}

// NaiveBatchedMatMulStridedU8 performs batched matrix multiplication for uint8 using direct loops with support for non-contiguous memory
// Strides are [batch_stride, row_stride, col_stride] for each tensor.
// Broadcasting over batch is supported if the batch_stride (strides[0]) is 0 for A or B.
func NaiveBatchedMatMulStridedU8(bSize, m, n, k int, a, b, c []uint8, aStrides, bStrides, cStrides []int) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		for i := range m {
			for j := range n {
				var sum uint8
				for l := range k {
					aIdx := bb*aStrides[0] + i*aStrides[1] + l*aStrides[2]
					bIdx := bb*bStrides[0] + l*bStrides[1] + j*bStrides[2]
					sum += a[aIdx] * b[bIdx]
				}
				cIdx := bb*cStrides[0] + i*cStrides[1] + j*cStrides[2]
				c[cIdx] = sum
			}
		}
	}
}

// NaiveBatchedMatMulStridedU32 performs batched matrix multiplication for uint32 using direct loops with support for non-contiguous memory
// Strides are [batch_stride, row_stride, col_stride] for each tensor.
// Broadcasting over batch is supported if the batch_stride (strides[0]) is 0 for A or B.
func NaiveBatchedMatMulStridedU32(bSize, m, n, k int, a, b, c []uint32, aStrides, bStrides, cStrides []int) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		for i := range m {
			for j := range n {
				var sum uint32
				for l := range k {
					aIdx := bb*aStrides[0] + i*aStrides[1] + l*aStrides[2]
					bIdx := bb*bStrides[0] + l*bStrides[1] + j*bStrides[2]
					sum += a[aIdx] * b[bIdx]
				}
				cIdx := bb*cStrides[0] + i*cStrides[1] + j*cStrides[2]
				c[cIdx] = sum
			}
		}
	}
}

// NaiveBatchedMatMulStridedI64 performs batched matrix multiplication for int64 using direct loops with support for non-contiguous memory
// Strides are [batch_stride, row_stride, col_stride] for each tensor.
// Broadcasting over batch is supported if the batch_stride (strides[0]) is 0 for A or B.
func NaiveBatchedMatMulStridedI64(bSize, m, n, k int, a, b, c []int64, aStrides, bStrides, cStrides []int) {
	// C[bb,i,j] = sum(A[bb,i,l] * B[bb,l,j]) for l in [0,k)
	for bb := range bSize {
		for i := range m {
			for j := range n {
				var sum int64
				for l := range k {
					aIdx := bb*aStrides[0] + i*aStrides[1] + l*aStrides[2]
					bIdx := bb*bStrides[0] + l*bStrides[1] + j*bStrides[2]
					sum += a[aIdx] * b[bIdx]
				}
				cIdx := bb*cStrides[0] + i*cStrides[1] + j*cStrides[2]
				c[cIdx] = sum
			}
		}
	}
}

// Im2colConv1dF32 performs 1D convolution for float32 using im2col + gemm with direct BLAS Gemm call
func Im2colConv1dF32(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []float32) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	colSize := bSize * lOut * cIn * kSize
	col := make([]float32, colSize)
	Im2col1dF32(bSize, cIn, lIn, lOut, kSize, stride, padding, dilation, src, col)
	m, n, k := bSize*lOut, cOut, cIn*kSize
	blas32.Gemm(blas.NoTrans, blas.Trans, m, n, k, 1.0, col, k, kernel, k, 0.0, dst, n)
}

// Im2colConv1dF64 performs 1D convolution for float64 using im2col + gemm with direct BLAS Gemm call
func Im2colConv1dF64(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []float64) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	colSize := bSize * lOut * cIn * kSize
	col := make([]float64, colSize)
	Im2col1dF64(bSize, cIn, lIn, lOut, kSize, stride, padding, dilation, src, col)
	m, n, k := bSize*lOut, cOut, cIn*kSize
	blas64.Gemm(blas.NoTrans, blas.Trans, m, n, k, 1.0, col, k, kernel, k, 0.0, dst, n)
}

// Im2colConv2dF32 performs 2D convolution for float32 using im2col + gemm with direct BLAS Gemm call
func Im2colConv2dF32(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []float32) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	colSize := bSize * hOut * wOut * cIn * hK * wK
	col := make([]float32, colSize)
	Im2colF32(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK, stride, padding, dilation, src, col)
	m, n, k := bSize*hOut*wOut, cOut, cIn*hK*wK
	blas32.Gemm(blas.NoTrans, blas.Trans, m, n, k, 1.0, col, k, kernel, k, 0.0, dst, n)
}

// Im2colConv2dF64 performs 2D convolution for float64 using im2col + gemm with direct BLAS Gemm call
func Im2colConv2dF64(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []float64) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	colSize := bSize * hOut * wOut * cIn * hK * wK
	col := make([]float64, colSize)
	Im2colF64(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK, stride, padding, dilation, src, col)
	m, n, k := bSize*hOut*wOut, cOut, cIn*hK*wK
	blas64.Gemm(blas.NoTrans, blas.Trans, m, n, k, 1.0, col, k, kernel, k, 0.0, dst, n)
}

// NaiveConv1d performs 1D convolution for any supported numeric type using direct loop
func NaiveConv1d[T D](bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []T) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum T
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							sum += src[b*cIn*lIn+ci*lIn+li] * kernel[co*cIn*kSize+ci*kSize+k]
						}
					}
				}
				dst[b*cOut*lOut+co*lOut+lo] = sum
			}
		}
	}
}

// NaiveConv1dF32 performs 1D convolution for float32 using direct loop
func NaiveConv1dF32(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []float32) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				sum := float32(0)
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							sum += src[b*cIn*lIn+ci*lIn+li] * kernel[co*cIn*kSize+ci*kSize+k]
						}
					}
				}
				dst[b*cOut*lOut+co*lOut+lo] = sum
			}
		}
	}
}

// NaiveConv1dF64 performs 1D convolution for float64 using direct loop
func NaiveConv1dF64(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []float64) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				sum := float64(0)
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							sum += src[b*cIn*lIn+ci*lIn+li] * kernel[co*cIn*kSize+ci*kSize+k]
						}
					}
				}
				dst[b*cOut*lOut+co*lOut+lo] = sum
			}
		}
	}
}

// NaiveConv1dU8 performs 1D convolution for uint8 using direct loop
func NaiveConv1dU8(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []uint8) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							sum += int64(src[b*cIn*lIn+ci*lIn+li]) * int64(kernel[co*cIn*kSize+ci*kSize+k])
						}
					}
				}
				if sum > math.MaxUint8 {
					sum = math.MaxUint8
				} else if sum < 0 {
					sum = 0
				}
				dst[b*cOut*lOut+co*lOut+lo] = uint8(sum)
			}
		}
	}
}

// NaiveConv1dU32 performs 1D convolution for uint32 using direct loop
func NaiveConv1dU32(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []uint32) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							sum += int64(src[b*cIn*lIn+ci*lIn+li]) * int64(kernel[co*cIn*kSize+ci*kSize+k])
						}
					}
				}
				if sum > math.MaxUint32 {
					sum = math.MaxUint32
				} else if sum < 0 {
					sum = 0
				}
				dst[b*cOut*lOut+co*lOut+lo] = uint32(sum)
			}
		}
	}
}

// NaiveConv1dI64 performs 1D convolution for int64 using direct loop
func NaiveConv1dI64(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []int64) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							sum += src[b*cIn*lIn+ci*lIn+li] * kernel[co*cIn*kSize+ci*kSize+k]
						}
					}
				}
				dst[b*cOut*lOut+co*lOut+lo] = sum
			}
		}
	}
}

// NaiveConv1dStrided performs 1D convolution for any supported numeric type using direct loop with support for non-contiguous memory
func NaiveConv1dStrided[T D](bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []T, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum T
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
							kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + k*kernelStrides[2]
							sum += src[srcIdx] * kernel[kernelIdx]
						}
					}
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = sum
			}
		}
	}
}

// NaiveConv1dStridedF32 performs 1D convolution for float32 using direct loop with support for non-contiguous memory
func NaiveConv1dStridedF32(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []float32, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				sum := float32(0)
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
							kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + k*kernelStrides[2]
							sum += src[srcIdx] * kernel[kernelIdx]
						}
					}
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = sum
			}
		}
	}
}

// NaiveConv1dStridedF64 performs 1D convolution for float64 using direct loop with support for non-contiguous memory
func NaiveConv1dStridedF64(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []float64, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				sum := float64(0)
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
							kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + k*kernelStrides[2]
							sum += src[srcIdx] * kernel[kernelIdx]
						}
					}
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = sum
			}
		}
	}
}

// NaiveConv1dStridedU8 performs 1D convolution for uint8 using direct loop with support for non-contiguous memory
func NaiveConv1dStridedU8(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []uint8, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
							kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + k*kernelStrides[2]
							sum += int64(src[srcIdx]) * int64(kernel[kernelIdx])
						}
					}
				}
				if sum > math.MaxUint8 {
					sum = math.MaxUint8
				} else if sum < 0 {
					sum = 0
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = uint8(sum)
			}
		}
	}
}

// NaiveConv1dStridedU32 performs 1D convolution for uint32 using direct loop with support for non-contiguous memory
func NaiveConv1dStridedU32(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []uint32, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
							kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + k*kernelStrides[2]
							sum += int64(src[srcIdx]) * int64(kernel[kernelIdx])
						}
					}
				}
				if sum > math.MaxUint32 {
					sum = math.MaxUint32
				} else if sum < 0 {
					sum = 0
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = uint32(sum)
			}
		}
	}
}

// NaiveConv1dStridedI64 performs 1D convolution for int64 using direct loop with support for non-contiguous memory
func NaiveConv1dStridedI64(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []int64, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						li := lo*stride + k*dilation - padding
						if li >= 0 && li < lIn {
							srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
							kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + k*kernelStrides[2]
							sum += src[srcIdx] * kernel[kernelIdx]
						}
					}
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = sum
			}
		}
	}
}

// NaiveConv2d performs 2D convolution for any supported numeric type using direct loop
func NaiveConv2d[T D](bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []T) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum T
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									sum += src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi] * kernel[co*cIn*hK*wK+ci*hK*wK+hk*wK+wk]
								}
							}
						}
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = sum
				}
			}
		}
	}
}

// NaiveConv2dF32 performs 2D convolution for float32 using direct loop
func NaiveConv2dF32(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []float32) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					sum := float32(0)
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									sum += src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi] * kernel[co*cIn*hK*wK+ci*hK*wK+hk*wK+wk]
								}
							}
						}
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = sum
				}
			}
		}
	}
}

// NaiveConv2dF64 performs 2D convolution for float64 using direct loop
func NaiveConv2dF64(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []float64) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					sum := float64(0)
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									sum += src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi] * kernel[co*cIn*hK*wK+ci*hK*wK+hk*wK+wk]
								}
							}
						}
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = sum
				}
			}
		}
	}
}

// NaiveConv2dU8 performs 2D convolution for uint8 using direct loop
func NaiveConv2dU8(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []uint8) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									sum += int64(src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi]) * int64(kernel[co*cIn*hK*wK+ci*hK*wK+hk*wK+wk])
								}
							}
						}
					}
					if sum > math.MaxUint8 {
						sum = math.MaxUint8
					} else if sum < 0 {
						sum = 0
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = uint8(sum)
				}
			}
		}
	}
}

// NaiveConv2dU32 performs 2D convolution for uint32 using direct loop
func NaiveConv2dU32(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []uint32) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									sum += int64(src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi]) * int64(kernel[co*cIn*hK*wK+ci*hK*wK+hk*wK+wk])
								}
							}
						}
					}
					if sum > math.MaxUint32 {
						sum = math.MaxUint32
					} else if sum < 0 {
						sum = 0
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = uint32(sum)
				}
			}
		}
	}
}

// NaiveConv2dI64 performs 2D convolution for int64 using direct loop
func NaiveConv2dI64(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []int64) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									sum += src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi] * kernel[co*cIn*hK*wK+ci*hK*wK+hk*wK+wk]
								}
							}
						}
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = sum
				}
			}
		}
	}
}

// NaiveConv2dStrided performs 2D convolution for any supported numeric type using direct loop with support for non-contiguous memory
func NaiveConv2dStrided[T D](bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []T, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum T
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
									kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
									sum += src[srcIdx] * kernel[kernelIdx]
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum
				}
			}
		}
	}
}

// NaiveConv2dStridedF32 performs 2D convolution for float32 using direct loop with support for non-contiguous memory
func NaiveConv2dStridedF32(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []float32, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					sum := float32(0)
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
									kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
									sum += src[srcIdx] * kernel[kernelIdx]
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum
				}
			}
		}
	}
}

// NaiveConv2dStridedF64 performs 2D convolution for float64 using direct loop with support for non-contiguous memory
func NaiveConv2dStridedF64(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []float64, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					sum := float64(0)
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
									kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
									sum += src[srcIdx] * kernel[kernelIdx]
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum
				}
			}
		}
	}
}

// NaiveConv2dStridedU8 performs 2D convolution for uint8 using direct loop with support for non-contiguous memory
func NaiveConv2dStridedU8(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []uint8, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
									kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
									sum += int64(src[srcIdx]) * int64(kernel[kernelIdx])
								}
							}
						}
					}
					if sum > math.MaxUint8 {
						sum = math.MaxUint8
					} else if sum < 0 {
						sum = 0
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = uint8(sum)
				}
			}
		}
	}
}

// NaiveConv2dStridedU32 performs 2D convolution for uint32 using direct loop with support for non-contiguous memory
func NaiveConv2dStridedU32(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []uint32, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
									kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
									sum += int64(src[srcIdx]) * int64(kernel[kernelIdx])
								}
							}
						}
					}
					if sum > math.MaxUint32 {
						sum = math.MaxUint32
					} else if sum < 0 {
						sum = 0
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = uint32(sum)
				}
			}
		}
	}
}

// NaiveConv2dStridedI64 performs 2D convolution for int64 using direct loop with support for non-contiguous memory
func NaiveConv2dStridedI64(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []int64, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hi := ho*stride + hk*dilation - padding
								wi := wo*stride + wk*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
									kernelIdx := co*kernelStrides[0] + ci*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
									sum += src[srcIdx] * kernel[kernelIdx]
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum
				}
			}
		}
	}
}

// NaiveConvTranspose1d performs 1D transpose convolution for any supported numeric type using direct loop
func NaiveConvTranspose1d[T D](bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []T) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum T
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								sum += src[b*cIn*lIn+ci*lIn+li] * kernel[ci*cOut*kSize+co*kSize+k]
							}
						}
					}
				}
				dst[b*cOut*lOut+co*lOut+lo] = sum
			}
		}
	}
}

// NaiveConvTranspose1dF32 performs 1D transpose convolution for float32 using direct loop
func NaiveConvTranspose1dF32(bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []float32) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				sum := float32(0)
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								sum += src[b*cIn*lIn+ci*lIn+li] * kernel[ci*cOut*kSize+co*kSize+k]
							}
						}
					}
				}
				dst[b*cOut*lOut+co*lOut+lo] = sum
			}
		}
	}
}

// NaiveConvTranspose1dF64 performs 1D transpose convolution for float64 using direct loop
func NaiveConvTranspose1dF64(bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []float64) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				sum := float64(0)
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								sum += src[b*cIn*lIn+ci*lIn+li] * kernel[ci*cOut*kSize+co*kSize+k]
							}
						}
					}
				}
				dst[b*cOut*lOut+co*lOut+lo] = sum
			}
		}
	}
}

// NaiveConvTranspose1dU8 performs 1D transpose convolution for uint8 using direct loop
func NaiveConvTranspose1dU8(bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []uint8) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								sum += int64(src[b*cIn*lIn+ci*lIn+li]) * int64(kernel[ci*cOut*kSize+co*kSize+k])
							}
						}
					}
				}
				if sum > math.MaxUint8 {
					sum = math.MaxUint8
				} else if sum < 0 {
					sum = 0
				}
				dst[b*cOut*lOut+co*lOut+lo] = uint8(sum)
			}
		}
	}
}

// NaiveConvTranspose1dU32 performs 1D transpose convolution for uint32 using direct loop
func NaiveConvTranspose1dU32(bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []uint32) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								sum += int64(src[b*cIn*lIn+ci*lIn+li]) * int64(kernel[ci*cOut*kSize+co*kSize+k])
							}
						}
					}
				}
				if sum > math.MaxUint32 {
					sum = math.MaxUint32
				} else if sum < 0 {
					sum = 0
				}
				dst[b*cOut*lOut+co*lOut+lo] = uint32(sum)
			}
		}
	}
}

// NaiveConvTranspose1dI64 performs 1D transpose convolution for int64 using direct loop
func NaiveConvTranspose1dI64(bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []int64) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								sum += src[b*cIn*lIn+ci*lIn+li] * kernel[ci*cOut*kSize+co*kSize+k]
							}
						}
					}
				}
				dst[b*cOut*lOut+co*lOut+lo] = sum
			}
		}
	}
}

// NaiveConvTranspose1dStrided performs 1D transpose convolution for any supported numeric type using direct loop with support for non-contiguous memory
func NaiveConvTranspose1dStrided[T D](bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []T, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum T
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
								kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + k*kernelStrides[2]
								sum += src[srcIdx] * kernel[kernelIdx]
							}
						}
					}
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = sum
			}
		}
	}
}

// NaiveConvTranspose1dStridedF32 performs 1D transpose convolution for float32 using direct loop with support for non-contiguous memory
func NaiveConvTranspose1dStridedF32(bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []float32, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				sum := float32(0)
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
								kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + k*kernelStrides[2]
								sum += src[srcIdx] * kernel[kernelIdx]
							}
						}
					}
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = sum
			}
		}
	}
}

// NaiveConvTranspose1dStridedF64 performs 1D transpose convolution for float64 using direct loop with support for non-contiguous memory
func NaiveConvTranspose1dStridedF64(bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []float64, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				sum := float64(0)
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
								kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + k*kernelStrides[2]
								sum += src[srcIdx] * kernel[kernelIdx]
							}
						}
					}
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = sum
			}
		}
	}
}

// NaiveConvTranspose1dStridedU8 performs 1D transpose convolution for uint8 using direct loop with support for non-contiguous memory
func NaiveConvTranspose1dStridedU8(bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []uint8, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
								kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + k*kernelStrides[2]
								sum += int64(src[srcIdx]) * int64(kernel[kernelIdx])
							}
						}
					}
				}
				if sum > math.MaxUint8 {
					sum = math.MaxUint8
				} else if sum < 0 {
					sum = 0
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = uint8(sum)
			}
		}
	}
}

// NaiveConvTranspose1dStridedU32 performs 1D transpose convolution for uint32 using direct loop with support for non-contiguous memory
func NaiveConvTranspose1dStridedU32(bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []uint32, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
								kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + k*kernelStrides[2]
								sum += int64(src[srcIdx]) * int64(kernel[kernelIdx])
							}
						}
					}
				}
				if sum > math.MaxUint32 {
					sum = math.MaxUint32
				} else if sum < 0 {
					sum = 0
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = uint32(sum)
			}
		}
	}
}

// NaiveConvTranspose1dStridedI64 performs 1D transpose convolution for int64 using direct loop with support for non-contiguous memory
func NaiveConvTranspose1dStridedI64(bSize, cIn, lIn, cOut, kSize int, stride, padding, outPadding, dilation int, src, kernel, dst []int64, srcStrides, kernelStrides, dstStrides []int) {
	lOut := (lIn-1)*stride + dilation*(kSize-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for lo := range lOut {
				var sum int64
				for ci := range cIn {
					for k := range kSize {
						liStride := lo + padding - k*dilation
						if liStride%stride == 0 {
							li := liStride / stride
							if li >= 0 && li < lIn {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
								kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + k*kernelStrides[2]
								sum += src[srcIdx] * kernel[kernelIdx]
							}
						}
					}
				}
				dstIdx := b*dstStrides[0] + co*dstStrides[1] + lo*dstStrides[2]
				dst[dstIdx] = sum
			}
		}
	}
}

// NaiveConvTranspose2d performs 2D transpose convolution for any supported numeric type using direct loop
func NaiveConvTranspose2d[T D](bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []T) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum T
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										sum += src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi] * kernel[ci*cOut*hK*wK+co*hK*wK+hk*wK+wk]
									}
								}
							}
						}
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = sum
				}
			}
		}
	}
}

// NaiveConvTranspose2dF32 performs 2D transpose convolution for float32 using direct loop
func NaiveConvTranspose2dF32(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []float32) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					sum := float32(0)
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										sum += src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi] * kernel[ci*cOut*hK*wK+co*hK*wK+hk*wK+wk]
									}
								}
							}
						}
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = sum
				}
			}
		}
	}
}

// NaiveConvTranspose2dF64 performs 2D transpose convolution for float64 using direct loop
func NaiveConvTranspose2dF64(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []float64) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					sum := float64(0)
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										sum += src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi] * kernel[ci*cOut*hK*wK+co*hK*wK+hk*wK+wk]
									}
								}
							}
						}
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = sum
				}
			}
		}
	}
}

// NaiveConvTranspose2dU8 performs 2D transpose convolution for uint8 using direct loop
func NaiveConvTranspose2dU8(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []uint8) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										sum += int64(src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi]) * int64(kernel[ci*cOut*hK*wK+co*hK*wK+hk*wK+wk])
									}
								}
							}
						}
					}
					if sum > math.MaxUint8 {
						sum = math.MaxUint8
					} else if sum < 0 {
						sum = 0
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = uint8(sum)
				}
			}
		}
	}
}

// NaiveConvTranspose2dU32 performs 2D transpose convolution for uint32 using direct loop
func NaiveConvTranspose2dU32(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []uint32) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										sum += int64(src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi]) * int64(kernel[ci*cOut*hK*wK+co*hK*wK+hk*wK+wk])
									}
								}
							}
						}
					}
					if sum > math.MaxUint32 {
						sum = math.MaxUint32
					} else if sum < 0 {
						sum = 0
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = uint32(sum)
				}
			}
		}
	}
}

// NaiveConvTranspose2dI64 performs 2D transpose convolution for int64 using direct loop
func NaiveConvTranspose2dI64(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []int64) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										sum += src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi] * kernel[ci*cOut*hK*wK+co*hK*wK+hk*wK+wk]
									}
								}
							}
						}
					}
					dst[b*cOut*hOut*wOut+co*hOut*wOut+ho*wOut+wo] = sum
				}
			}
		}
	}
}

// NaiveConvTranspose2dStrided performs 2D transpose convolution for any supported numeric type using direct loop with support for non-contiguous memory
func NaiveConvTranspose2dStrided[T D](bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []T, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum T
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
										kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
										sum += src[srcIdx] * kernel[kernelIdx]
									}
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum
				}
			}
		}
	}
}

// NaiveConvTranspose2dStridedF32 performs 2D transpose convolution for float32 using direct loop with support for non-contiguous memory
func NaiveConvTranspose2dStridedF32(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []float32, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					sum := float32(0)
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
										kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
										sum += src[srcIdx] * kernel[kernelIdx]
									}
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum
				}
			}
		}
	}
}

// NaiveConvTranspose2dStridedF64 performs 2D transpose convolution for float64 using direct loop with support for non-contiguous memory
func NaiveConvTranspose2dStridedF64(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []float64, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					sum := float64(0)
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
										kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
										sum += src[srcIdx] * kernel[kernelIdx]
									}
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum
				}
			}
		}
	}
}

// NaiveConvTranspose2dStridedU8 performs 2D transpose convolution for uint8 using direct loop with support for non-contiguous memory
func NaiveConvTranspose2dStridedU8(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []uint8, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
										kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
										sum += int64(src[srcIdx]) * int64(kernel[kernelIdx])
									}
								}
							}
						}
					}
					if sum > math.MaxUint8 {
						sum = math.MaxUint8
					} else if sum < 0 {
						sum = 0
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = uint8(sum)
				}
			}
		}
	}
}

// NaiveConvTranspose2dStridedU32 performs 2D transpose convolution for uint32 using direct loop with support for non-contiguous memory
func NaiveConvTranspose2dStridedU32(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []uint32, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
										kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
										sum += int64(src[srcIdx]) * int64(kernel[kernelIdx])
									}
								}
							}
						}
					}
					if sum > math.MaxUint32 {
						sum = math.MaxUint32
					} else if sum < 0 {
						sum = 0
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = uint32(sum)
				}
			}
		}
	}
}

// NaiveConvTranspose2dStridedI64 performs 2D transpose convolution for int64 using direct loop with support for non-contiguous memory
func NaiveConvTranspose2dStridedI64(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, outPadding, dilation int, src, kernel, dst []int64, srcStrides, kernelStrides, dstStrides []int) {
	hOut := (hIn-1)*stride + dilation*(hK-1) + outPadding - 2*padding + 1
	wOut := (wIn-1)*stride + dilation*(wK-1) + outPadding - 2*padding + 1
	for b := range bSize {
		for co := range cOut {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					for ci := range cIn {
						for hk := range hK {
							for wk := range wK {
								hiStride := ho + padding - hk*dilation
								wiStride := wo + padding - wk*dilation
								if hiStride%stride == 0 && wiStride%stride == 0 {
									hi := hiStride / stride
									wi := wiStride / stride
									if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
										srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
										kernelIdx := ci*kernelStrides[0] + co*kernelStrides[1] + hk*kernelStrides[2] + wk*kernelStrides[3]
										sum += src[srcIdx] * kernel[kernelIdx]
									}
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + co*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum
				}
			}
		}
	}
}

// AvgPool2d performs 2D average pooling for any supported numeric type
func AvgPool2d[T D](bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []T) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var sum T
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								sum += src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
								count++
							}
						}
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = sum / T(count)
				}
			}
		}
	}
}

// AvgPool2dF32 performs 2D average pooling for float32
func AvgPool2dF32(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []float32) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					sum := float32(0)
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								sum += src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
								count++
							}
						}
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = sum / float32(count)
				}
			}
		}
	}
}

// AvgPool2dF64 performs 2D average pooling for float64
func AvgPool2dF64(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []float64) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					sum := float64(0)
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								sum += src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
								count++
							}
						}
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = sum / float64(count)
				}
			}
		}
	}
}

// AvgPool2dU8 performs 2D average pooling for uint8
func AvgPool2dU8(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []uint8) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								sum += int64(src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi])
								count++
							}
						}
					}
					avg := min(sum/int64(count), math.MaxUint8)
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = uint8(avg)
				}
			}
		}
	}
}

// AvgPool2dU32 performs 2D average pooling for uint32
func AvgPool2dU32(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []uint32) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								sum += int64(src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi])
								count++
							}
						}
					}
					avg := min(sum/int64(count), math.MaxUint32)
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = uint32(avg)
				}
			}
		}
	}
}

// AvgPool2dI64 performs 2D average pooling for int64
func AvgPool2dI64(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []int64) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								sum += src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
								count++
							}
						}
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = sum / int64(count)
				}
			}
		}
	}
}

// AvgPool2dStrided performs 2D average pooling for any supported numeric type with support for non-contiguous memory
func AvgPool2dStrided[T D](bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []T, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var sum T
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								sum += src[srcIdx]
								count++
							}
						}
					}
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum / T(count)
				}
			}
		}
	}
}

// AvgPool2dStridedF32 performs 2D average pooling for float32 with support for non-contiguous memory
func AvgPool2dStridedF32(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []float32, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					sum := float32(0)
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								sum += src[srcIdx]
								count++
							}
						}
					}
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum / float32(count)
				}
			}
		}
	}
}

// AvgPool2dStridedF64 performs 2D average pooling for float64 with support for non-contiguous memory
func AvgPool2dStridedF64(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []float64, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					sum := float64(0)
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								sum += src[srcIdx]
								count++
							}
						}
					}
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum / float64(count)
				}
			}
		}
	}
}

// AvgPool2dStridedU8 performs 2D average pooling for uint8 with support for non-contiguous memory
func AvgPool2dStridedU8(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []uint8, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								sum += int64(src[srcIdx])
								count++
							}
						}
					}
					avg := min(sum/int64(count), math.MaxUint8)
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = uint8(avg)
				}
			}
		}
	}
}

// AvgPool2dStridedU32 performs 2D average pooling for uint32 with support for non-contiguous memory
func AvgPool2dStridedU32(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []uint32, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								sum += int64(src[srcIdx])
								count++
							}
						}
					}
					avg := min(sum/int64(count), math.MaxUint32)
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = uint32(avg)
				}
			}
		}
	}
}

// AvgPool2dStridedI64 performs 2D average pooling for int64 with support for non-contiguous memory
func AvgPool2dStridedI64(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []int64, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var sum int64
					count := 0
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								sum += src[srcIdx]
								count++
							}
						}
					}
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = sum / int64(count)
				}
			}
		}
	}
}

// MaxPool2d performs 2D max pooling for any supported numeric type
func MaxPool2d[T D](bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []T) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var maxVal T
					first := true
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								val := src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
								if first {
									maxVal = val
									first = false
								} else if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = maxVal
				}
			}
		}
	}
}

// MaxPool2dF32 performs 2D max pooling for float32
func MaxPool2dF32(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []float32) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					maxVal := float32(math.Inf(-1))
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								val := src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
								if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = maxVal
				}
			}
		}
	}
}

// MaxPool2dF64 performs 2D max pooling for float64
func MaxPool2dF64(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []float64) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					maxVal := float64(math.Inf(-1))
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								val := src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
								if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = maxVal
				}
			}
		}
	}
}

// MaxPool2dU8 performs 2D max pooling for uint8
func MaxPool2dU8(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []uint8) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var maxVal uint8
					first := true
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								val := src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
								if first {
									maxVal = val
									first = false
								} else if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = maxVal
				}
			}
		}
	}
}

// MaxPool2dU32 performs 2D max pooling for uint32
func MaxPool2dU32(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []uint32) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var maxVal uint32
					first := true
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								val := src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
								if first {
									maxVal = val
									first = false
								} else if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = maxVal
				}
			}
		}
	}
}

// MaxPool2dI64 performs 2D max pooling for int64
func MaxPool2dI64(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []int64) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var maxVal int64
					first := true
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								val := src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
								if first {
									maxVal = val
									first = false
								} else if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = maxVal
				}
			}
		}
	}
}

// MaxPool2dStrided performs 2D max pooling for any supported numeric type with support for non-contiguous memory
func MaxPool2dStrided[T D](bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []T, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var maxVal T
					first := true
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								val := src[srcIdx]
								if first {
									maxVal = val
									first = false
								} else if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = maxVal
				}
			}
		}
	}
}

// MaxPool2dStridedF32 performs 2D max pooling for float32 with support for non-contiguous memory
func MaxPool2dStridedF32(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []float32, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					maxVal := float32(math.Inf(-1))
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								val := src[srcIdx]
								if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = maxVal
				}
			}
		}
	}
}

// MaxPool2dStridedF64 performs 2D max pooling for float64 with support for non-contiguous memory
func MaxPool2dStridedF64(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []float64, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					maxVal := float64(math.Inf(-1))
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								val := src[srcIdx]
								if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = maxVal
				}
			}
		}
	}
}

// MaxPool2dStridedU8 performs 2D max pooling for uint8 with support for non-contiguous memory
func MaxPool2dStridedU8(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []uint8, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var maxVal uint8
					first := true
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								val := src[srcIdx]
								if first {
									maxVal = val
									first = false
								} else if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = maxVal
				}
			}
		}
	}
}

// MaxPool2dStridedU32 performs 2D max pooling for uint32 with support for non-contiguous memory
func MaxPool2dStridedU32(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []uint32, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var maxVal uint32
					first := true
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								val := src[srcIdx]
								if first {
									maxVal = val
									first = false
								} else if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = maxVal
				}
			}
		}
	}
}

// MaxPool2dStridedI64 performs 2D max pooling for int64 with support for non-contiguous memory
func MaxPool2dStridedI64(bSize, c, hIn, wIn, hK, wK, hStride, wStride int, src, dst []int64, srcStrides, dstStrides []int) {
	hOut := (hIn-hK)/hStride + 1
	wOut := (wIn-wK)/wStride + 1
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					var maxVal int64
					first := true
					for hk := range hK {
						for wk := range wK {
							hi := ho*hStride + hk
							wi := wo*wStride + wk
							if hi < hIn && wi < wIn {
								srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								val := src[srcIdx]
								if first {
									maxVal = val
									first = false
								} else if val > maxVal {
									maxVal = val
								}
							}
						}
					}
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = maxVal
				}
			}
		}
	}
}

// UpsampleNearest2d performs 2D nearest neighbor upsampling for any supported numeric type
func UpsampleNearest2d[T D](bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []T) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
				}
			}
		}
	}
}

// UpsampleNearest2dF32 performs 2D nearest neighbor upsampling for float32
func UpsampleNearest2dF32(bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []float32) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
				}
			}
		}
	}
}

// UpsampleNearest2dF64 performs 2D nearest neighbor upsampling for float64
func UpsampleNearest2dF64(bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []float64) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
				}
			}
		}
	}
}

// UpsampleNearest2dU8 performs 2D nearest neighbor upsampling for uint8
func UpsampleNearest2dU8(bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []uint8) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
				}
			}
		}
	}
}

// UpsampleNearest2dU32 performs 2D nearest neighbor upsampling for uint32
func UpsampleNearest2dU32(bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []uint32) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
				}
			}
		}
	}
}

// UpsampleNearest2dI64 performs 2D nearest neighbor upsampling for int64
func UpsampleNearest2dI64(bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []int64) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					dst[b*c*hOut*wOut+ch*hOut*wOut+ho*wOut+wo] = src[b*c*hIn*wIn+ch*hIn*wIn+hi*wIn+wi]
				}
			}
		}
	}
}

// UpsampleNearest2dStrided performs 2D nearest neighbor upsampling for any supported numeric type with support for non-contiguous memory
func UpsampleNearest2dStrided[T D](bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []T, srcStrides, dstStrides []int) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = src[srcIdx]
				}
			}
		}
	}
}

// UpsampleNearest2dStridedF32 performs 2D nearest neighbor upsampling for float32 with support for non-contiguous memory
func UpsampleNearest2dStridedF32(bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []float32, srcStrides, dstStrides []int) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = src[srcIdx]
				}
			}
		}
	}
}

// UpsampleNearest2dStridedF64 performs 2D nearest neighbor upsampling for float64 with support for non-contiguous memory
func UpsampleNearest2dStridedF64(bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []float64, srcStrides, dstStrides []int) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = src[srcIdx]
				}
			}
		}
	}
}

// UpsampleNearest2dStridedU8 performs 2D nearest neighbor upsampling for uint8 with support for non-contiguous memory
func UpsampleNearest2dStridedU8(bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []uint8, srcStrides, dstStrides []int) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = src[srcIdx]
				}
			}
		}
	}
}

// UpsampleNearest2dStridedU32 performs 2D nearest neighbor upsampling for uint32 with support for non-contiguous memory
func UpsampleNearest2dStridedU32(bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []uint32, srcStrides, dstStrides []int) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = src[srcIdx]
				}
			}
		}
	}
}

// UpsampleNearest2dStridedI64 performs 2D nearest neighbor upsampling for int64 with support for non-contiguous memory
func UpsampleNearest2dStridedI64(bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []int64, srcStrides, dstStrides []int) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor((float64(ho) + 0.5) * float64(hIn) / float64(hOut)))
					wi := int(math.Floor((float64(wo) + 0.5) * float64(wIn) / float64(wOut)))
					if hi >= hIn {
						hi = hIn - 1
					}
					if wi >= wIn {
						wi = wIn - 1
					}
					srcIdx := b*srcStrides[0] + ch*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
					dstIdx := b*dstStrides[0] + ch*dstStrides[1] + ho*dstStrides[2] + wo*dstStrides[3]
					dst[dstIdx] = src[srcIdx]
				}
			}
		}
	}
}

// Im2col extracts columns for 2D convolution (im2col) for any supported numeric type
func Im2col[T D](bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []T) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = 0
							} else {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi]
							}
						}
					}
				}
			}
		}
	}
}

// Im2colF32 extracts columns for 2D convolution (im2col) for float32
func Im2colF32(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []float32) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = 0
							} else {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi]
							}
						}
					}
				}
			}
		}
	}
}

// Im2colF64 extracts columns for 2D convolution (im2col) for float64
func Im2colF64(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []float64) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = 0
							} else {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi]
							}
						}
					}
				}
			}
		}
	}
}

// Im2colU8 extracts columns for 2D convolution (im2col) for uint8
func Im2colU8(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []uint8) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = 0
							} else {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi]
							}
						}
					}
				}
			}
		}
	}
}

// Im2colU32 extracts columns for 2D convolution (im2col) for uint32
func Im2colU32(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []uint32) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = 0
							} else {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi]
							}
						}
					}
				}
			}
		}
	}
}

// Im2colI64 extracts columns for 2D convolution (im2col) for int64
func Im2colI64(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []int64) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = 0
							} else {
								col[b*hOut*wOut*cIn*hK*wK+ho*wOut*cIn*hK*wK+wo*cIn*hK*wK+ci*hK*wK+hk*wK+wk] = src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi]
							}
						}
					}
				}
			}
		}
	}
}

// Im2colStrided extracts columns for 2D convolution (im2col) for any supported numeric type with support for non-contiguous src
func Im2colStrided[T D](bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []T, srcStrides []int) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							colIdx := b*hOut*wOut*cIn*hK*wK + ho*wOut*cIn*hK*wK + wo*cIn*hK*wK + ci*hK*wK + hk*wK + wk
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[colIdx] = 0
							} else {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								col[colIdx] = src[srcIdx]
							}
						}
					}
				}
			}
		}
	}
}

// Im2colStridedF32 extracts columns for 2D convolution (im2col) for float32 with support for non-contiguous src
func Im2colStridedF32(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []float32, srcStrides []int) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							colIdx := b*hOut*wOut*cIn*hK*wK + ho*wOut*cIn*hK*wK + wo*cIn*hK*wK + ci*hK*wK + hk*wK + wk
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[colIdx] = 0
							} else {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								col[colIdx] = src[srcIdx]
							}
						}
					}
				}
			}
		}
	}
}

// Im2colStridedF64 extracts columns for 2D convolution (im2col) for float64 with support for non-contiguous src
func Im2colStridedF64(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []float64, srcStrides []int) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							colIdx := b*hOut*wOut*cIn*hK*wK + ho*wOut*cIn*hK*wK + wo*cIn*hK*wK + ci*hK*wK + hk*wK + wk
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[colIdx] = 0
							} else {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								col[colIdx] = src[srcIdx]
							}
						}
					}
				}
			}
		}
	}
}

// Im2colStridedU8 extracts columns for 2D convolution (im2col) for uint8 with support for non-contiguous src
func Im2colStridedU8(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []uint8, srcStrides []int) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							colIdx := b*hOut*wOut*cIn*hK*wK + ho*wOut*cIn*hK*wK + wo*cIn*hK*wK + ci*hK*wK + hk*wK + wk
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[colIdx] = 0
							} else {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								col[colIdx] = src[srcIdx]
							}
						}
					}
				}
			}
		}
	}
}

// Im2colStridedU32 extracts columns for 2D convolution (im2col) for uint32 with support for non-contiguous src
func Im2colStridedU32(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []uint32, srcStrides []int) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							colIdx := b*hOut*wOut*cIn*hK*wK + ho*wOut*cIn*hK*wK + wo*cIn*hK*wK + ci*hK*wK + hk*wK + wk
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[colIdx] = 0
							} else {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								col[colIdx] = src[srcIdx]
							}
						}
					}
				}
			}
		}
	}
}

// Im2colStridedI64 extracts columns for 2D convolution (im2col) for int64 with support for non-contiguous src
func Im2colStridedI64(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK int, stride, padding, dilation int, src, col []int64, srcStrides []int) {
	for b := range bSize {
		for ho := range hOut {
			for wo := range wOut {
				for ci := range cIn {
					for hk := range hK {
						for wk := range wK {
							hi := ho*stride + hk*dilation - padding
							wi := wo*stride + wk*dilation - padding
							colIdx := b*hOut*wOut*cIn*hK*wK + ho*wOut*cIn*hK*wK + wo*cIn*hK*wK + ci*hK*wK + hk*wK + wk
							if hi < 0 || hi >= hIn || wi < 0 || wi >= wIn {
								col[colIdx] = 0
							} else {
								srcIdx := b*srcStrides[0] + ci*srcStrides[1] + hi*srcStrides[2] + wi*srcStrides[3]
								col[colIdx] = src[srcIdx]
							}
						}
					}
				}
			}
		}
	}
}

// Im2col1d performs im2col transformation for any supported numeric type
func Im2col1d[T D](bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []T) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li < 0 || li >= lIn {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = 0
					} else {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = src[b*cIn*lIn+ci*lIn+li]
					}
				}
			}
		}
	}
}

// Im2col1dF32 extracts columns for 1D convolution (im2col) for float32
func Im2col1dF32(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []float32) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li < 0 || li >= lIn {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = 0
					} else {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = src[b*cIn*lIn+ci*lIn+li]
					}
				}
			}
		}
	}
}

// Im2col1dF64 extracts columns for 1D convolution (im2col) for float64
func Im2col1dF64(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []float64) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li < 0 || li >= lIn {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = 0
					} else {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = src[b*cIn*lIn+ci*lIn+li]
					}
				}
			}
		}
	}
}

// Im2col1dU8 extracts columns for 1D convolution (im2col) for uint8
func Im2col1dU8(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []uint8) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li < 0 || li >= lIn {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = 0
					} else {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = src[b*cIn*lIn+ci*lIn+li]
					}
				}
			}
		}
	}
}

// Im2col1dU32 extracts columns for 1D convolution (im2col) for uint32
func Im2col1dU32(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []uint32) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li < 0 || li >= lIn {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = 0
					} else {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = src[b*cIn*lIn+ci*lIn+li]
					}
				}
			}
		}
	}
}

// Im2col1dI64 extracts columns for 1D convolution (im2col) for int64
func Im2col1dI64(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []int64) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li < 0 || li >= lIn {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = 0
					} else {
						col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k] = src[b*cIn*lIn+ci*lIn+li]
					}
				}
			}
		}
	}
}

// Im2col1dStrided performs im2col transformation for any supported numeric type with support for non-contiguous src
func Im2col1dStrided[T D](bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []T, srcStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
					if li < 0 || li >= lIn {
						col[colIdx] = 0
					} else {
						srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
						col[colIdx] = src[srcIdx]
					}
				}
			}
		}
	}
}

// Im2col1dStridedF32 extracts columns for 1D convolution (im2col) for float32 with support for non-contiguous src
func Im2col1dStridedF32(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []float32, srcStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
					if li < 0 || li >= lIn {
						col[colIdx] = 0
					} else {
						srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
						col[colIdx] = src[srcIdx]
					}
				}
			}
		}
	}
}

// Im2col1dStridedF64 performs 1D col2im for float64 with non-contiguous src
func Im2col1dStridedF64(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []float64, srcStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
					if li < 0 || li >= lIn {
						col[colIdx] = 0
					} else {
						srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
						col[colIdx] = src[srcIdx]
					}
				}
			}
		}
	}
}

// Im2col1dStridedU8 extracts columns for 1D convolution (im2col) for uint8 with support for non-contiguous src
func Im2col1dStridedU8(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []uint8, srcStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
					if li < 0 || li >= lIn {
						col[colIdx] = 0
					} else {
						srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
						col[colIdx] = src[srcIdx]
					}
				}
			}
		}
	}
}

// Im2col1dStridedU32 extracts columns for 1D convolution (im2col) for uint32 with support for non-contiguous src
func Im2col1dStridedU32(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []uint32, srcStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
					if li < 0 || li >= lIn {
						col[colIdx] = 0
					} else {
						srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
						col[colIdx] = src[srcIdx]
					}
				}
			}
		}
	}
}

// Im2col1dStridedI64 extracts columns for 1D convolution (im2col) for int64 with support for non-contiguous src
func Im2col1dStridedI64(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, src, col []int64, srcStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
					if li < 0 || li >= lIn {
						col[colIdx] = 0
					} else {
						srcIdx := b*srcStrides[0] + ci*srcStrides[1] + li*srcStrides[2]
						col[colIdx] = src[srcIdx]
					}
				}
			}
		}
	}
}

// Col2im1d performs standard 1D col2im for any supported numeric type (e.g., for conv backward data)
func Col2im1d[T D](bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []T) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						im[b*cIn*lIn+ci*lIn+li] += col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k]
					}
				}
			}
		}
	}
}

// Col2im1dF32 performs standard 1D col2im for float32 (e.g., for conv backward data)
func Col2im1dF32(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []float32) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						im[b*cIn*lIn+ci*lIn+li] += col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k]
					}
				}
			}
		}
	}
}

// Col2im1dF64 performs standard 1D col2im for float64 (e.g., for conv backward data)
func Col2im1dF64(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []float64) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						im[b*cIn*lIn+ci*lIn+li] += col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k]
					}
				}
			}
		}
	}
}

// Col2im1dU8 performs standard 1D col2im for uint8 (e.g., for conv backward data)
func Col2im1dU8(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []uint8) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						sum := int64(im[b*cIn*lIn+ci*lIn+li]) + int64(col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k])
						if sum > math.MaxUint8 {
							sum = math.MaxUint8
						}
						im[b*cIn*lIn+ci*lIn+li] = uint8(sum)
					}
				}
			}
		}
	}
}

// Col2im1dU32 performs standard 1D col2im for uint32 (e.g., for conv backward data)
func Col2im1dU32(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []uint32) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						sum := int64(im[b*cIn*lIn+ci*lIn+li]) + int64(col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k])
						if sum > math.MaxUint32 {
							sum = math.MaxUint32
						}
						im[b*cIn*lIn+ci*lIn+li] = uint32(sum)
					}
				}
			}
		}
	}
}

// Col2im1dI64 performs standard 1D col2im for int64 (e.g., for conv backward data)
func Col2im1dI64(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []int64) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						im[b*cIn*lIn+ci*lIn+li] += col[b*lOut*cIn*kSize+lo*cIn*kSize+ci*kSize+k]
					}
				}
			}
		}
	}
}

// Col2im1dStrided performs standard 1D col2im for any supported numeric type with support for non-contiguous im
func Col2im1dStrided[T D](bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []T, imStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
						imIdx := b*imStrides[0] + ci*imStrides[1] + li*imStrides[2]
						im[imIdx] += col[colIdx]
					}
				}
			}
		}
	}
}

// Col2im1dStridedF32 performs standard 1D col2im for float32 with support for non-contiguous im
func Col2im1dStridedF32(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []float32, imStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
						imIdx := b*imStrides[0] + ci*imStrides[1] + li*imStrides[2]
						im[imIdx] += col[colIdx]
					}
				}
			}
		}
	}
}

// Col2im1dStridedF64 performs standard 1D col2im for float64 with support for non-contiguous im
func Col2im1dStridedF64(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []float64, imStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
						imIdx := b*imStrides[0] + ci*imStrides[1] + li*imStrides[2]
						im[imIdx] += col[colIdx]
					}
				}
			}
		}
	}
}

// Col2im1dStridedU8 performs standard 1D col2im for uint8 with support for non-contiguous im
func Col2im1dStridedU8(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []uint8, imStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
						imIdx := b*imStrides[0] + ci*imStrides[1] + li*imStrides[2]
						sum := int64(im[imIdx]) + int64(col[colIdx])
						if sum > math.MaxUint8 {
							sum = math.MaxUint8
						}
						im[imIdx] = uint8(sum)
					}
				}
			}
		}
	}
}

// Col2im1dStridedU32 performs standard 1D col2im for uint32 with support for non-contiguous im
func Col2im1dStridedU32(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []uint32, imStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
						imIdx := b*imStrides[0] + ci*imStrides[1] + li*imStrides[2]
						sum := int64(im[imIdx]) + int64(col[colIdx])
						if sum > math.MaxUint32 {
							sum = math.MaxUint32
						}
						im[imIdx] = uint32(sum)
					}
				}
			}
		}
	}
}

// Col2im1dStridedI64 performs standard 1D col2im for int64 with support for non-contiguous im
func Col2im1dStridedI64(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []int64, imStrides []int) {
	for b := range bSize {
		for lo := range lOut {
			for ci := range cIn {
				for k := range kSize {
					li := lo*stride + k*dilation - padding
					if li >= 0 && li < lIn {
						colIdx := b*lOut*cIn*kSize + lo*cIn*kSize + ci*kSize + k
						imIdx := b*imStrides[0] + ci*imStrides[1] + li*imStrides[2]
						im[imIdx] += col[colIdx]
					}
				}
			}
		}
	}
}

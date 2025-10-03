package kernels

import (
	"math"

	"github.com/gocnn/gomat/blas"
	"github.com/gocnn/gomat/blas/blas32"
	"github.com/gocnn/gomat/blas/blas64"
)

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

// Im2colConv1d performs 1D convolution for any supported numeric type using im2col + manual gemm
//
// GEMM Configuration: col:(b*lOut, cIn*kSize) × kernel^T:(cIn*kSize, cOut) = dst:(b*lOut, cOut)
// Layout: Optimized for maximum performance with minimal memory reshaping
func Im2colConv1d[T D](bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []T) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	colSize := bSize * lOut * cIn * kSize
	col := make([]T, colSize)
	Im2col1d(bSize, cIn, lIn, lOut, kSize, stride, padding, dilation, src, col)
	// Manual GEMM: dst = col * kernel^T (col: (b*lOut, cIn*kSize), kernel: (cOut, cIn*kSize), dst: (b*lOut, cOut))
	m, n, k := bSize*lOut, cOut, cIn*kSize
	for i := range m {
		for j := range n {
			sum := T(0)
			for l := range k {
				sum += col[i*k+l] * kernel[j*k+l]
			}
			dst[i*n+j] = sum
		}
	}
}

// Im2colConv1dF32 performs 1D convolution for float32 using im2col + gemm with direct BLAS Gemm call
//
// GEMM Configuration: col:(b*lOut, cIn*kSize) × kernel^T:(cIn*kSize, cOut) = dst:(b*lOut, cOut)
// Layout: Optimized for maximum BLAS performance with minimal memory reshaping
func Im2colConv1dF32(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []float32) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	colSize := bSize * lOut * cIn * kSize
	col := make([]float32, colSize)
	Im2col1dF32(bSize, cIn, lIn, lOut, kSize, stride, padding, dilation, src, col)
	// GEMM: dst = col * kernel^T (col: (b*lOut, cIn*kSize), kernel: (cOut, cIn*kSize), dst: (b*lOut, cOut))
	m, n, k := bSize*lOut, cOut, cIn*kSize
	blas32.Gemm(blas.NoTrans, blas.Trans, m, n, k, 1.0, col, k, kernel, k, 0.0, dst, n)
}

// Im2colConv1dF64 performs 1D convolution for float64 using im2col + gemm with direct BLAS Gemm call
//
// GEMM Configuration: col:(b*lOut, cIn*kSize) × kernel^T:(cIn*kSize, cOut) = dst:(b*lOut, cOut)
// Layout: Optimized for maximum BLAS performance with minimal memory reshaping
func Im2colConv1dF64(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []float64) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	colSize := bSize * lOut * cIn * kSize
	col := make([]float64, colSize)
	Im2col1dF64(bSize, cIn, lIn, lOut, kSize, stride, padding, dilation, src, col)
	// GEMM: dst = col * kernel^T (col: (b*lOut, cIn*kSize), kernel: (cOut, cIn*kSize), dst: (b*lOut, cOut))
	m, n, k := bSize*lOut, cOut, cIn*kSize
	blas64.Gemm(blas.NoTrans, blas.Trans, m, n, k, 1.0, col, k, kernel, k, 0.0, dst, n)
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
						for kh := range hK {
							for kw := range wK {
								hi := ho*stride + kh*dilation - padding
								wi := wo*stride + kw*dilation - padding
								if hi >= 0 && hi < hIn && wi >= 0 && wi < wIn {
									sum += src[b*cIn*hIn*wIn+ci*hIn*wIn+hi*wIn+wi] * kernel[co*cIn*hK*wK+ci*hK*wK+kh*wK+kw]
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

// Im2colConv2d performs 2D convolution for any supported numeric type using im2col + manual gemm
//
// GEMM Configuration: col:(b*hOut*wOut, cIn*hK*wK) × kernel^T:(cIn*hK*wK, cOut) = dst:(b*hOut*wOut, cOut)
// Layout: Optimized for maximum performance with minimal memory reshaping
func Im2colConv2d[T D](bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []T) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	colSize := bSize * hOut * wOut * cIn * hK * wK
	col := make([]T, colSize)
	Im2col[T](bSize, cIn, hIn, wIn, hOut, wOut, hK, wK, stride, padding, dilation, src, col)
	// Manual GEMM: dst = col * kernel^T (col: (b*hOut*wOut, cIn*hK*wK), kernel: (cOut, cIn*hK*wK), dst: (b*hOut*wOut, cOut))
	m, n, k := bSize*hOut*wOut, cOut, cIn*hK*wK
	for i := range m {
		for j := range n {
			sum := T(0)
			for l := range k {
				sum += col[i*k+l] * kernel[j*k+l]
			}
			dst[i*n+j] = sum
		}
	}
}

// Im2colConv2dF32 performs 2D convolution for float32 using im2col + gemm with direct BLAS Gemm call
//
// GEMM Configuration: col:(b*hOut*wOut, cIn*hK*wK) × kernel^T:(cIn*hK*wK, cOut) = dst:(b*hOut*wOut, cOut)
// Layout: Optimized for maximum BLAS performance with minimal memory reshaping
func Im2colConv2dF32(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []float32) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	colSize := bSize * hOut * wOut * cIn * hK * wK
	col := make([]float32, colSize)
	Im2colF32(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK, stride, padding, dilation, src, col)
	// GEMM: dst = col * kernel^T (col: (b*hOut*wOut, cIn*hK*wK), kernel: (cOut, cIn*hK*wK), dst: (b*hOut*wOut, cOut))
	m, n, k := bSize*hOut*wOut, cOut, cIn*hK*wK
	blas32.Gemm(blas.NoTrans, blas.Trans, m, n, k, 1.0, col, k, kernel, k, 0.0, dst, n)
}

// Im2colConv2dF64 performs 2D convolution for float64 using im2col + gemm with direct BLAS Gemm call
//
// GEMM Configuration: col:(b*hOut*wOut, cIn*hK*wK) × kernel^T:(cIn*hK*wK, cOut) = dst:(b*hOut*wOut, cOut)
// Layout: Optimized for maximum BLAS performance with minimal memory reshaping
func Im2colConv2dF64(bSize, cIn, hIn, wIn, cOut, hK, wK int, stride, padding, dilation int, src, kernel, dst []float64) {
	hOut := (hIn+2*padding-dilation*(hK-1)-1)/stride + 1
	wOut := (wIn+2*padding-dilation*(wK-1)-1)/stride + 1
	colSize := bSize * hOut * wOut * cIn * hK * wK
	col := make([]float64, colSize)
	Im2colF64(bSize, cIn, hIn, wIn, hOut, wOut, hK, wK, stride, padding, dilation, src, col)
	// GEMM: dst = col * kernel^T (col: (b*hOut*wOut, cIn*hK*wK), kernel: (cOut, cIn*hK*wK), dst: (b*hOut*wOut, cOut))
	m, n, k := bSize*hOut*wOut, cOut, cIn*hK*wK
	blas64.Gemm(blas.NoTrans, blas.Trans, m, n, k, 1.0, col, k, kernel, k, 0.0, dst, n)
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
	for b := 0; b < bSize; b++ {
		for co := 0; co < cOut; co++ {
			for lo := 0; lo < lOut; lo++ {
				sum := float32(0)
				for ci := 0; ci < cIn; ci++ {
					for k := 0; k < kSize; k++ {
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
	for b := 0; b < bSize; b++ {
		for co := 0; co < cOut; co++ {
			for lo := 0; lo < lOut; lo++ {
				sum := float64(0)
				for ci := 0; ci < cIn; ci++ {
					for k := 0; k < kSize; k++ {
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

// UpsampleNearest2d performs 2D nearest neighbor upsampling for any supported numeric type
//
// Algorithm: Standard nearest neighbor interpolation with (ho+0.5)*hIn/hOut mapping
// Compatibility: Fully compatible with PyTorch F.interpolate(mode='nearest')
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
//
// Algorithm: Standard nearest neighbor interpolation with (ho+0.5)*hIn/hOut mapping
// Compatibility: Fully compatible with PyTorch F.interpolate(mode='nearest')
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
//
// Algorithm: Standard nearest neighbor interpolation with (ho+0.5)*hIn/hOut mapping
// Compatibility: Fully compatible with PyTorch F.interpolate(mode='nearest')
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

// UpsampleNearest2dStrided performs 2D nearest neighbor upsampling for any supported numeric type with support for non-contiguous memory
//
// Algorithm: Standard nearest neighbor interpolation with (ho+0.5)*hIn/hOut mapping
// Compatibility: Fully compatible with PyTorch F.interpolate(mode='nearest')
// srcStrides: [batch_stride, channel_stride, hIn_stride, wIn_stride]
// dstStrides: [batch_stride, channel_stride, hOut_stride, wOut_stride]
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
//
// Algorithm: Standard nearest neighbor interpolation with (ho+0.5)*hIn/hOut mapping
// Compatibility: Fully compatible with PyTorch F.interpolate(mode='nearest')
// srcStrides: [batch_stride, channel_stride, hIn_stride, wIn_stride]
// dstStrides: [batch_stride, channel_stride, hOut_stride, wOut_stride]
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
//
// Algorithm: Standard nearest neighbor interpolation with (ho+0.5)*hIn/hOut mapping
// Compatibility: Fully compatible with PyTorch F.interpolate(mode='nearest')
// srcStrides: [batch_stride, channel_stride, hIn_stride, wIn_stride]
// dstStrides: [batch_stride, channel_stride, hOut_stride, wOut_stride]
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

// Im2col extracts columns for 2D convolution (im2col) for any supported numeric type
//
// Memory Layout: (batch, hOut, wOut, cIn, hK, wK) - optimized for GEMM performance
// PyTorch Layout: (batch, cIn, hK, wK, hOut, wOut) - standard unfold output
//
// Note: Layout difference does not affect algorithm correctness but impacts
// direct data exchange with PyTorch. Current layout optimizes GEMM operations.
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
//
// Memory Layout: (batch, hOut, wOut, cIn, hK, wK) - optimized for GEMM performance
// PyTorch Layout: (batch, cIn, hK, wK, hOut, wOut) - standard unfold output
//
// Note: Layout difference does not affect algorithm correctness but impacts
// direct data exchange with PyTorch. Current layout optimizes GEMM operations.
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
//
// Memory Layout: (batch, hOut, wOut, cIn, hK, wK) - optimized for GEMM performance
// PyTorch Layout: (batch, cIn, hK, wK, hOut, wOut) - standard unfold output
//
// Note: Layout difference does not affect algorithm correctness but impacts
// direct data exchange with PyTorch. Current layout optimizes GEMM operations.
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

// Im2colStrided extracts columns for 2D convolution (im2col) for any supported numeric type with support for non-contiguous src
//
// Memory Layout for col: (batch, hOut, wOut, cIn, hK, wK) - optimized for GEMM performance, assumed contiguous
// srcStrides: [batch_stride, cIn_stride, hIn_stride, wIn_stride]
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
//
// Memory Layout for col: (batch, hOut, wOut, cIn, hK, wK) - optimized for GEMM performance, assumed contiguous
// srcStrides: [batch_stride, cIn_stride, hIn_stride, wIn_stride]
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
//
// Memory Layout for col: (batch, hOut, wOut, cIn, hK, wK) - optimized for GEMM performance, assumed contiguous
// srcStrides: [batch_stride, cIn_stride, hIn_stride, wIn_stride]
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
//
// Memory Layout: (batch, lOut, cIn, kSize) - optimized for GEMM performance
// PyTorch Layout: (batch, cIn, kSize, lOut) - standard unfold output
//
// Note: Layout difference does not affect algorithm correctness but impacts
// direct data exchange with PyTorch. Current layout optimizes GEMM operations.
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
//
// Memory Layout: (batch, lOut, cIn, kSize) - optimized for GEMM performance
// PyTorch Layout: (batch, cIn, kSize, lOut) - standard unfold output
//
// Note: Layout difference does not affect algorithm correctness but impacts
// direct data exchange with PyTorch. Current layout optimizes GEMM operations.
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

// Im2col1dStrided performs im2col transformation for any supported numeric type with support for non-contiguous src
//
// Memory Layout for col: (batch, lOut, cIn, kSize) - optimized for GEMM performance, assumed contiguous
// srcStrides: [batch_stride, cIn_stride, lIn_stride]
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
//
// Memory Layout for col: (batch, lOut, cIn, kSize) - optimized for GEMM performance, assumed contiguous
// srcStrides: [batch_stride, cIn_stride, lIn_stride]
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

// Col2im1d performs standard 1D col2im for any supported numeric type (e.g., for conv backward data)
//
// Input Layout: (batch, lOut, cIn, kSize) - matches Im2col1d output
// Output Layout: (batch, cIn, lIn) - standard tensor format
//
// Note: This is the inverse operation of Im2col1d with proper accumulation
// for overlapping regions during the reconstruction process.
func Col2im1d[T D](bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []T) {
	// Clear im to zero first
	for i := range im {
		im[i] = 0
	}

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
//
// Input Layout: (batch, lOut, cIn, kSize) - matches Im2col1d output
// Output Layout: (batch, cIn, lIn) - standard tensor format
//
// Note: This is the inverse operation of Im2col1d with proper accumulation
// for overlapping regions during the reconstruction process.
func Col2im1dF32(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []float32) {
	// Clear im to zero first
	for i := range im {
		im[i] = 0
	}

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
//
// Input Layout: (batch, lOut, cIn, kSize) - matches Im2col1d output
// Output Layout: (batch, cIn, lIn) - standard tensor format
//
// Note: This is the inverse operation of Im2col1d with proper accumulation
// for overlapping regions during the reconstruction process.
func Col2im1dF64(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []float64) {
	// Clear im to zero first
	for i := range im {
		im[i] = 0
	}

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

// Col2im1dStrided performs standard 1D col2im for any supported numeric type (e.g., for conv backward data) with support for non-contiguous im
//
// Input Layout: col:(batch, lOut, cIn, kSize) - assumed contiguous, matches Im2col1dStrided output
// Output Layout: im:(batch, cIn, lIn) - supports non-contiguous via imStrides
// imStrides: [batch_stride, cIn_stride, lIn_stride]
//
// Note: This is the inverse operation of Im2col1dStrided with proper accumulation
// for overlapping regions during the reconstruction process. Clears im to zero first.
func Col2im1dStrided[T D](bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []T, imStrides []int) {
	// Clear im to zero first (works even if im is non-contiguous view, as it's the underlying slice)
	for b := range bSize {
		for ci := range cIn {
			for li := range lIn {
				imIdx := b*imStrides[0] + ci*imStrides[1] + li*imStrides[2]
				im[imIdx] = 0
			}
		}
	}

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

// Col2im1dStridedF32 performs standard 1D col2im for float32 (e.g., for conv backward data) with support for non-contiguous im
//
// Input Layout: col:(batch, lOut, cIn, kSize) - assumed contiguous, matches Im2col1dStrided output
// Output Layout: im:(batch, cIn, lIn) - supports non-contiguous via imStrides
// imStrides: [batch_stride, cIn_stride, lIn_stride]
//
// Note: This is the inverse operation of Im2col1dStrided with proper accumulation
// for overlapping regions during the reconstruction process. Clears im to zero first.
func Col2im1dStridedF32(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []float32, imStrides []int) {
	// Clear im to zero first (works even if im is non-contiguous view, as it's the underlying slice)
	for b := range bSize {
		for ci := range cIn {
			for li := range lIn {
				imIdx := b*imStrides[0] + ci*imStrides[1] + li*imStrides[2]
				im[imIdx] = 0
			}
		}
	}

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

// Col2im1dStridedF64 performs standard 1D col2im for float64 (e.g., for conv backward data) with support for non-contiguous im
//
// Input Layout: col:(batch, lOut, cIn, kSize) - assumed contiguous, matches Im2col1dStrided output
// Output Layout: im:(batch, cIn, lIn) - supports non-contiguous via imStrides
// imStrides: [batch_stride, cIn_stride, lIn_stride]
//
// Note: This is the inverse operation of Im2col1dStrided with proper accumulation
// for overlapping regions during the reconstruction process. Clears im to zero first.
func Col2im1dStridedF64(bSize, cIn, lIn, lOut, kSize int, stride, padding, dilation int, col, im []float64, imStrides []int) {
	// Clear im to zero first (works even if im is non-contiguous view, as it's the underlying slice)
	for b := range bSize {
		for ci := range cIn {
			for li := range lIn {
				imIdx := b*imStrides[0] + ci*imStrides[1] + li*imStrides[2]
				im[imIdx] = 0
			}
		}
	}

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

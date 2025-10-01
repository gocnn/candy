package kernels

import (
	"math"

	"github.com/gocnn/gomat/blas"
	"github.com/gocnn/gomat/blas/blas32"
	"github.com/gocnn/gomat/blas/blas64"
)

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

// Im2colConv1dF32 performs 1D convolution for float32 using im2col + gemm with direct BLAS Gemm call
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
func Im2colConv1dF64(bSize, cIn, lIn, cOut, kSize int, stride, padding, dilation int, src, kernel, dst []float64) {
	lOut := (lIn+2*padding-dilation*(kSize-1)-1)/stride + 1
	colSize := bSize * lOut * cIn * kSize
	col := make([]float64, colSize)
	Im2col1dF64(bSize, cIn, lIn, lOut, kSize, stride, padding, dilation, src, col)
	// GEMM: dst = col * kernel^T (col: (b*lOut, cIn*kSize), kernel: (cOut, cIn*kSize), dst: (b*lOut, cOut))
	m, n, k := bSize*lOut, cOut, cIn*kSize
	blas64.Gemm(blas.NoTrans, blas.Trans, m, n, k, 1.0, col, k, kernel, k, 0.0, dst, n)
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

// Im2colConv2dF32 performs 2D convolution for float32 using im2col + gemm with direct BLAS Gemm call
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

// UpsampleNearest2dF32 performs 2D nearest neighbor upsampling for float32
func UpsampleNearest2dF32(bSize, c, hIn, wIn, hOut, wOut int, hScale, wScale float64, src, dst []float32) {
	for b := range bSize {
		for ch := range c {
			for ho := range hOut {
				for wo := range wOut {
					hi := int(math.Floor(float64(ho) * hScale))
					wi := int(math.Floor(float64(wo) * wScale))
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
					hi := int(math.Floor(float64(ho) * hScale))
					wi := int(math.Floor(float64(wo) * wScale))
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

// Col2im1dF32 performs standard 1D col2im for float32 (e.g., for conv backward data)
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

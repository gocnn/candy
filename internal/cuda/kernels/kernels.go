package kernels

import (
	"embed"
)

// Kernel function name constants for cuModuleGetFunction calls
// Keep in sync with the actual kernel function names in *.cu files
const (
	// Affine operations
	AffineF32    = "affine_f32"
	AffineF64    = "affine_f64"
	AffineU8     = "affine_u8"
	AffineU32    = "affine_u32"
	AffineI16    = "affine_i16"
	AffineI32    = "affine_i32"
	AffineI64    = "affine_i64"
	AffineF16    = "affine_f16"     // Available when __CUDA_ARCH__ >= 530
	AffineBF16   = "affine_bf16"    // Available when __CUDA_ARCH__ >= 800
	AffineF8E4M3 = "affine_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
)

// Binary operation kernel function names
const (
	// Binary arithmetic operations
	BAddF32    = "badd_f32"
	BAddF64    = "badd_f64"
	BAddF16    = "badd_f16"     // Available when __CUDA_ARCH__ >= 530
	BAddBF16   = "badd_bf16"    // Available when __CUDA_ARCH__ >= 800
	BAddF8E4M3 = "badd_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	BSubF32    = "bsub_f32"
	BSubF64    = "bsub_f64"
	BSubF16    = "bsub_f16"     // Available when __CUDA_ARCH__ >= 530
	BSubBF16   = "bsub_bf16"    // Available when __CUDA_ARCH__ >= 800
	BSubF8E4M3 = "bsub_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	BMulF32    = "bmul_f32"
	BMulF64    = "bmul_f64"
	BMulF16    = "bmul_f16"     // Available when __CUDA_ARCH__ >= 530
	BMulBF16   = "bmul_bf16"    // Available when __CUDA_ARCH__ >= 800
	BMulF8E4M3 = "bmul_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	BDivF32    = "bdiv_f32"
	BDivF64    = "bdiv_f64"
	BDivF16    = "bdiv_f16"     // Available when __CUDA_ARCH__ >= 530
	BDivBF16   = "bdiv_bf16"    // Available when __CUDA_ARCH__ >= 800
	BDivF8E4M3 = "bdiv_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	BMaximumF32    = "bmaximum_f32"
	BMaximumF64    = "bmaximum_f64"
	BMaximumF16    = "bmaximum_f16"     // Available when __CUDA_ARCH__ >= 530
	BMaximumBF16   = "bmaximum_bf16"    // Available when __CUDA_ARCH__ >= 800
	BMaximumF8E4M3 = "bmaximum_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	BMinimumF32    = "bminimum_f32"
	BMinimumF64    = "bminimum_f64"
	BMinimumF16    = "bminimum_f16"     // Available when __CUDA_ARCH__ >= 530
	BMinimumBF16   = "bminimum_bf16"    // Available when __CUDA_ARCH__ >= 800
	BMinimumF8E4M3 = "bminimum_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	// Binary comparison operations (output uint8_t)
	EqF32    = "eq_f32"
	EqF64    = "eq_f64"
	EqF16    = "eq_f16"     // Available when __CUDA_ARCH__ >= 530
	EqBF16   = "eq_bf16"    // Available when __CUDA_ARCH__ >= 800
	EqF8E4M3 = "eq_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	NeF32    = "ne_f32"
	NeF64    = "ne_f64"
	NeF16    = "ne_f16"     // Available when __CUDA_ARCH__ >= 530
	NeBF16   = "ne_bf16"    // Available when __CUDA_ARCH__ >= 800
	NeF8E4M3 = "ne_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	LtF32    = "lt_f32"
	LtF64    = "lt_f64"
	LtF16    = "lt_f16"     // Available when __CUDA_ARCH__ >= 530
	LtBF16   = "lt_bf16"    // Available when __CUDA_ARCH__ >= 800
	LtF8E4M3 = "lt_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	LeF32    = "le_f32"
	LeF64    = "le_f64"
	LeF16    = "le_f16"     // Available when __CUDA_ARCH__ >= 530
	LeBF16   = "le_bf16"    // Available when __CUDA_ARCH__ >= 800
	LeF8E4M3 = "le_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	GtF32    = "gt_f32"
	GtF64    = "gt_f64"
	GtF16    = "gt_f16"     // Available when __CUDA_ARCH__ >= 530
	GtBF16   = "gt_bf16"    // Available when __CUDA_ARCH__ >= 800
	GtF8E4M3 = "gt_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	GeF32    = "ge_f32"
	GeF64    = "ge_f64"
	GeF16    = "ge_f16"     // Available when __CUDA_ARCH__ >= 530
	GeBF16   = "ge_bf16"    // Available when __CUDA_ARCH__ >= 800
	GeF8E4M3 = "ge_f8_e4m3" // Available when __CUDA_ARCH__ >= 800
)

// Cast operation kernel function names
const (
	// Basic type casting operations
	CastF32F32  = "cast_f32_f32"
	CastF32F64  = "cast_f32_f64"
	CastF32U8   = "cast_f32_u8"
	CastF32U32  = "cast_f32_u32"
	CastF32I64  = "cast_f32_i64"
	CastF32F16  = "cast_f32_f16"  // Available when __CUDA_ARCH__ >= 530
	CastF32BF16 = "cast_f32_bf16" // Available when __CUDA_ARCH__ >= 800

	CastF64F32  = "cast_f64_f32"
	CastF64F64  = "cast_f64_f64"
	CastF64U8   = "cast_f64_u8"
	CastF64U32  = "cast_f64_u32"
	CastF64I64  = "cast_f64_i64"
	CastF64F16  = "cast_f64_f16"  // Available when __CUDA_ARCH__ >= 530
	CastF64BF16 = "cast_f64_bf16" // Available when __CUDA_ARCH__ >= 800

	CastU8U8   = "cast_u8_u8"
	CastU8U32  = "cast_u8_u32"
	CastU8I64  = "cast_u8_i64"
	CastU8F32  = "cast_u8_f32"
	CastU8F64  = "cast_u8_f64"
	CastU8F16  = "cast_u8_f16"  // Available when __CUDA_ARCH__ >= 530
	CastU8BF16 = "cast_u8_bf16" // Available when __CUDA_ARCH__ >= 800

	CastU32U8  = "cast_u32_u8"
	CastU32U32 = "cast_u32_u32"
	CastU32I64 = "cast_u32_i64"
	CastU32F32 = "cast_u32_f32"
	CastU32F64 = "cast_u32_f64"
	CastU32F16 = "cast_u32_f16" // Available when __CUDA_ARCH__ >= 530

	CastI64U8  = "cast_i64_u8"
	CastI64U32 = "cast_i64_u32"
	CastI64I64 = "cast_i64_i64"
	CastI64F32 = "cast_i64_f32"
	CastI64F64 = "cast_i64_f64"

	// F16 casting operations (available when __CUDA_ARCH__ >= 530)
	CastF16F16  = "cast_f16_f16"  // Available when __CUDA_ARCH__ >= 530
	CastF16U8   = "cast_f16_u8"   // Available when __CUDA_ARCH__ >= 530
	CastF16U32  = "cast_f16_u32"  // Available when __CUDA_ARCH__ >= 530
	CastF16F32  = "cast_f16_f32"  // Available when __CUDA_ARCH__ >= 530
	CastF16F64  = "cast_f16_f64"  // Available when __CUDA_ARCH__ >= 530
	CastF16BF16 = "cast_f16_bf16" // Available when __CUDA_ARCH__ >= 530

	// BF16 casting operations (available when __CUDA_ARCH__ >= 800)
	CastBF16F32 = "cast_bf16_f32" // Available when __CUDA_ARCH__ >= 800
	CastBF16U8  = "cast_bf16_u8"  // Available when __CUDA_ARCH__ >= 800
	CastBF16F16 = "cast_bf16_f16" // Available when __CUDA_ARCH__ >= 800
	CastBF16F64 = "cast_bf16_f64" // Available when __CUDA_ARCH__ >= 800

	// F8E4M3 casting operations (available when __CUDA_ARCH__ >= 890)
	CastF8E4M3F32  = "cast_f8_e4m3_f32"  // Available when __CUDA_ARCH__ >= 890
	CastF8E4M3U8   = "cast_f8_e4m3_u8"   // Available when __CUDA_ARCH__ >= 890
	CastF8E4M3F16  = "cast_f8_e4m3_f16"  // Available when __CUDA_ARCH__ >= 890
	CastF8E4M3F64  = "cast_f8_e4m3_f64"  // Available when __CUDA_ARCH__ >= 890
	CastF8E4M3I32  = "cast_f8_e4m3_i32"  // Available when __CUDA_ARCH__ >= 890
	CastF8E4M3BF16 = "cast_f8_e4m3_bf16" // Available when __CUDA_ARCH__ >= 890

	CastF32F8E4M3  = "cast_f32_f8_e4m3"  // Available when __CUDA_ARCH__ >= 890
	CastF16F8E4M3  = "cast_f16_f8_e4m3"  // Available when __CUDA_ARCH__ >= 890
	CastF64F8E4M3  = "cast_f64_f8_e4m3"  // Available when __CUDA_ARCH__ >= 890
	CastU8F8E4M3   = "cast_u8_f8_e4m3"   // Available when __CUDA_ARCH__ >= 890
	CastI32F8E4M3  = "cast_i32_f8_e4m3"  // Available when __CUDA_ARCH__ >= 890
	CastBF16F8E4M3 = "cast_bf16_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
)

// Convolution operation kernel function names
const (
	// 1D convolution operations
	Conv1DF32 = "conv1d_f32"
	Conv1DF64 = "conv1d_f64"
	Conv1DU8  = "conv1d_u8"
	Conv1DU32 = "conv1d_u32"
	Conv1DF16 = "conv1d_f16" // Available when __CUDA_ARCH__ >= 530

	// 2D convolution operations
	Conv2DF32 = "conv2d_f32"
	Conv2DF64 = "conv2d_f64"
	Conv2DU8  = "conv2d_u8"
	Conv2DU32 = "conv2d_u32"
	Conv2DF16 = "conv2d_f16" // Available when __CUDA_ARCH__ >= 530

	// 1D transposed convolution operations
	ConvTranspose1DF32 = "conv_transpose1d_f32"
	ConvTranspose1DF64 = "conv_transpose1d_f64"
	ConvTranspose1DU8  = "conv_transpose1d_u8"
	ConvTranspose1DU32 = "conv_transpose1d_u32"
	ConvTranspose1DF16 = "conv_transpose1d_f16" // Available when __CUDA_ARCH__ >= 530

	// 2D transposed convolution operations
	ConvTranspose2DF32 = "conv_transpose2d_f32"
	ConvTranspose2DF64 = "conv_transpose2d_f64"
	ConvTranspose2DU8  = "conv_transpose2d_u8"
	ConvTranspose2DU32 = "conv_transpose2d_u32"
	ConvTranspose2DF16 = "conv_transpose2d_f16" // Available when __CUDA_ARCH__ >= 530

	// Average pooling operations
	AvgPool2DF32 = "avg_pool2d_f32"
	AvgPool2DF64 = "avg_pool2d_f64"
	AvgPool2DU8  = "avg_pool2d_u8"
	AvgPool2DU32 = "avg_pool2d_u32"
	AvgPool2DF16 = "avg_pool2d_f16" // Available when __CUDA_ARCH__ >= 530

	// Max pooling operations
	MaxPool2DF32 = "max_pool2d_f32"
	MaxPool2DF64 = "max_pool2d_f64"
	MaxPool2DU8  = "max_pool2d_u8"
	MaxPool2DU32 = "max_pool2d_u32"
	MaxPool2DF16 = "max_pool2d_f16" // Available when __CUDA_ARCH__ >= 530

	// Nearest neighbor upsampling operations
	UpsampleNearest2DF32 = "upsample_nearest2d_f32"
	UpsampleNearest2DF64 = "upsample_nearest2d_f64"
	UpsampleNearest2DU8  = "upsample_nearest2d_u8"
	UpsampleNearest2DU32 = "upsample_nearest2d_u32"
	UpsampleNearest2DF16 = "upsample_nearest2d_f16" // Available when __CUDA_ARCH__ >= 530

	// Im2col operations (2D)
	Im2ColF32 = "im2col_f32"
	Im2ColF64 = "im2col_f64"
	Im2ColU8  = "im2col_u8"
	Im2ColU32 = "im2col_u32"
	Im2ColF16 = "im2col_f16" // Available when __CUDA_ARCH__ >= 530

	// Im2col operations (1D)
	Im2Col1DF32 = "im2col1d_f32"
	Im2Col1DF64 = "im2col1d_f64"
	Im2Col1DU8  = "im2col1d_u8"
	Im2Col1DU32 = "im2col1d_u32"
	Im2Col1DF16 = "im2col1d_f16" // Available when __CUDA_ARCH__ >= 530

	// Col2im operations (1D)
	Col2Im1DF32 = "col2im1d_f32"
	Col2Im1DF64 = "col2im1d_f64"
	Col2Im1DU8  = "col2im1d_u8"
	Col2Im1DU32 = "col2im1d_u32"
	Col2Im1DF16 = "col2im1d_f16" // Available when __CUDA_ARCH__ >= 530
)

// Fill operation kernel function names
const (
	// Fill operations
	FillU8  = "fill_u8"
	FillU32 = "fill_u32"
	FillI64 = "fill_i64"
	FillF32 = "fill_f32"
	FillF64 = "fill_f64"

	// Copy2D operations
	Copy2DF32 = "copy2d_f32"
	Copy2DF64 = "copy2d_f64"
	Copy2DU8  = "copy2d_u8"
	Copy2DU32 = "copy2d_u32"
	Copy2DI64 = "copy2d_i64"

	// Constant set operations
	ConstSetU8  = "const_set_u8"
	ConstSetU32 = "const_set_u32"
	ConstSetI64 = "const_set_i64"
	ConstSetF32 = "const_set_f32"
	ConstSetF64 = "const_set_f64"
)

// Indexing operation kernel function names
const (
	// Index select operations with different index types and data types
	IndexSelectU32F32 = "is_u32_f32"
	IndexSelectU32F64 = "is_u32_f64"
	IndexSelectU32U8  = "is_u32_u8"
	IndexSelectU32U32 = "is_u32_u32"
	IndexSelectU32I64 = "is_u32_i64"

	IndexSelectU8F32 = "is_u8_f32"
	IndexSelectU8F64 = "is_u8_f64"
	IndexSelectU8U8  = "is_u8_u8"
	IndexSelectU8U32 = "is_u8_u32"
	IndexSelectU8I64 = "is_u8_i64"

	IndexSelectI32F32 = "is_i32_f32"
	IndexSelectI32F64 = "is_i32_f64"
	IndexSelectI32U8  = "is_i32_u8"
	IndexSelectI32U32 = "is_i32_u32"
	IndexSelectI32I64 = "is_i32_i64"

	IndexSelectI64F32 = "is_i64_f32"
	IndexSelectI64F64 = "is_i64_f64"
	IndexSelectI64U8  = "is_i64_u8"
	IndexSelectI64U32 = "is_i64_u32"
	IndexSelectI64I64 = "is_i64_i64"

	// Gather operations with different index types and data types
	GatherU32F32 = "gather_u32_f32"
	GatherU32F64 = "gather_u32_f64"
	GatherU32U8  = "gather_u32_u8"
	GatherU32U32 = "gather_u32_u32"
	GatherU32I64 = "gather_u32_i64"

	GatherU8F32 = "gather_u8_f32"
	GatherU8F64 = "gather_u8_f64"
	GatherU8U8  = "gather_u8_u8"
	GatherU8U32 = "gather_u8_u32"
	GatherU8I64 = "gather_u8_i64"

	GatherI32F32 = "gather_i32_f32"
	GatherI32F64 = "gather_i32_f64"
	GatherI32U8  = "gather_i32_u8"
	GatherI32U32 = "gather_i32_u32"
	GatherI32I64 = "gather_i32_i64"

	GatherI64F32 = "gather_i64_f32"
	GatherI64F64 = "gather_i64_f64"
	GatherI64U8  = "gather_i64_u8"
	GatherI64U32 = "gather_i64_u32"
	GatherI64I64 = "gather_i64_i64"

	// Scatter operations with different index types and data types
	ScatterU32F32 = "scatter_u32_f32"
	ScatterU32F64 = "scatter_u32_f64"
	ScatterU32U8  = "scatter_u32_u8"
	ScatterU32U32 = "scatter_u32_u32"
	ScatterU32I64 = "scatter_u32_i64"

	ScatterU8F32 = "scatter_u8_f32"
	ScatterU8F64 = "scatter_u8_f64"
	ScatterU8U8  = "scatter_u8_u8"
	ScatterU8U32 = "scatter_u8_u32"
	ScatterU8I64 = "scatter_u8_i64"

	ScatterI32F32 = "scatter_i32_f32"
	ScatterI32F64 = "scatter_i32_f64"
	ScatterI32U8  = "scatter_i32_u8"
	ScatterI32U32 = "scatter_i32_u32"
	ScatterI32I64 = "scatter_i32_i64"

	ScatterI64F32 = "scatter_i64_f32"
	ScatterI64F64 = "scatter_i64_f64"
	ScatterI64U8  = "scatter_i64_u8"
	ScatterI64U32 = "scatter_i64_u32"
	ScatterI64I64 = "scatter_i64_i64"

	// Index add operations with different index types and data types
	IndexAddU32F32 = "index_add_u32_f32"
	IndexAddU32F64 = "index_add_u32_f64"
	IndexAddU32U8  = "index_add_u32_u8"
	IndexAddU32U32 = "index_add_u32_u32"
	IndexAddU32I64 = "index_add_u32_i64"

	IndexAddU8F32 = "index_add_u8_f32"
	IndexAddU8F64 = "index_add_u8_f64"
	IndexAddU8U8  = "index_add_u8_u8"
	IndexAddU8U32 = "index_add_u8_u32"
	IndexAddU8I64 = "index_add_u8_i64"

	IndexAddI32F32 = "index_add_i32_f32"
	IndexAddI32F64 = "index_add_i32_f64"
	IndexAddI32U8  = "index_add_i32_u8"
	IndexAddI32U32 = "index_add_i32_u32"
	IndexAddI32I64 = "index_add_i32_i64"

	IndexAddI64F32 = "index_add_i64_f32"
	IndexAddI64F64 = "index_add_i64_f64"
	IndexAddI64U8  = "index_add_i64_u8"
	IndexAddI64U32 = "index_add_i64_u32"
	IndexAddI64I64 = "index_add_i64_i64"
)

// Quantized operation kernel function names (adapted from llama.cpp)
const (
	// 4-bit quantization operations
	QuantizeQ4_0   = "quantize_q4_0"
	QuantizeQ4_1   = "quantize_q4_1"
	DequantizeQ4_0 = "dequantize_q4_0"
	DequantizeQ4_1 = "dequantize_q4_1"

	// 5-bit quantization operations
	QuantizeQ5_0   = "quantize_q5_0"
	QuantizeQ5_1   = "quantize_q5_1"
	DequantizeQ5_0 = "dequantize_q5_0"
	DequantizeQ5_1 = "dequantize_q5_1"

	// 8-bit quantization operations
	QuantizeQ8_0   = "quantize_q8_0"
	QuantizeQ8_1   = "quantize_q8_1"
	DequantizeQ8_0 = "dequantize_q8_0"
	DequantizeQ8_1 = "dequantize_q8_1"

	// K-quantization operations (super-block quantization)
	QuantizeQ2_K = "quantize_q2_k"
	QuantizeQ3_K = "quantize_q3_k"
	QuantizeQ4_K = "quantize_q4_k"
	QuantizeQ5_K = "quantize_q5_k"
	QuantizeQ6_K = "quantize_q6_k"
	QuantizeQ8_K = "quantize_q8_k"

	DequantizeQ2_K = "dequantize_q2_k"
	DequantizeQ3_K = "dequantize_q3_k"
	DequantizeQ4_K = "dequantize_q4_k"
	DequantizeQ5_K = "dequantize_q5_k"
	DequantizeQ6_K = "dequantize_q6_k"
	DequantizeQ8_K = "dequantize_q8_k"

	// Quantized matrix multiplication operations
	MulMatQ4_0 = "mul_mat_q4_0"
	MulMatQ4_1 = "mul_mat_q4_1"
	MulMatQ5_0 = "mul_mat_q5_0"
	MulMatQ5_1 = "mul_mat_q5_1"
	MulMatQ8_0 = "mul_mat_q8_0"
	MulMatQ8_1 = "mul_mat_q8_1"

	// Quantized matrix multiplication with K-quantization
	MulMatQ2_K = "mul_mat_q2_k"
	MulMatQ3_K = "mul_mat_q3_k"
	MulMatQ4_K = "mul_mat_q4_k"
	MulMatQ5_K = "mul_mat_q5_k"
	MulMatQ6_K = "mul_mat_q6_k"
	MulMatQ8_K = "mul_mat_q8_k"

	// Vector-matrix multiplication operations
	MulMatVecQ4_0 = "mul_mat_vec_q4_0"
	MulMatVecQ4_1 = "mul_mat_vec_q4_1"
	MulMatVecQ5_0 = "mul_mat_vec_q5_0"
	MulMatVecQ5_1 = "mul_mat_vec_q5_1"
	MulMatVecQ8_0 = "mul_mat_vec_q8_0"
	MulMatVecQ8_1 = "mul_mat_vec_q8_1"

	// Vector-matrix multiplication with K-quantization
	MulMatVecQ2_K = "mul_mat_vec_q2_k"
	MulMatVecQ3_K = "mul_mat_vec_q3_k"
	MulMatVecQ4_K = "mul_mat_vec_q4_k"
	MulMatVecQ5_K = "mul_mat_vec_q5_k"
	MulMatVecQ6_K = "mul_mat_vec_q6_k"
	MulMatVecQ8_K = "mul_mat_vec_q8_k"

	// Mixed precision quantized operations
	MulMatVecQ4_0_F32 = "mul_mat_vec_q4_0_f32"
	MulMatVecQ4_1_F32 = "mul_mat_vec_q4_1_f32"
	MulMatVecQ8_0_F32 = "mul_mat_vec_q8_0_f32"
	MulMatVecQ4_0_F16 = "mul_mat_vec_q4_0_f16"
	MulMatVecQ4_1_F16 = "mul_mat_vec_q4_1_f16"
	MulMatVecQ8_0_F16 = "mul_mat_vec_q8_0_f16"

	// Quantization utility operations
	QuantizeRowQ4_0 = "quantize_row_q4_0"
	QuantizeRowQ4_1 = "quantize_row_q4_1"
	QuantizeRowQ8_0 = "quantize_row_q8_0"
	QuantizeRowQ8_1 = "quantize_row_q8_1"

	DequantizeRowQ4_0 = "dequantize_row_q4_0"
	DequantizeRowQ4_1 = "dequantize_row_q4_1"
	DequantizeRowQ8_0 = "dequantize_row_q8_0"
	DequantizeRowQ8_1 = "dequantize_row_q8_1"

	// Block-wise quantization operations
	QuantizeBlockQ4_0 = "quantize_block_q4_0"
	QuantizeBlockQ4_1 = "quantize_block_q4_1"
	QuantizeBlockQ8_0 = "quantize_block_q8_0"

	DequantizeBlockQ4_0 = "dequantize_block_q4_0"
	DequantizeBlockQ4_1 = "dequantize_block_q4_1"
	DequantizeBlockQ8_0 = "dequantize_block_q8_0"
)

// Reduce operation kernel function names
const (
	// Sum reduction operations
	SumF32 = "sum_f32"
	SumF64 = "sum_f64"
	SumU8  = "sum_u8"
	SumU32 = "sum_u32"
	SumI64 = "sum_i64"
	SumF16 = "sum_f16" // Available when __CUDA_ARCH__ >= 530

	// Fast sum operations (optimized for contiguous data)
	FastSumF32 = "fast_sum_f32"
	FastSumF64 = "fast_sum_f64"
	FastSumF16 = "fast_sum_f16" // Available when __CUDA_ARCH__ >= 530

	// Min reduction operations
	MinF32 = "min_f32"
	MinF64 = "min_f64"
	MinU8  = "min_u8"
	MinU32 = "min_u32"
	MinI64 = "min_i64"
	MinF16 = "min_f16" // Available when __CUDA_ARCH__ >= 530

	// Max reduction operations
	MaxF32 = "max_f32"
	MaxF64 = "max_f64"
	MaxU8  = "max_u8"
	MaxU32 = "max_u32"
	MaxI64 = "max_i64"
	MaxF16 = "max_f16" // Available when __CUDA_ARCH__ >= 530

	// ArgMin reduction operations (return indices)
	ArgMinF32 = "argmin_f32"
	ArgMinF64 = "argmin_f64"
	ArgMinU8  = "argmin_u8"
	ArgMinU32 = "argmin_u32"
	ArgMinI64 = "argmin_i64"
	ArgMinF16 = "argmin_f16" // Available when __CUDA_ARCH__ >= 530

	// ArgMax reduction operations (return indices)
	ArgMaxF32 = "argmax_f32"
	ArgMaxF64 = "argmax_f64"
	ArgMaxU8  = "argmax_u8"
	ArgMaxU32 = "argmax_u32"
	ArgMaxI64 = "argmax_i64"
	ArgMaxF16 = "argmax_f16" // Available when __CUDA_ARCH__ >= 530

	// Softmax operations
	SoftmaxF32 = "softmax_f32"
	SoftmaxF64 = "softmax_f64"
	SoftmaxF16 = "softmax_f16" // Available when __CUDA_ARCH__ >= 530

	// LogSoftmax operations
	LogSoftmaxF32 = "log_softmax_f32"
	LogSoftmaxF64 = "log_softmax_f64"
	LogSoftmaxF16 = "log_softmax_f16" // Available when __CUDA_ARCH__ >= 530

	// Layer normalization operations
	LayerNormF32 = "layernorm_f32"
	LayerNormF64 = "layernorm_f64"
	LayerNormF16 = "layernorm_f16" // Available when __CUDA_ARCH__ >= 530

	// RMS normalization operations
	RMSNormF32 = "rmsnorm_f32"
	RMSNormF64 = "rmsnorm_f64"
	RMSNormF16 = "rmsnorm_f16" // Available when __CUDA_ARCH__ >= 530

	// Group normalization operations
	GroupNormF32 = "groupnorm_f32"
	GroupNormF64 = "groupnorm_f64"
	GroupNormF16 = "groupnorm_f16" // Available when __CUDA_ARCH__ >= 530

	// Variance reduction operations
	VarF32 = "var_f32"
	VarF64 = "var_f64"
	VarF16 = "var_f16" // Available when __CUDA_ARCH__ >= 530

	// Standard deviation operations
	StdF32 = "std_f32"
	StdF64 = "std_f64"
	StdF16 = "std_f16" // Available when __CUDA_ARCH__ >= 530

	// Mean reduction operations
	MeanF32 = "mean_f32"
	MeanF64 = "mean_f64"
	MeanF16 = "mean_f16" // Available when __CUDA_ARCH__ >= 530

	// L1 norm operations
	L1NormF32 = "l1_norm_f32"
	L1NormF64 = "l1_norm_f64"
	L1NormF16 = "l1_norm_f16" // Available when __CUDA_ARCH__ >= 530

	// L2 norm operations
	L2NormF32 = "l2_norm_f32"
	L2NormF64 = "l2_norm_f64"
	L2NormF16 = "l2_norm_f16" // Available when __CUDA_ARCH__ >= 530

	// LogSumExp operations
	LogSumExpF32 = "logsumexp_f32"
	LogSumExpF64 = "logsumexp_f64"
	LogSumExpF16 = "logsumexp_f16" // Available when __CUDA_ARCH__ >= 530

	// Welford variance operations (numerically stable)
	WelfordVarF32 = "welford_var_f32"
	WelfordVarF64 = "welford_var_f64"
	WelfordVarF16 = "welford_var_f16" // Available when __CUDA_ARCH__ >= 530
)

// Sort operation kernel function names (adapted from llama.cpp)
const (
	// Ascending argsort operations
	ArgSortAscF32  = "asort_asc_f32"
	ArgSortAscF64  = "asort_asc_f64"
	ArgSortAscU8   = "asort_asc_u8"
	ArgSortAscU32  = "asort_asc_u32"
	ArgSortAscI64  = "asort_asc_i64"
	ArgSortAscF16  = "asort_asc_f16"  // Available when __CUDA_ARCH__ >= 530
	ArgSortAscBF16 = "asort_asc_bf16" // Available when __CUDA_ARCH__ >= 800

	// Descending argsort operations
	ArgSortDescF32  = "asort_desc_f32"
	ArgSortDescF64  = "asort_desc_f64"
	ArgSortDescU8   = "asort_desc_u8"
	ArgSortDescU32  = "asort_desc_u32"
	ArgSortDescI64  = "asort_desc_i64"
	ArgSortDescF16  = "asort_desc_f16"  // Available when __CUDA_ARCH__ >= 530
	ArgSortDescBF16 = "asort_desc_bf16" // Available when __CUDA_ARCH__ >= 800
)

// Ternary operation kernel function names (where/select operations)
const (
	// Where operations with uint32_t condition
	WhereU32F32  = "where_u32_f32"
	WhereU32F64  = "where_u32_f64"
	WhereU32U8   = "where_u32_u8"
	WhereU32U32  = "where_u32_u32"
	WhereU32I64  = "where_u32_i64"
	WhereU32F16  = "where_u32_f16"  // Available when __CUDA_ARCH__ >= 530
	WhereU32BF16 = "where_u32_bf16" // Available when __CUDA_ARCH__ >= 800

	// Where operations with uint8_t condition
	WhereU8F32  = "where_u8_f32"
	WhereU8F64  = "where_u8_f64"
	WhereU8U8   = "where_u8_u8"
	WhereU8U32  = "where_u8_u32"
	WhereU8I64  = "where_u8_i64"
	WhereU8F16  = "where_u8_f16"  // Available when __CUDA_ARCH__ >= 530
	WhereU8BF16 = "where_u8_bf16" // Available when __CUDA_ARCH__ >= 800

	// Where operations with int64_t condition
	WhereI64F32  = "where_i64_f32"
	WhereI64F64  = "where_i64_f64"
	WhereI64U8   = "where_i64_u8"
	WhereI64U32  = "where_i64_u32"
	WhereI64I64  = "where_i64_i64"
	WhereI64F16  = "where_i64_f16"  // Available when __CUDA_ARCH__ >= 530
	WhereI64BF16 = "where_i64_bf16" // Available when __CUDA_ARCH__ >= 800

	// Where operations with int32_t condition
	WhereI32F32  = "where_i32_f32"
	WhereI32F64  = "where_i32_f64"
	WhereI32U8   = "where_i32_u8"
	WhereI32U32  = "where_i32_u32"
	WhereI32I64  = "where_i32_i64"
	WhereI32F16  = "where_i32_f16"  // Available when __CUDA_ARCH__ >= 530
	WhereI32BF16 = "where_i32_bf16" // Available when __CUDA_ARCH__ >= 800
)

// Unary operation kernel function names
const (
	// Basic mathematical operations
	UExpF32  = "uexp_f32"
	UExpF64  = "uexp_f64"
	UExpF16  = "uexp_f16"  // Available when __CUDA_ARCH__ >= 530
	UExpBF16 = "uexp_bf16" // Available when __CUDA_ARCH__ >= 800

	ULogF32  = "ulog_f32"
	ULogF64  = "ulog_f64"
	ULogF16  = "ulog_f16"  // Available when __CUDA_ARCH__ >= 530
	ULogBF16 = "ulog_bf16" // Available when __CUDA_ARCH__ >= 800

	USinF32  = "usin_f32"
	USinF64  = "usin_f64"
	USinF16  = "usin_f16"  // Available when __CUDA_ARCH__ >= 530
	USinBF16 = "usin_bf16" // Available when __CUDA_ARCH__ >= 800

	UCosF32  = "ucos_f32"
	UCosF64  = "ucos_f64"
	UCosF16  = "ucos_f16"  // Available when __CUDA_ARCH__ >= 530
	UCosBF16 = "ucos_bf16" // Available when __CUDA_ARCH__ >= 800

	UTanhF32  = "utanh_f32"
	UTanhF64  = "utanh_f64"
	UTanhF16  = "utanh_f16"  // Available when __CUDA_ARCH__ >= 530
	UTanhBF16 = "utanh_bf16" // Available when __CUDA_ARCH__ >= 800

	USqrtF32  = "usqrt_f32"
	USqrtF64  = "usqrt_f64"
	USqrtF16  = "usqrt_f16"  // Available when __CUDA_ARCH__ >= 530
	USqrtBF16 = "usqrt_bf16" // Available when __CUDA_ARCH__ >= 800

	// Activation functions
	UReluF32  = "urelu_f32"
	UReluF64  = "urelu_f64"
	UReluF16  = "urelu_f16"  // Available when __CUDA_ARCH__ >= 530
	UReluBF16 = "urelu_bf16" // Available when __CUDA_ARCH__ >= 800

	UGeluF32  = "ugelu_f32"
	UGeluF64  = "ugelu_f64"
	UGeluF16  = "ugelu_f16"  // Available when __CUDA_ARCH__ >= 530
	UGeluBF16 = "ugelu_bf16" // Available when __CUDA_ARCH__ >= 800

	UGeluErfF32  = "ugelu_erf_f32"
	UGeluErfF64  = "ugelu_erf_f64"
	UGeluErfF16  = "ugelu_erf_f16"  // Available when __CUDA_ARCH__ >= 530
	UGeluErfBF16 = "ugelu_erf_bf16" // Available when __CUDA_ARCH__ >= 800

	USiluF32  = "usilu_f32"
	USiluF64  = "usilu_f64"
	USiluF16  = "usilu_f16"  // Available when __CUDA_ARCH__ >= 530
	USiluBF16 = "usilu_bf16" // Available when __CUDA_ARCH__ >= 800

	USigmoidF32  = "usigmoid_f32"
	USigmoidF64  = "usigmoid_f64"
	USigmoidF16  = "usigmoid_f16"  // Available when __CUDA_ARCH__ >= 530
	USigmoidBF16 = "usigmoid_bf16" // Available when __CUDA_ARCH__ >= 800

	UEluF32  = "uelu_f32"
	UEluF64  = "uelu_f64"
	UEluF16  = "uelu_f16"  // Available when __CUDA_ARCH__ >= 530
	UEluBF16 = "uelu_bf16" // Available when __CUDA_ARCH__ >= 800

	// Basic operations
	UAbsF32  = "uabs_f32"
	UAbsF64  = "uabs_f64"
	UAbsI32  = "uabs_i32"
	UAbsI64  = "uabs_i64"
	UAbsF16  = "uabs_f16"  // Available when __CUDA_ARCH__ >= 530
	UAbsBF16 = "uabs_bf16" // Available when __CUDA_ARCH__ >= 800

	UNegF32  = "uneg_f32"
	UNegF64  = "uneg_f64"
	UNegI32  = "uneg_i32"
	UNegI64  = "uneg_i64"
	UNegF16  = "uneg_f16"  // Available when __CUDA_ARCH__ >= 530
	UNegBF16 = "uneg_bf16" // Available when __CUDA_ARCH__ >= 800

	USqrF32  = "usqr_f32"
	USqrF64  = "usqr_f64"
	USqrF16  = "usqr_f16"  // Available when __CUDA_ARCH__ >= 530
	USqrBF16 = "usqr_bf16" // Available when __CUDA_ARCH__ >= 800

	URecipF32  = "urecip_f32"
	URecipF64  = "urecip_f64"
	URecipF16  = "urecip_f16"  // Available when __CUDA_ARCH__ >= 530
	URecipBF16 = "urecip_bf16" // Available when __CUDA_ARCH__ >= 800

	// Copy operations
	UCopyF32    = "ucopy_f32"
	UCopyF64    = "ucopy_f64"
	UCopyU8     = "ucopy_u8"
	UCopyU32    = "ucopy_u32"
	UCopyI32    = "ucopy_i32"
	UCopyI64    = "ucopy_i64"
	UCopyF16    = "ucopy_f16"     // Available when __CUDA_ARCH__ >= 530
	UCopyBF16   = "ucopy_bf16"    // Available when __CUDA_ARCH__ >= 800
	UCopyF8E4M3 = "ucopy_f8_e4m3" // Available when __CUDA_ARCH__ >= 890

	// Error function
	UErfF32  = "uerf_f32"
	UErfF64  = "uerf_f64"
	UErfF16  = "uerf_f16"  // Available when __CUDA_ARCH__ >= 530
	UErfBF16 = "uerf_bf16" // Available when __CUDA_ARCH__ >= 800
)

//go:embed *.ptx
var Kernels embed.FS

// KernelFile represents available CUDA kernel files
type KernelFile string

const (
	// Core operation kernels
	AffineKernel    KernelFile = "affine"
	BinaryKernel    KernelFile = "binary"
	CastKernel      KernelFile = "cast"
	ConvKernel      KernelFile = "conv"
	FillKernel      KernelFile = "fill"
	IndexingKernel  KernelFile = "indexing"
	QuantizedKernel KernelFile = "quantized"
	ReduceKernel    KernelFile = "reduce"
	SortKernel      KernelFile = "sort"
	TernaryKernel   KernelFile = "ternary"
	UnaryKernel     KernelFile = "unary"
)

// String returns the kernel file name
func (k KernelFile) String() string {
	return string(k)
}

// AllKernels returns all predefined kernel file enums
func AllKernels() []KernelFile {
	return []KernelFile{
		AffineKernel,
		BinaryKernel,
		CastKernel,
		ConvKernel,
		FillKernel,
		IndexingKernel,
		QuantizedKernel,
		ReduceKernel,
		SortKernel,
		TernaryKernel,
		UnaryKernel,
	}
}

// GetKernel loads a CUDA kernel file by enum
func GetKernel(kernel KernelFile) ([]byte, error) {
	filename := string(kernel) + ".ptx"
	return Kernels.ReadFile(filename)
}

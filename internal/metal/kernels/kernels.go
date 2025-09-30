package kernels

import "embed"

// Affine kernel function names
const (
	AffineU8          = "affine_u8"
	AffineU8Strided   = "affine_u8_strided"
	AffineU32         = "affine_u32"
	AffineU32Strided  = "affine_u32_strided"
	AffineF32         = "affine_f32"
	AffineF32Strided  = "affine_f32_strided"
	AffineF16         = "affine_f16"
	AffineF16Strided  = "affine_f16_strided"
	AffineBF16        = "affine_bf16" // Available when __HAVE_BFLOAT__ is defined
	AffineBF16Strided = "affine_bf16_strided"

	// Power function kernel names
	PowfF32         = "powf_f32"
	PowfF32Strided  = "powf_f32_strided"
	PowfF16         = "powf_f16"
	PowfF16Strided  = "powf_f16_strided"
	PowfBF16        = "powf_bf16" // Available when __HAVE_BFLOAT__ is defined
	PowfBF16Strided = "powf_bf16_strided"

	// ELU (Exponential Linear Unit) kernel names
	EluF32         = "elu_f32"
	EluF32Strided  = "elu_f32_strided"
	EluF16         = "elu_f16"
	EluF16Strided  = "elu_f16_strided"
	EluBF16        = "elu_bf16" // Available when __HAVE_BFLOAT__ is defined
	EluBF16Strided = "elu_bf16_strided"
)

// Binary arithmetic operations (add, sub, mul, div, min, max)
const (
	// Add operations
	AddF32         = "add_f32"
	AddF32Strided  = "add_f32_strided"
	AddF16         = "add_f16"
	AddF16Strided  = "add_f16_strided"
	AddU32         = "add_u32"
	AddU32Strided  = "add_u32_strided"
	AddU8          = "add_u8"
	AddU8Strided   = "add_u8_strided"
	AddI64         = "add_i64" // Available when __METAL_VERSION__ >= 220
	AddI64Strided  = "add_i64_strided"
	AddBF16        = "add_bf16" // Available when __HAVE_BFLOAT__ is defined
	AddBF16Strided = "add_bf16_strided"

	// Sub operations
	SubF32         = "sub_f32"
	SubF32Strided  = "sub_f32_strided"
	SubF16         = "sub_f16"
	SubF16Strided  = "sub_f16_strided"
	SubU32         = "sub_u32"
	SubU32Strided  = "sub_u32_strided"
	SubU8          = "sub_u8"
	SubU8Strided   = "sub_u8_strided"
	SubI64         = "sub_i64" // Available when __METAL_VERSION__ >= 220
	SubI64Strided  = "sub_i64_strided"
	SubBF16        = "sub_bf16" // Available when __HAVE_BFLOAT__ is defined
	SubBF16Strided = "sub_bf16_strided"

	// Mul operations
	MulF32         = "mul_f32"
	MulF32Strided  = "mul_f32_strided"
	MulF16         = "mul_f16"
	MulF16Strided  = "mul_f16_strided"
	MulU32         = "mul_u32"
	MulU32Strided  = "mul_u32_strided"
	MulU8          = "mul_u8"
	MulU8Strided   = "mul_u8_strided"
	MulI64         = "mul_i64" // Available when __METAL_VERSION__ >= 220
	MulI64Strided  = "mul_i64_strided"
	MulBF16        = "mul_bf16" // Available when __HAVE_BFLOAT__ is defined
	MulBF16Strided = "mul_bf16_strided"

	// Div operations
	DivF32         = "div_f32"
	DivF32Strided  = "div_f32_strided"
	DivF16         = "div_f16"
	DivF16Strided  = "div_f16_strided"
	DivU32         = "div_u32"
	DivU32Strided  = "div_u32_strided"
	DivU8          = "div_u8"
	DivU8Strided   = "div_u8_strided"
	DivI64         = "div_i64" // Available when __METAL_VERSION__ >= 220
	DivI64Strided  = "div_i64_strided"
	DivBF16        = "div_bf16" // Available when __HAVE_BFLOAT__ is defined
	DivBF16Strided = "div_bf16_strided"

	// Min operations
	MinF32         = "min_f32"
	MinF32Strided  = "min_f32_strided"
	MinF16         = "min_f16"
	MinF16Strided  = "min_f16_strided"
	MinU32         = "min_u32"
	MinU32Strided  = "min_u32_strided"
	MinU8          = "min_u8"
	MinU8Strided   = "min_u8_strided"
	MinI64         = "min_i64" // Available when __METAL_VERSION__ >= 220
	MinI64Strided  = "min_i64_strided"
	MinBF16        = "min_bf16" // Available when __HAVE_BFLOAT__ is defined
	MinBF16Strided = "min_bf16_strided"

	// Max operations
	MaxF32         = "max_f32"
	MaxF32Strided  = "max_f32_strided"
	MaxF16         = "max_f16"
	MaxF16Strided  = "max_f16_strided"
	MaxU32         = "max_u32"
	MaxU32Strided  = "max_u32_strided"
	MaxU8          = "max_u8"
	MaxU8Strided   = "max_u8_strided"
	MaxI64         = "max_i64" // Available when __METAL_VERSION__ >= 220
	MaxI64Strided  = "max_i64_strided"
	MaxBF16        = "max_bf16" // Available when __HAVE_BFLOAT__ is defined
	MaxBF16Strided = "max_bf16_strided"
)

// Binary comparison operations (output uint8_t)
const (
	// Equal operations
	EqF32         = "eq_f32"
	EqF32Strided  = "eq_f32_strided"
	EqF16         = "eq_f16"
	EqF16Strided  = "eq_f16_strided"
	EqU32         = "eq_u32"
	EqU32Strided  = "eq_u32_strided"
	EqU8          = "eq_u8"
	EqU8Strided   = "eq_u8_strided"
	EqI64         = "eq_i64" // Available when __METAL_VERSION__ >= 220
	EqI64Strided  = "eq_i64_strided"
	EqBF16        = "eq_bf16" // Available when __HAVE_BFLOAT__ is defined
	EqBF16Strided = "eq_bf16_strided"

	// Not equal operations
	NeF32         = "ne_f32"
	NeF32Strided  = "ne_f32_strided"
	NeF16         = "ne_f16"
	NeF16Strided  = "ne_f16_strided"
	NeU32         = "ne_u32"
	NeU32Strided  = "ne_u32_strided"
	NeU8          = "ne_u8"
	NeU8Strided   = "ne_u8_strided"
	NeI64         = "ne_i64" // Available when __METAL_VERSION__ >= 220
	NeI64Strided  = "ne_i64_strided"
	NeBF16        = "ne_bf16" // Available when __HAVE_BFLOAT__ is defined
	NeBF16Strided = "ne_bf16_strided"

	// Less than or equal operations
	LeF32         = "le_f32"
	LeF32Strided  = "le_f32_strided"
	LeF16         = "le_f16"
	LeF16Strided  = "le_f16_strided"
	LeU32         = "le_u32"
	LeU32Strided  = "le_u32_strided"
	LeU8          = "le_u8"
	LeU8Strided   = "le_u8_strided"
	LeI64         = "le_i64" // Available when __METAL_VERSION__ >= 220
	LeI64Strided  = "le_i64_strided"
	LeBF16        = "le_bf16" // Available when __HAVE_BFLOAT__ is defined
	LeBF16Strided = "le_bf16_strided"

	// Less than operations
	LtF32         = "lt_f32"
	LtF32Strided  = "lt_f32_strided"
	LtF16         = "lt_f16"
	LtF16Strided  = "lt_f16_strided"
	LtU32         = "lt_u32"
	LtU32Strided  = "lt_u32_strided"
	LtU8          = "lt_u8"
	LtU8Strided   = "lt_u8_strided"
	LtI64         = "lt_i64" // Available when __METAL_VERSION__ >= 220
	LtI64Strided  = "lt_i64_strided"
	LtBF16        = "lt_bf16" // Available when __HAVE_BFLOAT__ is defined
	LtBF16Strided = "lt_bf16_strided"

	// Greater than or equal operations
	GeF32         = "ge_f32"
	GeF32Strided  = "ge_f32_strided"
	GeF16         = "ge_f16"
	GeF16Strided  = "ge_f16_strided"
	GeU32         = "ge_u32"
	GeU32Strided  = "ge_u32_strided"
	GeU8          = "ge_u8"
	GeU8Strided   = "ge_u8_strided"
	GeI64         = "ge_i64" // Available when __METAL_VERSION__ >= 220
	GeI64Strided  = "ge_i64_strided"
	GeBF16        = "ge_bf16" // Available when __HAVE_BFLOAT__ is defined
	GeBF16Strided = "ge_bf16_strided"

	// Greater than operations
	GtF32         = "gt_f32"
	GtF32Strided  = "gt_f32_strided"
	GtF16         = "gt_f16"
	GtF16Strided  = "gt_f16_strided"
	GtU32         = "gt_u32"
	GtU32Strided  = "gt_u32_strided"
	GtU8          = "gt_u8"
	GtU8Strided   = "gt_u8_strided"
	GtI64         = "gt_i64" // Available when __METAL_VERSION__ >= 220
	GtI64Strided  = "gt_i64_strided"
	GtBF16        = "gt_bf16" // Available when __HAVE_BFLOAT__ is defined
	GtBF16Strided = "gt_bf16_strided"
)

// Type casting operation kernel names
const (
	// u32 casting operations
	CastU32F32         = "cast_u32_f32"
	CastU32F32Strided  = "cast_u32_f32_strided"
	CastU32U8          = "cast_u32_u8"
	CastU32U8Strided   = "cast_u32_u8_strided"
	CastU32F16         = "cast_u32_f16"
	CastU32F16Strided  = "cast_u32_f16_strided"
	CastU32I64         = "cast_u32_i64" // Available when __METAL_VERSION__ >= 220
	CastU32I64Strided  = "cast_u32_i64_strided"
	CastU32BF16        = "cast_u32_bf16" // Available when __HAVE_BFLOAT__ is defined
	CastU32BF16Strided = "cast_u32_bf16_strided"

	// u8 casting operations
	CastU8U32         = "cast_u8_u32"
	CastU8U32Strided  = "cast_u8_u32_strided"
	CastU8F32         = "cast_u8_f32"
	CastU8F32Strided  = "cast_u8_f32_strided"
	CastU8F16         = "cast_u8_f16"
	CastU8F16Strided  = "cast_u8_f16_strided"
	CastU8I64         = "cast_u8_i64" // Available when __METAL_VERSION__ >= 220
	CastU8I64Strided  = "cast_u8_i64_strided"
	CastU8BF16        = "cast_u8_bf16" // Available when __HAVE_BFLOAT__ is defined
	CastU8BF16Strided = "cast_u8_bf16_strided"

	// f16 casting operations
	CastF16F32         = "cast_f16_f32"
	CastF16F32Strided  = "cast_f16_f32_strided"
	CastF16U8          = "cast_f16_u8"
	CastF16U8Strided   = "cast_f16_u8_strided"
	CastF16U32         = "cast_f16_u32"
	CastF16U32Strided  = "cast_f16_u32_strided"
	CastF16I64         = "cast_f16_i64"
	CastF16I64Strided  = "cast_f16_i64_strided"
	CastF16BF16        = "cast_f16_bf16" // Available when __HAVE_BFLOAT__ is defined
	CastF16BF16Strided = "cast_f16_bf16_strided"

	// i64 casting operations
	CastI64F32         = "cast_i64_f32"
	CastI64F32Strided  = "cast_i64_f32_strided"
	CastI64U8          = "cast_i64_u8"
	CastI64U8Strided   = "cast_i64_u8_strided"
	CastI64U32         = "cast_i64_u32"
	CastI64U32Strided  = "cast_i64_u32_strided"
	CastI64F16         = "cast_i64_f16"
	CastI64F16Strided  = "cast_i64_f16_strided"
	CastI64BF16        = "cast_i64_bf16" // Available when __HAVE_BFLOAT__ is defined
	CastI64BF16Strided = "cast_i64_bf16_strided"

	// f32 casting operations
	CastF32F16         = "cast_f32_f16"
	CastF32F16Strided  = "cast_f32_f16_strided"
	CastF32U32         = "cast_f32_u32"
	CastF32U32Strided  = "cast_f32_u32_strided"
	CastF32U8          = "cast_f32_u8"
	CastF32U8Strided   = "cast_f32_u8_strided"
	CastF32I64         = "cast_f32_i64"
	CastF32I64Strided  = "cast_f32_i64_strided"
	CastF32BF16        = "cast_f32_bf16" // Available when __HAVE_BFLOAT__ is defined
	CastF32BF16Strided = "cast_f32_bf16_strided"

	// bf16 casting operations
	CastBF16U32        = "cast_bf16_u32" // Available when __HAVE_BFLOAT__ is defined
	CastBF16U32Strided = "cast_bf16_u32_strided"
	CastBF16U8         = "cast_bf16_u8"
	CastBF16U8Strided  = "cast_bf16_u8_strided"
	CastBF16F32        = "cast_bf16_f32"
	CastBF16F32Strided = "cast_bf16_f32_strided"
	CastBF16F16        = "cast_bf16_f16"
	CastBF16F16Strided = "cast_bf16_f16_strided"
	CastBF16I64        = "cast_bf16_i64"
	CastBF16I64Strided = "cast_bf16_i64_strided"
)

// Convolution operation kernel names
const (
	// 2D im2col operations
	Im2ColF32  = "im2col_f32"
	Im2ColF16  = "im2col_f16"
	Im2ColU32  = "im2col_u32"
	Im2ColU8   = "im2col_u8"
	Im2ColI64  = "im2col_i64"  // Available when __METAL_VERSION__ >= 220
	Im2ColBF16 = "im2col_bf16" // Available when __HAVE_BFLOAT__ is defined

	// 1D im2col operations
	Im2Col1DF32  = "im2col1d_f32"
	Im2Col1DF16  = "im2col1d_f16"
	Im2Col1DU32  = "im2col1d_u32"
	Im2Col1DU8   = "im2col1d_u8"
	Im2Col1DI64  = "im2col1d_i64"  // Available when __METAL_VERSION__ >= 220
	Im2Col1DBF16 = "im2col1d_bf16" // Available when __HAVE_BFLOAT__ is defined

	// 1D col2im operations
	Col2Im1DF32  = "col2im1d_f32"
	Col2Im1DF16  = "col2im1d_f16"
	Col2Im1DU32  = "col2im1d_u32"
	Col2Im1DU8   = "col2im1d_u8"
	Col2Im1DI64  = "col2im1d_i64"  // Available when __METAL_VERSION__ >= 220
	Col2Im1DBF16 = "col2im1d_bf16" // Available when __HAVE_BFLOAT__ is defined

	// 2D convolution operations
	Conv2DF32  = "conv2d_f32"
	Conv2DF16  = "conv2d_f16"
	Conv2DU32  = "conv2d_u32"
	Conv2DU8   = "conv2d_u8"
	Conv2DI64  = "conv2d_i64"  // Available when __METAL_VERSION__ >= 220
	Conv2DBF16 = "conv2d_bf16" // Available when __HAVE_BFLOAT__ is defined

	// 1D convolution operations
	Conv1DF32  = "conv1d_f32"
	Conv1DF16  = "conv1d_f16"
	Conv1DU32  = "conv1d_u32"
	Conv1DU8   = "conv1d_u8"
	Conv1DI64  = "conv1d_i64"  // Available when __METAL_VERSION__ >= 220
	Conv1DBF16 = "conv1d_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Transposed 2D convolution operations
	ConvTranspose2DF32  = "conv_transpose2d_f32"
	ConvTranspose2DF16  = "conv_transpose2d_f16"
	ConvTranspose2DU32  = "conv_transpose2d_u32"
	ConvTranspose2DU8   = "conv_transpose2d_u8"
	ConvTranspose2DI64  = "conv_transpose2d_i64"  // Available when __METAL_VERSION__ >= 220
	ConvTranspose2DBF16 = "conv_transpose2d_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Transposed 1D convolution operations
	ConvTranspose1DF32  = "conv_transpose1d_f32"
	ConvTranspose1DF16  = "conv_transpose1d_f16"
	ConvTranspose1DU32  = "conv_transpose1d_u32"
	ConvTranspose1DU8   = "conv_transpose1d_u8"
	ConvTranspose1DI64  = "conv_transpose1d_i64"  // Available when __METAL_VERSION__ >= 220
	ConvTranspose1DBF16 = "conv_transpose1d_bf16" // Available when __HAVE_BFLOAT__ is defined
)

// Fill operation kernel names
const (
	FillU8   = "fill_u8"
	FillU32  = "fill_u32"
	FillI64  = "fill_i64"
	FillF16  = "fill_f16"
	FillF32  = "fill_f32"
	FillBF16 = "fill_bf16" // Available when __METAL_VERSION__ >= 310
)

// Indexing operation kernel names
const (
	// Index operations with different index types and data types
	IndexU32U8   = "index_u32_u8"
	IndexU32U32  = "index_u32_u32"
	IndexU32F16  = "index_u32_f16"
	IndexU32F32  = "index_u32_f32"
	IndexU32I64  = "index_u32_i64"
	IndexU32BF16 = "index_u32_bf16" // Available when __HAVE_BFLOAT__ is defined

	IndexU8U8   = "index_u8_u8"
	IndexU8U32  = "index_u8_u32"
	IndexU8F16  = "index_u8_f16"
	IndexU8F32  = "index_u8_f32"
	IndexU8I64  = "index_u8_i64"
	IndexU8BF16 = "index_u8_bf16" // Available when __HAVE_BFLOAT__ is defined

	IndexI64U8   = "index_i64_u8"
	IndexI64U32  = "index_i64_u32"
	IndexI64F16  = "index_i64_f16"
	IndexI64F32  = "index_i64_f32"
	IndexI64I64  = "index_i64_i64"
	IndexI64BF16 = "index_i64_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Gather operations with different index types and data types
	GatherU32U8   = "gather_u32_u8"
	GatherU32U32  = "gather_u32_u32"
	GatherU32F16  = "gather_u32_f16"
	GatherU32F32  = "gather_u32_f32"
	GatherU32I64  = "gather_u32_i64"
	GatherU32BF16 = "gather_u32_bf16" // Available when __HAVE_BFLOAT__ is defined

	GatherU8U8   = "gather_u8_u8"
	GatherU8U32  = "gather_u8_u32"
	GatherU8F16  = "gather_u8_f16"
	GatherU8F32  = "gather_u8_f32"
	GatherU8I64  = "gather_u8_i64"
	GatherU8BF16 = "gather_u8_bf16" // Available when __HAVE_BFLOAT__ is defined

	GatherI64U8   = "gather_i64_u8"
	GatherI64U32  = "gather_i64_u32"
	GatherI64F16  = "gather_i64_f16"
	GatherI64F32  = "gather_i64_f32"
	GatherI64I64  = "gather_i64_i64"
	GatherI64BF16 = "gather_i64_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Scatter operations with different index types and data types
	ScatterU32U8   = "scatter_u32_u8"
	ScatterU32U32  = "scatter_u32_u32"
	ScatterU32F16  = "scatter_u32_f16"
	ScatterU32F32  = "scatter_u32_f32"
	ScatterU32I64  = "scatter_u32_i64"
	ScatterU32BF16 = "scatter_u32_bf16" // Available when __HAVE_BFLOAT__ is defined

	ScatterU8U8   = "scatter_u8_u8"
	ScatterU8U32  = "scatter_u8_u32"
	ScatterU8F16  = "scatter_u8_f16"
	ScatterU8F32  = "scatter_u8_f32"
	ScatterU8I64  = "scatter_u8_i64"
	ScatterU8BF16 = "scatter_u8_bf16" // Available when __HAVE_BFLOAT__ is defined

	ScatterI64U8   = "scatter_i64_u8"
	ScatterI64U32  = "scatter_i64_u32"
	ScatterI64F16  = "scatter_i64_f16"
	ScatterI64F32  = "scatter_i64_f32"
	ScatterI64I64  = "scatter_i64_i64"
	ScatterI64BF16 = "scatter_i64_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Index add operations with different index types and data types
	IndexAddU32U8   = "index_add_u32_u8"
	IndexAddU32U32  = "index_add_u32_u32"
	IndexAddU32F16  = "index_add_u32_f16"
	IndexAddU32F32  = "index_add_u32_f32"
	IndexAddU32I64  = "index_add_u32_i64"
	IndexAddU32BF16 = "index_add_u32_bf16" // Available when __HAVE_BFLOAT__ is defined

	IndexAddU8U8   = "index_add_u8_u8"
	IndexAddU8U32  = "index_add_u8_u32"
	IndexAddU8F16  = "index_add_u8_f16"
	IndexAddU8F32  = "index_add_u8_f32"
	IndexAddU8I64  = "index_add_u8_i64"
	IndexAddU8BF16 = "index_add_u8_bf16" // Available when __HAVE_BFLOAT__ is defined

	IndexAddI64U8   = "index_add_i64_u8"
	IndexAddI64U32  = "index_add_i64_u32"
	IndexAddI64F16  = "index_add_i64_f16"
	IndexAddI64F32  = "index_add_i64_f32"
	IndexAddI64I64  = "index_add_i64_i64"
	IndexAddI64BF16 = "index_add_i64_bf16" // Available when __HAVE_BFLOAT__ is defined
)

// GEMM (General Matrix Multiply) operation kernel names from MLX
const (
	// Basic GEMM operations
	GemmF32  = "gemm_f32"
	GemmF16  = "gemm_f16"
	GemmBF16 = "gemm_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Split-K GEMM operations for large matrices
	GemmSplitKF32  = "gemm_splitk_f32"
	GemmSplitKF16  = "gemm_splitk_f16"
	GemmSplitKBF16 = "gemm_splitk_bf16" // Available when __HAVE_BFLOAT__ is defined

	// GEMM with addition (C = alpha * A * B + beta * C)
	GemmAddMMF32  = "gemm_addmm_f32"
	GemmAddMMF16  = "gemm_addmm_f16"
	GemmAddMMBF16 = "gemm_addmm_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Batched GEMM operations
	BatchedGemmF32  = "batched_gemm_f32"
	BatchedGemmF16  = "batched_gemm_f16"
	BatchedGemmBF16 = "batched_gemm_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Optimized GEMM variants with different tile sizes
	GemmTiled16x16F32 = "gemm_tiled_16x16_f32"
	GemmTiled16x16F16 = "gemm_tiled_16x16_f16"
	GemmTiled32x32F32 = "gemm_tiled_32x32_f32"
	GemmTiled32x32F16 = "gemm_tiled_32x32_f16"
	GemmTiled64x64F32 = "gemm_tiled_64x64_f32"
	GemmTiled64x64F16 = "gemm_tiled_64x64_f16"

	// Transposed GEMM operations
	GemmTransAF32  = "gemm_trans_a_f32"
	GemmTransAF16  = "gemm_trans_a_f16"
	GemmTransBF32  = "gemm_trans_b_f32"
	GemmTransBF16  = "gemm_trans_b_f16"
	GemmTransABF32 = "gemm_trans_ab_f32"
	GemmTransABF16 = "gemm_trans_ab_f16"

	// Specialized GEMM operations
	GemmSteelF32  = "gemm_steel_f32" // MLX Steel optimized version
	GemmSteelF16  = "gemm_steel_f16"
	GemmSteelBF16 = "gemm_steel_bf16"
)

// Sort operation kernel names from MLX
const (
	// Basic sort operations
	SortF32  = "sort_f32"
	SortF16  = "sort_f16"
	SortU32  = "sort_u32"
	SortU8   = "sort_u8"
	SortI32  = "sort_i32"
	SortI64  = "sort_i64"
	SortBF16 = "sort_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Argsort operations (return indices of sorted elements)
	ArgSortF32  = "argsort_f32"
	ArgSortF16  = "argsort_f16"
	ArgSortU32  = "argsort_u32"
	ArgSortU8   = "argsort_u8"
	ArgSortI32  = "argsort_i32"
	ArgSortI64  = "argsort_i64"
	ArgSortBF16 = "argsort_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Partial sort operations (top-k)
	PartialSortF32  = "partial_sort_f32"
	PartialSortF16  = "partial_sort_f16"
	PartialSortU32  = "partial_sort_u32"
	PartialSortU8   = "partial_sort_u8"
	PartialSortI32  = "partial_sort_i32"
	PartialSortI64  = "partial_sort_i64"
	PartialSortBF16 = "partial_sort_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Merge sort operations
	MergeSortF32  = "merge_sort_f32"
	MergeSortF16  = "merge_sort_f16"
	MergeSortU32  = "merge_sort_u32"
	MergeSortU8   = "merge_sort_u8"
	MergeSortI32  = "merge_sort_i32"
	MergeSortI64  = "merge_sort_i64"
	MergeSortBF16 = "merge_sort_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Radix sort operations (for integer types)
	RadixSortU32 = "radix_sort_u32"
	RadixSortU8  = "radix_sort_u8"
	RadixSortI32 = "radix_sort_i32"
	RadixSortI64 = "radix_sort_i64"

	// Bitonic sort operations (for small arrays)
	BitonicSortF32  = "bitonic_sort_f32"
	BitonicSortF16  = "bitonic_sort_f16"
	BitonicSortU32  = "bitonic_sort_u32"
	BitonicSortU8   = "bitonic_sort_u8"
	BitonicSortI32  = "bitonic_sort_i32"
	BitonicSortI64  = "bitonic_sort_i64"
	BitonicSortBF16 = "bitonic_sort_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Stable sort operations
	StableSortF32  = "stable_sort_f32"
	StableSortF16  = "stable_sort_f16"
	StableSortU32  = "stable_sort_u32"
	StableSortU8   = "stable_sort_u8"
	StableSortI32  = "stable_sort_i32"
	StableSortI64  = "stable_sort_i64"
	StableSortBF16 = "stable_sort_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Multi-dimensional sort operations
	SortMultiDimF32  = "sort_multidim_f32"
	SortMultiDimF16  = "sort_multidim_f16"
	SortMultiDimU32  = "sort_multidim_u32"
	SortMultiDimU8   = "sort_multidim_u8"
	SortMultiDimI32  = "sort_multidim_i32"
	SortMultiDimI64  = "sort_multidim_i64"
	SortMultiDimBF16 = "sort_multidim_bf16" // Available when __HAVE_BFLOAT__ is defined
)

// Quantized operation kernel names
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

	// Ternary quantization operations
	QuantizeTQ1_0   = "quantize_tq1_0" // 1.6875 bpw
	QuantizeTQ2_0   = "quantize_tq2_0" // 2.0625 bpw
	DequantizeTQ1_0 = "dequantize_tq1_0"
	DequantizeTQ2_0 = "dequantize_tq2_0"

	// Super-block quantization operations
	QuantizeQ2_K = "quantize_q2_k" // 2.625 bpw
	QuantizeQ3_K = "quantize_q3_k" // 3.4375 bpw
	QuantizeQ4_K = "quantize_q4_k" // 4.5 bpw
	QuantizeQ5_K = "quantize_q5_k" // 5.5 bpw
	QuantizeQ6_K = "quantize_q6_k" // 6.5625 bpw
	QuantizeQ8_K = "quantize_q8_k" // 8.5 bpw

	DequantizeQ2_K = "dequantize_q2_k"
	DequantizeQ3_K = "dequantize_q3_k"
	DequantizeQ4_K = "dequantize_q4_k"
	DequantizeQ5_K = "dequantize_q5_k"
	DequantizeQ6_K = "dequantize_q6_k"
	DequantizeQ8_K = "dequantize_q8_k"

	// Packed quantization operations (multiple blocks)
	QuantizeQ4_0x4 = "quantize_q4_0x4" // 4 blocks
	QuantizeQ4_0x8 = "quantize_q4_0x8" // 8 blocks
	QuantizeQ8_0x4 = "quantize_q8_0x4" // 4 blocks
	QuantizeQ8_0x8 = "quantize_q8_0x8" // 8 blocks

	DequantizeQ4_0x4 = "dequantize_q4_0x4"
	DequantizeQ4_0x8 = "dequantize_q4_0x8"
	DequantizeQ8_0x4 = "dequantize_q8_0x4"
	DequantizeQ8_0x8 = "dequantize_q8_0x8"

	// Quantized matrix multiplication operations
	QGemmQ4_0 = "qgemm_q4_0"
	QGemmQ4_1 = "qgemm_q4_1"
	QGemmQ5_0 = "qgemm_q5_0"
	QGemmQ5_1 = "qgemm_q5_1"
	QGemmQ8_0 = "qgemm_q8_0"
	QGemmQ8_1 = "qgemm_q8_1"

	// Quantized matrix multiplication with K-quantization
	QGemmQ2_K = "qgemm_q2_k"
	QGemmQ3_K = "qgemm_q3_k"
	QGemmQ4_K = "qgemm_q4_k"
	QGemmQ5_K = "qgemm_q5_k"
	QGemmQ6_K = "qgemm_q6_k"
	QGemmQ8_K = "qgemm_q8_k"

	// Mixed precision quantized operations
	QGemmF16Q4_0 = "qgemm_f16_q4_0"
	QGemmF16Q4_1 = "qgemm_f16_q4_1"
	QGemmF16Q8_0 = "qgemm_f16_q8_0"
	QGemmF32Q4_0 = "qgemm_f32_q4_0"
	QGemmF32Q4_1 = "qgemm_f32_q4_1"
	QGemmF32Q8_0 = "qgemm_f32_q8_0"

	// Quantization utility operations
	QuantizeRowwise      = "quantize_rowwise"
	QuantizeColumnwise   = "quantize_columnwise"
	DequantizeRowwise    = "dequantize_rowwise"
	DequantizeColumnwise = "dequantize_columnwise"

	// Dynamic quantization operations
	DynamicQuantizeF32   = "dynamic_quantize_f32"
	DynamicQuantizeF16   = "dynamic_quantize_f16"
	DynamicDequantizeF32 = "dynamic_dequantize_f32"
	DynamicDequantizeF16 = "dynamic_dequantize_f16"
)

// Random number generation kernel names
const (
	// Uniform random number generation
	RandomUniformF32 = "random_uniform_f32"
	RandomUniformF16 = "random_uniform_f16"
	RandomUniformU32 = "random_uniform_u32"
	RandomUniformU8  = "random_uniform_u8"
	RandomUniformI32 = "random_uniform_i32"
	RandomUniformI64 = "random_uniform_i64"

	// Normal (Gaussian) random number generation
	RandomNormalF32 = "random_normal_f32"
	RandomNormalF16 = "random_normal_f16"

	// Bernoulli random number generation
	RandomBernoulliF32 = "random_bernoulli_f32"
	RandomBernoulliF16 = "random_bernoulli_f16"

	// Exponential random number generation
	RandomExponentialF32 = "random_exponential_f32"
	RandomExponentialF16 = "random_exponential_f16"

	// Gamma random number generation
	RandomGammaF32 = "random_gamma_f32"
	RandomGammaF16 = "random_gamma_f16"

	// Beta random number generation
	RandomBetaF32 = "random_beta_f32"
	RandomBetaF16 = "random_beta_f16"

	// Categorical random number generation
	RandomCategoricalU32 = "random_categorical_u32"
	RandomCategoricalU8  = "random_categorical_u8"

	// Multinomial random number generation
	RandomMultinomialU32 = "random_multinomial_u32"
	RandomMultinomialU8  = "random_multinomial_u8"

	// Poisson random number generation
	RandomPoissonU32 = "random_poisson_u32"
	RandomPoissonU8  = "random_poisson_u8"

	// Random sampling operations
	RandomSampleF32 = "random_sample_f32"
	RandomSampleF16 = "random_sample_f16"
	RandomSampleU32 = "random_sample_u32"
	RandomSampleU8  = "random_sample_u8"

	// Random permutation operations
	RandomPermuteU32 = "random_permute_u32"
	RandomPermuteU8  = "random_permute_u8"
	RandomPermuteI32 = "random_permute_i32"
	RandomPermuteI64 = "random_permute_i64"

	// Random shuffle operations
	RandomShuffleF32 = "random_shuffle_f32"
	RandomShuffleF16 = "random_shuffle_f16"
	RandomShuffleU32 = "random_shuffle_u32"
	RandomShuffleU8  = "random_shuffle_u8"

	// Dropout operations
	DropoutF32 = "dropout_f32"
	DropoutF16 = "dropout_f16"

	// Random mask generation
	RandomMaskU8  = "random_mask_u8"
	RandomMaskU32 = "random_mask_u32"

	// Seed initialization operations
	InitSeedBuffer   = "init_seed_buffer"
	UpdateSeedBuffer = "update_seed_buffer"
	GetRandomSeed    = "get_random_seed"

	// HybridTaus RNG operations
	HybridTausInit    = "hybrid_taus_init"
	HybridTausRand    = "hybrid_taus_rand"
	HybridTausUniform = "hybrid_taus_uniform"
	HybridTausNormal  = "hybrid_taus_normal"

	// Atomic seed operations
	AtomicLoadSeed   = "atomic_load_seed"
	AtomicStoreSeed  = "atomic_store_seed"
	AtomicUpdateSeed = "atomic_update_seed"
)

// Reduce operation kernel names
const (
	// Sum reduction operations
	ReduceSumF32  = "reduce_sum_f32"
	ReduceSumF16  = "reduce_sum_f16"
	ReduceSumU32  = "reduce_sum_u32"
	ReduceSumU8   = "reduce_sum_u8"
	ReduceSumI32  = "reduce_sum_i32"
	ReduceSumI64  = "reduce_sum_i64"
	ReduceSumBF16 = "reduce_sum_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Mean reduction operations
	ReduceMeanF32  = "reduce_mean_f32"
	ReduceMeanF16  = "reduce_mean_f16"
	ReduceMeanBF16 = "reduce_mean_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Max reduction operations
	ReduceMaxF32  = "reduce_max_f32"
	ReduceMaxF16  = "reduce_max_f16"
	ReduceMaxU32  = "reduce_max_u32"
	ReduceMaxU8   = "reduce_max_u8"
	ReduceMaxI32  = "reduce_max_i32"
	ReduceMaxI64  = "reduce_max_i64"
	ReduceMaxBF16 = "reduce_max_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Min reduction operations
	ReduceMinF32  = "reduce_min_f32"
	ReduceMinF16  = "reduce_min_f16"
	ReduceMinU32  = "reduce_min_u32"
	ReduceMinU8   = "reduce_min_u8"
	ReduceMinI32  = "reduce_min_i32"
	ReduceMinI64  = "reduce_min_i64"
	ReduceMinBF16 = "reduce_min_bf16" // Available when __HAVE_BFLOAT__ is defined

	// ArgMax reduction operations (return indices)
	ReduceArgMaxF32  = "reduce_argmax_f32"
	ReduceArgMaxF16  = "reduce_argmax_f16"
	ReduceArgMaxU32  = "reduce_argmax_u32"
	ReduceArgMaxU8   = "reduce_argmax_u8"
	ReduceArgMaxI32  = "reduce_argmax_i32"
	ReduceArgMaxI64  = "reduce_argmax_i64"
	ReduceArgMaxBF16 = "reduce_argmax_bf16" // Available when __HAVE_BFLOAT__ is defined

	// ArgMin reduction operations (return indices)
	ReduceArgMinF32  = "reduce_argmin_f32"
	ReduceArgMinF16  = "reduce_argmin_f16"
	ReduceArgMinU32  = "reduce_argmin_u32"
	ReduceArgMinU8   = "reduce_argmin_u8"
	ReduceArgMinI32  = "reduce_argmin_i32"
	ReduceArgMinI64  = "reduce_argmin_i64"
	ReduceArgMinBF16 = "reduce_argmin_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Product reduction operations
	ReduceProdF32  = "reduce_prod_f32"
	ReduceProdF16  = "reduce_prod_f16"
	ReduceProdU32  = "reduce_prod_u32"
	ReduceProdU8   = "reduce_prod_u8"
	ReduceProdI32  = "reduce_prod_i32"
	ReduceProdI64  = "reduce_prod_i64"
	ReduceProdBF16 = "reduce_prod_bf16" // Available when __HAVE_BFLOAT__ is defined

	// All reduction operations (logical AND)
	ReduceAllU8  = "reduce_all_u8"
	ReduceAllU32 = "reduce_all_u32"

	// Any reduction operations (logical OR)
	ReduceAnyU8  = "reduce_any_u8"
	ReduceAnyU32 = "reduce_any_u32"

	// Variance reduction operations
	ReduceVarF32  = "reduce_var_f32"
	ReduceVarF16  = "reduce_var_f16"
	ReduceVarBF16 = "reduce_var_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Standard deviation reduction operations
	ReduceStdF32  = "reduce_std_f32"
	ReduceStdF16  = "reduce_std_f16"
	ReduceStdBF16 = "reduce_std_bf16" // Available when __HAVE_BFLOAT__ is defined

	// L1 norm reduction operations
	ReduceL1NormF32  = "reduce_l1_norm_f32"
	ReduceL1NormF16  = "reduce_l1_norm_f16"
	ReduceL1NormBF16 = "reduce_l1_norm_bf16" // Available when __HAVE_BFLOAT__ is defined

	// L2 norm reduction operations
	ReduceL2NormF32  = "reduce_l2_norm_f32"
	ReduceL2NormF16  = "reduce_l2_norm_f16"
	ReduceL2NormBF16 = "reduce_l2_norm_bf16" // Available when __HAVE_BFLOAT__ is defined

	// LogSumExp reduction operations
	ReduceLogSumExpF32  = "reduce_logsumexp_f32"
	ReduceLogSumExpF16  = "reduce_logsumexp_f16"
	ReduceLogSumExpBF16 = "reduce_logsumexp_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Softmax reduction operations
	SoftmaxF32  = "softmax_f32"
	SoftmaxF16  = "softmax_f16"
	SoftmaxBF16 = "softmax_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Multi-dimensional reduction operations
	ReduceMultiDimSumF32 = "reduce_multidim_sum_f32"
	ReduceMultiDimSumF16 = "reduce_multidim_sum_f16"
	ReduceMultiDimMaxF32 = "reduce_multidim_max_f32"
	ReduceMultiDimMaxF16 = "reduce_multidim_max_f16"
	ReduceMultiDimMinF32 = "reduce_multidim_min_f32"
	ReduceMultiDimMinF16 = "reduce_multidim_min_f16"

	// Strided reduction operations
	ReduceStridedSumF32 = "reduce_strided_sum_f32"
	ReduceStridedSumF16 = "reduce_strided_sum_f16"
	ReduceStridedMaxF32 = "reduce_strided_max_f32"
	ReduceStridedMaxF16 = "reduce_strided_max_f16"
	ReduceStridedMinF32 = "reduce_strided_min_f32"
	ReduceStridedMinF16 = "reduce_strided_min_f16"
)

// Scaled Dot Product Attention kernel names from MLX
const (
	// Basic SDPA operations
	SDPAF32  = "sdpa_f32"
	SDPAF16  = "sdpa_f16"
	SDPABF16 = "sdpa_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Vector SDPA operations (optimized for small dimensions)
	SDPAVectorF32  = "sdpa_vector_f32"
	SDPAVectorF16  = "sdpa_vector_f16"
	SDPAVectorBF16 = "sdpa_vector_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Fast attention operations (MLX optimized)
	FastAttentionF32  = "fast_attention_f32"
	FastAttentionF16  = "fast_attention_f16"
	FastAttentionBF16 = "fast_attention_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Multi-head attention operations
	MultiHeadAttentionF32  = "multi_head_attention_f32"
	MultiHeadAttentionF16  = "multi_head_attention_f16"
	MultiHeadAttentionBF16 = "multi_head_attention_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Grouped query attention operations
	GroupedQueryAttentionF32  = "grouped_query_attention_f32"
	GroupedQueryAttentionF16  = "grouped_query_attention_f16"
	GroupedQueryAttentionBF16 = "grouped_query_attention_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Causal attention operations (with causal mask)
	CausalAttentionF32  = "causal_attention_f32"
	CausalAttentionF16  = "causal_attention_f16"
	CausalAttentionBF16 = "causal_attention_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Attention with custom mask operations
	MaskedAttentionF32  = "masked_attention_f32"
	MaskedAttentionF16  = "masked_attention_f16"
	MaskedAttentionBF16 = "masked_attention_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Flash attention operations (memory efficient)
	FlashAttentionF32  = "flash_attention_f32"
	FlashAttentionF16  = "flash_attention_f16"
	FlashAttentionBF16 = "flash_attention_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Attention backward operations
	AttentionBackwardF32  = "attention_backward_f32"
	AttentionBackwardF16  = "attention_backward_f16"
	AttentionBackwardBF16 = "attention_backward_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Attention with RoPE (Rotary Position Embedding)
	AttentionRoPEF32  = "attention_rope_f32"
	AttentionRoPEF16  = "attention_rope_f16"
	AttentionRoPEBF16 = "attention_rope_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Attention with ALiBi (Attention with Linear Biases)
	AttentionALiBiF32  = "attention_alibi_f32"
	AttentionALiBiF16  = "attention_alibi_f16"
	AttentionALiBiBF16 = "attention_alibi_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Sparse attention operations
	SparseAttentionF32  = "sparse_attention_f32"
	SparseAttentionF16  = "sparse_attention_f16"
	SparseAttentionBF16 = "sparse_attention_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Local attention operations (sliding window)
	LocalAttentionF32  = "local_attention_f32"
	LocalAttentionF16  = "local_attention_f16"
	LocalAttentionBF16 = "local_attention_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Cross attention operations
	CrossAttentionF32  = "cross_attention_f32"
	CrossAttentionF16  = "cross_attention_f16"
	CrossAttentionBF16 = "cross_attention_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Attention with soft capping
	SoftCappedAttentionF32  = "soft_capped_attention_f32"
	SoftCappedAttentionF16  = "soft_capped_attention_f16"
	SoftCappedAttentionBF16 = "soft_capped_attention_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Attention utility operations
	AttentionScoresF32  = "attention_scores_f32"
	AttentionScoresF16  = "attention_scores_f16"
	AttentionWeightsF32 = "attention_weights_f32"
	AttentionWeightsF16 = "attention_weights_f16"
	AttentionSoftmaxF32 = "attention_softmax_f32"
	AttentionSoftmaxF16 = "attention_softmax_f16"
)

// Sort operation kernel names (from llama.cpp)
const (
	// Ascending argsort operations
	ArgSortAscF32  = "asort_asc_f32"
	ArgSortAscF16  = "asort_asc_f16"
	ArgSortAscU8   = "asort_asc_u8"
	ArgSortAscU32  = "asort_asc_u32"
	ArgSortAscI64  = "asort_asc_i64"  // Available when __METAL_VERSION__ >= 220
	ArgSortAscBF16 = "asort_asc_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Descending argsort operations
	ArgSortDescF32  = "asort_desc_f32"
	ArgSortDescF16  = "asort_desc_f16"
	ArgSortDescU8   = "asort_desc_u8"
	ArgSortDescU32  = "asort_desc_u32"
	ArgSortDescI64  = "asort_desc_i64"  // Available when __METAL_VERSION__ >= 220
	ArgSortDescBF16 = "asort_desc_bf16" // Available when __HAVE_BFLOAT__ is defined
)

// Ternary operation kernel names (where/select operations)
const (
	// Where operations with uint32_t condition
	WhereU32F16  = "where_u32_f16"
	WhereU32F32  = "where_u32_f32"
	WhereU32U8   = "where_u32_u8"
	WhereU32U32  = "where_u32_u32"
	WhereU32I64  = "where_u32_i64"  // Available when __METAL_VERSION__ >= 220
	WhereU32BF16 = "where_u32_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Where operations with uint8_t condition
	WhereU8F16  = "where_u8_f16"
	WhereU8F32  = "where_u8_f32"
	WhereU8U8   = "where_u8_u8"
	WhereU8U32  = "where_u8_u32"
	WhereU8I64  = "where_u8_i64"  // Available when __METAL_VERSION__ >= 220
	WhereU8BF16 = "where_u8_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Where operations with int64_t condition
	WhereI64F16  = "where_i64_f16"  // Available when __METAL_VERSION__ >= 220
	WhereI64F32  = "where_i64_f32"  // Available when __METAL_VERSION__ >= 220
	WhereI64U8   = "where_i64_u8"   // Available when __METAL_VERSION__ >= 220
	WhereI64U32  = "where_i64_u32"  // Available when __METAL_VERSION__ >= 220
	WhereI64I64  = "where_i64_i64"  // Available when __METAL_VERSION__ >= 220
	WhereI64BF16 = "where_i64_bf16" // Available when __METAL_VERSION__ >= 220 and __HAVE_BFLOAT__
)

// Unary operation kernel names
const (
	// Basic unary operations
	UnaryAbsF32  = "uabs_f32"
	UnaryAbsF16  = "uabs_f16"
	UnaryAbsU32  = "uabs_u32"
	UnaryAbsU8   = "uabs_u8"
	UnaryAbsI64  = "uabs_i64"  // Available when __METAL_VERSION__ >= 220
	UnaryAbsBF16 = "uabs_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Strided versions
	UnaryAbsF32Strided  = "uabs_f32_strided"
	UnaryAbsF16Strided  = "uabs_f16_strided"
	UnaryAbsU32Strided  = "uabs_u32_strided"
	UnaryAbsU8Strided   = "uabs_u8_strided"
	UnaryAbsI64Strided  = "uabs_i64_strided"  // Available when __METAL_VERSION__ >= 220
	UnaryAbsBF16Strided = "uabs_bf16_strided" // Available when __HAVE_BFLOAT__ is defined

	// Tiled versions
	UnaryAbsF32Tiled = "uabs_f32_tiled"
	UnaryAbsF16Tiled = "uabs_f16_tiled"
	UnaryAbsU32Tiled = "uabs_u32_tiled"
	UnaryAbsU8Tiled  = "uabs_u8_tiled"

	// Activation functions
	UReluF32  = "urelu_f32"
	UReluF16  = "urelu_f16"
	UReluU32  = "urelu_u32"
	UReluU8   = "urelu_u8"
	UReluBF16 = "urelu_bf16" // Available when __HAVE_BFLOAT__ is defined

	UReluF32Strided  = "urelu_f32_strided"
	UReluF16Strided  = "urelu_f16_strided"
	UReluU32Strided  = "urelu_u32_strided"
	UReluU8Strided   = "urelu_u8_strided"
	UReluBF16Strided = "urelu_bf16_strided"

	// GELU activation
	UGeluF32  = "ugelu_f32"
	UGeluF16  = "ugelu_f16"
	UGeluBF16 = "ugelu_bf16" // Available when __HAVE_BFLOAT__ is defined

	UGeluF32Strided  = "ugelu_f32_strided"
	UGeluF16Strided  = "ugelu_f16_strided"
	UGeluBF16Strided = "ugelu_bf16_strided"

	// GELU ERF version
	UGeluErfF32  = "ugelu_erf_f32"
	UGeluErfF16  = "ugelu_erf_f16"
	UGeluErfBF16 = "ugelu_erf_bf16" // Available when __HAVE_BFLOAT__ is defined

	UGeluErfF32Strided  = "ugelu_erf_f32_strided"
	UGeluErfF16Strided  = "ugelu_erf_f16_strided"
	UGeluErfBF16Strided = "ugelu_erf_bf16_strided"

	// SiLU (Swish) activation
	USiluF32  = "usilu_f32"
	USiluF16  = "usilu_f16"
	USiluBF16 = "usilu_bf16" // Available when __HAVE_BFLOAT__ is defined

	USiluF32Strided  = "usilu_f32_strided"
	USiluF16Strided  = "usilu_f16_strided"
	USiluBF16Strided = "usilu_bf16_strided"

	// Sigmoid activation
	USigmoidF32  = "usigmoid_f32"
	USigmoidF16  = "usigmoid_f16"
	USigmoidBF16 = "usigmoid_bf16" // Available when __HAVE_BFLOAT__ is defined

	USigmoidF32Strided  = "usigmoid_f32_strided"
	USigmoidF16Strided  = "usigmoid_f16_strided"
	USigmoidBF16Strided = "usigmoid_bf16_strided"

	// Mathematical functions
	UExpF32  = "uexp_f32"
	UExpF16  = "uexp_f16"
	UExpBF16 = "uexp_bf16" // Available when __HAVE_BFLOAT__ is defined

	ULogF32  = "ulog_f32"
	ULogF16  = "ulog_f16"
	ULogBF16 = "ulog_bf16" // Available when __HAVE_BFLOAT__ is defined

	USinF32 = "usin_f32"
	USinF16 = "usin_f16"
	UCosF32 = "ucos_f32"
	UCosF16 = "ucos_f16"

	UTanhF32  = "utanh_f32"
	UTanhF16  = "utanh_f16"
	UTanhBF16 = "utanh_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Square and reciprocal
	USqrF32  = "usqr_f32"
	USqrF16  = "usqr_f16"
	USqrBF16 = "usqr_bf16" // Available when __HAVE_BFLOAT__ is defined

	URecipF32  = "urecip_f32"
	URecipF16  = "urecip_f16"
	URecipBF16 = "urecip_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Negation
	UNegF32  = "uneg_f32"
	UNegF16  = "uneg_f16"
	UNegI32  = "uneg_i32"
	UNegI64  = "uneg_i64"  // Available when __METAL_VERSION__ >= 220
	UNegBF16 = "uneg_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Error function
	UErfF32  = "uerf_f32"
	UErfF16  = "uerf_f16"
	UErfBF16 = "uerf_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Identity (copy)
	UIdF32  = "uid_f32"
	UIdF16  = "uid_f16"
	UIdU32  = "uid_u32"
	UIdU8   = "uid_u8"
	UIdI64  = "uid_i64"  // Available when __METAL_VERSION__ >= 220
	UIdBF16 = "uid_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Constant set operations
	ConstSetF32  = "const_set_f32"
	ConstSetF16  = "const_set_f16"
	ConstSetU32  = "const_set_u32"
	ConstSetU8   = "const_set_u8"
	ConstSetI64  = "const_set_i64"  // Available when __METAL_VERSION__ >= 220
	ConstSetBF16 = "const_set_bf16" // Available when __HAVE_BFLOAT__ is defined

	// Strided constant set operations
	ConstSetF32Strided  = "const_set_f32_strided"
	ConstSetF16Strided  = "const_set_f16_strided"
	ConstSetU32Strided  = "const_set_u32_strided"
	ConstSetU8Strided   = "const_set_u8_strided"
	ConstSetI64Strided  = "const_set_i64_strided"  // Available when __METAL_VERSION__ >= 220
	ConstSetBF16Strided = "const_set_bf16_strided" // Available when __HAVE_BFLOAT__ is defined

	// Tiled constant set operations
	ConstSetF32Tiled = "const_set_f32_tiled"
	ConstSetF16Tiled = "const_set_f16_tiled"
	ConstSetU32Tiled = "const_set_u32_tiled"
	ConstSetU8Tiled  = "const_set_u8_tiled"
)

//go:embed *.metal
var Kernels embed.FS

// KernelFile represents available Metal kernel files
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
	RandomKernel    KernelFile = "random"
	ReduceKernel    KernelFile = "reduce"
	SortKernel      KernelFile = "sort"
	TernaryKernel   KernelFile = "ternary"
	UnaryKernel     KernelFile = "unary"

	// Advanced kernels
	MLXGemmKernel KernelFile = "mlx_gemm"
	MLXSortKernel KernelFile = "mlx_sort"
	SDPAKernel    KernelFile = "scaled_dot_product_attention"

	// Utility kernels
	UtilsKernel KernelFile = "utils"
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
		RandomKernel,
		ReduceKernel,
		SortKernel,
		TernaryKernel,
		UnaryKernel,
		MLXGemmKernel,
		MLXSortKernel,
		SDPAKernel,
		UtilsKernel,
	}
}

// GetKernel loads a Metal kernel file by enum
func GetKernel(kernel KernelFile) ([]byte, error) {
	filename := string(kernel) + ".metal"
	return Kernels.ReadFile(filename)
}

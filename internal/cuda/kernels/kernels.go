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
	BAddU8     = "badd_u8"
	BAddU32    = "badd_u32"
	BAddI64    = "badd_i64"
	BAddF16    = "badd_f16"     // Available when __CUDA_ARCH__ >= 530
	BAddBF16   = "badd_bf16"    // Available when __CUDA_ARCH__ >= 800
	BAddF8E4M3 = "badd_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	BSubF32    = "bsub_f32"
	BSubF64    = "bsub_f64"
	BSubU8     = "bsub_u8"
	BSubU32    = "bsub_u32"
	BSubI64    = "bsub_i64"
	BSubF16    = "bsub_f16"     // Available when __CUDA_ARCH__ >= 530
	BSubBF16   = "bsub_bf16"    // Available when __CUDA_ARCH__ >= 800
	BSubF8E4M3 = "bsub_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	BMulF32    = "bmul_f32"
	BMulF64    = "bmul_f64"
	BMulU8     = "bmul_u8"
	BMulU32    = "bmul_u32"
	BMulI64    = "bmul_i64"
	BMulF16    = "bmul_f16"     // Available when __CUDA_ARCH__ >= 530
	BMulBF16   = "bmul_bf16"    // Available when __CUDA_ARCH__ >= 800
	BMulF8E4M3 = "bmul_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	BDivF32    = "bdiv_f32"
	BDivF64    = "bdiv_f64"
	BDivU8     = "bdiv_u8"
	BDivU32    = "bdiv_u32"
	BDivI64    = "bdiv_i64"
	BDivF16    = "bdiv_f16"     // Available when __CUDA_ARCH__ >= 530
	BDivBF16   = "bdiv_bf16"    // Available when __CUDA_ARCH__ >= 800
	BDivF8E4M3 = "bdiv_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	BMaximumF32    = "bmaximum_f32"
	BMaximumF64    = "bmaximum_f64"
	BMaximumU8     = "bmaximum_u8"
	BMaximumU32    = "bmaximum_u32"
	BMaximumI64    = "bmaximum_i64"
	BMaximumF16    = "bmaximum_f16"     // Available when __CUDA_ARCH__ >= 530
	BMaximumBF16   = "bmaximum_bf16"    // Available when __CUDA_ARCH__ >= 800
	BMaximumF8E4M3 = "bmaximum_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	BMinimumF32    = "bminimum_f32"
	BMinimumF64    = "bminimum_f64"
	BMinimumU8     = "bminimum_u8"
	BMinimumU32    = "bminimum_u32"
	BMinimumI64    = "bminimum_i64"
	BMinimumF16    = "bminimum_f16"     // Available when __CUDA_ARCH__ >= 530
	BMinimumBF16   = "bminimum_bf16"    // Available when __CUDA_ARCH__ >= 800
	BMinimumF8E4M3 = "bminimum_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	// Binary comparison operations (output uint8_t)
	EqF32    = "eq_f32"
	EqF64    = "eq_f64"
	EqU8     = "eq_u8"
	EqU32    = "eq_u32"
	EqI64    = "eq_i64"
	EqF16    = "eq_f16"     // Available when __CUDA_ARCH__ >= 530
	EqBF16   = "eq_bf16"    // Available when __CUDA_ARCH__ >= 800
	EqF8E4M3 = "eq_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	NeF32    = "ne_f32"
	NeF64    = "ne_f64"
	NeU8     = "ne_u8"
	NeU32    = "ne_u32"
	NeI64    = "ne_i64"
	NeF16    = "ne_f16"     // Available when __CUDA_ARCH__ >= 530
	NeBF16   = "ne_bf16"    // Available when __CUDA_ARCH__ >= 800
	NeF8E4M3 = "ne_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	LtF32    = "lt_f32"
	LtF64    = "lt_f64"
	LtU8     = "lt_u8"
	LtU32    = "lt_u32"
	LtI64    = "lt_i64"
	LtF16    = "lt_f16"     // Available when __CUDA_ARCH__ >= 530
	LtBF16   = "lt_bf16"    // Available when __CUDA_ARCH__ >= 800
	LtF8E4M3 = "lt_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	LeF32    = "le_f32"
	LeF64    = "le_f64"
	LeU8     = "le_u8"
	LeU32    = "le_u32"
	LeI64    = "le_i64"
	LeF16    = "le_f16"     // Available when __CUDA_ARCH__ >= 530
	LeBF16   = "le_bf16"    // Available when __CUDA_ARCH__ >= 800
	LeF8E4M3 = "le_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	GtF32    = "gt_f32"
	GtF64    = "gt_f64"
	GtU8     = "gt_u8"
	GtU32    = "gt_u32"
	GtI64    = "gt_i64"
	GtF16    = "gt_f16"     // Available when __CUDA_ARCH__ >= 530
	GtBF16   = "gt_bf16"    // Available when __CUDA_ARCH__ >= 800
	GtF8E4M3 = "gt_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	GeF32    = "ge_f32"
	GeF64    = "ge_f64"
	GeU8     = "ge_u8"
	GeU32    = "ge_u32"
	GeI64    = "ge_i64"
	GeF16    = "ge_f16"     // Available when __CUDA_ARCH__ >= 530
	GeBF16   = "ge_bf16"    // Available when __CUDA_ARCH__ >= 800
	GeF8E4M3 = "ge_f8_e4m3" // Available when __CUDA_ARCH__ >= 800
)

// Cast operation kernel function names
// Keep in sync with the actual kernel function names in *.cu files
const (
	// Cast operations for uint32_t
	CastU32U32 = "cast_u32_u32"
	CastU32U8  = "cast_u32_u8"
	CastU32I64 = "cast_u32_i64"
	CastU32F32 = "cast_u32_f32"
	CastU32F64 = "cast_u32_f64"

	// Cast operations for uint8_t
	CastU8U32 = "cast_u8_u32"
	CastU8U8  = "cast_u8_u8"
	CastU8I64 = "cast_u8_i64"
	CastU8F32 = "cast_u8_f32"
	CastU8F64 = "cast_u8_f64"

	// Cast operations for int64_t
	CastI64U32 = "cast_i64_u32"
	CastI64U8  = "cast_i64_u8"
	CastI64I64 = "cast_i64_i64"
	CastI64F32 = "cast_i64_f32"
	CastI64F64 = "cast_i64_f64"

	// Cast operations for float
	CastF32U8  = "cast_f32_u8"
	CastF32U32 = "cast_f32_u32"
	CastF32I64 = "cast_f32_i64"
	CastF32F32 = "cast_f32_f32"
	CastF32F64 = "cast_f32_f64"

	// Cast operations for double
	CastF64U8  = "cast_f64_u8"
	CastF64U32 = "cast_f64_u32"
	CastF64I64 = "cast_f64_i64"
	CastF64F32 = "cast_f64_f32"
	CastF64F64 = "cast_f64_f64"

	// Cast operations for f16 (Available when __CUDA_ARCH__ >= 530)
	CastF16F16 = "cast_f16_f16"
	CastF16U8  = "cast_f16_u8" // Through float
	CastF16U32 = "cast_f16_u32"
	CastF16F32 = "cast_f16_f32"
	CastF16F64 = "cast_f16_f64"
	CastU8F16  = "cast_u8_f16"
	CastU32F16 = "cast_u32_f16"
	CastF32F16 = "cast_f32_f16"
	CastF64F16 = "cast_f64_f16"

	// Cast operations for bf16 (Available when __CUDA_ARCH__ >= 800)
	CastBF16BF16 = "cast_bf16_bf16"
	CastBF16U32  = "cast_bf16_u32"
	CastBF16F32  = "cast_bf16_f32"
	CastBF16F64  = "cast_bf16_f64"
	CastU8BF16   = "cast_u8_bf16"
	CastU32BF16  = "cast_u32_bf16"
	CastF32BF16  = "cast_f32_bf16"
	CastF64BF16  = "cast_f64_bf16"
	CastBF16U8   = "cast_bf16_u8"  // Through float
	CastBF16F16  = "cast_bf16_f16" // Through float
	CastF16BF16  = "cast_f16_bf16" // Through float

	// Cast operations for fp8_e4m3 (Available when __CUDA_ARCH__ >= 800)
	CastF8E4M3F8E4M3 = "cast_f8_e4m3_f8_e4m3"
	CastF8E4M3F32    = "cast_f8_e4m3_f32"
	CastF32F8E4M3    = "cast_f32_f8_e4m3"
	CastF8E4M3U8     = "cast_f8_e4m3_u8"
	CastF8E4M3F16    = "cast_f8_e4m3_f16"
	CastF8E4M3F64    = "cast_f8_e4m3_f64"
	CastF16F8E4M3    = "cast_f16_f8_e4m3"
	CastF64F8E4M3    = "cast_f64_f8_e4m3"
	CastU8F8E4M3     = "cast_u8_f8_e4m3"
	CastI32F8E4M3    = "cast_i32_f8_e4m3"
	CastF8E4M3I32    = "cast_f8_e4m3_i32"
	CastF8E4M3BF16   = "cast_f8_e4m3_bf16"
	CastBF16F8E4M3   = "cast_bf16_f8_e4m3"

	// Cast operations for bf16 (Available when CUDA_VERSION >= 11000 and __CUDA_ARCH__ < 800)
	CastBF16F32CUDA11    = "cast_bf16_f32"     // Also available when CUDA_VERSION >= 11000
	CastF32BF16CUDA11    = "cast_f32_bf16"     // Also available when CUDA_VERSION >= 11000
	CastBF16U8CUDA11     = "cast_bf16_u8"      // Through float
	CastBF16F16CUDA11    = "cast_bf16_f16"     // Through float
	CastBF16F64CUDA11    = "cast_bf16_f64"     // Through float
	CastF16BF16CUDA11    = "cast_f16_bf16"     // Through float
	CastF64BF16CUDA11    = "cast_f64_bf16"     // Through float
	CastU8BF16CUDA11     = "cast_u8_bf16"      // Through float
	CastBF16F8E4M3CUDA11 = "cast_bf16_f8_e4m3" // Through float
)

// Convolution and related operation kernel function names
// Keep in sync with the actual kernel function names in *.cu files
const (
	// Conv1d operations
	Conv1dF32  = "conv1d_f32"
	Conv1dF64  = "conv1d_f64"
	Conv1dU8   = "conv1d_u8"
	Conv1dU32  = "conv1d_u32"
	Conv1dF16  = "conv1d_f16"  // Available when __CUDA_ARCH__ >= 530
	Conv1dBF16 = "conv1d_bf16" // Available when __CUDA_ARCH__ >= 800

	// Conv2d operations
	Conv2dF32  = "conv2d_f32"
	Conv2dF64  = "conv2d_f64"
	Conv2dU8   = "conv2d_u8"
	Conv2dU32  = "conv2d_u32"
	Conv2dF16  = "conv2d_f16"  // Available when __CUDA_ARCH__ >= 530
	Conv2dBF16 = "conv2d_bf16" // Available when __CUDA_ARCH__ >= 800

	// Conv_transpose1d operations
	ConvTranspose1dF32  = "conv_transpose1d_f32"
	ConvTranspose1dF64  = "conv_transpose1d_f64"
	ConvTranspose1dU8   = "conv_transpose1d_u8"
	ConvTranspose1dU32  = "conv_transpose1d_u32"
	ConvTranspose1dF16  = "conv_transpose1d_f16"  // Available when __CUDA_ARCH__ >= 530
	ConvTranspose1dBF16 = "conv_transpose1d_bf16" // Available when __CUDA_ARCH__ >= 800

	// Conv_transpose2d operations
	ConvTranspose2dF32  = "conv_transpose2d_f32"
	ConvTranspose2dF64  = "conv_transpose2d_f64"
	ConvTranspose2dU8   = "conv_transpose2d_u8"
	ConvTranspose2dU32  = "conv_transpose2d_u32"
	ConvTranspose2dF16  = "conv_transpose2d_f16"  // Available when __CUDA_ARCH__ >= 530
	ConvTranspose2dBF16 = "conv_transpose2d_bf16" // Available when __CUDA_ARCH__ >= 800

	// Avg_pool2d operations
	AvgPool2dF32  = "avg_pool2d_f32"
	AvgPool2dF64  = "avg_pool2d_f64"
	AvgPool2dU8   = "avg_pool2d_u8"
	AvgPool2dU32  = "avg_pool2d_u32"
	AvgPool2dF16  = "avg_pool2d_f16"  // Available when __CUDA_ARCH__ >= 530
	AvgPool2dBF16 = "avg_pool2d_bf16" // Available when __CUDA_ARCH__ >= 800

	// Max_pool2d operations
	MaxPool2dF32  = "max_pool2d_f32"
	MaxPool2dF64  = "max_pool2d_f64"
	MaxPool2dU8   = "max_pool2d_u8"
	MaxPool2dU32  = "max_pool2d_u32"
	MaxPool2dF16  = "max_pool2d_f16"  // Available when __CUDA_ARCH__ >= 530
	MaxPool2dBF16 = "max_pool2d_bf16" // Available when __CUDA_ARCH__ >= 800

	// Upsample_nearest2d operations
	UpsampleNearest2dF32  = "upsample_nearest2d_f32"
	UpsampleNearest2dF64  = "upsample_nearest2d_f64"
	UpsampleNearest2dU8   = "upsample_nearest2d_u8"
	UpsampleNearest2dU32  = "upsample_nearest2d_u32"
	UpsampleNearest2dF16  = "upsample_nearest2d_f16"  // Available when __CUDA_ARCH__ >= 530
	UpsampleNearest2dBF16 = "upsample_nearest2d_bf16" // Available when __CUDA_ARCH__ >= 800

	// Im2col operations
	Im2colF32  = "im2col_f32"
	Im2colF64  = "im2col_f64"
	Im2colU8   = "im2col_u8"
	Im2colU32  = "im2col_u32"
	Im2colF16  = "im2col_f16"  // Available when __CUDA_ARCH__ >= 530
	Im2colBF16 = "im2col_bf16" // Available when __CUDA_ARCH__ >= 800

	// Im2col1d operations
	Im2col1dF32  = "im2col1d_f32"
	Im2col1dF64  = "im2col1d_f64"
	Im2col1dU8   = "im2col1d_u8"
	Im2col1dU32  = "im2col1d_u32"
	Im2col1dF16  = "im2col1d_f16"  // Available when __CUDA_ARCH__ >= 530
	Im2col1dBF16 = "im2col1d_bf16" // Available when __CUDA_ARCH__ >= 800

	// Col2im1d operations
	Col2im1dF32  = "col2im1d_f32"
	Col2im1dF64  = "col2im1d_f64"
	Col2im1dU8   = "col2im1d_u8"
	Col2im1dU32  = "col2im1d_u32"
	Col2im1dF16  = "col2im1d_f16"  // Available when __CUDA_ARCH__ >= 530
	Col2im1dBF16 = "col2im1d_bf16" // Available when __CUDA_ARCH__ >= 800
)

// Fill, copy2d, and const_set operation kernel function names
// Keep in sync with the actual kernel function names in *.cu files
const (
	// Fill operations
	FillF32    = "fill_f32"
	FillF64    = "fill_f64"
	FillU8     = "fill_u8"
	FillU32    = "fill_u32"
	FillI64    = "fill_i64"
	FillF16    = "fill_f16"     // Available when __CUDA_ARCH__ >= 530
	FillBF16   = "fill_bf16"    // Available when __CUDA_ARCH__ >= 800
	FillF8E4M3 = "fill_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	// Copy2d operations
	Copy2dF32    = "copy2d_f32"
	Copy2dF64    = "copy2d_f64"
	Copy2dU8     = "copy2d_u8"
	Copy2dU32    = "copy2d_u32"
	Copy2dI64    = "copy2d_i64"
	Copy2dF16    = "copy2d_f16"     // Available when __CUDA_ARCH__ >= 530
	Copy2dBF16   = "copy2d_bf16"    // Available when __CUDA_ARCH__ >= 800
	Copy2dF8E4M3 = "copy2d_f8_e4m3" // Available when __CUDA_ARCH__ >= 800

	// Const_set operations
	ConstSetF32    = "const_set_f32"
	ConstSetF64    = "const_set_f64"
	ConstSetU8     = "const_set_u8"
	ConstSetU32    = "const_set_u32"
	ConstSetI64    = "const_set_i64"
	ConstSetF16    = "const_set_f16"     // Available when __CUDA_ARCH__ >= 530
	ConstSetBF16   = "const_set_bf16"    // Available when __CUDA_ARCH__ >= 800
	ConstSetF8E4M3 = "const_set_f8_e4m3" // Available when __CUDA_ARCH__ >= 800
)

// Index-related operation kernel function names
// Keep in sync with the actual kernel function names in *.cu files
const (
	// Index_select operations
	IndexSelectI64F32    = "is_i64_f32"
	IndexSelectI64F64    = "is_i64_f64"
	IndexSelectI64U8     = "is_i64_u8"
	IndexSelectI64U32    = "is_i64_u32"
	IndexSelectI64I64    = "is_i64_i64"
	IndexSelectU32F32    = "is_u32_f32"
	IndexSelectU32F64    = "is_u32_f64"
	IndexSelectU32U8     = "is_u32_u8"
	IndexSelectU32I64    = "is_u32_i64"
	IndexSelectU32U32    = "is_u32_u32"
	IndexSelectU8F32     = "is_u8_f32"
	IndexSelectU8F64     = "is_u8_f64"
	IndexSelectU8U8      = "is_u8_u8"
	IndexSelectU8U32     = "is_u8_u32"
	IndexSelectU8I64     = "is_u8_i64"
	IndexSelectI64F16    = "is_i64_f16"     // Available when __CUDA_ARCH__ >= 530
	IndexSelectU32F16    = "is_u32_f16"     // Available when __CUDA_ARCH__ >= 530
	IndexSelectU8F16     = "is_u8_f16"      // Available when __CUDA_ARCH__ >= 530
	IndexSelectI64BF16   = "is_i64_bf16"    // Available when __CUDA_ARCH__ >= 800
	IndexSelectU32BF16   = "is_u32_bf16"    // Available when __CUDA_ARCH__ >= 800
	IndexSelectU8BF16    = "is_u8_bf16"     // Available when __CUDA_ARCH__ >= 800
	IndexSelectI16F8E4M3 = "is_i16_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	IndexSelectI32F8E4M3 = "is_i32_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	IndexSelectI64F8E4M3 = "is_i64_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	IndexSelectU32F8E4M3 = "is_u32_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	IndexSelectU8F8E4M3  = "is_u8_f8_e4m3"  // Available when __CUDA_ARCH__ >= 890

	// Gather operations
	GatherI64F32    = "gather_i64_f32"
	GatherI64F64    = "gather_i64_f64"
	GatherI64U8     = "gather_i64_u8"
	GatherI64U32    = "gather_i64_u32"
	GatherI64I64    = "gather_i64_i64"
	GatherU32F32    = "gather_u32_f32"
	GatherU32F64    = "gather_u32_f64"
	GatherU32U8     = "gather_u32_u8"
	GatherU32I64    = "gather_u32_i64"
	GatherU32U32    = "gather_u32_u32"
	GatherU8F32     = "gather_u8_f32"
	GatherU8F64     = "gather_u8_f64"
	GatherU8U8      = "gather_u8_u8"
	GatherU8U32     = "gather_u8_u32"
	GatherU8I64     = "gather_u8_i64"
	GatherI64F16    = "gather_i64_f16"     // Available when __CUDA_ARCH__ >= 530
	GatherU32F16    = "gather_u32_f16"     // Available when __CUDA_ARCH__ >= 530
	GatherU8F16     = "gather_u8_f16"      // Available when __CUDA_ARCH__ >= 530
	GatherI64BF16   = "gather_i64_bf16"    // Available when __CUDA_ARCH__ >= 800
	GatherU32BF16   = "gather_u32_bf16"    // Available when __CUDA_ARCH__ >= 800
	GatherU8BF16    = "gather_u8_bf16"     // Available when __CUDA_ARCH__ >= 800
	GatherI16F8E4M3 = "gather_i16_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	GatherI32F8E4M3 = "gather_i32_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	GatherI64F8E4M3 = "gather_i64_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	GatherU32F8E4M3 = "gather_u32_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	GatherU8F8E4M3  = "gather_u8_f8_e4m3"  // Available when __CUDA_ARCH__ >= 890

	// Index_add operations
	IndexAddI64F32    = "ia_i64_f32"
	IndexAddI64F64    = "ia_i64_f64"
	IndexAddI64U8     = "ia_i64_u8"
	IndexAddI64U32    = "ia_i64_u32"
	IndexAddI64I64    = "ia_i64_i64"
	IndexAddU32F32    = "ia_u32_f32"
	IndexAddU32F64    = "ia_u32_f64"
	IndexAddU32U8     = "ia_u32_u8"
	IndexAddU32I64    = "ia_u32_i64"
	IndexAddU32U32    = "ia_u32_u32"
	IndexAddU8F32     = "ia_u8_f32"
	IndexAddU8F64     = "ia_u8_f64"
	IndexAddU8U8      = "ia_u8_u8"
	IndexAddU8U32     = "ia_u8_u32"
	IndexAddU8I64     = "ia_u8_i64"
	IndexAddI64F16    = "ia_i64_f16"     // Available when __CUDA_ARCH__ >= 530
	IndexAddU32F16    = "ia_u32_f16"     // Available when __CUDA_ARCH__ >= 530
	IndexAddU8F16     = "ia_u8_f16"      // Available when __CUDA_ARCH__ >= 530
	IndexAddI64BF16   = "ia_i64_bf16"    // Available when __CUDA_ARCH__ >= 800
	IndexAddU32BF16   = "ia_u32_bf16"    // Available when __CUDA_ARCH__ >= 800
	IndexAddU8BF16    = "ia_u8_bf16"     // Available when __CUDA_ARCH__ >= 800
	IndexAddI16F8E4M3 = "ia_i16_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	IndexAddI32F8E4M3 = "ia_i32_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	IndexAddI64F8E4M3 = "ia_i64_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	IndexAddU32F8E4M3 = "ia_u32_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	IndexAddU8F8E4M3  = "ia_u8_f8_e4m3"  // Available when __CUDA_ARCH__ >= 890

	// Scatter operations
	ScatterI64F32  = "s_i64_f32"
	ScatterI64F64  = "s_i64_f64"
	ScatterI64U8   = "s_i64_u8"
	ScatterI64U32  = "s_i64_u32"
	ScatterI64I64  = "s_i64_i64"
	ScatterU32F32  = "s_u32_f32"
	ScatterU32F64  = "s_u32_f64"
	ScatterU32U8   = "s_u32_u8"
	ScatterU32I64  = "s_u32_i64"
	ScatterU32U32  = "s_u32_u32"
	ScatterU8F32   = "s_u8_f32"
	ScatterU8F64   = "s_u8_f64"
	ScatterU8U8    = "s_u8_u8"
	ScatterU8U32   = "s_u8_u32"
	ScatterU8I64   = "s_u8_i64"
	ScatterI64F16  = "s_i64_f16"  // Available when __CUDA_ARCH__ >= 530
	ScatterU32F16  = "s_u32_f16"  // Available when __CUDA_ARCH__ >= 530
	ScatterU8F16   = "s_u8_f16"   // Available when __CUDA_ARCH__ >= 530
	ScatterI64BF16 = "s_i64_bf16" // Available when __CUDA_ARCH__ >= 800
	ScatterU32BF16 = "s_u32_bf16" // Available when __CUDA_ARCH__ >= 800
	ScatterU8BF16  = "s_u8_bf16"  // Available when __CUDA_ARCH__ >= 800

	// Scatter_add operations
	ScatterAddI64F32    = "sa_i64_f32"
	ScatterAddI64F64    = "sa_i64_f64"
	ScatterAddI64U8     = "sa_i64_u8"
	ScatterAddI64U32    = "sa_i64_u32"
	ScatterAddI64I64    = "sa_i64_i64"
	ScatterAddU32F32    = "sa_u32_f32"
	ScatterAddU32F64    = "sa_u32_f64"
	ScatterAddU32U8     = "sa_u32_u8"
	ScatterAddU32I64    = "sa_u32_i64"
	ScatterAddU32U32    = "sa_u32_u32"
	ScatterAddU8F32     = "sa_u8_f32"
	ScatterAddU8F64     = "sa_u8_f64"
	ScatterAddU8U8      = "sa_u8_u8"
	ScatterAddU8U32     = "sa_u8_u32"
	ScatterAddU8I64     = "sa_u8_i64"
	ScatterAddI64F16    = "sa_i64_f16"     // Available when __CUDA_ARCH__ >= 530
	ScatterAddU32F16    = "sa_u32_f16"     // Available when __CUDA_ARCH__ >= 530
	ScatterAddU8F16     = "sa_u8_f16"      // Available when __CUDA_ARCH__ >= 530
	ScatterAddI64BF16   = "sa_i64_bf16"    // Available when __CUDA_ARCH__ >= 800
	ScatterAddU32BF16   = "sa_u32_bf16"    // Available when __CUDA_ARCH__ >= 800
	ScatterAddU8BF16    = "sa_u8_bf16"     // Available when __CUDA_ARCH__ >= 800
	ScatterAddI16F8E4M3 = "sa_i16_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	ScatterAddI32F8E4M3 = "sa_i32_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	ScatterAddI64F8E4M3 = "sa_i64_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	ScatterAddU32F8E4M3 = "sa_u32_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	ScatterAddU8F8E4M3  = "sa_u8_f8_e4m3"  // Available when __CUDA_ARCH__ >= 890
)

// Quantization and matrix multiplication kernel function names
// Keep in sync with the actual kernel function names in *.cu files
const (
	// Dequantize operations
	DequantizeQ4_0F32 = "dequantize_block_q4_0_f32"
	DequantizeQ4_0F16 = "dequantize_block_q4_0_f16" // Available when __CUDA_ARCH__ >= 530
	DequantizeQ4_1F32 = "dequantize_block_q4_1_f32"
	DequantizeQ4_1F16 = "dequantize_block_q4_1_f16" // Available when __CUDA_ARCH__ >= 530
	DequantizeQ5_0F32 = "dequantize_block_q5_0_f32"
	DequantizeQ5_0F16 = "dequantize_block_q5_0_f16" // Available when __CUDA_ARCH__ >= 530
	DequantizeQ5_1F32 = "dequantize_block_q5_1_f32"
	DequantizeQ5_1F16 = "dequantize_block_q5_1_f16" // Available when __CUDA_ARCH__ >= 530
	DequantizeQ8_0F32 = "dequantize_block_q8_0_f32"
	DequantizeQ8_0F16 = "dequantize_block_q8_0_f16" // Available when __CUDA_ARCH__ >= 530
	DequantizeQ2_KF32 = "dequantize_block_q2_K_f32"
	DequantizeQ2_KF16 = "dequantize_block_q2_K_f16" // Available when __CUDA_ARCH__ >= 530
	DequantizeQ3_KF32 = "dequantize_block_q3_K_f32"
	DequantizeQ3_KF16 = "dequantize_block_q3_K_f16" // Available when __CUDA_ARCH__ >= 530
	DequantizeQ4_KF32 = "dequantize_block_q4_K_f32"
	DequantizeQ4_KF16 = "dequantize_block_q4_K_f16" // Available when __CUDA_ARCH__ >= 530
	DequantizeQ5_KF32 = "dequantize_block_q5_K_f32"
	DequantizeQ5_KF16 = "dequantize_block_q5_K_f16" // Available when __CUDA_ARCH__ >= 530
	DequantizeQ6_KF32 = "dequantize_block_q6_K_f32"
	DequantizeQ6_KF16 = "dequantize_block_q6_K_f16" // Available when __CUDA_ARCH__ >= 530
	DequantizeQ8_KF32 = "dequantize_block_q8_K_f32"
	DequantizeQ8_KF16 = "dequantize_block_q8_K_f16" // Available when __CUDA_ARCH__ >= 530

	// Dequantize and matrix-vector multiplication operations
	DequantizeMulMatVecQ4_0 = "dequantize_mul_mat_vec_q4_0_cuda"
	DequantizeMulMatVecQ4_1 = "dequantize_mul_mat_vec_q4_1_cuda"
	DequantizeMulMatVecQ5_0 = "dequantize_mul_mat_vec_q5_0_cuda"
	DequantizeMulMatVecQ5_1 = "dequantize_mul_mat_vec_q5_1_cuda"
	DequantizeMulMatVecQ8_0 = "dequantize_mul_mat_vec_q8_0_cuda"
	DequantizeMulMatVecQ2_K = "dequantize_mul_mat_vec_q2_k"
	DequantizeMulMatVecQ3_K = "dequantize_mul_mat_vec_q3_k"
	DequantizeMulMatVecQ4_K = "dequantize_mul_mat_vec_q4_k"
	DequantizeMulMatVecQ5_K = "dequantize_mul_mat_vec_q5_k"
	DequantizeMulMatVecQ6_K = "dequantize_mul_mat_vec_q6_k"

	// Matrix-vector multiplication operations (batch size 1)
	MulMatVecQ4_0Q8_1Cuda1 = "mul_mat_vec_q4_0_q8_1_cuda1"
	MulMatVecQ4_1Q8_1Cuda1 = "mul_mat_vec_q4_1_q8_1_cuda1"
	MulMatVecQ5_0Q8_1Cuda1 = "mul_mat_vec_q5_0_q8_1_cuda1"
	MulMatVecQ5_1Q8_1Cuda1 = "mul_mat_vec_q5_1_q8_1_cuda1"
	MulMatVecQ8_0Q8_1Cuda1 = "mul_mat_vec_q8_0_q8_1_cuda1"
	MulMatVecQ2_KQ8_1Cuda1 = "mul_mat_vec_q2_K_q8_1_cuda1"
	MulMatVecQ3_KQ8_1Cuda1 = "mul_mat_vec_q3_K_q8_1_cuda1"
	MulMatVecQ4_KQ8_1Cuda1 = "mul_mat_vec_q4_K_q8_1_cuda1"
	MulMatVecQ5_KQ8_1Cuda1 = "mul_mat_vec_q5_K_q8_1_cuda1"
	MulMatVecQ6_KQ8_1Cuda1 = "mul_mat_vec_q6_K_q8_1_cuda1"

	// Matrix-vector multiplication operations (batch size 2)
	MulMatVecQ4_0Q8_1Cuda2 = "mul_mat_vec_q4_0_q8_1_cuda2"
	MulMatVecQ4_1Q8_1Cuda2 = "mul_mat_vec_q4_1_q8_1_cuda2"
	MulMatVecQ5_0Q8_1Cuda2 = "mul_mat_vec_q5_0_q8_1_cuda2"
	MulMatVecQ5_1Q8_1Cuda2 = "mul_mat_vec_q5_1_q8_1_cuda2"
	MulMatVecQ8_0Q8_1Cuda2 = "mul_mat_vec_q8_0_q8_1_cuda2"
	MulMatVecQ2_KQ8_1Cuda2 = "mul_mat_vec_q2_K_q8_1_cuda2"
	MulMatVecQ3_KQ8_1Cuda2 = "mul_mat_vec_q3_K_q8_1_cuda2"
	MulMatVecQ4_KQ8_1Cuda2 = "mul_mat_vec_q4_K_q8_1_cuda2"
	MulMatVecQ5_KQ8_1Cuda2 = "mul_mat_vec_q5_K_q8_1_cuda2"
	MulMatVecQ6_KQ8_1Cuda2 = "mul_mat_vec_q6_K_q8_1_cuda2"

	// Matrix-vector multiplication operations (batch size 3)
	MulMatVecQ4_0Q8_1Cuda3 = "mul_mat_vec_q4_0_q8_1_cuda3"
	MulMatVecQ4_1Q8_1Cuda3 = "mul_mat_vec_q4_1_q8_1_cuda3"
	MulMatVecQ5_0Q8_1Cuda3 = "mul_mat_vec_q5_0_q8_1_cuda3"
	MulMatVecQ5_1Q8_1Cuda3 = "mul_mat_vec_q5_1_q8_1_cuda3"
	MulMatVecQ8_0Q8_1Cuda3 = "mul_mat_vec_q8_0_q8_1_cuda3"
	MulMatVecQ2_KQ8_1Cuda3 = "mul_mat_vec_q2_K_q8_1_cuda3"
	MulMatVecQ3_KQ8_1Cuda3 = "mul_mat_vec_q3_K_q8_1_cuda3"
	MulMatVecQ4_KQ8_1Cuda3 = "mul_mat_vec_q4_K_q8_1_cuda3"
	MulMatVecQ5_KQ8_1Cuda3 = "mul_mat_vec_q5_K_q8_1_cuda3"
	MulMatVecQ6_KQ8_1Cuda3 = "mul_mat_vec_q6_K_q8_1_cuda3"

	// Matrix-vector multiplication operations (batch size 4)
	MulMatVecQ4_0Q8_1Cuda4 = "mul_mat_vec_q4_0_q8_1_cuda4"
	MulMatVecQ4_1Q8_1Cuda4 = "mul_mat_vec_q4_1_q8_1_cuda4"
	MulMatVecQ5_0Q8_1Cuda4 = "mul_mat_vec_q5_0_q8_1_cuda4"
	MulMatVecQ5_1Q8_1Cuda4 = "mul_mat_vec_q5_1_q8_1_cuda4"
	MulMatVecQ8_0Q8_1Cuda4 = "mul_mat_vec_q8_0_q8_1_cuda4"
	MulMatVecQ2_KQ8_1Cuda4 = "mul_mat_vec_q2_K_q8_1_cuda4"
	MulMatVecQ3_KQ8_1Cuda4 = "mul_mat_vec_q3_K_q8_1_cuda4"
	MulMatVecQ4_KQ8_1Cuda4 = "mul_mat_vec_q4_K_q8_1_cuda4"
	MulMatVecQ5_KQ8_1Cuda4 = "mul_mat_vec_q5_K_q8_1_cuda4"
	MulMatVecQ6_KQ8_1Cuda4 = "mul_mat_vec_q6_K_q8_1_cuda4"

	// Matrix-vector multiplication operations (batch size 5)
	MulMatVecQ4_0Q8_1Cuda5 = "mul_mat_vec_q4_0_q8_1_cuda5"
	MulMatVecQ4_1Q8_1Cuda5 = "mul_mat_vec_q4_1_q8_1_cuda5"
	MulMatVecQ5_0Q8_1Cuda5 = "mul_mat_vec_q5_0_q8_1_cuda5"
	MulMatVecQ5_1Q8_1Cuda5 = "mul_mat_vec_q5_1_q8_1_cuda5"
	MulMatVecQ8_0Q8_1Cuda5 = "mul_mat_vec_q8_0_q8_1_cuda5"
	MulMatVecQ2_KQ8_1Cuda5 = "mul_mat_vec_q2_K_q8_1_cuda5"
	MulMatVecQ3_KQ8_1Cuda5 = "mul_mat_vec_q3_K_q8_1_cuda5"
	MulMatVecQ4_KQ8_1Cuda5 = "mul_mat_vec_q4_K_q8_1_cuda5"
	MulMatVecQ5_KQ8_1Cuda5 = "mul_mat_vec_q5_K_q8_1_cuda5"
	MulMatVecQ6_KQ8_1Cuda5 = "mul_mat_vec_q6_K_q8_1_cuda5"

	// Matrix-vector multiplication operations (batch size 6)
	MulMatVecQ4_0Q8_1Cuda6 = "mul_mat_vec_q4_0_q8_1_cuda6"
	MulMatVecQ4_1Q8_1Cuda6 = "mul_mat_vec_q4_1_q8_1_cuda6"
	MulMatVecQ5_0Q8_1Cuda6 = "mul_mat_vec_q5_0_q8_1_cuda6"
	MulMatVecQ5_1Q8_1Cuda6 = "mul_mat_vec_q5_1_q8_1_cuda6"
	MulMatVecQ8_0Q8_1Cuda6 = "mul_mat_vec_q8_0_q8_1_cuda6"
	MulMatVecQ2_KQ8_1Cuda6 = "mul_mat_vec_q2_K_q8_1_cuda6"
	MulMatVecQ3_KQ8_1Cuda6 = "mul_mat_vec_q3_K_q8_1_cuda6"
	MulMatVecQ4_KQ8_1Cuda6 = "mul_mat_vec_q4_K_q8_1_cuda6"
	MulMatVecQ5_KQ8_1Cuda6 = "mul_mat_vec_q5_K_q8_1_cuda6"
	MulMatVecQ6_KQ8_1Cuda6 = "mul_mat_vec_q6_K_q8_1_cuda6"

	// Matrix-vector multiplication operations (batch size 7)
	MulMatVecQ4_0Q8_1Cuda7 = "mul_mat_vec_q4_0_q8_1_cuda7"
	MulMatVecQ4_1Q8_1Cuda7 = "mul_mat_vec_q4_1_q8_1_cuda7"
	MulMatVecQ5_0Q8_1Cuda7 = "mul_mat_vec_q5_0_q8_1_cuda7"
	MulMatVecQ5_1Q8_1Cuda7 = "mul_mat_vec_q5_1_q8_1_cuda7"
	MulMatVecQ8_0Q8_1Cuda7 = "mul_mat_vec_q8_0_q8_1_cuda7"
	MulMatVecQ2_KQ8_1Cuda7 = "mul_mat_vec_q2_K_q8_1_cuda7"
	MulMatVecQ3_KQ8_1Cuda7 = "mul_mat_vec_q3_K_q8_1_cuda7"
	MulMatVecQ4_KQ8_1Cuda7 = "mul_mat_vec_q4_K_q8_1_cuda7"
	MulMatVecQ5_KQ8_1Cuda7 = "mul_mat_vec_q5_K_q8_1_cuda7"
	MulMatVecQ6_KQ8_1Cuda7 = "mul_mat_vec_q6_K_q8_1_cuda7"

	// Matrix-vector multiplication operations (batch size 8)
	MulMatVecQ4_0Q8_1Cuda8 = "mul_mat_vec_q4_0_q8_1_cuda8"
	MulMatVecQ4_1Q8_1Cuda8 = "mul_mat_vec_q4_1_q8_1_cuda8"
	MulMatVecQ5_0Q8_1Cuda8 = "mul_mat_vec_q5_0_q8_1_cuda8"
	MulMatVecQ5_1Q8_1Cuda8 = "mul_mat_vec_q5_1_q8_1_cuda8"
	MulMatVecQ8_0Q8_1Cuda8 = "mul_mat_vec_q8_0_q8_1_cuda8"
	MulMatVecQ2_KQ8_1Cuda8 = "mul_mat_vec_q2_K_q8_1_cuda8"
	MulMatVecQ3_KQ8_1Cuda8 = "mul_mat_vec_q3_K_q8_1_cuda8"
	MulMatVecQ4_KQ8_1Cuda8 = "mul_mat_vec_q4_K_q8_1_cuda8"
	MulMatVecQ5_KQ8_1Cuda8 = "mul_mat_vec_q5_K_q8_1_cuda8"
	MulMatVecQ6_KQ8_1Cuda8 = "mul_mat_vec_q6_K_q8_1_cuda8"

	// Matrix-matrix multiplication operations
	MulMatQ4_0 = "mul_mat_q4_0"
	MulMatQ4_1 = "mul_mat_q4_1"
	MulMatQ5_0 = "mul_mat_q5_0"
	MulMatQ5_1 = "mul_mat_q5_1"
	MulMatQ8_0 = "mul_mat_q8_0"
	MulMatQ2_K = "mul_mat_q2_K"
	MulMatQ3_K = "mul_mat_q3_K"
	MulMatQ4_K = "mul_mat_q4_K"
	MulMatQ5_K = "mul_mat_q5_K"
	MulMatQ6_K = "mul_mat_q6_K"

	// Quantization operation
	QuantizeQ8_1 = "quantize_q8_1"
)

// Reduction, normalization, and rotary embedding kernel function names
// Keep in sync with the actual kernel function names in *.cu files
const (
	// Fast_sum operations
	FastSumF32  = "fast_sum_f32"
	FastSumF64  = "fast_sum_f64"
	FastSumU32  = "fast_sum_u32"
	FastSumI64  = "fast_sum_i64"
	FastSumU8   = "fast_sum_u8"
	FastSumF16  = "fast_sum_f16"  // Available when __CUDA_ARCH__ >= 530
	FastSumBF16 = "fast_sum_bf16" // Available when __CUDA_ARCH__ >= 800

	// Fast_min operations
	FastMinF32  = "fast_min_f32"
	FastMinF64  = "fast_min_f64"
	FastMinU32  = "fast_min_u32"
	FastMinI64  = "fast_min_i64"
	FastMinU8   = "fast_min_u8"
	FastMinF16  = "fast_min_f16"  // Available when __CUDA_ARCH__ >= 530
	FastMinBF16 = "fast_min_bf16" // Available when __CUDA_ARCH__ >= 800

	// Fast_max operations
	FastMaxF32  = "fast_max_f32"
	FastMaxF64  = "fast_max_f64"
	FastMaxU32  = "fast_max_u32"
	FastMaxI64  = "fast_max_i64"
	FastMaxU8   = "fast_max_u8"
	FastMaxF16  = "fast_max_f16"  // Available when __CUDA_ARCH__ >= 530
	FastMaxBF16 = "fast_max_bf16" // Available when __CUDA_ARCH__ >= 800

	// Fast_argmin operations
	FastArgminF32  = "fast_argmin_f32"
	FastArgminF64  = "fast_argmin_f64"
	FastArgminU32  = "fast_argmin_u32"
	FastArgminI64  = "fast_argmin_i64"
	FastArgminU8   = "fast_argmin_u8"
	FastArgminF16  = "fast_argmin_f16"  // Available when __CUDA_ARCH__ >= 530
	FastArgminBF16 = "fast_argmin_bf16" // Available when __CUDA_ARCH__ >= 800

	// Fast_argmax operations
	FastArgmaxF32  = "fast_argmax_f32"
	FastArgmaxF64  = "fast_argmax_f64"
	FastArgmaxU32  = "fast_argmax_u32"
	FastArgmaxI64  = "fast_argmax_i64"
	FastArgmaxU8   = "fast_argmax_u8"
	FastArgmaxF16  = "fast_argmax_f16"  // Available when __CUDA_ARCH__ >= 530
	FastArgmaxBF16 = "fast_argmax_bf16" // Available when __CUDA_ARCH__ >= 800

	// Sum operations
	SumF32  = "sum_f32"
	SumF64  = "sum_f64"
	SumU32  = "sum_u32"
	SumF16  = "sum_f16"  // Available when __CUDA_ARCH__ >= 530
	SumBF16 = "sum_bf16" // Available when __CUDA_ARCH__ >= 800

	// Softmax operations
	SoftmaxF32  = "softmax_f32"
	SoftmaxF64  = "softmax_f64"
	SoftmaxF16  = "softmax_f16"  // Available when __CUDA_ARCH__ >= 530
	SoftmaxBF16 = "softmax_bf16" // Available when __CUDA_ARCH__ >= 800

	// Rmsnorm operations
	RmsnormF32  = "rmsnorm_f32"
	RmsnormF64  = "rmsnorm_f64"
	RmsnormF16  = "rmsnorm_f16"  // Available when __CUDA_ARCH__ >= 530
	RmsnormBF16 = "rmsnorm_bf16" // Available when __CUDA_ARCH__ >= 800

	// Layernorm operations
	LayernormF32  = "layernorm_f32"
	LayernormF64  = "layernorm_f64"
	LayernormF16  = "layernorm_f16"  // Available when __CUDA_ARCH__ >= 530
	LayernormBF16 = "layernorm_bf16" // Available when __CUDA_ARCH__ >= 800

	// Rope operations
	RopeF32     = "rope_f32"
	RopeF64     = "rope_f64"
	RopeF16     = "rope_f16"  // Available when __CUDA_ARCH__ >= 530
	RopeBF16    = "rope_bf16" // Available when __CUDA_ARCH__ >= 800
	RopeIF32    = "rope_i_f32"
	RopeIF64    = "rope_i_f64"
	RopeIF16    = "rope_i_f16"  // Available when __CUDA_ARCH__ >= 530
	RopeIBF16   = "rope_i_bf16" // Available when __CUDA_ARCH__ >= 800
	RopeThdF32  = "rope_thd_f32"
	RopeThdF64  = "rope_thd_f64"
	RopeThdF16  = "rope_thd_f16"  // Available when __CUDA_ARCH__ >= 530
	RopeThdBF16 = "rope_thd_bf16" // Available when __CUDA_ARCH__ >= 800
)

// Argsort kernel function names
// Keep in sync with the actual kernel function names in *.cu files
const (
	// Ascending argsort operations
	AsortAscF32  = "asort_asc_f32"
	AsortAscF64  = "asort_asc_f64"
	AsortAscU8   = "asort_asc_u8"
	AsortAscU32  = "asort_asc_u32"
	AsortAscI64  = "asort_asc_i64"
	AsortAscF16  = "asort_asc_f16"  // Available when __CUDA_ARCH__ >= 530
	AsortAscBF16 = "asort_asc_bf16" // Available when __CUDA_ARCH__ >= 800

	// Descending argsort operations
	AsortDescF32  = "asort_desc_f32"
	AsortDescF64  = "asort_desc_f64"
	AsortDescU8   = "asort_desc_u8"
	AsortDescU32  = "asort_desc_u32"
	AsortDescI64  = "asort_desc_i64"
	AsortDescF16  = "asort_desc_f16"  // Available when __CUDA_ARCH__ >= 530
	AsortDescBF16 = "asort_desc_bf16" // Available when __CUDA_ARCH__ >= 800
)

// Where kernel function names
// Keep in sync with the actual kernel function names in *.cu files
const (
	// Where operations with int64_t index
	WhereI64F32    = "where_i64_f32"
	WhereI64F64    = "where_i64_f64"
	WhereI64U8     = "where_i64_u8"
	WhereI64U32    = "where_i64_u32"
	WhereI64I64    = "where_i64_i64"
	WhereI64F16    = "where_i64_f16"      // Available when __CUDA_ARCH__ >= 530
	WhereI64BF16   = "where_i64_bf16"     // Available when __CUDA_ARCH__ >= 800
	WhereI64F8E4M3 = "where_i64_fp8_e4m3" // Available when __CUDA_ARCH__ >= 890

	// Where operations with uint32_t index
	WhereU32F32    = "where_u32_f32"
	WhereU32F64    = "where_u32_f64"
	WhereU32U8     = "where_u32_u8"
	WhereU32U32    = "where_u32_u32"
	WhereU32I64    = "where_u32_i64"
	WhereU32F16    = "where_u32_f16"      // Available when __CUDA_ARCH__ >= 530
	WhereU32BF16   = "where_u32_bf16"     // Available when __CUDA_ARCH__ >= 800
	WhereU32F8E4M3 = "where_u32_fp8_e4m3" // Available when __CUDA_ARCH__ >= 890

	// Where operations with uint8_t index
	WhereU8F32    = "where_u8_f32"
	WhereU8F64    = "where_u8_f64"
	WhereU8U8     = "where_u8_u8"
	WhereU8U32    = "where_u8_u32"
	WhereU8I64    = "where_u8_i64"
	WhereU8F16    = "where_u8_f16"      // Available when __CUDA_ARCH__ >= 530
	WhereU8BF16   = "where_u8_bf16"     // Available when __CUDA_ARCH__ >= 800
	WhereU8F8E4M3 = "where_u8_fp8_e4m3" // Available when __CUDA_ARCH__ >= 890

	// Where operations with int16_t index
	WhereI16F8E4M3 = "where_i16_fp8_e4m3" // Available when __CUDA_ARCH__ >= 890

	// Where operations with int32_t index
	WhereI32F8E4M3 = "where_i32_fp8_e4m3" // Available when __CUDA_ARCH__ >= 890
)

// Unary operation kernel function names
// Keep in sync with the actual kernel function names in *.cu files
const (
	// Copy operations
	UcopyF32    = "ucopy_f32"
	UcopyF64    = "ucopy_f64"
	UcopyU8     = "ucopy_u8"
	UcopyU32    = "ucopy_u32"
	UcopyI64    = "ucopy_i64"
	UcopyF16    = "ucopy_f16"     // Available when __CUDA_ARCH__ >= 530
	UcopyBF16   = "ucopy_bf16"    // Available when __CUDA_ARCH__ >= 800
	UcopyF8E4M3 = "ucopy_f8_e4m3" // Available when __CUDA_ARCH__ >= 890

	// Negation operations
	UnegF32    = "uneg_f32"
	UnegF64    = "uneg_f64"
	UnegF16    = "uneg_f16"     // Available when __CUDA_ARCH__ >= 530
	UnegF8E4M3 = "uneg_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UnegBF16   = "uneg_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Reciprocal operations
	UrecipF32    = "urecip_f32"
	UrecipF64    = "urecip_f64"
	UrecipF16    = "urecip_f16"     // Available when __CUDA_ARCH__ >= 530
	UrecipF8E4M3 = "urecip_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UrecipBF16   = "urecip_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Exponential operations
	UexpF32    = "uexp_f32"
	UexpF64    = "uexp_f64"
	UexpF16    = "uexp_f16"     // Available when __CUDA_ARCH__ >= 530
	UexpF8E4M3 = "uexp_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UexpBF16   = "uexp_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Logarithm operations
	UlogF32    = "ulog_f32"
	UlogF64    = "ulog_f64"
	UlogF16    = "ulog_f16"     // Available when __CUDA_ARCH__ >= 530
	UlogF8E4M3 = "ulog_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UlogBF16   = "ulog_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Sine operations
	UsinF32    = "usin_f32"
	UsinF64    = "usin_f64"
	UsinF16    = "usin_f16"     // Available when __CUDA_ARCH__ >= 530
	UsinF8E4M3 = "usin_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UsinBF16   = "usin_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Cosine operations
	UcosF32    = "ucos_f32"
	UcosF64    = "ucos_f64"
	UcosF16    = "ucos_f16"     // Available when __CUDA_ARCH__ >= 530
	UcosF8E4M3 = "ucos_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UcosBF16   = "ucos_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Tangent hyperbolic operations
	UtanhF32    = "utanh_f32"
	UtanhF64    = "utanh_f64"
	UtanhF16    = "utanh_f16"     // Available when __CUDA_ARCH__ >= 530
	UtanhF8E4M3 = "utanh_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UtanhBF16   = "utanh_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Error function operations
	UerfF32    = "uerf_f32"
	UerfF64    = "uerf_f64"
	UerfF16    = "uerf_f16"     // Available when __CUDA_ARCH__ >= 530
	UerfF8E4M3 = "uerf_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UerfBF16   = "uerf_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Ceiling operations
	UceilF32    = "uceil_f32"
	UceilF64    = "uceil_f64"
	UceilF16    = "uceil_f16"     // Available when __CUDA_ARCH__ >= 530
	UceilF8E4M3 = "uceil_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UceilBF16   = "uceil_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Floor operations
	UfloorF32    = "ufloor_f32"
	UfloorF64    = "ufloor_f64"
	UfloorF16    = "ufloor_f16"     // Available when __CUDA_ARCH__ >= 530
	UfloorF8E4M3 = "ufloor_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UfloorBF16   = "ufloor_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Round operations
	UroundF32    = "uround_f32"
	UroundF64    = "uround_f64"
	UroundF16    = "uround_f16"     // Available when __CUDA_ARCH__ >= 530
	UroundF8E4M3 = "uround_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UroundBF16   = "uround_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Normal CDF operations
	UnormcdfF32    = "unormcdf_f32"
	UnormcdfF64    = "unormcdf_f64"
	UnormcdfF16    = "unormcdf_f16"     // Available when __CUDA_ARCH__ >= 530
	UnormcdfF8E4M3 = "unormcdf_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UnormcdfBF16   = "unormcdf_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Absolute value operations
	UabsF32    = "uabs_f32"
	UabsF64    = "uabs_f64"
	UabsF16    = "uabs_f16"     // Available when __CUDA_ARCH__ >= 530
	UabsF8E4M3 = "uabs_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UabsBF16   = "uabs_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Square operations
	UsqrF32    = "usqr_f32"
	UsqrF64    = "usqr_f64"
	UsqrF16    = "usqr_f16"     // Available when __CUDA_ARCH__ >= 530
	UsqrF8E4M3 = "usqr_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UsqrBF16   = "usqr_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Square root operations
	UsqrtF32    = "usqrt_f32"
	UsqrtF64    = "usqrt_f64"
	UsqrtF16    = "usqrt_f16"     // Available when __CUDA_ARCH__ >= 530
	UsqrtF8E4M3 = "usqrt_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UsqrtBF16   = "usqrt_bf16"    // Available when __CUDA_ARCH__ >= 800

	// GELU operations
	UgeluF32    = "ugelu_f32"
	UgeluF64    = "ugelu_f64"
	UgeluF16    = "ugelu_f16"     // Available when __CUDA_ARCH__ >= 530
	UgeluF8E4M3 = "ugelu_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UgeluBF16   = "ugelu_bf16"    // Available when __CUDA_ARCH__ >= 800

	// GELU (ERF-based) operations
	UgeluErfF32    = "ugelu_erf_f32"
	UgeluErfF64    = "ugelu_erf_f64"
	UgeluErfF16    = "ugelu_erf_f16"     // Available when __CUDA_ARCH__ >= 530
	UgeluErfF8E4M3 = "ugelu_erf_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UgeluErfBF16   = "ugelu_erf_bf16"    // Available when __CUDA_ARCH__ >= 800

	// ReLU operations
	UreluF32    = "urelu_f32"
	UreluF64    = "urelu_f64"
	UreluF16    = "urelu_f16"     // Available when __CUDA_ARCH__ >= 530
	UreluF8E4M3 = "urelu_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UreluBF16   = "urelu_bf16"    // Available when __CUDA_ARCH__ >= 800

	// ELU operations
	UeluF32    = "uelu_f32"
	UeluF64    = "uelu_f64"
	UeluF16    = "uelu_f16"     // Available when __CUDA_ARCH__ >= 530
	UeluF8E4M3 = "uelu_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UeluBF16   = "uelu_bf16"    // Available when __CUDA_ARCH__ >= 800

	// SiLU operations
	UsiluF32    = "usilu_f32"
	UsiluF64    = "usilu_f64"
	UsiluF16    = "usilu_f16"     // Available when __CUDA_ARCH__ >= 530
	UsiluF8E4M3 = "usilu_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UsiluBF16   = "usilu_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Power operations
	UpowfF32    = "upowf_f32"
	UpowfF64    = "upowf_f64"
	UpowfF16    = "upowf_f16"     // Available when __CUDA_ARCH__ >= 530
	UpowfF8E4M3 = "upowf_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UpowfBF16   = "upowf_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Sign operations
	UsignF32    = "usign_f32"
	UsignF64    = "usign_f64"
	UsignF16    = "usign_f16"     // Available when __CUDA_ARCH__ >= 530
	UsignF8E4M3 = "usign_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UsignBF16   = "usign_bf16"    // Available when __CUDA_ARCH__ >= 800

	// Sigmoid operations
	UsigmoidF32    = "usigmoid_f32"
	UsigmoidF64    = "usigmoid_f64"
	UsigmoidF16    = "usigmoid_f16"     // Available when __CUDA_ARCH__ >= 530
	UsigmoidF8E4M3 = "usigmoid_f8_e4m3" // Available when __CUDA_ARCH__ >= 890
	UsigmoidBF16   = "usigmoid_bf16"    // Available when __CUDA_ARCH__ >= 800
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

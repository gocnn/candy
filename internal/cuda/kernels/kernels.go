package kernels

import (
	"embed"
)

// Kernel function name constants for cuModuleGetFunction calls
// Keep in sync with the actual kernel function names in *.cu files
const (
	// Affine operations
	AffineF32 = "affine_f32"
	AffineF64 = "affine_f64"
	AffineU8  = "affine_u8"
	AffineU32 = "affine_u32"
	AffineI16 = "affine_i16"
	AffineI32 = "affine_i32"
	AffineI64 = "affine_i64"

	// Binary operations (examples - there are many more)
	AddF32 = "badd_f32"
	AddF64 = "badd_f64"
	MulF32 = "bmul_f32"
	MulF64 = "bmul_f64"
	SubF32 = "bsub_f32"
	SubF64 = "bsub_f64"
	DivF32 = "bdiv_f32"
	DivF64 = "bdiv_f64"

	// Unary operations
	UCopyF32    = "ucopy_f32"
	UCopyF64    = "ucopy_f64"
	UCopyF16    = "ucopy_f16"
	UCopyBF16   = "ucopy_bf16"
	UCopyF8E4M3 = "ucopy_f8_e4m3"
	UCopyU8     = "ucopy_u8"
	UCopyU32    = "ucopy_u32"
	UCopyI64    = "ucopy_i64"

	UNEGF32     = "uneg_f32"
	UNEGF64     = "uneg_f64"
	UNEGF16     = "uneg_f16"
	UNEGBF16    = "uneg_bf16"
	UNEGFP8E4M3 = "uneg_fp8_e4m3"

	UExpF32  = "uexp_f32"
	UExpF64  = "uexp_f64"
	UExpF16  = "uexp_f16"
	UExpBF16 = "uexp_bf16"

	ULogF32  = "ulog_f32"
	ULOGF64  = "ulog_f64"
	ULOGF16  = "ulog_f16"
	ULOGBF16 = "ulog_bf16"

	USinF32  = "usin_f32"
	USinF64  = "usin_f64"
	USinF16  = "usin_f16"
	USinBF16 = "usin_bf16"

	UCosF32  = "ucos_f32"
	UCosF64  = "ucos_f64"
	UCosF16  = "ucos_f16"
	UCosBF16 = "ucos_bf16"

	UTanhF32  = "utanh_f32"
	UTanhF64  = "utanh_f64"
	UTanhF16  = "utanh_f16"
	UTanhBF16 = "utanh_bf16"

	UReluF32  = "urelu_f32"
	UReluF64  = "urelu_f64"
	UReluF16  = "urelu_f16"
	UReluBF16 = "urelu_bf16"

	UGeluF32  = "ugelu_f32"
	UGeluF64  = "ugelu_f64"
	UGeluF16  = "ugelu_f16"
	UGeluBF16 = "ugelu_bf16"

	USigmoidF32  = "usigmoid_f32"
	USigmoidF64  = "usigmoid_f64"
	USigmoidF16  = "usigmoid_f16"
	USigmoidBF16 = "usigmoid_bf16"

	USqrtF32  = "usqrt_f32"
	USqrtF64  = "usqrt_f64"
	USqrtF16  = "usqrt_f16"
	USqrtBF16 = "usqrt_bf16"

	// Fill operations
	FillF32 = "fill_f32"
	FillF64 = "fill_f64"
	FillU8  = "fill_u8"
	FillU32 = "fill_u32"
	FillI64 = "fill_i64"

	// Copy operations
	Copy2DF32 = "copy2d_f32"
	Copy2DF64 = "copy2d_f64"
	Copy2DF16 = "copy2d_f16"
	Copy2DU8  = "copy2d_u8"
	Copy2DU32 = "copy2d_u32"
	Copy2DI64 = "copy2d_i64"

	// Reduce operations
	SumF32    = "sum_f32"
	SumF64    = "sum_f64"
	MinF32    = "min_f32"
	MinF64    = "min_f64"
	MaxF32    = "max_f32"
	MaxF64    = "max_f64"
	ArgMinF32 = "argmin_f32"
	ArgMinF64 = "argmin_f64"
	ArgMaxF32 = "argmax_f32"
	ArgMaxF64 = "argmax_f64"

	// Softmax operations
	SoftmaxF32 = "softmax_f32"
	SoftmaxF64 = "softmax_f64"
	SoftmaxF16 = "softmax_f16"

	// Layer normalization
	LayerNormF32 = "layernorm_f32"
	LayerNormF64 = "layernorm_f64"
	LayerNormF16 = "layernorm_f16"

	// RMS normalization
	RMSNormF32 = "rmsnorm_f32"
	RMSNormF64 = "rmsnorm_f64"
	RMSNormF16 = "rmsnorm_f16"

	// Sort operations
	ArgSortAscF32  = "asort_asc_f32"
	ArgSortDescF32 = "asort_desc_f32"
	ArgSortAscF64  = "asort_asc_f64"
	ArgSortDescF64 = "asort_desc_f64"

	// Ternary operations (where)
	WhereI64F32 = "where_i64_f32"
	WhereI64F64 = "where_i64_f64"
	WhereU32F32 = "where_u32_f32"
	WhereU32F64 = "where_u32_f64"

	// Cast operations
	CastF32ToF64 = "cast_f32_f64"
	CastF64ToF32 = "cast_f64_f32"
	CastF32ToF16 = "cast_f32_f16"
	CastF16ToF32 = "cast_f16_f32"
	CastF32ToU8  = "cast_f32_u8"
	CastU8ToF32  = "cast_u8_f32"

	// Convolution operations
	Conv1DF32 = "conv1d_f32"
	Conv1DF64 = "conv1d_f64"
	Conv2DF32 = "conv2d_f32"
	Conv2DF64 = "conv2d_f64"

	// Indexing operations
	IndexSelectF32 = "index_select_f32"
	IndexSelectF64 = "index_select_f64"
	IndexAddF32    = "index_add_f32"
	IndexAddF64    = "index_add_f64"
	GatherF32      = "gather_f32"
	GatherF64      = "gather_f64"
	ScatterF32     = "scatter_f32"
	ScatterF64     = "scatter_f64"
)

//go:embed *.ptx
var Kernels embed.FS

func GetKernel(name string) ([]byte, error) {
	if len(name) < 4 || name[len(name)-4:] != ".ptx" {
		name = name + ".ptx"
	}
	return Kernels.ReadFile(name)
}

// ListKernels returns all available kernel names
func ListKernels() ([]string, error) {
	entries, err := Kernels.ReadDir(".")
	if err != nil {
		return nil, err
	}

	var kernels []string
	for _, entry := range entries {
		if !entry.IsDir() && len(entry.Name()) > 4 && entry.Name()[len(entry.Name())-4:] == ".ptx" {
			kernels = append(kernels, entry.Name()[:len(entry.Name())-4])
		}
	}
	return kernels, nil
}

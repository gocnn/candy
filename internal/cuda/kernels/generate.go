package kernels

//go:generate nvcc --ptx -Wno-deprecated-gpu-targets -diag-suppress 173 affine.cu -o affine.ptx
//go:generate nvcc --ptx -Wno-deprecated-gpu-targets -diag-suppress 173 binary.cu -o binary.ptx
//go:generate nvcc --ptx -Wno-deprecated-gpu-targets -diag-suppress 173 cast.cu -o cast.ptx
//go:generate nvcc --ptx -Wno-deprecated-gpu-targets -diag-suppress 173 conv.cu -o conv.ptx
//go:generate nvcc --ptx -Wno-deprecated-gpu-targets -diag-suppress 173 fill.cu -o fill.ptx
//go:generate nvcc --ptx -Wno-deprecated-gpu-targets -diag-suppress 173 indexing.cu -o indexing.ptx
//go:generate nvcc --ptx -Wno-deprecated-gpu-targets -diag-suppress 173 quantized.cu -o quantized.ptx
//go:generate nvcc --ptx -Wno-deprecated-gpu-targets -diag-suppress 173 reduce.cu -o reduce.ptx
//go:generate nvcc --ptx -Wno-deprecated-gpu-targets -diag-suppress 173 sort.cu -o sort.ptx
//go:generate nvcc --ptx -Wno-deprecated-gpu-targets -diag-suppress 173 ternary.cu -o ternary.ptx
//go:generate nvcc --ptx -Wno-deprecated-gpu-targets -diag-suppress 173 unary.cu -o unary.ptx

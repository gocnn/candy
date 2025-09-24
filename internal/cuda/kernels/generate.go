package kernels

//go:generate nvcc --ptx affine.cu -o affine.ptx
//go:generate nvcc --ptx binary.cu -o binary.ptx
//go:generate nvcc --ptx cast.cu -o cast.ptx
//go:generate nvcc --ptx conv.cu -o conv.ptx
//go:generate nvcc --ptx fill.cu -o fill.ptx
//go:generate nvcc --ptx indexing.cu -o indexing.ptx
//go:generate nvcc --ptx quantized.cu -o quantized.ptx
//go:generate nvcc --ptx reduce.cu -o reduce.ptx
//go:generate nvcc --ptx sort.cu -o sort.ptx
//go:generate nvcc --ptx ternary.cu -o ternary.ptx
//go:generate nvcc --ptx unary.cu -o unary.ptx

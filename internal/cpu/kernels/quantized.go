package kernels

import (
	"math"
)

// BlockQ4_0 represents a q4_0 block
type BlockQ4_0 struct {
	D  uint16   // delta (as ggml_fp16_t)
	Qs [16]byte // nibbles / quants
}

// BlockQ4_1 represents a q4_1 block
type BlockQ4_1 struct {
	DM [2]uint16 // dm.x = delta, dm.y = min (as ggml_fp16_t)
	Qs [16]byte  // nibbles / quants
}

// BlockQ5_0 represents a q5_0 block
type BlockQ5_0 struct {
	D  uint16   // delta (as ggml_fp16_t)
	Qh [4]byte  // 5-th bit of quants
	Qs [16]byte // nibbles / quants
}

// BlockQ5_1 represents a q5_1 block
type BlockQ5_1 struct {
	DM [2]uint16 // dm.x = delta, dm.y = min (as ggml_fp16_t)
	Qh [4]byte   // 5-th bit of quants
	Qs [16]byte  // nibbles / quants
}

// BlockQ8_0 represents a q8_0 block
type BlockQ8_0 struct {
	D  uint16   // delta (as ggml_fp16_t)
	Qs [32]int8 // quants
}

// BlockQ2K represents a q2_K block
type BlockQ2K struct {
	Scales [16]byte  // scales and mins, quantized with 4 bits
	Qs     [64]byte  // quants
	DM     [2]uint16 // super-block scale for quantized scales/mins (as ggml_fp16_t)
}

// BlockQ3K represents a q3_K block
type BlockQ3K struct {
	Hmask  [32]byte // quants - high bit
	Qs     [64]byte // quants - low 2 bits
	Scales [12]byte // scales, quantized with 6 bits
	D      uint16   // super-block scale (as ggml_fp16_t)
}

// BlockQ4K represents a q4_K block
type BlockQ4K struct {
	DM     [2]uint16 // super-block scale for quantized scales/mins (as ggml_fp16_t)
	Scales [12]byte  // scales and mins, quantized with 6 bits
	Qs     [128]byte // 4-bit quants
}

// BlockQ5K represents a q5_K block
type BlockQ5K struct {
	DM     [2]uint16 // super-block scale for quantized scales/mins (as ggml_fp16_t)
	Scales [12]byte  // scales and mins, quantized with 6 bits
	Qh     [32]byte  // quants, high bit
	Qs     [128]byte // quants, low 4 bits
}

// BlockQ6K represents a q6_K block
type BlockQ6K struct {
	Ql     [128]byte // quants, lower 4 bits
	Qh     [64]byte  // quants, upper 2 bits
	Scales [16]int8  // scales
	D      uint16    // delta (as ggml_fp16_t)
}

// BlockQ8K represents a q8_K block
type BlockQ8K struct {
	D     float32   // delta
	Qs    [256]int8 // quants
	Bsums [16]int16 // sum of quants in groups of 16
}

// DequantizeBlockQ4_0F32 dequantizes a q4_0 block to float32 (contiguous)
func DequantizeBlockQ4_0F32(b *BlockQ4_0, y []float32) {
	d := fp16ToFloat32(b.D)
	for i := 0; i < 32; i += 2 {
		q1 := int(b.Qs[i/2]) & 0xF
		q2 := int(b.Qs[i/2]) >> 4
		y[i] = d * (float32(q1) - 8)
		y[i+1] = d * (float32(q2) - 8)
	}
}

// DequantizeBlockQ4_0F64 dequantizes a q4_0 block to float64 (contiguous)
func DequantizeBlockQ4_0F64(b *BlockQ4_0, y []float64) {
	d := float64(fp16ToFloat32(b.D))
	for i := 0; i < 32; i += 2 {
		q1 := int(b.Qs[i/2]) & 0xF
		q2 := int(b.Qs[i/2]) >> 4
		y[i] = d * (float64(q1) - 8)
		y[i+1] = d * (float64(q2) - 8)
	}
}

// DequantizeBlockQ4_1F32 dequantizes a q4_1 block to float32 (contiguous)
func DequantizeBlockQ4_1F32(b *BlockQ4_1, y []float32) {
	d := fp16ToFloat32(b.DM[0])
	m := fp16ToFloat32(b.DM[1])
	for i := 0; i < 32; i += 2 {
		q1 := int(b.Qs[i/2]) & 0xF
		q2 := int(b.Qs[i/2]) >> 4
		y[i] = d*float32(q1) + m
		y[i+1] = d*float32(q2) + m
	}
}

// DequantizeBlockQ4_1F64 dequantizes a q4_1 block to float64 (contiguous)
func DequantizeBlockQ4_1F64(b *BlockQ4_1, y []float64) {
	d := float64(fp16ToFloat32(b.DM[0]))
	m := float64(fp16ToFloat32(b.DM[1]))
	for i := 0; i < 32; i += 2 {
		q1 := int(b.Qs[i/2]) & 0xF
		q2 := int(b.Qs[i/2]) >> 4
		y[i] = d*float64(q1) + m
		y[i+1] = d*float64(q2) + m
	}
}

// DequantizeBlockQ8_0F32 dequantizes a q8_0 block to float32 (contiguous)
func DequantizeBlockQ8_0F32(b *BlockQ8_0, y []float32) {
	d := fp16ToFloat32(b.D)
	for i := range 32 {
		y[i] = d * float32(b.Qs[i])
	}
}

// DequantizeBlockQ8_0F64 dequantizes a q8_0 block to float64 (contiguous)
func DequantizeBlockQ8_0F64(b *BlockQ8_0, y []float64) {
	d := float64(fp16ToFloat32(b.D))
	for i := range 32 {
		y[i] = d * float64(b.Qs[i])
	}
}

// fp16ToFloat32 converts an IEEE 754 half-precision (FP16) float to single-precision (float32).
// FP16 format: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits.
// float32 format: 1 sign bit, 8 exponent bits (bias 127), 23 mantissa bits.
func fp16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16 // Extract sign bit and shift to float32 position
	exp := uint32(h & 0x7C00)      // Extract 5-bit exponent
	mant := uint32(h & 0x03FF)     // Extract 10-bit mantissa

	switch exp {
	case 0x7C00: // Infinity or NaN
		return math.Float32frombits(sign | 0x7F800000 | (mant << 13))
	case 0: // Zero or denormal
		if mant == 0 { // Zero
			return math.Float32frombits(sign)
		}
		// Convert denormal to normalized float32
		exp = (127 - 14) << 23 // 113 << 23 = 0x3F100000
		for mant&0x400 == 0 {  // Normalize mantissa (find leading 1)
			mant <<= 1
			exp -= 1 << 23 // Decrease exponent by 1
		}
		mant &= 0x3FF // Remove the leading 1
		return math.Float32frombits(sign | exp | (mant << 13))
	default: // Normalized number
		exp32 := (((exp >> 10) - 15 + 127) << 23)
		return math.Float32frombits(sign | exp32 | (mant << 13))
	}
}

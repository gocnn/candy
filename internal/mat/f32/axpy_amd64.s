#include "textflag.h"

// func AxpyUnitary(alpha float32, x, y []float32)
TEXT ·AxpyUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+8(FP), SI  // SI = &x
	MOVQ    y_base+32(FP), DI // DI = &y
	MOVQ    x_len+16(FP), BX  // BX = min( len(x), len(y) )
	CMPQ    y_len+40(FP), BX
	CMOVQLE y_len+40(FP), BX
	CMPQ    BX, $0            // if BX == 0 { return }
	JE      axpy_end
	MOVSS   alpha+0(FP), X0
	SHUFPS  $0, X0, X0        // X0 = { a, a, a, a }
	XORQ    AX, AX            // i = 0
	PXOR    X2, X2            // 2 NOP instructions (PXOR) to align
	PXOR    X3, X3            // loop to cache line
	MOVQ    DI, CX
	ANDQ    $0xF, CX          // Align on 16-byte boundary for ADDPS
	JZ      axpy_no_trim      // if CX == 0 { goto axpy_no_trim }

	XORQ $0xF, CX // CX = 4 - floor( BX % 16 / 4 )
	INCQ CX
	SHRQ $2, CX

axpy_align: // Trim first value(s) in unaligned buffer  do {
	MOVSS (SI)(AX*4), X2 // X2 = x[i]
	MULSS X0, X2         // X2 *= a
	ADDSS (DI)(AX*4), X2 // X2 += y[i]
	MOVSS X2, (DI)(AX*4) // y[i] = X2
	INCQ  AX             // i++
	DECQ  BX
	JZ    axpy_end       // if --BX == 0 { return }
	LOOP  axpy_align     // } while --CX > 0

axpy_no_trim:
	MOVUPS X0, X1           // Copy X0 to X1 for pipelining
	MOVQ   BX, CX
	ANDQ   $0xF, BX         // BX = len % 16
	SHRQ   $4, CX           // CX = int( len / 16 )
	JZ     axpy_tail4_start // if CX == 0 { return }

axpy_loop: // Loop unrolled 16x   do {
	MOVUPS (SI)(AX*4), X2   // X2 = x[i:i+4]
	MOVUPS 16(SI)(AX*4), X3
	MOVUPS 32(SI)(AX*4), X4
	MOVUPS 48(SI)(AX*4), X5
	MULPS  X0, X2           // X2 *= a
	MULPS  X1, X3
	MULPS  X0, X4
	MULPS  X1, X5
	ADDPS  (DI)(AX*4), X2   // X2 += y[i:i+4]
	ADDPS  16(DI)(AX*4), X3
	ADDPS  32(DI)(AX*4), X4
	ADDPS  48(DI)(AX*4), X5
	MOVUPS X2, (DI)(AX*4)   // dst[i:i+4] = X2
	MOVUPS X3, 16(DI)(AX*4)
	MOVUPS X4, 32(DI)(AX*4)
	MOVUPS X5, 48(DI)(AX*4)
	ADDQ   $16, AX          // i += 16
	LOOP   axpy_loop        // while (--CX) > 0
	CMPQ   BX, $0           // if BX == 0 { return }
	JE     axpy_end

axpy_tail4_start: // Reset loop counter for 4-wide tail loop
	MOVQ BX, CX          // CX = floor( BX / 4 )
	SHRQ $2, CX
	JZ   axpy_tail_start // if CX == 0 { goto axpy_tail_start }

axpy_tail4: // Loop unrolled 4x   do {
	MOVUPS (SI)(AX*4), X2 // X2 = x[i]
	MULPS  X0, X2         // X2 *= a
	ADDPS  (DI)(AX*4), X2 // X2 += y[i]
	MOVUPS X2, (DI)(AX*4) // y[i] = X2
	ADDQ   $4, AX         // i += 4
	LOOP   axpy_tail4     // } while --CX > 0

axpy_tail_start: // Reset loop counter for 1-wide tail loop
	MOVQ BX, CX   // CX = BX % 4
	ANDQ $3, CX
	JZ   axpy_end // if CX == 0 { return }

axpy_tail:
	MOVSS (SI)(AX*4), X1 // X1 = x[i]
	MULSS X0, X1         // X1 *= a
	ADDSS (DI)(AX*4), X1 // X1 += y[i]
	MOVSS X1, (DI)(AX*4) // y[i] = X1
	INCQ  AX             // i++
	LOOP  axpy_tail      // } while --CX > 0

axpy_end:
	RET

// func AxpyUnitaryTo(dst []float32, alpha float32, x, y []float32)
TEXT ·AxpyUnitaryTo(SB), NOSPLIT, $0
	MOVQ    dst_base+0(FP), DI // DI = &dst
	MOVQ    x_base+32(FP), SI  // SI = &x
	MOVQ    y_base+56(FP), DX  // DX = &y
	MOVQ    x_len+40(FP), BX   // BX = min( len(x), len(y), len(dst) )
	CMPQ    y_len+64(FP), BX
	CMOVQLE y_len+64(FP), BX
	CMPQ    dst_len+8(FP), BX
	CMOVQLE dst_len+8(FP), BX
	CMPQ    BX, $0             // if BX == 0 { return }
	JE      axpy_end
	MOVSS   alpha+24(FP), X0
	SHUFPS  $0, X0, X0         // X0 = { a, a, a, a, }
	XORQ    AX, AX             // i = 0
	MOVQ    DX, CX
	ANDQ    $0xF, CX           // Align on 16-byte boundary for ADDPS
	JZ      axpy_no_trim       // if CX == 0 { goto axpy_no_trim }

	XORQ $0xF, CX // CX = 4 - floor ( B % 16 / 4 )
	INCQ CX
	SHRQ $2, CX

axpy_align: // Trim first value(s) in unaligned buffer  do {
	MOVSS (SI)(AX*4), X2 // X2 = x[i]
	MULSS X0, X2         // X2 *= a
	ADDSS (DX)(AX*4), X2 // X2 += y[i]
	MOVSS X2, (DI)(AX*4) // y[i] = X2
	INCQ  AX             // i++
	DECQ  BX
	JZ    axpy_end       // if --BX == 0 { return }
	LOOP  axpy_align     // } while --CX > 0

axpy_no_trim:
	MOVUPS X0, X1           // Copy X0 to X1 for pipelining
	MOVQ   BX, CX
	ANDQ   $0xF, BX         // BX = len % 16
	SHRQ   $4, CX           // CX = floor( len / 16 )
	JZ     axpy_tail4_start // if CX == 0 { return }

axpy_loop: // Loop unrolled 16x  do {
	MOVUPS (SI)(AX*4), X2   // X2 = x[i:i+4]
	MOVUPS 16(SI)(AX*4), X3
	MOVUPS 32(SI)(AX*4), X4
	MOVUPS 48(SI)(AX*4), X5
	MULPS  X0, X2           // X2 *= a
	MULPS  X1, X3
	MULPS  X0, X4
	MULPS  X1, X5
	ADDPS  (DX)(AX*4), X2   // X2 += y[i:i+4]
	ADDPS  16(DX)(AX*4), X3
	ADDPS  32(DX)(AX*4), X4
	ADDPS  48(DX)(AX*4), X5
	MOVUPS X2, (DI)(AX*4)   // dst[i:i+4] = X2
	MOVUPS X3, 16(DI)(AX*4)
	MOVUPS X4, 32(DI)(AX*4)
	MOVUPS X5, 48(DI)(AX*4)
	ADDQ   $16, AX          // i += 16
	LOOP   axpy_loop        // while (--CX) > 0
	CMPQ   BX, $0           // if BX == 0 { return }
	JE     axpy_end

axpy_tail4_start: // Reset loop counter for 4-wide tail loop
	MOVQ BX, CX          // CX = floor( BX / 4 )
	SHRQ $2, CX
	JZ   axpy_tail_start // if CX == 0 { goto axpy_tail_start }

axpy_tail4: // Loop unrolled 4x  do {
	MOVUPS (SI)(AX*4), X2 // X2 = x[i]
	MULPS  X0, X2         // X2 *= a
	ADDPS  (DX)(AX*4), X2 // X2 += y[i]
	MOVUPS X2, (DI)(AX*4) // y[i] = X2
	ADDQ   $4, AX         // i += 4
	LOOP   axpy_tail4     // } while --CX > 0

axpy_tail_start: // Reset loop counter for 1-wide tail loop
	MOVQ BX, CX   // CX = BX % 4
	ANDQ $3, CX
	JZ   axpy_end // if CX == 0 { return }

axpy_tail:
	MOVSS (SI)(AX*4), X1 // X1 = x[i]
	MULSS X0, X1         // X1 *= a
	ADDSS (DX)(AX*4), X1 // X1 += y[i]
	MOVSS X1, (DI)(AX*4) // y[i] = X1
	INCQ  AX             // i++
	LOOP  axpy_tail      // } while --CX > 0

axpy_end:
	RET

// func AxpyInc(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr)
TEXT ·AxpyInc(SB), NOSPLIT, $0
	MOVQ  n+56(FP), CX      // CX = n
	CMPQ  CX, $0            // if n==0 { return }
	JLE   axpyi_end
	MOVQ  x_base+8(FP), SI  // SI = &x
	MOVQ  y_base+32(FP), DI // DI = &y
	MOVQ  ix+80(FP), R8     // R8 = ix
	MOVQ  iy+88(FP), R9     // R9 = iy
	LEAQ  (SI)(R8*4), SI    // SI = &(x[ix])
	LEAQ  (DI)(R9*4), DI    // DI = &(y[iy])
	MOVQ  DI, DX            // DX = DI   Read Pointer for y
	MOVQ  incX+64(FP), R8   // R8 = incX
	SHLQ  $2, R8            // R8 *= sizeof(float32)
	MOVQ  incY+72(FP), R9   // R9 = incY
	SHLQ  $2, R9            // R9 *= sizeof(float32)
	MOVSS alpha+0(FP), X0   // X0 = alpha
	MOVSS X0, X1            // X1 = X0  // for pipelining
	MOVQ  CX, BX
	ANDQ  $3, BX            // BX = n % 4
	SHRQ  $2, CX            // CX = floor( n / 4 )
	JZ    axpyi_tail_start  // if CX == 0 { goto axpyi_tail_start }

axpyi_loop: // Loop unrolled 4x   do {
	MOVSS (SI), X2       // X_i = x[i]
	MOVSS (SI)(R8*1), X3
	LEAQ  (SI)(R8*2), SI // SI = &(SI[incX*2])
	MOVSS (SI), X4
	MOVSS (SI)(R8*1), X5
	MULSS X1, X2         // X_i *= a
	MULSS X0, X3
	MULSS X1, X4
	MULSS X0, X5
	ADDSS (DX), X2       // X_i += y[i]
	ADDSS (DX)(R9*1), X3
	LEAQ  (DX)(R9*2), DX // DX = &(DX[incY*2])
	ADDSS (DX), X4
	ADDSS (DX)(R9*1), X5
	MOVSS X2, (DI)       // y[i] = X_i
	MOVSS X3, (DI)(R9*1)
	LEAQ  (DI)(R9*2), DI // DI = &(DI[incY*2])
	MOVSS X4, (DI)
	MOVSS X5, (DI)(R9*1)
	LEAQ  (SI)(R8*2), SI // SI = &(SI[incX*2])  // Increment addresses
	LEAQ  (DX)(R9*2), DX // DX = &(DX[incY*2])
	LEAQ  (DI)(R9*2), DI // DI = &(DI[incY*2])
	LOOP  axpyi_loop     // } while --CX > 0
	CMPQ  BX, $0         // if BX == 0 { return }
	JE    axpyi_end

axpyi_tail_start: // Reset loop registers
	MOVQ BX, CX // Loop counter: CX = BX

axpyi_tail: // do {
	MOVSS (SI), X2   // X2 = x[i]
	MULSS X1, X2     // X2 *= a
	ADDSS (DI), X2   // X2 += y[i]
	MOVSS X2, (DI)   // y[i] = X2
	ADDQ  R8, SI     // SI = &(SI[incX])
	ADDQ  R9, DI     // DI = &(DI[incY])
	LOOP  axpyi_tail // } while --CX > 0

axpyi_end:
	RET

// func AxpyIncTo(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr)
TEXT ·AxpyIncTo(SB), NOSPLIT, $0
	MOVQ  n+96(FP), CX       // CX = n
	CMPQ  CX, $0             // if n==0 { return }
	JLE   axpyi_end
	MOVQ  dst_base+0(FP), DI // DI = &dst
	MOVQ  x_base+48(FP), SI  // SI = &x
	MOVQ  y_base+72(FP), DX  // DX = &y
	MOVQ  ix+120(FP), R8     // R8 = ix  // Load the first index
	MOVQ  iy+128(FP), R9     // R9 = iy
	MOVQ  idst+32(FP), R10   // R10 = idst
	LEAQ  (SI)(R8*4), SI     // SI = &(x[ix])
	LEAQ  (DX)(R9*4), DX     // DX = &(y[iy])
	LEAQ  (DI)(R10*4), DI    // DI = &(dst[idst])
	MOVQ  incX+104(FP), R8   // R8 = incX
	SHLQ  $2, R8             // R8 *= sizeof(float32)
	MOVQ  incY+112(FP), R9   // R9 = incY
	SHLQ  $2, R9             // R9 *= sizeof(float32)
	MOVQ  incDst+24(FP), R10 // R10 = incDst
	SHLQ  $2, R10            // R10 *= sizeof(float32)
	MOVSS alpha+40(FP), X0   // X0 = alpha
	MOVSS X0, X1             // X1 = X0  // for pipelining
	MOVQ  CX, BX
	ANDQ  $3, BX             // BX = n % 4
	SHRQ  $2, CX             // CX = floor( n / 4 )
	JZ    axpyi_tail_start   // if CX == 0 { goto axpyi_tail_start }

axpyi_loop: // Loop unrolled 4x   do {
	MOVSS (SI), X2        // X_i = x[i]
	MOVSS (SI)(R8*1), X3
	LEAQ  (SI)(R8*2), SI  // SI = &(SI[incX*2])
	MOVSS (SI), X4
	MOVSS (SI)(R8*1), X5
	MULSS X1, X2          // X_i *= a
	MULSS X0, X3
	MULSS X1, X4
	MULSS X0, X5
	ADDSS (DX), X2        // X_i += y[i]
	ADDSS (DX)(R9*1), X3
	LEAQ  (DX)(R9*2), DX  // DX = &(DX[incY*2])
	ADDSS (DX), X4
	ADDSS (DX)(R9*1), X5
	MOVSS X2, (DI)        // dst[i] = X_i
	MOVSS X3, (DI)(R10*1)
	LEAQ  (DI)(R10*2), DI // DI = &(DI[incDst*2])
	MOVSS X4, (DI)
	MOVSS X5, (DI)(R10*1)
	LEAQ  (SI)(R8*2), SI  // SI = &(SI[incX*2])  // Increment addresses
	LEAQ  (DX)(R9*2), DX  // DX = &(DX[incY*2])
	LEAQ  (DI)(R10*2), DI // DI = &(DI[incDst*2])
	LOOP  axpyi_loop      // } while --CX > 0
	CMPQ  BX, $0          // if BX == 0 { return }
	JE    axpyi_end

axpyi_tail_start: // Reset loop registers
	MOVQ BX, CX // Loop counter: CX = BX

axpyi_tail: // do {
	MOVSS (SI), X2   // X2 = x[i]
	MULSS X1, X2     // X2 *= a
	ADDSS (DX), X2   // X2 += y[i]
	MOVSS X2, (DI)   // dst[i] = X2
	ADDQ  R8, SI     // SI = &(SI[incX])
	ADDQ  R9, DX     // DX = &(DX[incY])
	ADDQ  R10, DI    // DI = &(DI[incY])
	LOOP  axpyi_tail // } while --CX > 0

axpyi_end:
	RET


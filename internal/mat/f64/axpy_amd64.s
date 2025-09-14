#include "textflag.h"

#define X_PTR SI
#define Y_PTR DI
#define DST_PTR DI
#define IDX AX
#define LEN CX
#define TAIL BX
#define INC_X R8
#define INCx3_X R11
#define INC_Y R9
#define INCx3_Y R12
#define INC_DST R10
#define INCx3_DST R13
#define ALPHA X0
#define ALPHA_2 X1

// func AxpyUnitary(alpha float64, x, y []float64)
TEXT ·AxpyUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+8(FP), X_PTR  // X_PTR := &x
	MOVQ    y_base+32(FP), Y_PTR // Y_PTR := &y
	MOVQ    x_len+16(FP), LEN    // LEN = min( len(x), len(y) )
	CMPQ    y_len+40(FP), LEN
	CMOVQLE y_len+40(FP), LEN
	CMPQ    LEN, $0              // if LEN == 0 { return }
	JE      end
	XORQ    IDX, IDX
	MOVSD   alpha+0(FP), ALPHA   // ALPHA := { alpha, alpha }
	SHUFPD  $0, ALPHA, ALPHA
	MOVUPS  ALPHA, ALPHA_2       // ALPHA_2 := ALPHA   for pipelining
	MOVQ    Y_PTR, TAIL          // Check memory alignment
	ANDQ    $15, TAIL            // TAIL = &y % 16
	JZ      no_trim              // if TAIL == 0 { goto no_trim }

	// Align on 16-byte boundary
	MOVSD (X_PTR), X2   // X2 := x[0]
	MULSD ALPHA, X2     // X2 *= a
	ADDSD (Y_PTR), X2   // X2 += y[0]
	MOVSD X2, (DST_PTR) // y[0] = X2
	INCQ  IDX           // i++
	DECQ  LEN           // LEN--
	JZ    end           // if LEN == 0 { return }

no_trim:
	MOVQ LEN, TAIL
	ANDQ $7, TAIL   // TAIL := n % 8
	SHRQ $3, LEN    // LEN = floor( n / 8 )
	JZ   tail_start // if LEN == 0 { goto tail2_start }

loop:  // do {
	// y[i] += alpha * x[i] unrolled 8x.
	MOVUPS (X_PTR)(IDX*8), X2   // X_i = x[i]
	MOVUPS 16(X_PTR)(IDX*8), X3
	MOVUPS 32(X_PTR)(IDX*8), X4
	MOVUPS 48(X_PTR)(IDX*8), X5

	MULPD ALPHA, X2   // X_i *= a
	MULPD ALPHA_2, X3
	MULPD ALPHA, X4
	MULPD ALPHA_2, X5

	ADDPD (Y_PTR)(IDX*8), X2   // X_i += y[i]
	ADDPD 16(Y_PTR)(IDX*8), X3
	ADDPD 32(Y_PTR)(IDX*8), X4
	ADDPD 48(Y_PTR)(IDX*8), X5

	MOVUPS X2, (DST_PTR)(IDX*8)   // y[i] = X_i
	MOVUPS X3, 16(DST_PTR)(IDX*8)
	MOVUPS X4, 32(DST_PTR)(IDX*8)
	MOVUPS X5, 48(DST_PTR)(IDX*8)

	ADDQ $8, IDX  // i += 8
	DECQ LEN
	JNZ  loop     // } while --LEN > 0
	CMPQ TAIL, $0 // if TAIL == 0 { return }
	JE   end

tail_start: // Reset loop registers
	MOVQ TAIL, LEN // Loop counter: LEN = TAIL
	SHRQ $1, LEN   // LEN = floor( TAIL / 2 )
	JZ   tail_one  // if TAIL == 0 { goto tail }

tail_two: // do {
	MOVUPS (X_PTR)(IDX*8), X2   // X2 = x[i]
	MULPD  ALPHA, X2            // X2 *= a
	ADDPD  (Y_PTR)(IDX*8), X2   // X2 += y[i]
	MOVUPS X2, (DST_PTR)(IDX*8) // y[i] = X2
	ADDQ   $2, IDX              // i += 2
	DECQ   LEN
	JNZ    tail_two             // } while --LEN > 0

	ANDQ $1, TAIL
	JZ   end      // if TAIL == 0 { goto end }

tail_one:
	MOVSD (X_PTR)(IDX*8), X2   // X2 = x[i]
	MULSD ALPHA, X2            // X2 *= a
	ADDSD (Y_PTR)(IDX*8), X2   // X2 += y[i]
	MOVSD X2, (DST_PTR)(IDX*8) // y[i] = X2

end:
	RET

// func AxpyUnitaryTo(dst []float64, alpha float64, x, y []float64)
TEXT ·AxpyUnitaryTo(SB), NOSPLIT, $0
	MOVQ    dst_base+0(FP), DST_PTR // DST_PTR := &dst
	MOVQ    x_base+32(FP), X_PTR    // X_PTR := &x
	MOVQ    y_base+56(FP), Y_PTR    // Y_PTR := &y
	MOVQ    x_len+40(FP), LEN       // LEN = min( len(x), len(y), len(dst) )
	CMPQ    y_len+64(FP), LEN
	CMOVQLE y_len+64(FP), LEN
	CMPQ    dst_len+8(FP), LEN
	CMOVQLE dst_len+8(FP), LEN

	CMPQ LEN, $0
	JE   end     // if LEN == 0 { return }

	XORQ   IDX, IDX            // IDX = 0
	MOVSD  alpha+24(FP), ALPHA
	SHUFPD $0, ALPHA, ALPHA    // ALPHA := { alpha, alpha }
	MOVQ   Y_PTR, TAIL         // Check memory alignment
	ANDQ   $15, TAIL           // TAIL = &y % 16
	JZ     no_trim             // if TAIL == 0 { goto no_trim }

	// Align on 16-byte boundary
	MOVSD (X_PTR), X2   // X2 := x[0]
	MULSD ALPHA, X2     // X2 *= a
	ADDSD (Y_PTR), X2   // X2 += y[0]
	MOVSD X2, (DST_PTR) // y[0] = X2
	INCQ  IDX           // i++
	DECQ  LEN           // LEN--
	JZ    end           // if LEN == 0 { return }

no_trim:
	MOVQ LEN, TAIL
	ANDQ $7, TAIL   // TAIL := n % 8
	SHRQ $3, LEN    // LEN = floor( n / 8 )
	JZ   tail_start // if LEN == 0 { goto tail_start }

	MOVUPS ALPHA, ALPHA_2 // ALPHA_2 := ALPHA  for pipelining

loop:  // do {
	// y[i] += alpha * x[i] unrolled 8x.
	MOVUPS (X_PTR)(IDX*8), X2   // X_i = x[i]
	MOVUPS 16(X_PTR)(IDX*8), X3
	MOVUPS 32(X_PTR)(IDX*8), X4
	MOVUPS 48(X_PTR)(IDX*8), X5

	MULPD ALPHA, X2   // X_i *= alpha
	MULPD ALPHA_2, X3
	MULPD ALPHA, X4
	MULPD ALPHA_2, X5

	ADDPD (Y_PTR)(IDX*8), X2   // X_i += y[i]
	ADDPD 16(Y_PTR)(IDX*8), X3
	ADDPD 32(Y_PTR)(IDX*8), X4
	ADDPD 48(Y_PTR)(IDX*8), X5

	MOVUPS X2, (DST_PTR)(IDX*8)   // y[i] = X_i
	MOVUPS X3, 16(DST_PTR)(IDX*8)
	MOVUPS X4, 32(DST_PTR)(IDX*8)
	MOVUPS X5, 48(DST_PTR)(IDX*8)

	ADDQ $8, IDX  // i += 8
	DECQ LEN
	JNZ  loop     // } while --LEN > 0
	CMPQ TAIL, $0 // if TAIL == 0 { return }
	JE   end

tail_start: // Reset loop registers
	MOVQ TAIL, LEN // Loop counter: LEN = TAIL
	SHRQ $1, LEN   // LEN = floor( TAIL / 2 )
	JZ   tail_one  // if LEN == 0 { goto tail }

tail_two: // do {
	MOVUPS (X_PTR)(IDX*8), X2   // X2 = x[i]
	MULPD  ALPHA, X2            // X2 *= alpha
	ADDPD  (Y_PTR)(IDX*8), X2   // X2 += y[i]
	MOVUPS X2, (DST_PTR)(IDX*8) // y[i] = X2
	ADDQ   $2, IDX              // i += 2
	DECQ   LEN
	JNZ    tail_two             // } while --LEN > 0

	ANDQ $1, TAIL
	JZ   end      // if TAIL == 0 { goto end }

tail_one:
	MOVSD (X_PTR)(IDX*8), X2   // X2 = x[i]
	MULSD ALPHA, X2            // X2 *= a
	ADDSD (Y_PTR)(IDX*8), X2   // X2 += y[i]
	MOVSD X2, (DST_PTR)(IDX*8) // y[i] = X2

end:
	RET

// func AxpyInc(alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr)
TEXT ·AxpyInc(SB), NOSPLIT, $0
	MOVQ x_base+8(FP), X_PTR  // X_PTR = &x
	MOVQ y_base+32(FP), Y_PTR // Y_PTR = &y
	MOVQ n+56(FP), LEN        // LEN = n
	CMPQ LEN, $0              // if LEN == 0 { return }
	JE   end

	MOVQ ix+80(FP), INC_X
	MOVQ iy+88(FP), INC_Y
	LEAQ (X_PTR)(INC_X*8), X_PTR // X_PTR = &(x[ix])
	LEAQ (Y_PTR)(INC_Y*8), Y_PTR // Y_PTR = &(y[iy])
	MOVQ Y_PTR, DST_PTR          // DST_PTR = Y_PTR  // Write pointer

	MOVQ incX+64(FP), INC_X // INC_X = incX * sizeof(float64)
	SHLQ $3, INC_X
	MOVQ incY+72(FP), INC_Y // INC_Y = incY * sizeof(float64)
	SHLQ $3, INC_Y

	MOVSD alpha+0(FP), ALPHA // ALPHA = alpha
	MOVQ  LEN, TAIL
	ANDQ  $3, TAIL           // TAIL = n % 4
	SHRQ  $2, LEN            // LEN = floor( n / 4 )
	JZ    tail_start         // if LEN == 0 { goto tail_start }

	MOVAPS ALPHA, ALPHA_2            // ALPHA_2 = ALPHA  for pipelining
	LEAQ   (INC_X)(INC_X*2), INCx3_X // INCx3_X = INC_X * 3
	LEAQ   (INC_Y)(INC_Y*2), INCx3_Y // INCx3_Y = INC_Y * 3

loop:  // do {  // y[i] += alpha * x[i] unrolled 4x.
	MOVSD (X_PTR), X2            // X_i = x[i]
	MOVSD (X_PTR)(INC_X*1), X3
	MOVSD (X_PTR)(INC_X*2), X4
	MOVSD (X_PTR)(INCx3_X*1), X5

	MULSD ALPHA, X2   // X_i *= a
	MULSD ALPHA_2, X3
	MULSD ALPHA, X4
	MULSD ALPHA_2, X5

	ADDSD (Y_PTR), X2            // X_i += y[i]
	ADDSD (Y_PTR)(INC_Y*1), X3
	ADDSD (Y_PTR)(INC_Y*2), X4
	ADDSD (Y_PTR)(INCx3_Y*1), X5

	MOVSD X2, (DST_PTR)              // y[i] = X_i
	MOVSD X3, (DST_PTR)(INC_DST*1)
	MOVSD X4, (DST_PTR)(INC_DST*2)
	MOVSD X5, (DST_PTR)(INCx3_DST*1)

	LEAQ (X_PTR)(INC_X*4), X_PTR // X_PTR = &(X_PTR[incX*4])
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR // Y_PTR = &(Y_PTR[incY*4])
	DECQ LEN
	JNZ  loop                    // } while --LEN > 0
	CMPQ TAIL, $0                // if TAIL == 0 { return }
	JE   end

tail_start: // Reset Loop registers
	MOVQ TAIL, LEN // Loop counter: LEN = TAIL
	SHRQ $1, LEN   // LEN = floor( LEN / 2 )
	JZ   tail_one

tail_two:
	MOVSD (X_PTR), X2              // X_i = x[i]
	MOVSD (X_PTR)(INC_X*1), X3
	MULSD ALPHA, X2                // X_i *= a
	MULSD ALPHA, X3
	ADDSD (Y_PTR), X2              // X_i += y[i]
	ADDSD (Y_PTR)(INC_Y*1), X3
	MOVSD X2, (DST_PTR)            // y[i] = X_i
	MOVSD X3, (DST_PTR)(INC_DST*1)

	LEAQ (X_PTR)(INC_X*2), X_PTR // X_PTR = &(X_PTR[incX*2])
	LEAQ (Y_PTR)(INC_Y*2), Y_PTR // Y_PTR = &(Y_PTR[incY*2])

	ANDQ $1, TAIL
	JZ   end      // if TAIL == 0 { goto end }

tail_one:
	// y[i] += alpha * x[i] for the last n % 4 iterations.
	MOVSD (X_PTR), X2   // X2 = x[i]
	MULSD ALPHA, X2     // X2 *= a
	ADDSD (Y_PTR), X2   // X2 += y[i]
	MOVSD X2, (DST_PTR) // y[i] = X2

end:
	RET

// func AxpyIncTo(dst []float64, incDst, idst uintptr, alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr)
TEXT ·AxpyIncTo(SB), NOSPLIT, $0
	MOVQ dst_base+0(FP), DST_PTR // DST_PTR := &dst
	MOVQ x_base+48(FP), X_PTR    // X_PTR := &x
	MOVQ y_base+72(FP), Y_PTR    // Y_PTR := &y
	MOVQ n+96(FP), LEN           // LEN := n
	CMPQ LEN, $0                 // if LEN == 0 { return }
	JE   end

	MOVQ ix+120(FP), INC_X
	LEAQ (X_PTR)(INC_X*8), X_PTR       // X_PTR = &(x[ix])
	MOVQ iy+128(FP), INC_Y
	LEAQ (Y_PTR)(INC_Y*8), Y_PTR       // Y_PTR = &(dst[idst])
	MOVQ idst+32(FP), INC_DST
	LEAQ (DST_PTR)(INC_DST*8), DST_PTR // DST_PTR = &(y[iy])

	MOVQ  incX+104(FP), INC_X    // INC_X = incX * sizeof(float64)
	SHLQ  $3, INC_X
	MOVQ  incY+112(FP), INC_Y    // INC_Y = incY * sizeof(float64)
	SHLQ  $3, INC_Y
	MOVQ  incDst+24(FP), INC_DST // INC_DST = incDst * sizeof(float64)
	SHLQ  $3, INC_DST
	MOVSD alpha+40(FP), ALPHA

	MOVQ LEN, TAIL
	ANDQ $3, TAIL   // TAIL = n % 4
	SHRQ $2, LEN    // LEN = floor( n / 4 )
	JZ   tail_start // if LEN == 0 { goto tail_start }

	MOVSD ALPHA, ALPHA_2                  // ALPHA_2 = ALPHA for pipelining
	LEAQ  (INC_X)(INC_X*2), INCx3_X       // INCx3_X = INC_X * 3
	LEAQ  (INC_Y)(INC_Y*2), INCx3_Y       // INCx3_Y = INC_Y * 3
	LEAQ  (INC_DST)(INC_DST*2), INCx3_DST // INCx3_DST = INC_DST * 3

loop:  // do {  // y[i] += alpha * x[i] unrolled 2x.
	MOVSD (X_PTR), X2            // X_i = x[i]
	MOVSD (X_PTR)(INC_X*1), X3
	MOVSD (X_PTR)(INC_X*2), X4
	MOVSD (X_PTR)(INCx3_X*1), X5

	MULSD ALPHA, X2   // X_i *= a
	MULSD ALPHA_2, X3
	MULSD ALPHA, X4
	MULSD ALPHA_2, X5

	ADDSD (Y_PTR), X2            // X_i += y[i]
	ADDSD (Y_PTR)(INC_Y*1), X3
	ADDSD (Y_PTR)(INC_Y*2), X4
	ADDSD (Y_PTR)(INCx3_Y*1), X5

	MOVSD X2, (DST_PTR)              // y[i] = X_i
	MOVSD X3, (DST_PTR)(INC_DST*1)
	MOVSD X4, (DST_PTR)(INC_DST*2)
	MOVSD X5, (DST_PTR)(INCx3_DST*1)

	LEAQ (X_PTR)(INC_X*4), X_PTR       // X_PTR = &(X_PTR[incX*4])
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR       // Y_PTR = &(Y_PTR[incY*4])
	LEAQ (DST_PTR)(INC_DST*4), DST_PTR // DST_PTR = &(DST_PTR[incDst*4]
	DECQ LEN
	JNZ  loop                          // } while --LEN > 0
	CMPQ TAIL, $0                      // if TAIL == 0 { return }
	JE   end

tail_start: // Reset Loop registers
	MOVQ TAIL, LEN // Loop counter: LEN = TAIL
	SHRQ $1, LEN   // LEN = floor( LEN / 2 )
	JZ   tail_one

tail_two:
	MOVSD (X_PTR), X2              // X_i = x[i]
	MOVSD (X_PTR)(INC_X*1), X3
	MULSD ALPHA, X2                // X_i *= a
	MULSD ALPHA, X3
	ADDSD (Y_PTR), X2              // X_i += y[i]
	ADDSD (Y_PTR)(INC_Y*1), X3
	MOVSD X2, (DST_PTR)            // y[i] = X_i
	MOVSD X3, (DST_PTR)(INC_DST*1)

	LEAQ (X_PTR)(INC_X*2), X_PTR       // X_PTR = &(X_PTR[incX*2])
	LEAQ (Y_PTR)(INC_Y*2), Y_PTR       // Y_PTR = &(Y_PTR[incY*2])
	LEAQ (DST_PTR)(INC_DST*2), DST_PTR // DST_PTR = &(DST_PTR[incY*2]

	ANDQ $1, TAIL
	JZ   end      // if TAIL == 0 { goto end }

tail_one:
	MOVSD (X_PTR), X2   // X2 = x[i]
	MULSD ALPHA, X2     // X2 *= a
	ADDSD (Y_PTR), X2   // X2 += y[i]
	MOVSD X2, (DST_PTR) // y[i] = X2

end:
	RET

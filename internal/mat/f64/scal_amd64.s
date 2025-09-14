#include "textflag.h"

#define MOVDDUP_ALPHA    LONG $0x44120FF2; WORD $0x0824 // @ MOVDDUP XMM0, 8[RSP]

#define X_PTR SI
#define DST_PTR DI
#define IDX AX
#define LEN CX
#define TAIL BX
#define INC_X R8
#define INCx3_X R9
#define INC_DST R10
#define INCx3_DST R11
#define ALPHA X0
#define ALPHA_2 X1

// func ScalUnitary(alpha float64, x []float64)
TEXT ·ScalUnitary(SB), NOSPLIT, $0
	MOVDDUP_ALPHA            // ALPHA = { alpha, alpha }
	MOVQ x_base+8(FP), X_PTR // X_PTR = &x
	MOVQ x_len+16(FP), LEN   // LEN = len(x)
	CMPQ LEN, $0
	JE   end                 // if LEN == 0 { return }
	XORQ IDX, IDX            // IDX = 0

	MOVQ LEN, TAIL
	ANDQ $7, TAIL   // TAIL = LEN % 8
	SHRQ $3, LEN    // LEN = floor( LEN / 8 )
	JZ   tail_start // if LEN == 0 { goto tail_start }

	MOVUPS ALPHA, ALPHA_2

loop:  // do {  // x[i] *= alpha unrolled 8x.
	MOVUPS (X_PTR)(IDX*8), X2   // X_i = x[i]
	MOVUPS 16(X_PTR)(IDX*8), X3
	MOVUPS 32(X_PTR)(IDX*8), X4
	MOVUPS 48(X_PTR)(IDX*8), X5

	MULPD ALPHA, X2   // X_i *= ALPHA
	MULPD ALPHA_2, X3
	MULPD ALPHA, X4
	MULPD ALPHA_2, X5

	MOVUPS X2, (X_PTR)(IDX*8)   // x[i] = X_i
	MOVUPS X3, 16(X_PTR)(IDX*8)
	MOVUPS X4, 32(X_PTR)(IDX*8)
	MOVUPS X5, 48(X_PTR)(IDX*8)

	ADDQ $8, IDX  // i += 8
	DECQ LEN
	JNZ  loop     // while --LEN > 0
	CMPQ TAIL, $0
	JE   end      // if TAIL == 0 { return }

tail_start: // Reset loop registers
	MOVQ TAIL, LEN // Loop counter: LEN = TAIL
	SHRQ $1, LEN   // LEN = floor( TAIL / 2 )
	JZ   tail_one  // if n == 0 goto end

tail_two: // do {
	MOVUPS (X_PTR)(IDX*8), X2 // X_i = x[i]
	MULPD  ALPHA, X2          // X_i *= ALPHA
	MOVUPS X2, (X_PTR)(IDX*8) // x[i] = X_i
	ADDQ   $2, IDX            // i += 2
	DECQ   LEN
	JNZ    tail_two           // while --LEN > 0

	ANDQ $1, TAIL
	JZ   end      // if TAIL == 0 { return }

tail_one:
	// x[i] *= alpha for the remaining element.
	MOVSD (X_PTR)(IDX*8), X2
	MULSD ALPHA, X2
	MOVSD X2, (X_PTR)(IDX*8)

end:
	RET

// func ScalUnitaryTo(dst []float64, alpha float64, x []float64)
// This function assumes len(dst) >= len(x).
TEXT ·ScalUnitaryTo(SB), NOSPLIT, $0
	MOVQ x_base+32(FP), X_PTR    // X_PTR = &x
	MOVQ dst_base+0(FP), DST_PTR // DST_PTR = &dst
	MOVDDUP_ALPHA                // ALPHA = { alpha, alpha }
	MOVQ x_len+40(FP), LEN       // LEN = len(x)
	CMPQ LEN, $0
	JE   end                     // if LEN == 0 { return }

	XORQ IDX, IDX   // IDX = 0
	MOVQ LEN, TAIL
	ANDQ $7, TAIL   // TAIL = LEN % 8
	SHRQ $3, LEN    // LEN = floor( LEN / 8 )
	JZ   tail_start // if LEN == 0 { goto tail_start }

	MOVUPS ALPHA, ALPHA_2 // ALPHA_2 = ALPHA for pipelining

loop:  // do { // dst[i] = alpha * x[i] unrolled 8x.
	MOVUPS (X_PTR)(IDX*8), X2   // X_i = x[i]
	MOVUPS 16(X_PTR)(IDX*8), X3
	MOVUPS 32(X_PTR)(IDX*8), X4
	MOVUPS 48(X_PTR)(IDX*8), X5

	MULPD ALPHA, X2   // X_i *= ALPHA
	MULPD ALPHA_2, X3
	MULPD ALPHA, X4
	MULPD ALPHA_2, X5

	MOVUPS X2, (DST_PTR)(IDX*8)   // dst[i] = X_i
	MOVUPS X3, 16(DST_PTR)(IDX*8)
	MOVUPS X4, 32(DST_PTR)(IDX*8)
	MOVUPS X5, 48(DST_PTR)(IDX*8)

	ADDQ $8, IDX  // i += 8
	DECQ LEN
	JNZ  loop     // while --LEN > 0
	CMPQ TAIL, $0
	JE   end      // if TAIL == 0 { return }

tail_start: // Reset loop counters
	MOVQ TAIL, LEN // Loop counter: LEN = TAIL
	SHRQ $1, LEN   // LEN = floor( TAIL / 2 )
	JZ   tail_one  // if LEN == 0 { goto tail_one }

tail_two: // do {
	MOVUPS (X_PTR)(IDX*8), X2   // X_i = x[i]
	MULPD  ALPHA, X2            // X_i *= ALPHA
	MOVUPS X2, (DST_PTR)(IDX*8) // dst[i] = X_i
	ADDQ   $2, IDX              // i += 2
	DECQ   LEN
	JNZ    tail_two             // while --LEN > 0

	ANDQ $1, TAIL
	JZ   end      // if TAIL == 0 { return }

tail_one:
	MOVSD (X_PTR)(IDX*8), X2   // X_i = x[i]
	MULSD ALPHA, X2            // X_i *= ALPHA
	MOVSD X2, (DST_PTR)(IDX*8) // dst[i] = X_i

end:
	RET

// func ScalInc(alpha float64, x []float64, n, incX uintptr)
TEXT ·ScalInc(SB), NOSPLIT, $0
	MOVSD alpha+0(FP), ALPHA  // ALPHA = alpha
	MOVQ  x_base+8(FP), X_PTR // X_PTR = &x
	MOVQ  incX+40(FP), INC_X  // INC_X = incX
	SHLQ  $3, INC_X           // INC_X *= sizeof(float64)
	MOVQ  n+32(FP), LEN       // LEN = n
	CMPQ  LEN, $0
	JE    end                 // if LEN == 0 { return }

	MOVQ LEN, TAIL
	ANDQ $3, TAIL   // TAIL = LEN % 4
	SHRQ $2, LEN    // LEN = floor( LEN / 4 )
	JZ   tail_start // if LEN == 0 { goto tail_start }

	MOVUPS ALPHA, ALPHA_2            // ALPHA_2 = ALPHA for pipelining
	LEAQ   (INC_X)(INC_X*2), INCx3_X // INCx3_X = INC_X * 3

loop:  // do { // x[i] *= alpha unrolled 4x.
	MOVSD (X_PTR), X2            // X_i = x[i]
	MOVSD (X_PTR)(INC_X*1), X3
	MOVSD (X_PTR)(INC_X*2), X4
	MOVSD (X_PTR)(INCx3_X*1), X5

	MULSD ALPHA, X2   // X_i *= a
	MULSD ALPHA_2, X3
	MULSD ALPHA, X4
	MULSD ALPHA_2, X5

	MOVSD X2, (X_PTR)            // x[i] = X_i
	MOVSD X3, (X_PTR)(INC_X*1)
	MOVSD X4, (X_PTR)(INC_X*2)
	MOVSD X5, (X_PTR)(INCx3_X*1)

	LEAQ (X_PTR)(INC_X*4), X_PTR // X_PTR = &(X_PTR[incX*4])
	DECQ LEN
	JNZ  loop                    // } while --LEN > 0
	CMPQ TAIL, $0
	JE   end                     // if TAIL == 0 { return }

tail_start: // Reset loop registers
	MOVQ TAIL, LEN // Loop counter: LEN = TAIL
	SHRQ $1, LEN   // LEN = floor( LEN / 2 )
	JZ   tail_one

tail_two: // do {
	MOVSD (X_PTR), X2          // X_i = x[i]
	MOVSD (X_PTR)(INC_X*1), X3
	MULSD ALPHA, X2            // X_i *= a
	MULSD ALPHA, X3
	MOVSD X2, (X_PTR)          // x[i] = X_i
	MOVSD X3, (X_PTR)(INC_X*1)

	LEAQ (X_PTR)(INC_X*2), X_PTR // X_PTR = &(X_PTR[incX*2])

	ANDQ $1, TAIL
	JZ   end

tail_one:
	MOVSD (X_PTR), X2 // X_i = x[i]
	MULSD ALPHA, X2   // X_i *= ALPHA
	MOVSD X2, (X_PTR) // x[i] = X_i

end:
	RET

// func ScalIncTo(dst []float64, incDst uintptr, alpha float64, x []float64, n, incX uintptr)
TEXT ·ScalIncTo(SB), NOSPLIT, $0
	MOVQ  dst_base+0(FP), DST_PTR // DST_PTR = &dst
	MOVQ  incDst+24(FP), INC_DST  // INC_DST = incDst
	SHLQ  $3, INC_DST             // INC_DST *= sizeof(float64)
	MOVSD alpha+32(FP), ALPHA     // ALPHA = alpha
	MOVQ  x_base+40(FP), X_PTR    // X_PTR = &x
	MOVQ  n+64(FP), LEN           // LEN = n
	MOVQ  incX+72(FP), INC_X      // INC_X = incX
	SHLQ  $3, INC_X               // INC_X *= sizeof(float64)
	CMPQ  LEN, $0
	JE    end                     // if LEN == 0 { return }

	MOVQ LEN, TAIL
	ANDQ $3, TAIL   // TAIL = LEN % 4
	SHRQ $2, LEN    // LEN = floor( LEN / 4 )
	JZ   tail_start // if LEN == 0 { goto tail_start }

	MOVUPS ALPHA, ALPHA_2                  // ALPHA_2 = ALPHA for pipelining
	LEAQ   (INC_X)(INC_X*2), INCx3_X       // INCx3_X = INC_X * 3
	LEAQ   (INC_DST)(INC_DST*2), INCx3_DST // INCx3_DST = INC_DST * 3

loop:  // do { // x[i] *= alpha unrolled 4x.
	MOVSD (X_PTR), X2            // X_i = x[i]
	MOVSD (X_PTR)(INC_X*1), X3
	MOVSD (X_PTR)(INC_X*2), X4
	MOVSD (X_PTR)(INCx3_X*1), X5

	MULSD ALPHA, X2   // X_i *= a
	MULSD ALPHA_2, X3
	MULSD ALPHA, X4
	MULSD ALPHA_2, X5

	MOVSD X2, (DST_PTR)              // dst[i] = X_i
	MOVSD X3, (DST_PTR)(INC_DST*1)
	MOVSD X4, (DST_PTR)(INC_DST*2)
	MOVSD X5, (DST_PTR)(INCx3_DST*1)

	LEAQ (X_PTR)(INC_X*4), X_PTR       // X_PTR = &(X_PTR[incX*4])
	LEAQ (DST_PTR)(INC_DST*4), DST_PTR // DST_PTR = &(DST_PTR[incDst*4])
	DECQ LEN
	JNZ  loop                          // } while --LEN > 0
	CMPQ TAIL, $0
	JE   end                           // if TAIL == 0 { return }

tail_start: // Reset loop registers
	MOVQ TAIL, LEN // Loop counter: LEN = TAIL
	SHRQ $1, LEN   // LEN = floor( LEN / 2 )
	JZ   tail_one

tail_two:
	MOVSD (X_PTR), X2              // X_i = x[i]
	MOVSD (X_PTR)(INC_X*1), X3
	MULSD ALPHA, X2                // X_i *= a
	MULSD ALPHA, X3
	MOVSD X2, (DST_PTR)            // dst[i] = X_i
	MOVSD X3, (DST_PTR)(INC_DST*1)

	LEAQ (X_PTR)(INC_X*2), X_PTR       // X_PTR = &(X_PTR[incX*2])
	LEAQ (DST_PTR)(INC_DST*2), DST_PTR // DST_PTR = &(DST_PTR[incDst*2])

	ANDQ $1, TAIL
	JZ   end

tail_one:
	MOVSD (X_PTR), X2   // X_i = x[i]
	MULSD ALPHA, X2     // X_i *= ALPHA
	MOVSD X2, (DST_PTR) // x[i] = X_i

end:
	RET

#include "textflag.h"

// func DdotUnitary(x, y []float64) (sum float64)
// This function assumes len(y) >= len(x).
TEXT ·DotUnitary(SB), NOSPLIT, $0
	MOVQ x+0(FP), R8
	MOVQ x_len+8(FP), DI // n = len(x)
	MOVQ y+24(FP), R9

	MOVSD $(0.0), X7 // sum = 0
	MOVSD $(0.0), X8 // sum = 0

	MOVQ $0, SI   // i = 0
	SUBQ $4, DI   // n -= 4
	JL   tail_uni // if n < 0 goto tail_uni

loop_uni:
	// sum += x[i] * y[i] unrolled 4x.
	MOVUPD 0(R8)(SI*8), X0
	MOVUPD 0(R9)(SI*8), X1
	MOVUPD 16(R8)(SI*8), X2
	MOVUPD 16(R9)(SI*8), X3
	MULPD  X1, X0
	MULPD  X3, X2
	ADDPD  X0, X7
	ADDPD  X2, X8

	ADDQ $4, SI   // i += 4
	SUBQ $4, DI   // n -= 4
	JGE  loop_uni // if n >= 0 goto loop_uni

tail_uni:
	ADDQ $4, DI  // n += 4
	JLE  end_uni // if n <= 0 goto end_uni

onemore_uni:
	// sum += x[i] * y[i] for the remaining 1-3 elements.
	MOVSD 0(R8)(SI*8), X0
	MOVSD 0(R9)(SI*8), X1
	MULSD X1, X0
	ADDSD X0, X7

	ADDQ $1, SI      // i++
	SUBQ $1, DI      // n--
	JNZ  onemore_uni // if n != 0 goto onemore_uni

end_uni:
	// Add the four sums together.
	ADDPD    X8, X7
	MOVSD    X7, X0
	UNPCKHPD X7, X7
	ADDSD    X0, X7
	MOVSD    X7, sum+48(FP) // Return final sum.
	RET

// func DdotInc(x, y []float64, n, incX, incY, ix, iy uintptr) (sum float64)
TEXT ·DotInc(SB), NOSPLIT, $0
	MOVQ x+0(FP), R8
	MOVQ y+24(FP), R9
	MOVQ n+48(FP), CX
	MOVQ incX+56(FP), R11
	MOVQ incY+64(FP), R12
	MOVQ ix+72(FP), R13
	MOVQ iy+80(FP), R14

	MOVSD $(0.0), X7      // sum = 0
	LEAQ  (R8)(R13*8), SI // p = &x[ix]
	LEAQ  (R9)(R14*8), DI // q = &y[ix]
	SHLQ  $3, R11         // incX *= sizeof(float64)
	SHLQ  $3, R12         // indY *= sizeof(float64)

	SUBQ $2, CX   // n -= 2
	JL   tail_inc // if n < 0 goto tail_inc

loop_inc:
	// sum += *p * *q unrolled 2x.
	MOVHPD (SI), X0
	MOVHPD (DI), X1
	ADDQ   R11, SI  // p += incX
	ADDQ   R12, DI  // q += incY
	MOVLPD (SI), X0
	MOVLPD (DI), X1
	ADDQ   R11, SI  // p += incX
	ADDQ   R12, DI  // q += incY

	MULPD X1, X0
	ADDPD X0, X7

	SUBQ $2, CX   // n -= 2
	JGE  loop_inc // if n >= 0 goto loop_inc

tail_inc:
	ADDQ $2, CX  // n += 2
	JLE  end_inc // if n <= 0 goto end_inc

	// sum += *p * *q for the last iteration if n is odd.
	MOVSD (SI), X0
	MULSD (DI), X0
	ADDSD X0, X7

end_inc:
	// Add the two sums together.
	MOVSD    X7, X0
	UNPCKHPD X7, X7
	ADDSD    X0, X7
	MOVSD    X7, sum+88(FP) // Return final sum.
	RET


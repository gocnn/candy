#include "textflag.h"

#define HADDPS_SUM_SUM    LONG $0xC07C0FF2 // @ HADDPS X0, X0
#define HADDPD_SUM_SUM    LONG $0xC07C0F66 // @ HADDPD X0, X0

#define X_PTR SI
#define Y_PTR DI
#define LEN CX
#define TAIL BX
#define IDX AX
#define INC_X R8
#define INCx3_X R10
#define INC_Y R9
#define INCx3_Y R11
#define SUM X0
#define P_SUM X1

// func DotUnitary(x, y []float32) (sum float32)
TEXT ·DotUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+0(FP), X_PTR  // X_PTR = &x
	MOVQ    y_base+24(FP), Y_PTR // Y_PTR = &y
	PXOR    SUM, SUM             // SUM = 0
	MOVQ    x_len+8(FP), LEN     // LEN = min( len(x), len(y) )
	CMPQ    y_len+32(FP), LEN
	CMOVQLE y_len+32(FP), LEN
	CMPQ    LEN, $0
	JE      dot_end

	XORQ IDX, IDX
	MOVQ Y_PTR, DX
	ANDQ $0xF, DX    // Align on 16-byte boundary for MULPS
	JZ   dot_no_trim // if DX == 0 { goto dot_no_trim }
	SUBQ $16, DX

dot_align: // Trim first value(s) in unaligned buffer  do {
	MOVSS (X_PTR)(IDX*4), X2 // X2 = x[i]
	MULSS (Y_PTR)(IDX*4), X2 // X2 *= y[i]
	ADDSS X2, SUM            // SUM += X2
	INCQ  IDX                // IDX++
	DECQ  LEN
	JZ    dot_end            // if --TAIL == 0 { return }
	ADDQ  $4, DX
	JNZ   dot_align          // } while --DX > 0

dot_no_trim:
	PXOR P_SUM, P_SUM    // P_SUM = 0  for pipelining
	MOVQ LEN, TAIL
	ANDQ $0xF, TAIL      // TAIL = LEN % 16
	SHRQ $4, LEN         // LEN = floor( LEN / 16 )
	JZ   dot_tail4_start // if LEN == 0 { goto dot_tail4_start }

dot_loop: // Loop unrolled 16x  do {
	MOVUPS (X_PTR)(IDX*4), X2   // X_i = x[i:i+1]
	MOVUPS 16(X_PTR)(IDX*4), X3
	MOVUPS 32(X_PTR)(IDX*4), X4
	MOVUPS 48(X_PTR)(IDX*4), X5

	MULPS (Y_PTR)(IDX*4), X2   // X_i *= y[i:i+1]
	MULPS 16(Y_PTR)(IDX*4), X3
	MULPS 32(Y_PTR)(IDX*4), X4
	MULPS 48(Y_PTR)(IDX*4), X5

	ADDPS X2, SUM   // SUM += X_i
	ADDPS X3, P_SUM
	ADDPS X4, SUM
	ADDPS X5, P_SUM

	ADDQ $16, IDX // IDX += 16
	DECQ LEN
	JNZ  dot_loop // } while --LEN > 0

	ADDPS P_SUM, SUM // SUM += P_SUM
	CMPQ  TAIL, $0   // if TAIL == 0 { return }
	JE    dot_end

dot_tail4_start: // Reset loop counter for 4-wide tail loop
	MOVQ TAIL, LEN      // LEN = floor( TAIL / 4 )
	SHRQ $2, LEN
	JZ   dot_tail_start // if LEN == 0 { goto dot_tail_start }

dot_tail4_loop: // Loop unrolled 4x  do {
	MOVUPS (X_PTR)(IDX*4), X2 // X_i = x[i:i+1]
	MULPS  (Y_PTR)(IDX*4), X2 // X_i *= y[i:i+1]
	ADDPS  X2, SUM            // SUM += X_i
	ADDQ   $4, IDX            // i += 4
	DECQ   LEN
	JNZ    dot_tail4_loop     // } while --LEN > 0

dot_tail_start: // Reset loop counter for 1-wide tail loop
	ANDQ $3, TAIL // TAIL = TAIL % 4
	JZ   dot_end  // if TAIL == 0 { return }

dot_tail: // do {
	MOVSS (X_PTR)(IDX*4), X2 // X2 = x[i]
	MULSS (Y_PTR)(IDX*4), X2 // X2 *= y[i]
	ADDSS X2, SUM            // psum += X2
	INCQ  IDX                // IDX++
	DECQ  TAIL
	JNZ   dot_tail           // } while --TAIL > 0

dot_end:
	HADDPS_SUM_SUM        // SUM = \sum{ SUM[i] }
	HADDPS_SUM_SUM
	MOVSS SUM, sum+48(FP) // return SUM
	RET

// func DotInc(x, y []float32, n, incX, incY, ix, iy uintptr) (sum float32)
TEXT ·DotInc(SB), NOSPLIT, $0
	MOVQ x_base+0(FP), X_PTR  // X_PTR = &x
	MOVQ y_base+24(FP), Y_PTR // Y_PTR = &y
	PXOR SUM, SUM             // SUM = 0
	MOVQ n+48(FP), LEN        // LEN = n
	CMPQ LEN, $0
	JE   dot_end

	MOVQ ix+72(FP), INC_X        // INC_X = ix
	MOVQ iy+80(FP), INC_Y        // INC_Y = iy
	LEAQ (X_PTR)(INC_X*4), X_PTR // X_PTR = &(x[ix])
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR // Y_PTR = &(y[iy])

	MOVQ incX+56(FP), INC_X // INC_X := incX * sizeof(float32)
	SHLQ $2, INC_X
	MOVQ incY+64(FP), INC_Y // INC_Y := incY * sizeof(float32)
	SHLQ $2, INC_Y

	MOVQ LEN, TAIL
	ANDQ $0x3, TAIL // TAIL = LEN % 4
	SHRQ $2, LEN    // LEN = floor( LEN / 4 )
	JZ   dot_tail   // if LEN == 0 { goto dot_tail }

	PXOR P_SUM, P_SUM              // P_SUM = 0  for pipelining
	LEAQ (INC_X)(INC_X*2), INCx3_X // INCx3_X = INC_X * 3
	LEAQ (INC_Y)(INC_Y*2), INCx3_Y // INCx3_Y = INC_Y * 3

dot_loop: // Loop unrolled 4x  do {
	MOVSS (X_PTR), X2            // X_i = x[i:i+1]
	MOVSS (X_PTR)(INC_X*1), X3
	MOVSS (X_PTR)(INC_X*2), X4
	MOVSS (X_PTR)(INCx3_X*1), X5

	MULSS (Y_PTR), X2            // X_i *= y[i:i+1]
	MULSS (Y_PTR)(INC_Y*1), X3
	MULSS (Y_PTR)(INC_Y*2), X4
	MULSS (Y_PTR)(INCx3_Y*1), X5

	ADDSS X2, SUM   // SUM += X_i
	ADDSS X3, P_SUM
	ADDSS X4, SUM
	ADDSS X5, P_SUM

	LEAQ (X_PTR)(INC_X*4), X_PTR // X_PTR = &(X_PTR[INC_X * 4])
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR // Y_PTR = &(Y_PTR[INC_Y * 4])

	DECQ LEN
	JNZ  dot_loop // } while --LEN > 0

	ADDSS P_SUM, SUM // P_SUM += SUM
	CMPQ  TAIL, $0   // if TAIL == 0 { return }
	JE    dot_end

dot_tail: // do {
	MOVSS (X_PTR), X2  // X2 = x[i]
	MULSS (Y_PTR), X2  // X2 *= y[i]
	ADDSS X2, SUM      // SUM += X2
	ADDQ  INC_X, X_PTR // X_PTR += INC_X
	ADDQ  INC_Y, Y_PTR // Y_PTR += INC_Y
	DECQ  TAIL
	JNZ   dot_tail     // } while --TAIL > 0

dot_end:
	MOVSS SUM, sum+88(FP) // return SUM
	RET

// func DdotUnitary(x, y []float32) (sum float32)
TEXT ·DdotUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+0(FP), X_PTR  // X_PTR = &x
	MOVQ    y_base+24(FP), Y_PTR // Y_PTR = &y
	MOVQ    x_len+8(FP), LEN     // LEN = min( len(x), len(y) )
	CMPQ    y_len+32(FP), LEN
	CMOVQLE y_len+32(FP), LEN
	PXOR    SUM, SUM             // psum = 0
	CMPQ    LEN, $0
	JE      dot_end

	XORQ IDX, IDX
	MOVQ Y_PTR, DX
	ANDQ $0xF, DX    // Align on 16-byte boundary for ADDPS
	JZ   dot_no_trim // if DX == 0 { goto dot_no_trim }

	SUBQ $16, DX

dot_align: // Trim first value(s) in unaligned buffer  do {
	CVTSS2SD (X_PTR)(IDX*4), X2 // X2 = float64(x[i])
	CVTSS2SD (Y_PTR)(IDX*4), X3 // X3 = float64(y[i])
	MULSD    X3, X2
	ADDSD    X2, SUM            // SUM += X2
	INCQ     IDX                // IDX++
	DECQ     LEN
	JZ       dot_end            // if --TAIL == 0 { return }
	ADDQ     $4, DX
	JNZ      dot_align          // } while --LEN > 0

dot_no_trim:
	PXOR P_SUM, P_SUM   // P_SUM = 0  for pipelining
	MOVQ LEN, TAIL
	ANDQ $0x7, TAIL     // TAIL = LEN % 8
	SHRQ $3, LEN        // LEN = floor( LEN / 8 )
	JZ   dot_tail_start // if LEN == 0 { goto dot_tail_start }

dot_loop: // Loop unrolled 8x  do {
	CVTPS2PD (X_PTR)(IDX*4), X2   // X_i = x[i:i+1]
	CVTPS2PD 8(X_PTR)(IDX*4), X3
	CVTPS2PD 16(X_PTR)(IDX*4), X4
	CVTPS2PD 24(X_PTR)(IDX*4), X5

	CVTPS2PD (Y_PTR)(IDX*4), X6   // X_j = y[i:i+1]
	CVTPS2PD 8(Y_PTR)(IDX*4), X7
	CVTPS2PD 16(Y_PTR)(IDX*4), X8
	CVTPS2PD 24(Y_PTR)(IDX*4), X9

	MULPD X6, X2 // X_i *= X_j
	MULPD X7, X3
	MULPD X8, X4
	MULPD X9, X5

	ADDPD X2, SUM   // SUM += X_i
	ADDPD X3, P_SUM
	ADDPD X4, SUM
	ADDPD X5, P_SUM

	ADDQ $8, IDX  // IDX += 8
	DECQ LEN
	JNZ  dot_loop // } while --LEN > 0

	ADDPD P_SUM, SUM // SUM += P_SUM
	CMPQ  TAIL, $0   // if TAIL == 0 { return }
	JE    dot_end

dot_tail_start:
	MOVQ TAIL, LEN
	SHRQ $1, LEN
	JZ   dot_tail_one

dot_tail_two:
	CVTPS2PD (X_PTR)(IDX*4), X2 // X_i = x[i:i+1]
	CVTPS2PD (Y_PTR)(IDX*4), X6 // X_j = y[i:i+1]
	MULPD    X6, X2             // X_i *= X_j
	ADDPD    X2, SUM            // SUM += X_i
	ADDQ     $2, IDX            // IDX += 2
	DECQ     LEN
	JNZ      dot_tail_two       // } while --LEN > 0

	ANDQ $1, TAIL
	JZ   dot_end

dot_tail_one:
	CVTSS2SD (X_PTR)(IDX*4), X2 // X2 = float64(x[i])
	CVTSS2SD (Y_PTR)(IDX*4), X3 // X3 = float64(y[i])
	MULSD    X3, X2             // X2 *= X3
	ADDSD    X2, SUM            // SUM += X2

dot_end:
	HADDPD_SUM_SUM        // SUM = \sum{ SUM[i] }
	MOVSD SUM, sum+48(FP) // return SUM
	RET

// func DdotInc(x, y []float32, n, incX, incY, ix, iy uintptr) (sum float64)
TEXT ·DdotInc(SB), NOSPLIT, $0
	MOVQ x_base+0(FP), X_PTR  // X_PTR = &x
	MOVQ y_base+24(FP), Y_PTR // Y_PTR = &y
	MOVQ n+48(FP), LEN        // LEN = n
	PXOR SUM, SUM             // SUM = 0
	CMPQ LEN, $0
	JE   dot_end

	MOVQ ix+72(FP), INC_X        // INC_X = ix
	MOVQ iy+80(FP), INC_Y        // INC_Y = iy
	LEAQ (X_PTR)(INC_X*4), X_PTR // X_PTR = &(x[ix])
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR // Y_PTR = &(y[iy])

	MOVQ incX+56(FP), INC_X // INC_X = incX * sizeof(float32)
	SHLQ $2, INC_X
	MOVQ incY+64(FP), INC_Y // INC_Y = incY * sizeof(float32)
	SHLQ $2, INC_Y

	MOVQ LEN, TAIL
	ANDQ $3, TAIL  // TAIL = LEN % 4
	SHRQ $2, LEN   // LEN = floor( LEN / 4 )
	JZ   dot_tail  // if LEN == 0 { goto dot_tail }

	PXOR P_SUM, P_SUM              // P_SUM = 0  for pipelining
	LEAQ (INC_X)(INC_X*2), INCx3_X // INCx3_X = INC_X * 3
	LEAQ (INC_Y)(INC_Y*2), INCx3_Y // INCx3_Y = INC_Y * 3

dot_loop: // Loop unrolled 4x  do {
	CVTSS2SD (X_PTR), X2            // X_i = x[i:i+1]
	CVTSS2SD (X_PTR)(INC_X*1), X3
	CVTSS2SD (X_PTR)(INC_X*2), X4
	CVTSS2SD (X_PTR)(INCx3_X*1), X5

	CVTSS2SD (Y_PTR), X6            // X_j = y[i:i+1]
	CVTSS2SD (Y_PTR)(INC_Y*1), X7
	CVTSS2SD (Y_PTR)(INC_Y*2), X8
	CVTSS2SD (Y_PTR)(INCx3_Y*1), X9

	MULSD X6, X2 // X_i *= X_j
	MULSD X7, X3
	MULSD X8, X4
	MULSD X9, X5

	ADDSD X2, SUM   // SUM += X_i
	ADDSD X3, P_SUM
	ADDSD X4, SUM
	ADDSD X5, P_SUM

	LEAQ (X_PTR)(INC_X*4), X_PTR // X_PTR = &(X_PTR[INC_X * 4])
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR // Y_PTR = &(Y_PTR[INC_Y * 4])

	DECQ LEN
	JNZ  dot_loop // } while --LEN > 0

	ADDSD P_SUM, SUM // SUM += P_SUM
	CMPQ  TAIL, $0   // if TAIL == 0 { return }
	JE    dot_end

dot_tail: // do {
	CVTSS2SD (X_PTR), X2  // X2 = x[i]
	CVTSS2SD (Y_PTR), X3  // X2 *= y[i]
	MULSD    X3, X2
	ADDSD    X2, SUM      // SUM += X2
	ADDQ     INC_X, X_PTR // X_PTR += INC_X
	ADDQ     INC_Y, Y_PTR // Y_PTR += INC_Y
	DECQ     TAIL
	JNZ      dot_tail     // } while --TAIL > 0

dot_end:
	MOVSD SUM, sum+88(FP) // return SUM
	RET

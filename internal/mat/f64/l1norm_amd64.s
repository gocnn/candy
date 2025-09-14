#include "textflag.h"

// func L1Dist(s, t []float64) float64
TEXT ·L1Dist(SB), NOSPLIT, $0
	MOVQ    s_base+0(FP), DI  // DI = &s
	MOVQ    t_base+24(FP), SI // SI = &t
	MOVQ    s_len+8(FP), CX   // CX = len(s)
	CMPQ    t_len+32(FP), CX  // CX = max( CX, len(t) )
	CMOVQLE t_len+32(FP), CX
	PXOR    X3, X3            // norm = 0
	CMPQ    CX, $0            // if CX == 0 { return 0 }
	JE      l1_end
	XORQ    AX, AX            // i = 0
	MOVQ    CX, BX
	ANDQ    $1, BX            // BX = CX % 2
	SHRQ    $1, CX            // CX = floor( CX / 2 )
	JZ      l1_tail_start     // if CX == 0 { return 0 }

l1_loop: // Loop unrolled 2x  do {
	MOVUPS (SI)(AX*8), X0 // X0 = t[i:i+1]
	MOVUPS (DI)(AX*8), X1 // X1 = s[i:i+1]
	MOVAPS X0, X2
	SUBPD  X1, X0
	SUBPD  X2, X1
	MAXPD  X1, X0         // X0 = max( X0 - X1, X1 - X0 )
	ADDPD  X0, X3         // norm += X0
	ADDQ   $2, AX         // i += 2
	LOOP   l1_loop        // } while --CX > 0
	CMPQ   BX, $0         // if BX == 0 { return }
	JE     l1_end

l1_tail_start: // Reset loop registers
	MOVQ BX, CX // Loop counter: CX = BX
	PXOR X0, X0 // reset X0, X1 to break dependencies
	PXOR X1, X1

l1_tail:
	MOVSD  (SI)(AX*8), X0 // X0 = t[i]
	MOVSD  (DI)(AX*8), X1 // x1 = s[i]
	MOVAPD X0, X2
	SUBSD  X1, X0
	SUBSD  X2, X1
	MAXSD  X1, X0         // X0 = max( X0 - X1, X1 - X0 )
	ADDSD  X0, X3         // norm += X0

l1_end:
	MOVAPS X3, X2
	SHUFPD $1, X2, X2
	ADDSD  X3, X2         // X2 = X3[1] + X3[0]
	MOVSD  X2, ret+48(FP) // return X2
	RET

// func L1Norm(x []float64) float64
TEXT ·L1Norm(SB), NOSPLIT, $0
	MOVQ x_base+0(FP), SI // SI = &x
	MOVQ x_len+8(FP), CX  // CX = len(x)
	XORQ AX, AX           // i = 0
	PXOR X0, X0           // p_sum_i = 0
	PXOR X1, X1
	PXOR X2, X2
	PXOR X3, X3
	PXOR X4, X4
	PXOR X5, X5
	PXOR X6, X6
	PXOR X7, X7
	CMPQ CX, $0           // if CX == 0 { return 0 }
	JE   absum_end
	MOVQ CX, BX
	ANDQ $7, BX           // BX = len(x) % 8
	SHRQ $3, CX           // CX = floor( len(x) / 8 )
	JZ   absum_tail_start // if CX == 0 { goto absum_tail_start }

absum_loop: // do {
	// p_sum += max( p_sum + x[i], p_sum - x[i] )
	MOVUPS (SI)(AX*8), X8    // X_i = x[i:i+1]
	MOVUPS 16(SI)(AX*8), X9
	MOVUPS 32(SI)(AX*8), X10
	MOVUPS 48(SI)(AX*8), X11
	ADDPD  X8, X0            // p_sum_i += X_i  ( positive values )
	ADDPD  X9, X2
	ADDPD  X10, X4
	ADDPD  X11, X6
	SUBPD  X8, X1            // p_sum_(i+1) -= X_i  ( negative values )
	SUBPD  X9, X3
	SUBPD  X10, X5
	SUBPD  X11, X7
	MAXPD  X1, X0            // p_sum_i = max( p_sum_i, p_sum_(i+1) )
	MAXPD  X3, X2
	MAXPD  X5, X4
	MAXPD  X7, X6
	MOVAPS X0, X1            // p_sum_(i+1) = p_sum_i
	MOVAPS X2, X3
	MOVAPS X4, X5
	MOVAPS X6, X7
	ADDQ   $8, AX            // i += 8
	LOOP   absum_loop        // } while --CX > 0

	// p_sum_0 = \sum_{i=1}^{3}( p_sum_(i*2) )
	ADDPD X3, X0
	ADDPD X5, X7
	ADDPD X7, X0

	// p_sum_0[0] = p_sum_0[0] + p_sum_0[1]
	MOVAPS X0, X1
	SHUFPD $0x3, X0, X0 // lower( p_sum_0 ) = upper( p_sum_0 )
	ADDSD  X1, X0
	CMPQ   BX, $0
	JE     absum_end    // if BX == 0 { goto absum_end }

absum_tail_start: // Reset loop registers
	MOVQ  BX, CX // Loop counter:  CX = BX
	XORPS X8, X8 // X_8 = 0

absum_tail: // do {
	// p_sum += max( p_sum + x[i], p_sum - x[i] )
	MOVSD (SI)(AX*8), X8 // X_8 = x[i]
	MOVSD X0, X1         // p_sum_1 = p_sum_0
	ADDSD X8, X0         // p_sum_0 += X_8
	SUBSD X8, X1         // p_sum_1 -= X_8
	MAXSD X1, X0         // p_sum_0 = max( p_sum_0, p_sum_1 )
	INCQ  AX             // i++
	LOOP  absum_tail     // } while --CX > 0

absum_end: // return p_sum_0
	MOVSD X0, sum+24(FP)
	RET

// func L1NormInc(x []float64, n, incX int) (sum float64)
TEXT ·L1NormInc(SB), NOSPLIT, $0
	MOVQ  x_base+0(FP), SI // SI = &x
	MOVQ  n+24(FP), CX     // CX = n
	MOVQ  incX+32(FP), AX  // AX =  increment * sizeof( float64 )
	SHLQ  $3, AX
	MOVQ  AX, DX           // DX = AX * 3
	IMULQ $3, DX
	PXOR  X0, X0           // p_sum_i = 0
	PXOR  X1, X1
	PXOR  X2, X2
	PXOR  X3, X3
	PXOR  X4, X4
	PXOR  X5, X5
	PXOR  X6, X6
	PXOR  X7, X7
	CMPQ  CX, $0           // if CX == 0 { return 0 }
	JE    absum_end
	MOVQ  CX, BX
	ANDQ  $7, BX           // BX = n % 8
	SHRQ  $3, CX           // CX = floor( n / 8 )
	JZ    absum_tail_start // if CX == 0 { goto absum_tail_start }

absum_loop: // do {
	// p_sum = max( p_sum + x[i], p_sum - x[i] )
	MOVSD  (SI), X8        // X_i[0] = x[i]
	MOVSD  (SI)(AX*1), X9
	MOVSD  (SI)(AX*2), X10
	MOVSD  (SI)(DX*1), X11
	LEAQ   (SI)(AX*4), SI  // SI = SI + 4
	MOVHPD (SI), X8        // X_i[1] = x[i+4]
	MOVHPD (SI)(AX*1), X9
	MOVHPD (SI)(AX*2), X10
	MOVHPD (SI)(DX*1), X11
	ADDPD  X8, X0          // p_sum_i += X_i  ( positive values )
	ADDPD  X9, X2
	ADDPD  X10, X4
	ADDPD  X11, X6
	SUBPD  X8, X1          // p_sum_(i+1) -= X_i  ( negative values )
	SUBPD  X9, X3
	SUBPD  X10, X5
	SUBPD  X11, X7
	MAXPD  X1, X0          // p_sum_i = max( p_sum_i, p_sum_(i+1) )
	MAXPD  X3, X2
	MAXPD  X5, X4
	MAXPD  X7, X6
	MOVAPS X0, X1          // p_sum_(i+1) = p_sum_i
	MOVAPS X2, X3
	MOVAPS X4, X5
	MOVAPS X6, X7
	LEAQ   (SI)(AX*4), SI  // SI = SI + 4
	LOOP   absum_loop      // } while --CX > 0

	// p_sum_0 = \sum_{i=1}^{3}( p_sum_(i*2) )
	ADDPD X3, X0
	ADDPD X5, X7
	ADDPD X7, X0

	// p_sum_0[0] = p_sum_0[0] + p_sum_0[1]
	MOVAPS X0, X1
	SHUFPD $0x3, X0, X0 // lower( p_sum_0 ) = upper( p_sum_0 )
	ADDSD  X1, X0
	CMPQ   BX, $0
	JE     absum_end    // if BX == 0 { goto absum_end }

absum_tail_start: // Reset loop registers
	MOVQ  BX, CX // Loop counter:  CX = BX
	XORPS X8, X8 // X_8 = 0

absum_tail: // do {
	// p_sum += max( p_sum + x[i], p_sum - x[i] )
	MOVSD (SI), X8   // X_8 = x[i]
	MOVSD X0, X1     // p_sum_1 = p_sum_0
	ADDSD X8, X0     // p_sum_0 += X_8
	SUBSD X8, X1     // p_sum_1 -= X_8
	MAXSD X1, X0     // p_sum_0 = max( p_sum_0, p_sum_1 )
	ADDQ  AX, SI     // i++
	LOOP  absum_tail // } while --CX > 0

absum_end: // return p_sum_0
	MOVSD X0, sum+40(FP)
	RET

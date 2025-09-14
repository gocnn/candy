#include "textflag.h"

// func Sqrt(x float32) float32
TEXT ·Sqrt(SB),NOSPLIT,$0
	FMOVS	x+0(FP), F0
	FSQRTS	F0, F0
	FMOVS	F0, ret+8(FP)
	RET

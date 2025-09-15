// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && gc && !purego

package mat

var (
	div32 = DivSSE32
	div64 = DivSSE64
)

func init() {
	if hasAVX {
		div32 = DivAVX32
		div64 = DivAVX64
	}
}

// Div32 divides x1 by x2, element-wise, storing the result in y (32 bits).
func Div32(x1, x2, y []float32) {
	div32(x1, x2, y)
}

// Div64 divides x1 by x2, element-wise, storing the result in y (64 bits).
func Div64(x1, x2, y []float64) {
	div64(x1, x2, y)
}

// DivAVX32 divides x1 by x2, element-wise, storing the result in y (32 bits, AVX required).
//
//go:noescape
func DivAVX32(x1 []float32, x2 []float32, y []float32)

// DivAVX64 divides x1 by x2, element-wise, storing the result in y (64 bits, AVX required).
//
//go:noescape
func DivAVX64(x1 []float64, x2 []float64, y []float64)

// DivSSE32 divides x1 by x2, element-wise, storing the result in y (32 bits, SSE required).
//
//go:noescape
func DivSSE32(x1 []float32, x2 []float32, y []float32)

// DivSSE64 divides x1 by x2, element-wise, storing the result in y (64 bits, SSE required).
//
//go:noescape
func DivSSE64(x1 []float64, x2 []float64, y []float64)

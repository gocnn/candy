// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && gc && !purego

package mat

var (
	addConst32 = AddConstSSE32
	addConst64 = AddConstSSE64
)

func init() {
	if hasAVX2 {
		addConst32 = AddConstAVX32
		addConst64 = AddConstAVX64
	}
}

// AddConst32 adds a constant value c to each element of x, storing the result in y (32 bits).
func AddConst32(c float32, x, y []float32) {
	addConst32(c, x, y)
}

// AddConst64 adds a constant value c to each element of x, storing the result in y (64 bits).
func AddConst64(c float64, x, y []float64) {
	addConst64(c, x, y)
}

// AddConstAVX32 adds a constant value c to each element of x, storing the result in y (32 bits, AVX required).
//
//go:noescape
func AddConstAVX32(c float32, x []float32, y []float32)

// AddConstAVX64 adds a constant value c to each element of x, storing the result in y (64 bits, AVX required).
//
//go:noescape
func AddConstAVX64(c float64, x []float64, y []float64)

// AddConstSSE32 adds a constant value c to each element of x, storing the result in y (32 bits, SSE required).
//
//go:noescape
func AddConstSSE32(c float32, x []float32, y []float32)

// AddConstSSE64 adds a constant value c to each element of x, storing the result in y (64 bits, SSE required).
//
//go:noescape
func AddConstSSE64(c float64, x []float64, y []float64)

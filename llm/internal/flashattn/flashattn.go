package flashattn

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L./lib -lflashattention -lcudart -lcublas -lcurand -lstdc++
#cgo linux LDFLAGS: -Wl,-rpath,./lib
#cgo windows LDFLAGS: -Wl,-rpath,./lib

#include <stdlib.h>
#include <stdint.h>
#include "flash_api.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/gocnn/gocu/cudart"
)

// SupportedHeadDims defines valid head dimensions for Flash Attention.
var SupportedHeadDims = []uint32{32, 64, 96, 128, 160, 192, 224, 256}

// DataType represents the data type for Flash Attention computation.
type DataType int

const (
	// FP16 represents 16-bit floating-point precision.
	FP16 DataType = iota
	// BF16 represents 16-bit brain floating-point precision.
	BF16
)

// Mem represents a GPU tensor with its memory pointer and strides.
type Mem struct {
	Ptr         cudart.DevicePtr
	BatchStride uint32
	RowStride   uint32
	HeadStride  uint32
}

// AttentionParams holds configuration for Flash Attention computation.
type AttentionParams struct {
	Query, Key, Value, Output Mem
	SoftmaxLSE                cudart.DevicePtr
	ALiBiSlopes               Mem
	CuSeqlensQ, CuSeqlensK    *int32
	BatchSize                 uint32
	NumHeads                  uint32
	NumHeadsKV                uint32
	HeadDim                   uint32
	HeadDimRounded            uint32
	SoftmaxScale              float32
	SeqlenQ, SeqlenK          uint32
	SeqlenQRounded            uint32
	SeqlenKRounded            uint32
	DataType                  DataType
	IsCausal                  bool
	UnpaddedLSE               bool
	WindowSizeLeft            int32
	WindowSizeRight           int32
	SoftCap                   float32
}

// NewStandardParams creates parameters for standard multi-head attention.
func NewStandardParams(
	q, k, v, o cudart.DevicePtr,
	batchSize, numHeads, seqlenQ, seqlenK, headDim uint32,
	dataType DataType,
	isCausal bool,
) *AttentionParams {
	query := calculateStrides(numHeads, seqlenQ, headDim)
	key := calculateStrides(numHeads, seqlenK, headDim)
	value := calculateStrides(numHeads, seqlenK, headDim)
	output := calculateStrides(numHeads, seqlenQ, headDim)

	return &AttentionParams{
		Query:           Mem{Ptr: q, BatchStride: query.BatchStride, RowStride: query.RowStride, HeadStride: query.HeadStride},
		Key:             Mem{Ptr: k, BatchStride: key.BatchStride, RowStride: key.RowStride, HeadStride: key.HeadStride},
		Value:           Mem{Ptr: v, BatchStride: value.BatchStride, RowStride: value.RowStride, HeadStride: value.HeadStride},
		Output:          Mem{Ptr: o, BatchStride: output.BatchStride, RowStride: output.RowStride, HeadStride: output.HeadStride},
		BatchSize:       batchSize,
		NumHeads:        numHeads,
		NumHeadsKV:      numHeads,
		HeadDim:         headDim,
		HeadDimRounded:  roundUp(headDim, 8),
		SoftmaxScale:    defaultSoftmaxScale(headDim),
		SeqlenQ:         seqlenQ,
		SeqlenK:         seqlenK,
		SeqlenQRounded:  roundUp(seqlenQ, 128),
		SeqlenKRounded:  roundUp(seqlenK, 128),
		DataType:        dataType,
		IsCausal:        isCausal,
		WindowSizeLeft:  -1,
		WindowSizeRight: -1,
	}
}

// NewGQAParams creates parameters for Grouped Query Attention.
func NewGQAParams(
	q, k, v, o cudart.DevicePtr,
	batchSize, numHeads, numHeadsKV, seqlenQ, seqlenK, headDim uint32,
	dataType DataType,
	isCausal bool,
) *AttentionParams {
	query := calculateStrides(numHeads, seqlenQ, headDim)
	key := calculateStrides(numHeadsKV, seqlenK, headDim)
	value := calculateStrides(numHeadsKV, seqlenK, headDim)
	output := calculateStrides(numHeads, seqlenQ, headDim)

	return &AttentionParams{
		Query:           Mem{Ptr: q, BatchStride: query.BatchStride, RowStride: query.RowStride, HeadStride: query.HeadStride},
		Key:             Mem{Ptr: k, BatchStride: key.BatchStride, RowStride: key.RowStride, HeadStride: key.HeadStride},
		Value:           Mem{Ptr: v, BatchStride: value.BatchStride, RowStride: value.RowStride, HeadStride: value.HeadStride},
		Output:          Mem{Ptr: o, BatchStride: output.BatchStride, RowStride: output.RowStride, HeadStride: output.HeadStride},
		BatchSize:       batchSize,
		NumHeads:        numHeads,
		NumHeadsKV:      numHeadsKV,
		HeadDim:         headDim,
		HeadDimRounded:  roundUp(headDim, 8),
		SoftmaxScale:    defaultSoftmaxScale(headDim),
		SeqlenQ:         seqlenQ,
		SeqlenK:         seqlenK,
		SeqlenQRounded:  roundUp(seqlenQ, 128),
		SeqlenKRounded:  roundUp(seqlenK, 128),
		DataType:        dataType,
		IsCausal:        isCausal,
		WindowSizeLeft:  -1,
		WindowSizeRight: -1,
	}
}

// FlashAttention provides the interface for Flash Attention computation.
type FlashAttention struct{}

// New creates a new FlashAttention instance.
func New() *FlashAttention {
	return &FlashAttention{}
}

// Forward executes the Flash Attention forward pass.
func (fa *FlashAttention) Forward(params *AttentionParams) error {
	if params == nil {
		return errors.New("parameters cannot be nil")
	}

	if err := fa.validateParams(params); err != nil {
		return fmt.Errorf("parameter validation failed: %w", err)
	}

	var cuSeqlensQ, cuSeqlensK *C.int32_t
	if params.CuSeqlensQ != nil {
		cuSeqlensQ = (*C.int32_t)(unsafe.Pointer(params.CuSeqlensQ))
	}
	if params.CuSeqlensK != nil {
		cuSeqlensK = (*C.int32_t)(unsafe.Pointer(params.CuSeqlensK))
	}

	C.run_mha(
		unsafe.Pointer(params.Query.Ptr),
		unsafe.Pointer(params.Key.Ptr),
		unsafe.Pointer(params.Value.Ptr),
		unsafe.Pointer(params.Output.Ptr),
		unsafe.Pointer(params.SoftmaxLSE),
		unsafe.Pointer(params.ALiBiSlopes.Ptr),
		cuSeqlensQ,
		cuSeqlensK,
		C.uint32_t(params.Query.BatchStride),
		C.uint32_t(params.Key.BatchStride),
		C.uint32_t(params.Value.BatchStride),
		C.uint32_t(params.Output.BatchStride),
		C.uint32_t(params.ALiBiSlopes.BatchStride),
		C.uint32_t(params.Query.RowStride),
		C.uint32_t(params.Key.RowStride),
		C.uint32_t(params.Value.RowStride),
		C.uint32_t(params.Output.RowStride),
		C.uint32_t(params.Query.HeadStride),
		C.uint32_t(params.Key.HeadStride),
		C.uint32_t(params.Value.HeadStride),
		C.uint32_t(params.Output.HeadStride),
		C.uint32_t(params.BatchSize),
		C.uint32_t(params.NumHeads),
		C.uint32_t(params.NumHeadsKV),
		C.uint32_t(params.HeadDim),
		C.uint32_t(params.HeadDimRounded),
		C.float(params.SoftmaxScale),
		C.uint32_t(params.SeqlenQ),
		C.uint32_t(params.SeqlenK),
		C.uint32_t(params.SeqlenQRounded),
		C.uint32_t(params.SeqlenKRounded),
		C.int(params.DataType),
		C.int(boolToInt(params.IsCausal)),
		C.int(boolToInt(params.UnpaddedLSE)),
		C.int(params.WindowSizeLeft),
		C.int(params.WindowSizeRight),
		C.float(params.SoftCap),
	)

	runtime.KeepAlive(params)
	return nil
}

// validateParams checks the validity of attention parameters.
func (fa *FlashAttention) validateParams(params *AttentionParams) error {
	if params.Query.Ptr == nil || params.Key.Ptr == nil || params.Value.Ptr == nil || params.Output.Ptr == nil {
		return errors.New("query, key, value, and output tensors must be non-nil")
	}

	if params.BatchSize == 0 || params.NumHeads == 0 || params.NumHeadsKV == 0 {
		return errors.New("batch size, number of heads, and number of KV heads must be positive")
	}

	if params.HeadDim == 0 || params.SeqlenQ == 0 || params.SeqlenK == 0 {
		return errors.New("head dimension and sequence lengths must be positive")
	}

	if !isSupportedHeadDim(params.HeadDim) {
		return fmt.Errorf("unsupported head dimension: %d", params.HeadDim)
	}

	if params.DataType != FP16 && params.DataType != BF16 {
		return fmt.Errorf("unsupported data type: %d", params.DataType)
	}

	if params.NumHeads%params.NumHeadsKV != 0 {
		return fmt.Errorf("num_heads (%d) must be divisible by num_heads_kv (%d)", params.NumHeads, params.NumHeadsKV)
	}

	return nil
}

// isSupportedHeadDim checks if the head dimension is supported.
func isSupportedHeadDim(headDim uint32) bool {
	for _, dim := range SupportedHeadDims {
		if headDim <= dim {
			return true
		}
	}
	return false
}

// boolToInt converts a boolean to a C-compatible integer.
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// calculateStrides computes strides for contiguous tensors.
func calculateStrides(numHeads, seqlen, headDim uint32) Mem {
	return Mem{
		HeadStride:  headDim,
		RowStride:   numHeads * headDim,
		BatchStride: seqlen * numHeads * headDim,
	}
}

// roundUp rounds a value up to the nearest multiple of alignment.
func roundUp(value, alignment uint32) uint32 {
	return ((value + alignment - 1) / alignment) * alignment
}

// defaultSoftmaxScale calculates the default softmax scale (1/sqrt(head_dim)).
func defaultSoftmaxScale(headDim uint32) float32 {
	return 1.0 / float32(math.Sqrt(float64(headDim)))
}

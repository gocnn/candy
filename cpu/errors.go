package cpu

import (
	"errors"
	"fmt"

	"github.com/qntx/goml"
)

// Sentinel errors for common tensor operations
var (
	ErrInvalidShape        = errors.New("invalid tensor shape")
	ErrIncompatibleShape   = errors.New("incompatible tensor shapes")
	ErrIndexOutOfRange     = errors.New("index out of range")
	ErrDimensionOutOfRange = errors.New("dimension out of range")
	ErrInvalidOperation    = errors.New("invalid tensor operation")
	ErrInvalidInput        = errors.New("invalid input")
	ErrZeroStep            = errors.New("step cannot be zero")
	ErrNegativeSize        = errors.New("size must be non-negative")
)

// ShapeError represents shape-related errors with context
type ShapeError struct {
	Op       string
	Expected goml.Shape
	Actual   goml.Shape
	Err      error
}

func (e *ShapeError) Error() string {
	if e.Expected.Size() > 0 && e.Actual.Size() > 0 {
		return fmt.Sprintf("%s: %v (expected %v, got %v)", e.Op, e.Err, e.Expected, e.Actual)
	}
	return fmt.Sprintf("%s: %v", e.Op, e.Err)
}

func (e *ShapeError) Unwrap() error {
	return e.Err
}

// DimensionError represents dimension-related errors with context
type DimensionError struct {
	Op        string
	Dimension int
	MaxDim    int
	Err       error
}

func (e *DimensionError) Error() string {
	return fmt.Sprintf("%s: %v (dimension %d, max %d)", e.Op, e.Err, e.Dimension, e.MaxDim)
}

func (e *DimensionError) Unwrap() error {
	return e.Err
}

// IndexError represents index-related errors with context
type IndexError struct {
	Op    string
	Index int
	Size  int
	Err   error
}

func (e *IndexError) Error() string {
	return fmt.Sprintf("%s: %v (index %d, size %d)", e.Op, e.Err, e.Index, e.Size)
}

func (e *IndexError) Unwrap() error {
	return e.Err
}

// OperationError represents general operation errors with context
type OperationError struct {
	Op      string
	Message string
	Err     error
}

func (e *OperationError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %v (%s)", e.Op, e.Err, e.Message)
	}
	return fmt.Sprintf("%s: %s", e.Op, e.Message)
}

func (e *OperationError) Unwrap() error {
	return e.Err
}

func NewShapeError(op string, sentinelErr error, expected, actual goml.Shape) error {
	return &ShapeError{
		Op:       op,
		Expected: expected,
		Actual:   actual,
		Err:      sentinelErr,
	}
}

func NewDimensionError(op string, sentinelErr error, dim, maxDim int) error {
	return &DimensionError{
		Op:        op,
		Dimension: dim,
		MaxDim:    maxDim,
		Err:       sentinelErr,
	}
}

func NewIndexError(op string, sentinelErr error, index, size int) error {
	return &IndexError{
		Op:    op,
		Index: index,
		Size:  size,
		Err:   sentinelErr,
	}
}

func NewOperationError(op string, message string) error {
	return &OperationError{
		Op:      op,
		Message: message,
		Err:     ErrInvalidOperation,
	}
}

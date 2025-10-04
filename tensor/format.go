package tensor

import (
	"fmt"
	"math"
	"strings"

	"github.com/gocnn/spark/internal/cpu"
)

// String returns a compact, readable string representation of the tensor.
func (t *Tensor[T]) String() string {
	shape := t.layout.Shape()

	if shape.ElemCount() == 0 {
		return fmt.Sprintf("tensor([], shape=%v, dtype=%T, device=%s)", shape.Dims(), *new(T), t.device)
	}

	if shape.Rank() == 0 {
		data := t.Data()
		return fmt.Sprintf("tensor(%v, shape=[], dtype=%T, device=%s)", data[0], *new(T), t.device)
	}

	var sb strings.Builder
	sb.WriteString("tensor(")
	t.format(&sb, 0, make([]int, shape.Rank()))
	sb.WriteString(fmt.Sprintf(", shape=%v, dtype=%T, device=%s)", shape.Dims(), *new(T), t.device))
	return sb.String()
}

// Data returns the underlying data as a slice
func (t *Tensor[T]) Data() []T {
	if cpuStorage, ok := t.storage.(*cpu.CpuStorage[T]); ok {
		return cpuStorage.Data()
	}
	panic("unsupported storage type for Data()")
}

// format recursively formats tensor dimensions.
func (t *Tensor[T]) format(sb *strings.Builder, dim int, idx []int) {
	shape := t.layout.Shape()

	if dim == shape.Rank()-1 {
		t.formatRow(sb, idx)
		return
	}

	sb.WriteByte('[')
	for i := 0; i < shape.Dims()[dim]; i++ {
		if i > 0 {
			sb.WriteString(",\n")
			sb.WriteString(strings.Repeat(" ", dim+8))
		}
		idx[dim] = i
		t.format(sb, dim+1, idx)
	}
	sb.WriteByte(']')
}

// formatRow formats a single row of data with aligned values.
func (t *Tensor[T]) formatRow(sb *strings.Builder, idx []int) {
	sb.WriteByte('[')
	shape := t.layout.Shape()
	rowSize := shape.Dims()[shape.Rank()-1]
	maxWidth := t.globalMaxWidth()
	data := t.Data()

	for i := 0; i < rowSize; i++ {
		if i > 0 {
			sb.WriteString(", ")
		}
		flatIdx := t.flatIndex(idx, i)
		sb.WriteString(t.formatValue(data[flatIdx], maxWidth))
	}
	sb.WriteByte(']')
}

// flatIndex converts multi-dimensional indices to a flat index.
func (t *Tensor[T]) flatIndex(idx []int, last int) int {
	shape := t.layout.Shape()
	flat := last
	stride := 1

	for i := shape.Rank() - 2; i >= 0; i-- {
		stride *= shape.Dims()[i+1]
		flat += idx[i] * stride
	}
	return flat
}

// globalMaxWidth calculates the maximum width for formatting values across the entire tensor.
func (t *Tensor[T]) globalMaxWidth() int {
	data := t.Data()
	max := 0

	for i := 0; i < len(data); i++ {
		if w := len(t.formatValue(data[i], 0)); w > max {
			max = w
		}
	}
	return max
}

// formatValue formats a value with consistent width and precision.
func (t *Tensor[T]) formatValue(v T, width int) string {
	var s string

	switch x := any(v).(type) {
	case float32:
		f := float64(x)
		s = t.formatFloat(f)
	case float64:
		s = t.formatFloat(x)
	default:
		s = fmt.Sprintf("%v", v)
	}

	if width > len(s) {
		return strings.Repeat(" ", width-len(s)) + s
	}
	return s
}

// formatFloat formats floating point numbers with consistent precision.
func (t *Tensor[T]) formatFloat(f float64) string {
	if math.IsNaN(f) {
		return "NaN"
	}
	if math.IsInf(f, 0) {
		if math.IsInf(f, 1) {
			return " Inf"
		}
		return "-Inf"
	}

	// For integers that fit in reasonable range, show as integer
	if f == float64(int64(f)) && math.Abs(f) < 1e6 {
		return fmt.Sprintf("%d", int64(f))
	}

	// For small decimals, use fixed precision
	s := fmt.Sprintf("%.4f", f)
	s = strings.TrimRight(s, "0")
	s = strings.TrimRight(s, ".")

	// Ensure minimum width for alignment
	if len(s) < 6 {
		return fmt.Sprintf("%6s", s)
	}
	return s
}

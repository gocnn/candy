package cpu

import (
	"fmt"
	"math"
	"strings"
)

// String returns a compact, readable string representation of the tensor.
func (t *Tensor[T]) String() string {
	if t.shape.Size() == 0 {
		return fmt.Sprintf("tensor([], shape=[0], dtype=%T)", *new(T))
	}
	if t.shape.Ndim() == 0 {
		return fmt.Sprintf("tensor(%v, shape=[], dtype=%T)", t.data[0], *new(T))
	}

	var sb strings.Builder
	sb.WriteString("tensor(")
	t.format(&sb, 0, make([]int, t.shape.Ndim()))
	sb.WriteString(fmt.Sprintf(", shape=%v, dtype=%T)", t.shape, *new(T)))
	return sb.String()
}

// format recursively formats tensor dimensions.
func (t *Tensor[T]) format(sb *strings.Builder, dim int, idx []int) {
	if dim == t.shape.Ndim()-1 {
		t.formatRow(sb, idx)
		return
	}

	sb.WriteByte('[')
	for i := 0; i < t.shape.At(dim); i++ {
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
	rowSize := t.shape.At(t.shape.Ndim() - 1)
	maxWidth := t.globalMaxWidth() // Use global max width

	for i := 0; i < rowSize; i++ {
		if i > 0 {
			sb.WriteString(", ")
		}
		flatIdx := t.flatIndex(idx, i)
		sb.WriteString(t.formatValue(t.data[flatIdx], maxWidth))
	}
	sb.WriteByte(']')
}

// flatIndex converts multi-dimensional indices to a flat index.
func (t *Tensor[T]) flatIndex(idx []int, last int) int {
	flat := last
	stride := 1
	for i := t.shape.Ndim() - 2; i >= 0; i-- {
		stride *= t.shape.At(i + 1)
		flat += idx[i] * stride
	}
	return flat
}

// globalMaxWidth calculates the maximum width for formatting values across the entire tensor.
func (t *Tensor[T]) globalMaxWidth() int {
	max := 0
	for i := 0; i < len(t.data); i++ {
		if w := len(t.formatValue(t.data[i], 0)); w > max {
			max = w
		}
	}
	return max
}

// formatValue formats a value with consistent width and precision.
func (t *Tensor[T]) formatValue(v T, width int) string {
	var s string
	switch x := any(v).(type) {
	case float32, float64:
		f := float64(x.(float32)) // float32 or float64
		if math.IsNaN(f) {
			s = "NaN"
		} else if math.IsInf(f, 0) {
			s = fmt.Sprintf("%+4s", "Inf")
		} else if f == float64(int64(f)) && math.Abs(f) < 1e6 {
			s = fmt.Sprintf("%d", int64(f))
		} else {
			s = fmt.Sprintf("%.4f", f)
			s = strings.TrimRight(s, "0.")
			if len(s) < 6 { // Ensure minimum width for floats
				s = fmt.Sprintf("%6s", s)
			}
		}
	default:
		s = fmt.Sprintf("%v", v)
	}
	if width > len(s) {
		return strings.Repeat(" ", width-len(s)) + s
	}
	return s
}

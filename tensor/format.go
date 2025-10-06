package tensor

import (
	"fmt"
	"math"
	"strings"
)

// String returns a compact, readable string representation of the tensor,
// mimicking PyTorch-style formatting for improved aesthetics and clarity.
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

// format recursively formats tensor dimensions with nested brackets and indentation.
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
			sb.WriteString(strings.Repeat(" ", dim+8)) // Aligns with "tensor([" (8 chars).
		}
		idx[dim] = i
		t.format(sb, dim+1, idx)
	}
	sb.WriteByte(']')
}

// formatRow formats a single row with right-aligned, space-separated values.
func (t *Tensor[T]) formatRow(sb *strings.Builder, idx []int) {
	sb.WriteByte('[')
	shape := t.layout.Shape()
	rowSize := shape.Dims()[shape.Rank()-1]
	maxWidth := t.globalMaxWidth()
	data := t.Data()

	for i := range rowSize {
		if i > 0 {
			sb.WriteString(", ")
		}
		flatIdx := t.flatIndex(idx, i)
		sb.WriteString(t.formatValue(data[flatIdx], maxWidth))
	}
	sb.WriteByte(']')
}

// flatIndex computes the flat index from multi-dimensional indices.
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

// globalMaxWidth determines the maximum string length of formatted values for alignment.
func (t *Tensor[T]) globalMaxWidth() int {
	data := t.Data()
	max := 0
	for _, v := range data {
		if w := len(t.formatValue(v, 0)); w > max {
			max = w
		}
	}
	return max
}

// formatValue formats a value to a string, right-aligned to the given width.
func (t *Tensor[T]) formatValue(v T, width int) string {
	var s string
	switch x := any(v).(type) {
	case float32:
		s = t.formatFloat(float64(x))
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

// formatFloat formats a float with PyTorch-like precision and notation.
func (t *Tensor[T]) formatFloat(f float64) string {
	if math.IsNaN(f) {
		return "nan"
	}
	if math.IsInf(f, 0) {
		if math.IsInf(f, 1) {
			return "inf"
		}
		return "-inf"
	}
	s := fmt.Sprintf("%.4f", f)
	s = strings.TrimRight(s, "0") // Trim trailing zeros, keep "." for integer-like floats.
	return s
}

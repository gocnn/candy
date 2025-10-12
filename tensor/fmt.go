package tensor

import (
	"fmt"
	"math"
	"strings"
)

// String returns a compact, PyTorch-style string representation of the tensor.
func (t *Tensor[T]) String() string {
	shape := t.layout.Shape()
	if shape.Numel() == 0 {
		return fmt.Sprintf("tensor([], shape=%v, dtype=%T, device=%s)", shape.Dims(), *new(T), t.device)
	}
	if shape.Rank() == 0 {
		return fmt.Sprintf("tensor(%v, shape=[], dtype=%T, device=%s)", t.Data()[0], *new(T), t.device)
	}

	var sb strings.Builder
	sb.WriteString("tensor(")
	t.format(&sb, 0, make([]int, shape.Rank()))
	sb.WriteString(fmt.Sprintf(", shape=%v, dtype=%T, device=%s)", shape.Dims(), *new(T), t.device))
	return sb.String()
}

// format writes nested tensor dimensions to the builder with proper indentation.
func (t *Tensor[T]) format(sb *strings.Builder, dim int, idx []int) {
	shape := t.layout.Shape()
	if dim == shape.Rank()-1 {
		t.formatRow(sb, idx)
		return
	}

	sb.WriteByte('[')
	for i := 0; i < shape.Dims()[dim]; i++ {
		if i > 0 {
			sb.WriteString(",\n" + strings.Repeat(" ", dim+8))
		}
		idx[dim] = i
		t.format(sb, dim+1, idx)
	}
	sb.WriteByte(']')
}

// formatRow writes a single row of values to the builder, right-aligned.
func (t *Tensor[T]) formatRow(sb *strings.Builder, idx []int) {
	sb.WriteByte('[')
	shape := t.layout.Shape()
	n := shape.Dims()[shape.Rank()-1]
	w := t.maxWidth()
	data := t.Data()

	for i := range n {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(t.formatValue(data[t.flatIndex(idx, i)], w))
	}
	sb.WriteByte(']')
}

// flatIndex returns the flat index for multi-dimensional indices.
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

// maxWidth returns the maximum string length of formatted values for alignment.
func (t *Tensor[T]) maxWidth() int {
	w := 0
	for _, v := range t.Data() {
		if n := len(t.formatValue(v, 0)); n > w {
			w = n
		}
	}
	return w
}

// formatValue returns a right-aligned string for a tensor value.
func (t *Tensor[T]) formatValue(v T, w int) string {
	s := t.formatFloat(v)
	if w > len(s) {
		return strings.Repeat(" ", w-len(s)) + s
	}
	return s
}

// formatFloat returns a PyTorch-style string for a float value.
func (t *Tensor[T]) formatFloat(v T) string {
	var f float64
	switch x := any(v).(type) {
	case float32:
		f = float64(x)
	case float64:
		f = x
	default:
		return fmt.Sprintf("%v", v)
	}
	if math.IsNaN(f) {
		return "nan"
	}
	if math.IsInf(f, 0) {
		if math.IsInf(f, 1) {
			return "inf"
		}
		return "-inf"
	}

	// Use scientific notation for very large or very small numbers
	absF := math.Abs(f)
	if absF != 0 && (absF >= 1e8 || absF < 1e-4) {
		s := fmt.Sprintf("%.4e", f)
		// Clean up scientific notation: remove trailing zeros in mantissa
		if idx := strings.Index(s, "e"); idx != -1 {
			mantissa := s[:idx]
			exponent := s[idx:]
			mantissa = strings.TrimRight(mantissa, "0")
			mantissa = strings.TrimRight(mantissa, ".")
			return mantissa + exponent
		}
		return s
	}

	// Use regular decimal notation for normal range numbers
	s := fmt.Sprintf("%.4f", f)
	s = strings.TrimRight(s, "0")
	// Preserve decimal point for floats (PyTorch style: 1. not 1)
	if strings.HasSuffix(s, ".") {
		return s
	}
	// If we trimmed all decimal places, add back the decimal point
	if !strings.Contains(s, ".") {
		return s + "."
	}
	return s
}

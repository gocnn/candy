package mat

import (
	"fmt"
	"iter"
	"math"
	"math/rand/v2"
)

// Matrix is a matrix of complex128.
type Matrix struct {
	Rows int
	Cols int
	Data []float64
}

func New(v ...[]float64) *Matrix {
	rows := len(v)
	var cols int
	if rows > 0 {
		cols = len(v[0])
	}

	data := make([]float64, rows*cols)
	for i := range rows {
		copy(data[i*cols:(i+1)*cols], v[i])
	}

	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

// Zero returns a zero matrix.
func Zero(rows, cols int) *Matrix {
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]float64, rows*cols),
	}
}

func ZeroLike(m *Matrix) *Matrix {
	return Zero(Dim(m))
}

func OneLike(m *Matrix) *Matrix {
	return AddC(1.0, ZeroLike(m))
}

// Rand returns a matrix with elements that pseudo-random number in the half-open interval [0.0,1.0).
// m, n is the dimension of the matrix.
// s is the source of the pseudo-random number.
func Rand(rows, cols int, s ...rand.Source) *Matrix {
	return F(Zero(rows, cols), func(_ float64) float64 { return rnd(s...).Float64() })
}

// Randn returns a matrix with elements that normally distributed float64 in the range [-math.MaxFloat64, +math.MaxFloat64] with standard normal distribution.
// m, n is the dimension of the matrix.
// s is the source of the pseudo-random number.
func Randn(rows, cols int, s ...rand.Source) *Matrix {
	return F(Zero(rows, cols), func(_ float64) float64 { return rnd(s...).NormFloat64() })
}

// rnd returns a pseudo-random number generator.
func rnd(s ...rand.Source) *rand.Rand {
	if len(s) == 0 || s[0] == nil {
		return rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}

	return rand.New(s[0])
}

// At returns a value of matrix at (i,j).
func (m *Matrix) At(i, j int) float64 {
	return m.Data[i*m.Cols+j]
}

func (m *Matrix) Row(i int) []float64 {
	return m.Data[i*m.Cols : (i+1)*m.Cols]
}

// Set sets a value of matrix at (i,j).
func (m *Matrix) Set(i, j int, v float64) {
	m.Data[i*m.Cols+j] = v
}

func (m *Matrix) SetRow(i int, v []float64) {
	copy(m.Data[i*m.Cols:(i+1)*m.Cols], v)
}

// AddAt adds a value of matrix at (i,j).
func (m *Matrix) AddAt(i, j int, v float64) {
	m.Data[i*m.Cols+j] += v
}

// N returns the number of rows.
func (m *Matrix) N() int {
	return m.Rows
}

// Seq2 returns a sequence of rows.
func (m *Matrix) Seq2() iter.Seq2[int, []float64] {
	return func(yield func(int, []float64) bool) {
		for i := range m.Rows {
			if !yield(i, m.Row(i)) {
				return
			}
		}
	}
}

// String returns a string representation of the matrix.
func (m *Matrix) String() string {
	out := make([][]float64, m.Rows)
	for i := range m.Rows {
		out[i] = m.Row(i)
	}

	return fmt.Sprintf("%v", out)
}

func Size(m *Matrix) int {
	s := 1
	for _, v := range Shape(m) {
		s = s * v
	}

	return s
}

func Shape(m *Matrix) []int {
	a, b := Dim(m)
	return []int{a, b}
}

func Dim(m *Matrix) (rows int, cols int) {
	return m.Rows, m.Cols
}

func AddC(c float64, m *Matrix) *Matrix {
	return F(m, func(v float64) float64 { return c + v })
}

// SubC returns c - m.
func SubC(c float64, m *Matrix) *Matrix {
	return F(m, func(v float64) float64 { return c - v })
}

func MulC(c float64, m *Matrix) *Matrix {
	return F(m, func(v float64) float64 { return c * v })
}

func Exp(m *Matrix) *Matrix {
	return F(m, func(v float64) float64 { return math.Exp(v) })
}

func Log(m *Matrix) *Matrix {
	return F(m, func(v float64) float64 { return math.Log(v) })
}

func Sin(m *Matrix) *Matrix {
	return F(m, func(v float64) float64 { return math.Sin(v) })
}

func Cos(m *Matrix) *Matrix {
	return F(m, func(v float64) float64 { return math.Cos(v) })
}

func Tanh(m *Matrix) *Matrix {
	return F(m, func(v float64) float64 { return math.Tanh(v) })
}

func Pow(c float64, m *Matrix) *Matrix {
	return F(m, func(v float64) float64 { return math.Pow(v, c) })
}

func Add(m, n *Matrix) *Matrix {
	return F2(m, n, func(a, b float64) float64 { return a + b })
}

func Sub(m, n *Matrix) *Matrix {
	return F2(m, n, func(a, b float64) float64 { return a - b })
}

func Mul(m, n *Matrix) *Matrix {
	return F2(m, n, func(a, b float64) float64 { return a * b })
}

func Div(m, n *Matrix) *Matrix {
	return F2(m, n, func(a, b float64) float64 { return a / b })
}

func Mean(m *Matrix) float64 {
	return Sum(m) / float64(Size(m))
}

func Sum(m *Matrix) float64 {
	var sum float64
	for _, v := range m.Data {
		sum = sum + v
	}

	return sum
}

func Max(m *Matrix) float64 {
	max := m.Data[0]
	for _, v := range m.Data {
		if v > max {
			max = v
		}
	}

	return max
}

func Min(m *Matrix) float64 {
	min := m.Data[0]
	for _, v := range m.Data {
		if v < min {
			min = v
		}
	}

	return min
}

func Argmax(m *Matrix) []int {
	rows, cols := Dim(m)

	out := make([]int, rows)
	for i := range rows {
		max := m.At(i, 0)
		for j := range cols {
			mij := m.At(i, j)
			if mij > max {
				max, out[i] = mij, j
			}
		}
	}

	return out
}

// MatMul returns the matrix product of m and n.
func MatMul(m, n *Matrix) *Matrix {
	a, b := m.Rows, m.Cols
	_, p := n.Rows, n.Cols

	out := Zero(a, p)
	for i := range a {
		for k := range b {
			mik := m.Data[i*b+k]
			for j := range p {
				out.Data[i*p+j] += mik * n.Data[k*p+j]
			}
		}
	}

	return out
}

func Clip(m *Matrix, min, max float64) *Matrix {
	return F(m, func(v float64) float64 {
		if v < min {
			return min
		}

		if v > max {
			return max
		}

		return v
	})
}

// Mask returns a matrix with elements that 1 if f() is true and 0 otherwise.
func Mask(m *Matrix, f func(x float64) bool) *Matrix {
	return F(m, func(v float64) float64 {
		if f(v) {
			return 1
		}

		return 0
	})
}

func Broadcast(m, n *Matrix) (*Matrix, *Matrix) {
	return BroadcastTo(Shape(n), m), BroadcastTo(Shape(m), n)
}

func BroadcastTo(shape []int, m *Matrix) *Matrix {
	rows, cols := shape[0], shape[1]

	if m.Rows == 1 && m.Cols == 1 {
		data := make([]float64, rows*cols)
		for i := range data {
			data[i] = m.At(0, 0)
		}

		return Reshape(shape, New(data))
	}

	if m.Rows == 1 {
		// b is ignored
		out := Zero(rows, m.Cols)
		for i := range rows {
			out.SetRow(i, m.Row(0))
		}

		return out
	}

	if m.Cols == 1 {
		// a is ignored
		out := Zero(m.Rows, cols)
		for i := range m.Rows {
			for j := range cols {
				out.Set(i, j, m.At(i, 0))
			}
		}

		return out
	}

	return m
}

func SumTo(shape []int, m *Matrix) *Matrix {
	rows, cols := shape[0], shape[1]

	if rows == 1 && cols == 1 {
		return New([]float64{Sum(m)})
	}

	if rows == 1 {
		return SumAxis0(m)
	}

	if cols == 1 {
		return SumAxis1(m)
	}

	return m
}

// SumAxis0 returns the sum of each column.
func SumAxis0(m *Matrix) *Matrix {
	rows, cols := Dim(m)

	data := make([]float64, cols)
	for i := range rows {
		for j := range cols {
			data[j] += m.At(i, j)
		}
	}

	return New(data)
}

// SumAxis1 returns the sum of each row.
func SumAxis1(m *Matrix) *Matrix {
	rows, cols := Dim(m)

	data := make([]float64, rows)
	for i := range rows {
		for j := range cols {
			data[i] += m.At(i, j)
		}
	}

	return Transpose(New(data))
}

func MaxAxis1(m *Matrix) *Matrix {
	rows, cols := Dim(m)

	v := make([]float64, rows)
	for i := range rows {
		max := m.At(i, 0)
		for j := range cols {
			mij := m.At(i, j)
			if mij > max {
				max = mij
			}
		}

		v[i] = max
	}

	return Transpose(New(v))
}

func Transpose(m *Matrix) *Matrix {
	rows, cols := Dim(m)

	out := Zero(cols, rows)
	for i := range rows {
		for j := range cols {
			out.Set(j, i, m.At(i, j))
		}
	}

	return out
}

// Reshape returns the matrix with the given shape.
func Reshape(shape []int, m *Matrix) *Matrix {
	rows, cols := Dim(m)
	a, b := shape[0], shape[1]

	if a < 1 {
		a = rows * cols / b
	}

	if b < 1 {
		b = rows * cols / a
	}

	out := Zero(a, b)
	copy(out.Data, m.Data)
	return out
}

func F(m *Matrix, f func(a float64) float64) *Matrix {
	out := ZeroLike(m)
	for i := range m.Data {
		out.Data[i] = f(m.Data[i])
	}

	return out
}

func F2(m, n *Matrix, f func(a, b float64) float64) *Matrix {
	x, y := Broadcast(m, n)

	out := ZeroLike(x)
	for i := range x.Data {
		out.Data[i] = f(x.Data[i], y.Data[i])
	}

	return out
}

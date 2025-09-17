package cpu

import (
	"math"
	"math/rand/v2"
	"sync"

	"github.com/qntx/goml"
)

var _ goml.Tensor[float64] = (*Tensor[float64])(nil)

// A Tensor matrix implementation.
type Tensor[T goml.D] struct {
	data         []T
	grad         *Tensor[T]
	layout       goml.Layout
	requiresGrad bool

	gradMu sync.RWMutex
}

// Ones creates a tensor filled with ones.
func Ones[T goml.D](shape goml.Shape) goml.Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = T(1)
	}
	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: false,
	}
}

// OnesLike creates a tensor with the same shape as the input, filled with ones.
func OnesLike[T goml.D](d *Tensor[T]) goml.Tensor[T] {
	return Ones[T](d.Shape())
}

// Zeros creates a tensor filled with zeros.
func Zeros[T goml.D](shape goml.Shape) goml.Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: false,
	}
}

// ZerosLike creates a tensor with the same shape as the input, filled with zeros.
func ZerosLike[T goml.D](d *Tensor[T]) goml.Tensor[T] {
	return Zeros[T](d.Shape())
}

// Rand creates a tensor with uniform random values in [lo, hi).
func Rand[T goml.D](lo, hi T, shape goml.Shape) goml.Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	rangeVal := float64(hi - lo)
	for i := range data {
		data[i] = T(float64(lo) + rand.Float64()*rangeVal)
	}
	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: false,
	}
}

// RandLike creates a tensor with the same shape as the input, filled with uniform random values in [lo, hi).
func RandLike[T goml.D](d *Tensor[T], lo, hi T) goml.Tensor[T] {
	return Rand(lo, hi, d.Shape())
}

// Randn creates a tensor with normal distribution (specified mean and std).
func Randn[T goml.D](mean, std T, shape goml.Shape) goml.Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = T(rand.NormFloat64()*float64(std) + float64(mean))
	}
	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: false,
	}
}

// RandnLike creates a tensor with the same shape as the input, filled with normal distribution.
func RandnLike[T goml.D](d *Tensor[T], mean, std T) goml.Tensor[T] {
	return Randn(mean, std, d.Shape())
}

// Full creates a tensor filled with the specified value.
func Full[T goml.D](value T, shape goml.Shape) goml.Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = value
	}
	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: false,
	}
}

// FullLike creates a tensor with the same shape as the input, filled with the specified value.
func FullLike[T goml.D](d *Tensor[T], value T) goml.Tensor[T] {
	return Full(value, d.Shape())
}

// Eye creates an identity matrix (2D tensor with 1s on diagonal, 0s elsewhere).
func Eye[T goml.D](n int) goml.Tensor[T] {
	shape := goml.NewShape(n, n)
	size := shape.Size()
	data := make([]T, size)
	for i := range n {
		data[i*n+i] = T(1) // Set diagonal elements to 1
	}
	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: false,
	}
}

// Arange creates a 1D tensor with linearly spaced values from start to end (exclusive).
func Arange[T goml.D](start, end T) goml.Tensor[T] {
	return ArangeStep(start, end, T(1))
}

// ArangeStep creates a 1D tensor with linearly spaced values from start to end (exclusive) with given step.
func ArangeStep[T goml.D](start, end, step T) goml.Tensor[T] {
	if step == 0 {
		panic("tensor: step cannot be zero")
	}

	var size int
	if step > 0 {
		size = int(math.Ceil(float64(end-start) / float64(step)))
	} else {
		size = int(math.Ceil(float64(start-end) / float64(-step)))
	}

	if size <= 0 {
		size = 0
	}

	shape := goml.NewShape(size)
	data := make([]T, size)
	for i := 0; i < size; i++ {
		data[i] = start + T(i)*step
	}

	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: false,
	}
}

// FromSlice creates a tensor from a Go slice with the given shape.
func FromSlice[T goml.D](slice []T) goml.Tensor[T] {
	shape := goml.NewShape(len(slice))
	data := make([]T, len(slice))
	copy(data, slice)

	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: false,
	}
}

// Tril creates a lower triangular matrix (elements below diagonal are 1, others are 0).
func Tril[T goml.D](n int) goml.Tensor[T] {
	shape := goml.NewShape(n, n)
	size := shape.Size()
	data := make([]T, size)

	for i := range n {
		for j := 0; j <= i; j++ {
			data[i*n+j] = T(1)
		}
	}

	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: false,
	}
}

// Triu creates an upper triangular matrix (elements above diagonal are 1, others are 0).
func Triu[T goml.D](n int) goml.Tensor[T] {
	shape := goml.NewShape(n, n)
	size := shape.Size()
	data := make([]T, size)

	for i := range n {
		for j := i; j < n; j++ {
			data[i*n+j] = T(1)
		}
	}

	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: false,
	}
}

// Linspace creates a 1D tensor with linearly spaced values from start to end (inclusive).
func Linspace[T goml.D](start, end T, steps int) goml.Tensor[T] {
	if steps <= 0 {
		panic("tensor: steps must be positive")
	}
	if steps == 1 {
		return FromSlice([]T{start})
	}

	shape := goml.NewShape(steps)
	data := make([]T, steps)
	stepSize := float64(end-start) / float64(steps-1)

	for i := range steps {
		data[i] = T(float64(start) + float64(i)*stepSize)
	}

	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: false,
	}
}

func New[T goml.D]() goml.Tensor[T] {
	return &Tensor[T]{
		requiresGrad: false,
	}
}

func NewWith[T goml.D](data []T, shape goml.Shape, requiresGrad bool) goml.Tensor[T] {
	return &Tensor[T]{
		data:         data,
		layout:       goml.Contiguous(shape),
		requiresGrad: requiresGrad,
	}
}

// OnesLike creates a tensor with the same shape as this tensor, filled with ones.
func (d *Tensor[T]) OnesLike() goml.Tensor[T] {
	return Ones[T](d.layout.Shape())
}

// ZerosLike creates a tensor with the same shape as this tensor, filled with zeros.
func (d *Tensor[T]) ZerosLike() goml.Tensor[T] {
	return Zeros[T](d.layout.Shape())
}

// RandLike creates a tensor with the same shape as this tensor, filled with uniform random values in [lo, hi).
func (d *Tensor[T]) RandLike(lo, hi T) goml.Tensor[T] {
	return Rand(lo, hi, d.layout.Shape())
}

// RandnLike creates a tensor with the same shape as this tensor, filled with normal distribution (mean=0, std=1).
func (d *Tensor[T]) RandnLike(mean, std T) goml.Tensor[T] {
	return Randn(mean, std, d.layout.Shape())
}

// FullLike creates a tensor with the same shape as this tensor, filled with the specified value.
func (d *Tensor[T]) FullLike(value T) goml.Tensor[T] {
	return Full(value, d.layout.Shape())
}

// Clone creates a deep copy of this tensor.
func (d *Tensor[T]) Clone() goml.Tensor[T] {
	data := make([]T, len(d.data))
	copy(data, d.data)
	return &Tensor[T]{
		data:         data,
		layout:       d.layout.Clone(),
		requiresGrad: d.requiresGrad,
	}
}

func (d *Tensor[T]) DType() goml.DType {
	var zero T
	switch any(zero).(type) {
	case float32:
		return goml.Float32
	case float64:
		return goml.Float64
	default:
		panic("unsupported type")
	}
}

func (d *Tensor[T]) Layout() goml.Layout {
	return d.layout
}

func (d *Tensor[T]) SetLayout(layout goml.Layout) {
	d.layout = layout
}

func (d *Tensor[T]) Device() goml.Device {
	return goml.CPU
}

func (d *Tensor[T]) Size() int {
	return d.layout.Shape().Size()
}

func (d *Tensor[T]) Data() []T {
	return d.data
}

func (d *Tensor[T]) SetData(data []T) {
	d.data = data
}

func (d *Tensor[T]) Shape() goml.Shape {
	return d.layout.Shape()
}

func (d *Tensor[T]) SetShape(shape goml.Shape) {
	d.layout = goml.Contiguous(shape)
}

func (d *Tensor[T]) RequiresGrad() bool {
	return d.requiresGrad
}

func (d *Tensor[T]) SetRequiresGrad(requiresGrad bool) {
	d.requiresGrad = requiresGrad
}

func (d *Tensor[T]) Grad() goml.Tensor[T] {
	d.gradMu.RLock()
	defer d.gradMu.RUnlock()
	return d.grad
}

func (d *Tensor[T]) SetGrad(grad goml.Tensor[T]) {
	d.gradMu.Lock()
	defer d.gradMu.Unlock()
	d.grad = grad.(*Tensor[T])
}

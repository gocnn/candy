package cpu

import (
	"math"
	"math/rand/v2"
	"sync"

	"github.com/qntx/spark"
)

var _ spark.Tensor[float64] = (*Tensor[float64])(nil)

// A Tensor matrix implementation.
type Tensor[T spark.D] struct {
	data         []T
	grad         *Tensor[T]
	layout       spark.Layout
	requiresGrad bool

	gradMu sync.RWMutex
}

// Ones creates a tensor filled with ones.
func Ones[T spark.D](shape spark.Shape) spark.Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = T(1)
	}
	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: false,
	}
}

// OnesLike creates a tensor with the same shape as the input, filled with ones.
func OnesLike[T spark.D](d *Tensor[T]) spark.Tensor[T] {
	return Ones[T](d.Shape())
}

// Zeros creates a tensor filled with zeros.
func Zeros[T spark.D](shape spark.Shape) spark.Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: false,
	}
}

// ZerosLike creates a tensor with the same shape as the input, filled with zeros.
func ZerosLike[T spark.D](d *Tensor[T]) spark.Tensor[T] {
	return Zeros[T](d.Shape())
}

// Rand creates a tensor with uniform random values in [lo, hi).
func Rand[T spark.D](lo, hi T, shape spark.Shape) spark.Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	rangeVal := float64(hi - lo)
	for i := range data {
		data[i] = T(float64(lo) + rand.Float64()*rangeVal)
	}
	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: false,
	}
}

// RandLike creates a tensor with the same shape as the input, filled with uniform random values in [lo, hi).
func RandLike[T spark.D](d *Tensor[T], lo, hi T) spark.Tensor[T] {
	return Rand(lo, hi, d.Shape())
}

// Randn creates a tensor with normal distribution (specified mean and std).
func Randn[T spark.D](mean, std T, shape spark.Shape) spark.Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = T(rand.NormFloat64()*float64(std) + float64(mean))
	}
	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: false,
	}
}

// RandnLike creates a tensor with the same shape as the input, filled with normal distribution.
func RandnLike[T spark.D](d *Tensor[T], mean, std T) spark.Tensor[T] {
	return Randn(mean, std, d.Shape())
}

// Full creates a tensor filled with the specified value.
func Full[T spark.D](value T, shape spark.Shape) spark.Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = value
	}
	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: false,
	}
}

// FullLike creates a tensor with the same shape as the input, filled with the specified value.
func FullLike[T spark.D](d *Tensor[T], value T) spark.Tensor[T] {
	return Full(value, d.Shape())
}

// Eye creates an identity matrix (2D tensor with 1s on diagonal, 0s elsewhere).
func Eye[T spark.D](n int) spark.Tensor[T] {
	shape := spark.NewShape(n, n)
	size := shape.Size()
	data := make([]T, size)
	for i := range n {
		data[i*n+i] = T(1) // Set diagonal elements to 1
	}
	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: false,
	}
}

// Arange creates a 1D tensor with linearly spaced values from start to end (exclusive).
func Arange[T spark.D](start, end T) spark.Tensor[T] {
	return ArangeStep(start, end, T(1))
}

// ArangeStep creates a 1D tensor with linearly spaced values from start to end (exclusive) with given step.
func ArangeStep[T spark.D](start, end, step T) spark.Tensor[T] {
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

	shape := spark.NewShape(size)
	data := make([]T, size)
	for i := 0; i < size; i++ {
		data[i] = start + T(i)*step
	}

	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: false,
	}
}

// FromSlice creates a tensor from a Go slice with the given shape.
func FromSlice[T spark.D](slice []T) spark.Tensor[T] {
	shape := spark.NewShape(len(slice))
	data := make([]T, len(slice))
	copy(data, slice)

	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: false,
	}
}

// Tril creates a lower triangular matrix (elements below diagonal are 1, others are 0).
func Tril[T spark.D](n int) spark.Tensor[T] {
	shape := spark.NewShape(n, n)
	size := shape.Size()
	data := make([]T, size)

	for i := range n {
		for j := 0; j <= i; j++ {
			data[i*n+j] = T(1)
		}
	}

	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: false,
	}
}

// Triu creates an upper triangular matrix (elements above diagonal are 1, others are 0).
func Triu[T spark.D](n int) spark.Tensor[T] {
	shape := spark.NewShape(n, n)
	size := shape.Size()
	data := make([]T, size)

	for i := range n {
		for j := i; j < n; j++ {
			data[i*n+j] = T(1)
		}
	}

	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: false,
	}
}

// Linspace creates a 1D tensor with linearly spaced values from start to end (inclusive).
func Linspace[T spark.D](start, end T, steps int) spark.Tensor[T] {
	if steps <= 0 {
		panic("tensor: steps must be positive")
	}
	if steps == 1 {
		return FromSlice([]T{start})
	}

	shape := spark.NewShape(steps)
	data := make([]T, steps)
	stepSize := float64(end-start) / float64(steps-1)

	for i := range steps {
		data[i] = T(float64(start) + float64(i)*stepSize)
	}

	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: false,
	}
}

func New[T spark.D]() spark.Tensor[T] {
	return &Tensor[T]{
		requiresGrad: false,
	}
}

func NewWith[T spark.D](data []T, shape spark.Shape, requiresGrad bool) spark.Tensor[T] {
	return &Tensor[T]{
		data:         data,
		layout:       spark.Contiguous(shape),
		requiresGrad: requiresGrad,
	}
}

// OnesLike creates a tensor with the same shape as this tensor, filled with ones.
func (d *Tensor[T]) OnesLike() spark.Tensor[T] {
	return Ones[T](d.layout.Shape())
}

// ZerosLike creates a tensor with the same shape as this tensor, filled with zeros.
func (d *Tensor[T]) ZerosLike() spark.Tensor[T] {
	return Zeros[T](d.layout.Shape())
}

// RandLike creates a tensor with the same shape as this tensor, filled with uniform random values in [lo, hi).
func (d *Tensor[T]) RandLike(lo, hi T) spark.Tensor[T] {
	return Rand(lo, hi, d.layout.Shape())
}

// RandnLike creates a tensor with the same shape as this tensor, filled with normal distribution (mean=0, std=1).
func (d *Tensor[T]) RandnLike(mean, std T) spark.Tensor[T] {
	return Randn(mean, std, d.layout.Shape())
}

// FullLike creates a tensor with the same shape as this tensor, filled with the specified value.
func (d *Tensor[T]) FullLike(value T) spark.Tensor[T] {
	return Full(value, d.layout.Shape())
}

// Clone creates a deep copy of this tensor.
func (d *Tensor[T]) Clone() spark.Tensor[T] {
	data := make([]T, len(d.data))
	copy(data, d.data)
	return &Tensor[T]{
		data:         data,
		layout:       d.layout.Clone(),
		requiresGrad: d.requiresGrad,
	}
}

func (d *Tensor[T]) DType() spark.DType {
	var zero T
	switch any(zero).(type) {
	case float32:
		return spark.Float32
	case float64:
		return spark.Float64
	default:
		panic("unsupported type")
	}
}

func (d *Tensor[T]) Layout() spark.Layout {
	return d.layout
}

func (d *Tensor[T]) SetLayout(layout spark.Layout) {
	d.layout = layout
}

func (d *Tensor[T]) Device() spark.Device {
	return spark.CPU
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

func (d *Tensor[T]) Shape() spark.Shape {
	return d.layout.Shape()
}

func (d *Tensor[T]) SetShape(shape spark.Shape) {
	d.layout = spark.Contiguous(shape)
}

func (d *Tensor[T]) RequiresGrad() bool {
	return d.requiresGrad
}

func (d *Tensor[T]) SetRequiresGrad(requiresGrad bool) {
	d.requiresGrad = requiresGrad
}

func (d *Tensor[T]) Grad() spark.Tensor[T] {
	d.gradMu.RLock()
	defer d.gradMu.RUnlock()
	return d.grad
}

func (d *Tensor[T]) SetGrad(grad spark.Tensor[T]) {
	d.gradMu.Lock()
	defer d.gradMu.Unlock()
	d.grad = grad.(*Tensor[T])
}

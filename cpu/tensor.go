package cpu

import (
	"math"
	"math/rand/v2"
	"sync"

	"github.com/qntx/goml"
)

// A Tensor matrix implementation.
type Tensor[T goml.D] struct {
	data         []T
	grad         *Tensor[T]
	shape        goml.Shape
	requiresGrad bool

	gradMu sync.RWMutex
}

// Ones creates a tensor filled with ones.
func Ones[T goml.D](shape goml.Shape) *Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = T(1)
	}
	return &Tensor[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// OnesLike creates a tensor with the same shape as the input, filled with ones.
func OnesLike[T goml.D](d *Tensor[T]) *Tensor[T] {
	return Ones[T](d.shape)
}

// Zeros creates a tensor filled with zeros.
func Zeros[T goml.D](shape goml.Shape) *Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	return &Tensor[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// ZerosLike creates a tensor with the same shape as the input, filled with zeros.
func ZerosLike[T goml.D](d *Tensor[T]) *Tensor[T] {
	return Zeros[T](d.shape)
}

// Rand creates a tensor with uniform random values in [lo, hi).
func Rand[T goml.D](lo, hi T, shape goml.Shape) *Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	rangeVal := float64(hi - lo)
	for i := range data {
		data[i] = T(float64(lo) + rand.Float64()*rangeVal)
	}
	return &Tensor[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// RandLike creates a tensor with the same shape as the input, filled with uniform random values in [lo, hi).
func RandLike[T goml.D](d *Tensor[T], lo, hi T) *Tensor[T] {
	return Rand(lo, hi, d.shape)
}

// Randn creates a tensor with normal distribution (specified mean and std).
func Randn[T goml.D](mean, std T, shape goml.Shape) *Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = T(rand.NormFloat64()*float64(std) + float64(mean))
	}
	return &Tensor[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// RandnLike creates a tensor with the same shape as the input, filled with normal distribution.
func RandnLike[T goml.D](d *Tensor[T], mean, std T) *Tensor[T] {
	return Randn(mean, std, d.shape)
}

// Full creates a tensor filled with the specified value.
func Full[T goml.D](value T, shape goml.Shape) *Tensor[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = value
	}
	return &Tensor[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// FullLike creates a tensor with the same shape as the input, filled with the specified value.
func FullLike[T goml.D](d *Tensor[T], value T) *Tensor[T] {
	return Full(value, d.shape)
}

// Eye creates an identity matrix (2D tensor with 1s on diagonal, 0s elsewhere).
func Eye[T goml.D](n int) *Tensor[T] {
	shape := goml.NewShape(n, n)
	size := shape.Size()
	data := make([]T, size)
	for i := range n {
		data[i*n+i] = T(1) // Set diagonal elements to 1
	}
	return &Tensor[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// Arange creates a 1D tensor with linearly spaced values from start to end (exclusive).
func Arange[T goml.D](start, end T) *Tensor[T] {
	return ArangeStep(start, end, T(1))
}

// ArangeStep creates a 1D tensor with linearly spaced values from start to end (exclusive) with given step.
func ArangeStep[T goml.D](start, end, step T) *Tensor[T] {
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
		shape:        shape,
		requiresGrad: false,
	}
}

// FromSlice creates a tensor from a Go slice with the given shape.
func FromSlice[T goml.D](slice []T) *Tensor[T] {
	shape := goml.NewShape(len(slice))
	data := make([]T, len(slice))
	copy(data, slice)

	return &Tensor[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// Tril creates a lower triangular matrix (elements below diagonal are 1, others are 0).
func Tril[T goml.D](n int) *Tensor[T] {
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
		shape:        shape,
		requiresGrad: false,
	}
}

// Triu creates an upper triangular matrix (elements above diagonal are 1, others are 0).
func Triu[T goml.D](n int) *Tensor[T] {
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
		shape:        shape,
		requiresGrad: false,
	}
}

// Linspace creates a 1D tensor with linearly spaced values from start to end (inclusive).
func Linspace[T goml.D](start, end T, steps int) *Tensor[T] {
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
		shape:        shape,
		requiresGrad: false,
	}
}

func New[T goml.D]() *Tensor[T] {
	return &Tensor[T]{
		requiresGrad: false,
	}
}

func NewWith[T goml.D](data []T, shape goml.Shape, requiresGrad bool) *Tensor[T] {
	return &Tensor[T]{
		data:         data,
		shape:        shape,
		requiresGrad: requiresGrad,
	}
}

func (d *Tensor[T]) Data() []T {
	return d.data
}

func (d *Tensor[T]) SetData(data []T) {
	d.data = data
}

func (d *Tensor[T]) Shape() goml.Shape {
	return d.shape
}

func (d *Tensor[T]) SetShape(shape goml.Shape) {
	d.shape = shape
}

func (d *Tensor[T]) RequiresGrad() bool {
	return d.requiresGrad
}

func (d *Tensor[T]) SetRequiresGrad(requiresGrad bool) {
	d.requiresGrad = requiresGrad
}

func (d *Tensor[T]) Grad() *Tensor[T] {
	d.gradMu.RLock()
	defer d.gradMu.RUnlock()
	return d.grad
}

func (d *Tensor[T]) SetGrad(grad *Tensor[T]) {
	d.gradMu.Lock()
	defer d.gradMu.Unlock()
	d.grad = grad
}

// OnesLike creates a tensor with the same shape as this tensor, filled with ones.
func (d *Tensor[T]) OnesLike() *Tensor[T] {
	return Ones[T](d.shape)
}

// ZerosLike creates a tensor with the same shape as this tensor, filled with zeros.
func (d *Tensor[T]) ZerosLike() *Tensor[T] {
	return Zeros[T](d.shape)
}

// RandLike creates a tensor with the same shape as this tensor, filled with uniform random values in [lo, hi).
func (d *Tensor[T]) RandLike(lo, hi T) *Tensor[T] {
	return Rand(lo, hi, d.shape)
}

// RandnLike creates a tensor with the same shape as this tensor, filled with normal distribution (mean=0, std=1).
func (d *Tensor[T]) RandnLike(mean, std T) *Tensor[T] {
	return Randn(mean, std, d.shape)
}

// FullLike creates a tensor with the same shape as this tensor, filled with the specified value.
func (d *Tensor[T]) FullLike(value T) *Tensor[T] {
	return Full(value, d.shape)
}

// Clone creates a deep copy of this tensor.
func (d *Tensor[T]) Clone() *Tensor[T] {
	data := make([]T, len(d.data))
	copy(data, d.data)
	return &Tensor[T]{
		data:         data,
		shape:        d.shape,
		requiresGrad: d.requiresGrad,
	}
}

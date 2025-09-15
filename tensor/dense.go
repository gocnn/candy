package tensor

import (
	"math"
	"math/rand/v2"
	"sync"
)

// A Dense matrix implementation.
type Dense[T DType] struct {
	data         []T
	grad         *Dense[T]
	shape        Shape
	requiresGrad bool

	gradMu sync.RWMutex
}

// Ones creates a tensor filled with ones.
func Ones[T DType](shape Shape) *Dense[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = T(1)
	}
	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// OnesLike creates a tensor with the same shape as the input, filled with ones.
func OnesLike[T DType](d *Dense[T]) *Dense[T] {
	return Ones[T](d.shape)
}

// Zeros creates a tensor filled with zeros.
func Zeros[T DType](shape Shape) *Dense[T] {
	size := shape.Size()
	data := make([]T, size)
	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// ZerosLike creates a tensor with the same shape as the input, filled with zeros.
func ZerosLike[T DType](d *Dense[T]) *Dense[T] {
	return Zeros[T](d.shape)
}

// Rand creates a tensor with uniform random values in [0, 1).
func Rand[T DType](shape Shape) *Dense[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = T(rand.Float64())
	}
	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// RandLike creates a tensor with the same shape as the input, filled with uniform random values in [0, 1).
func RandLike[T DType](d *Dense[T]) *Dense[T] {
	return Rand[T](d.shape)
}

// RandRange creates a tensor with uniform random values in [lo, hi).
func RandRange[T DType](lo, hi T, shape Shape) *Dense[T] {
	size := shape.Size()
	data := make([]T, size)
	rangeVal := float64(hi - lo)
	for i := range data {
		data[i] = T(float64(lo) + rand.Float64()*rangeVal)
	}
	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// RandRangeLike creates a tensor with the same shape as the input, filled with uniform random values in [lo, hi).
func RandRangeLike[T DType](d *Dense[T], lo, hi T) *Dense[T] {
	return RandRange(lo, hi, d.shape)
}

// RandN creates a tensor with normal distribution (mean=0, std=1).
func RandN[T DType](shape Shape) *Dense[T] {
	return RandNormal[T](0, 1, shape)
}

// RandNLike creates a tensor with the same shape as the input, filled with normal distribution (mean=0, std=1).
func RandNLike[T DType](d *Dense[T]) *Dense[T] {
	return RandN[T](d.shape)
}

// RandNormal creates a tensor with normal distribution (specified mean and std).
func RandNormal[T DType](mean, std T, shape Shape) *Dense[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = T(rand.NormFloat64()*float64(std) + float64(mean))
	}
	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// RandNormalLike creates a tensor with the same shape as the input, filled with normal distribution.
func RandNormalLike[T DType](d *Dense[T], mean, std T) *Dense[T] {
	return RandNormal(mean, std, d.shape)
}

// Full creates a tensor filled with the specified value.
func Full[T DType](value T, shape Shape) *Dense[T] {
	size := shape.Size()
	data := make([]T, size)
	for i := range data {
		data[i] = value
	}
	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// FullLike creates a tensor with the same shape as the input, filled with the specified value.
func FullLike[T DType](d *Dense[T], value T) *Dense[T] {
	return Full(value, d.shape)
}

// Eye creates an identity matrix (2D tensor with 1s on diagonal, 0s elsewhere).
func Eye[T DType](n int) *Dense[T] {
	shape := NewShape(n, n)
	size := shape.Size()
	data := make([]T, size)
	for i := range n {
		data[i*n+i] = T(1) // Set diagonal elements to 1
	}
	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// Arange creates a 1D tensor with linearly spaced values from start to end (exclusive).
func Arange[T DType](start, end T) *Dense[T] {
	return ArangeStep(start, end, T(1))
}

// ArangeStep creates a 1D tensor with linearly spaced values from start to end (exclusive) with given step.
func ArangeStep[T DType](start, end, step T) *Dense[T] {
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

	shape := NewShape(size)
	data := make([]T, size)
	for i := 0; i < size; i++ {
		data[i] = start + T(i)*step
	}

	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// FromSlice creates a tensor from a Go slice with the given shape.
func FromSlice[T DType](slice []T) *Dense[T] {
	shape := NewShape(len(slice))
	data := make([]T, len(slice))
	copy(data, slice)

	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// Tril creates a lower triangular matrix (elements below diagonal are 1, others are 0).
func Tril[T DType](n int) *Dense[T] {
	shape := NewShape(n, n)
	size := shape.Size()
	data := make([]T, size)

	for i := range n {
		for j := 0; j <= i; j++ {
			data[i*n+j] = T(1)
		}
	}

	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// Triu creates an upper triangular matrix (elements above diagonal are 1, others are 0).
func Triu[T DType](n int) *Dense[T] {
	shape := NewShape(n, n)
	size := shape.Size()
	data := make([]T, size)

	for i := range n {
		for j := i; j < n; j++ {
			data[i*n+j] = T(1)
		}
	}

	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

// Linspace creates a 1D tensor with linearly spaced values from start to end (inclusive).
func Linspace[T DType](start, end T, steps int) *Dense[T] {
	if steps <= 0 {
		panic("tensor: steps must be positive")
	}
	if steps == 1 {
		return FromSlice([]T{start})
	}

	shape := NewShape(steps)
	data := make([]T, steps)
	stepSize := float64(end-start) / float64(steps-1)

	for i := range steps {
		data[i] = T(float64(start) + float64(i)*stepSize)
	}

	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: false,
	}
}

func New[T DType]() *Dense[T] {
	return &Dense[T]{
		requiresGrad: false,
	}
}

func NewWith[T DType](data []T, shape Shape, requiresGrad bool) *Dense[T] {
	return &Dense[T]{
		data:         data,
		shape:        shape,
		requiresGrad: requiresGrad,
	}
}

func (d *Dense[T]) Data() []T {
	return d.data
}

func (d *Dense[T]) SetData(data []T) {
	d.data = data
}

func (d *Dense[T]) Shape() Shape {
	return d.shape
}

func (d *Dense[T]) SetShape(shape Shape) {
	d.shape = shape
}

func (d *Dense[T]) RequiresGrad() bool {
	return d.requiresGrad
}

func (d *Dense[T]) SetRequiresGrad(requiresGrad bool) {
	d.requiresGrad = requiresGrad
}

func (d *Dense[T]) Grad() *Dense[T] {
	d.gradMu.RLock()
	defer d.gradMu.RUnlock()
	return d.grad
}

func (d *Dense[T]) SetGrad(grad *Dense[T]) {
	d.gradMu.Lock()
	defer d.gradMu.Unlock()
	d.grad = grad
}

// OnesLike creates a tensor with the same shape as this tensor, filled with ones.
func (d *Dense[T]) OnesLike() *Dense[T] {
	return Ones[T](d.shape)
}

// ZerosLike creates a tensor with the same shape as this tensor, filled with zeros.
func (d *Dense[T]) ZerosLike() *Dense[T] {
	return Zeros[T](d.shape)
}

// RandLike creates a tensor with the same shape as this tensor, filled with uniform random values in [0, 1).
func (d *Dense[T]) RandLike() *Dense[T] {
	return Rand[T](d.shape)
}

// RandRangeLike creates a tensor with the same shape as this tensor, filled with uniform random values in [lo, hi).
func (d *Dense[T]) RandRangeLike(lo, hi T) *Dense[T] {
	return RandRange[T](lo, hi, d.shape)
}

// RandNLike creates a tensor with the same shape as this tensor, filled with normal distribution (mean=0, std=1).
func (d *Dense[T]) RandNLike() *Dense[T] {
	return RandN[T](d.shape)
}

// RandNormalLike creates a tensor with the same shape as this tensor, filled with normal distribution.
func (d *Dense[T]) RandNormalLike(mean, std T) *Dense[T] {
	return RandNormal(mean, std, d.shape)
}

// FullLike creates a tensor with the same shape as this tensor, filled with the specified value.
func (d *Dense[T]) FullLike(value T) *Dense[T] {
	return Full(value, d.shape)
}

// Clone creates a deep copy of this tensor.
func (d *Dense[T]) Clone() *Dense[T] {
	data := make([]T, len(d.data))
	copy(data, d.data)
	return &Dense[T]{
		data:         data,
		shape:        d.shape,
		requiresGrad: d.requiresGrad,
	}
}

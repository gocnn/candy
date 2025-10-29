package nn

import (
	"fmt"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/tensor"
)

type BatchNorm2d[T candy.D] struct {
	numFeatures int
	runningMean *tensor.Tensor[T]
	runningVar  *tensor.Tensor[T]
	weight      *tensor.Tensor[T] // gamma, optional
	bias        *tensor.Tensor[T] // beta, optional
	eps         float64
	momentum    float64
	train       bool
}

func NewBatchNorm2d[T candy.D](numFeatures int, device candy.Device) *BatchNorm2d[T] {
	if numFeatures <= 0 {
		panic("batchnorm2d: numFeatures must be > 0")
	}
	rm, err := tensor.Zeros[T](candy.NewShape(numFeatures), device)
	if err != nil {
		panic(fmt.Errorf("batchnorm2d: running mean: %w", err))
	}
	rv, err := tensor.Ones[T](candy.NewShape(numFeatures), device)
	if err != nil {
		panic(fmt.Errorf("batchnorm2d: running var: %w", err))
	}
	w, err := tensor.Ones[T](candy.NewShape(numFeatures), device)
	if err != nil {
		panic(fmt.Errorf("batchnorm2d: weight: %w", err))
	}
	w.SetIsVar(true)
	b, err := tensor.Zeros[T](candy.NewShape(numFeatures), device)
	if err != nil {
		panic(fmt.Errorf("batchnorm2d: bias: %w", err))
	}
	b.SetIsVar(true)
	return &BatchNorm2d[T]{
		numFeatures: numFeatures,
		runningMean: rm,
		runningVar:  rv,
		weight:      w,
		bias:        b,
		eps:         1e-5,
		momentum:    0.1,
		train:       true,
	}
}

func NewBatchNorm2dNoAffine[T candy.D](numFeatures int, device candy.Device) *BatchNorm2d[T] {
	if numFeatures <= 0 {
		panic("batchnorm2d: numFeatures must be > 0")
	}
	rm, err := tensor.Zeros[T](candy.NewShape(numFeatures), device)
	if err != nil {
		panic(fmt.Errorf("batchnorm2d: running mean: %w", err))
	}
	rv, err := tensor.Ones[T](candy.NewShape(numFeatures), device)
	if err != nil {
		panic(fmt.Errorf("batchnorm2d: running var: %w", err))
	}
	return &BatchNorm2d[T]{
		numFeatures: numFeatures,
		runningMean: rm,
		runningVar:  rv,
		eps:         1e-5,
		momentum:    0.1,
		train:       true,
	}
}

func (bn *BatchNorm2d[T]) Train() { bn.train = true }
func (bn *BatchNorm2d[T]) Eval()  { bn.train = false }

func (bn *BatchNorm2d[T]) Weight() *tensor.Tensor[T]      { return bn.weight }
func (bn *BatchNorm2d[T]) Bias() *tensor.Tensor[T]        { return bn.bias }
func (bn *BatchNorm2d[T]) RunningMean() *tensor.Tensor[T] { return bn.runningMean }
func (bn *BatchNorm2d[T]) RunningVar() *tensor.Tensor[T]  { return bn.runningVar }

func (bn *BatchNorm2d[T]) Parameters() []*tensor.Tensor[T] {
	var ps []*tensor.Tensor[T]
	if bn.weight != nil {
		ps = append(ps, bn.weight)
	}
	if bn.bias != nil {
		ps = append(ps, bn.bias)
	}
	return ps
}

func (bn *BatchNorm2d[T]) forwardTrain(x *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	dims := x.Dims()
	if len(dims) < 2 {
		return nil, fmt.Errorf("batchnorm2d: expected rank>=2, got %d", len(dims))
	}
	if dims[1] != bn.numFeatures {
		return nil, fmt.Errorf("batchnorm2d: channel mismatch: got %d want %d", dims[1], bn.numFeatures)
	}
	// reduce over all dims except channel (dim=1)
	reduce := make([]int, 0, len(dims)-1)
	for i := range dims {
		if i != 1 {
			reduce = append(reduce, i)
		}
	}
	mean, err := x.MeanKeep(reduce)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: mean: %w", err)
	}
	diff, err := x.BroadcastSub(mean)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: x-mean: %w", err)
	}
	v, err := diff.Sqr()
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: sqr: %w", err)
	}
	varKeep, err := v.MeanKeep(reduce)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: var mean: %w", err)
	}
	// update running stats (reshape mean/var to [C])
	c := dims[1]
	meanC, err := mean.Reshape(c)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: reshape mean: %w", err)
	}
	varC, err := varKeep.Reshape(c)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: reshape var: %w", err)
	}
	// Bessel correction factor
	bs := 1
	for i, d := range dims {
		if i != 1 {
			bs *= d
		}
	}
	momVar := bn.momentum
	if bs > 1 {
		momVar = bn.momentum * float64(bs) / float64(bs-1)
	}
	rmScaled, err := bn.runningMean.MulScalar(1.0 - bn.momentum)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: scale running mean: %w", err)
	}
	mScaled, err := meanC.MulScalar(bn.momentum)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: scale batch mean: %w", err)
	}
	rmNew, err := rmScaled.Add(mScaled)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: update running mean: %w", err)
	}
	bn.runningMean.SetStorage(rmNew.Storage())

	rvScaled, err := bn.runningVar.MulScalar(1.0 - bn.momentum)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: scale running var: %w", err)
	}
	vvScaled, err := varC.MulScalar(momVar)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: scale batch var: %w", err)
	}
	rvNew, err := rvScaled.Add(vvScaled)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: update running var: %w", err)
	}
	bn.runningVar.SetStorage(rvNew.Storage())

	den, err := varKeep.AddScalar(bn.eps)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: add eps: %w", err)
	}
	den, err = den.Sqrt()
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: sqrt: %w", err)
	}
	norm, err := diff.BroadcastDiv(den)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: div: %w", err)
	}
	// affine
	if bn.weight != nil {
		ts := make([]int, len(dims))
		for i := range dims {
			if i == 1 {
				ts[i] = c
			} else {
				ts[i] = 1
			}
		}
		w, err := bn.weight.Reshape(ts...)
		if err != nil {
			return nil, fmt.Errorf("batchnorm2d: reshape weight: %w", err)
		}
		norm, err = norm.BroadcastMul(w)
		if err != nil {
			return nil, fmt.Errorf("batchnorm2d: mul gamma: %w", err)
		}
	}
	if bn.bias != nil {
		ts := make([]int, len(dims))
		for i := range dims {
			if i == 1 {
				ts[i] = c
			} else {
				ts[i] = 1
			}
		}
		b, err := bn.bias.Reshape(ts...)
		if err != nil {
			return nil, fmt.Errorf("batchnorm2d: reshape bias: %w", err)
		}
		norm, err = norm.BroadcastAdd(b)
		if err != nil {
			return nil, fmt.Errorf("batchnorm2d: add beta: %w", err)
		}
	}
	return norm, nil
}

func (bn *BatchNorm2d[T]) forwardEval(x *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	dims := x.Dims()
	if len(dims) < 2 {
		return nil, fmt.Errorf("batchnorm2d: expected rank>=2, got %d", len(dims))
	}
	if dims[1] != bn.numFeatures {
		return nil, fmt.Errorf("batchnorm2d: channel mismatch: got %d want %d", dims[1], bn.numFeatures)
	}
	c := dims[1]
	// build target shape same rank as input: [1,C,1,1,...]
	ts := make([]int, len(dims))
	for i := range dims {
		if i == 1 {
			ts[i] = c
		} else {
			ts[i] = 1
		}
	}
	rm, err := bn.runningMean.Reshape(ts...)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: reshape running mean: %w", err)
	}
	rv, err := bn.runningVar.Reshape(ts...)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: reshape running var: %w", err)
	}
	y, err := x.BroadcastSub(rm)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: sub mean: %w", err)
	}
	den, err := rv.AddScalar(bn.eps)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: add eps: %w", err)
	}
	den, err = den.Sqrt()
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: sqrt: %w", err)
	}
	y, err = y.BroadcastDiv(den)
	if err != nil {
		return nil, fmt.Errorf("batchnorm2d: div: %w", err)
	}
	if bn.weight != nil {
		ts := make([]int, len(dims))
		for i := range dims {
			if i == 1 {
				ts[i] = c
			} else {
				ts[i] = 1
			}
		}
		w, err := bn.weight.Reshape(ts...)
		if err != nil {
			return nil, fmt.Errorf("batchnorm2d: reshape weight: %w", err)
		}
		y, err = y.BroadcastMul(w)
		if err != nil {
			return nil, fmt.Errorf("batchnorm2d: mul gamma: %w", err)
		}
	}
	if bn.bias != nil {
		ts := make([]int, len(dims))
		for i := range dims {
			if i == 1 {
				ts[i] = c
			} else {
				ts[i] = 1
			}
		}
		b, err := bn.bias.Reshape(ts...)
		if err != nil {
			return nil, fmt.Errorf("batchnorm2d: reshape bias: %w", err)
		}
		y, err = y.BroadcastAdd(b)
		if err != nil {
			return nil, fmt.Errorf("batchnorm2d: add beta: %w", err)
		}
	}
	return y, nil
}

func (bn *BatchNorm2d[T]) Forward(x *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if bn.train {
		return bn.forwardTrain(x)
	}
	return bn.forwardEval(x)
}

func (bn *BatchNorm2d[T]) MustForward(x *tensor.Tensor[T]) *tensor.Tensor[T] {
	y, err := bn.Forward(x)
	if err != nil {
		panic(err)
	}
	return y
}

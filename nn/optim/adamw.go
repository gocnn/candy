package optim

import (
	"errors"
	"fmt"
	"math"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/tensor"
)

// AdamWParams holds parameters for the AdamW optimizer.
type AdamWParams struct {
	LearningRate float64 // Learning rate
	Beta1        float64 // First moment decay rate
	Beta2        float64 // Second moment decay rate
	Epsilon      float64 // Numerical stability constant
	WeightDecay  float64 // Weight decay coefficient
}

// DefaultAdamWParams returns default AdamW parameters.
func DefaultAdamWParams() AdamWParams {
	return AdamWParams{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.01,
	}
}

// adamVar holds a variable and its momentum states for AdamW.
type adamVar[T candy.D] struct {
	v *tensor.Tensor[T] // Variable tensor
	m *tensor.Tensor[T] // First moment
	s *tensor.Tensor[T] // Second moment
}

var _ Optimizer[float32] = (*AdamW[float32])(nil)
var _ Optimizer[float64] = (*AdamW[float64])(nil)

// AdamW implements the AdamW optimizer.
type AdamW[T candy.D] struct {
	vars []adamVar[T]
	step int
	p    AdamWParams
}

// NewAdamW creates a new AdamW optimizer with the given parameters.
func NewAdamW[T candy.D](vars []*tensor.Tensor[T], p AdamWParams) (*AdamW[T], error) {
	vs := make([]adamVar[T], 0, len(vars))
	for _, v := range vars {
		if !v.IsVar() {
			continue
		}
		m, err := v.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("adamw: failed to create first moment: %w", err)
		}
		s, err := v.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("adamw: failed to create second moment: %w", err)
		}
		vs = append(vs, adamVar[T]{v: v, m: m, s: s})
	}
	return &AdamW[T]{vars: vs, p: p}, nil
}

// NewAdamWWithLR creates an AdamW optimizer with a custom learning rate.
func NewAdamWWithLR[T candy.D](vars []*tensor.Tensor[T], lr float64) (*AdamW[T], error) {
	p := DefaultAdamWParams()
	p.LearningRate = lr
	return NewAdamW(vars, p)
}

// Params returns the current AdamW parameters.
func (a *AdamW[T]) Params() AdamWParams {
	return a.p
}

// SetParams sets the AdamW parameters.
func (a *AdamW[T]) SetParams(p AdamWParams) {
	a.p = p
}

// LearningRate returns the current learning rate.
func (a *AdamW[T]) LearningRate() float64 {
	return a.p.LearningRate
}

// SetLearningRate sets the learning rate.
func (a *AdamW[T]) SetLearningRate(lr float64) {
	a.p.LearningRate = lr
}

// Add adds a variable to be optimized.
func (a *AdamW[T]) Add(v *tensor.Tensor[T]) error {
	if !v.IsVar() {
		return errors.New("adamw: not a variable")
	}
	m, err := v.ZerosLike()
	if err != nil {
		return fmt.Errorf("adamw: failed to create first moment: %w", err)
	}
	s, err := v.ZerosLike()
	if err != nil {
		return fmt.Errorf("adamw: failed to create second moment: %w", err)
	}
	a.vars = append(a.vars, adamVar[T]{v: v, m: m, s: s})
	return nil
}

// Vars returns all variables being optimized.
func (a *AdamW[T]) Vars() []*tensor.Tensor[T] {
	vs := make([]*tensor.Tensor[T], 0, len(a.vars))
	for _, av := range a.vars {
		vs = append(vs, av.v)
	}
	return vs
}

// Optimize performs backward propagation and an AdamW step.
func (a *AdamW[T]) Optimize(loss *tensor.Tensor[T]) error {
	gs := tensor.NewGradStore[T]()
	if err := tensor.Backward(loss, gs); err != nil {
		return fmt.Errorf("adamw: failed to backward: %w", err)
	}
	return a.Step(gs)
}

// MustOptimize performs backward propagation and an AdamW step, panics on error.
func (a *AdamW[T]) MustOptimize(loss *tensor.Tensor[T]) {
	if err := a.Optimize(loss); err != nil {
		panic(err)
	}
}

// Step performs an AdamW optimization step.
func (a *AdamW[T]) Step(gs *tensor.GradStore[T]) error {
	a.step++
	p := a.p
	mScale := 1.0 / (1.0 - math.Pow(p.Beta1, float64(a.step)))
	vScale := 1.0 / (1.0 - math.Pow(p.Beta2, float64(a.step)))

	for _, av := range a.vars {
		v := av.v
		g := gs.Get(v)
		if g == nil {
			continue
		}

		// First moment: m = beta1*m + (1-beta1)*g
		m, err := av.m.MulScalar(p.Beta1)
		if err != nil {
			return fmt.Errorf("adamw: failed to scale first moment: %w", err)
		}
		g1, err := g.MulScalar(1.0 - p.Beta1)
		if err != nil {
			return fmt.Errorf("adamw: failed to scale gradient: %w", err)
		}
		m, err = m.Add(g1)
		if err != nil {
			return fmt.Errorf("adamw: failed to update first moment: %w", err)
		}

		// Second moment: s = beta2*s + (1-beta2)*g²
		s, err := av.s.MulScalar(p.Beta2)
		if err != nil {
			return fmt.Errorf("adamw: failed to scale second moment: %w", err)
		}
		g2, err := g.Mul(g)
		if err != nil {
			return fmt.Errorf("adamw: failed to square gradient: %w", err)
		}
		g2, err = g2.MulScalar(1.0 - p.Beta2)
		if err != nil {
			return fmt.Errorf("adamw: failed to scale squared gradient: %w", err)
		}
		s, err = s.Add(g2)
		if err != nil {
			return fmt.Errorf("adamw: failed to update second moment: %w", err)
		}

		// Bias correction
		mh, err := m.MulScalar(mScale)
		if err != nil {
			return fmt.Errorf("adamw: failed to correct first moment: %w", err)
		}
		vh, err := s.MulScalar(vScale)
		if err != nil {
			return fmt.Errorf("adamw: failed to correct second moment: %w", err)
		}

		// Weight decay: v = v * (1 - lr * wd)
		v, err = v.MulScalar(1.0 - p.LearningRate*p.WeightDecay)
		if err != nil {
			return fmt.Errorf("adamw: failed to apply weight decay: %w", err)
		}

		// Adjusted gradient: mh / (√vh + eps)
		r, err := vh.Sqrt()
		if err != nil {
			return fmt.Errorf("adamw: failed to sqrt second moment: %w", err)
		}
		r, err = r.AddScalar(p.Epsilon)
		if err != nil {
			return fmt.Errorf("adamw: failed to add epsilon: %w", err)
		}
		u, err := mh.Div(r)
		if err != nil {
			return fmt.Errorf("adamw: failed to compute update: %w", err)
		}

		// Update: v = v - lr * u
		u, err = u.MulScalar(p.LearningRate)
		if err != nil {
			return fmt.Errorf("adamw: failed to scale update: %w", err)
		}
		v, err = v.Sub(u)
		if err != nil {
			return fmt.Errorf("adamw: failed to update variable: %w", err)
		}

		// Update tensors
		av.m.SetStorage(m.Storage())
		av.s.SetStorage(s.Storage())
		av.v.SetStorage(v.Storage())
	}
	return nil
}

// MustStep performs an AdamW optimization step, panics on error.
func (a *AdamW[T]) MustStep(gs *tensor.GradStore[T]) {
	if err := a.Step(gs); err != nil {
		panic(err)
	}
}

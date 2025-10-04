package optim

import (
	"errors"
	"fmt"
	"math"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

// AdamWParams holds parameters for the AdamW optimizer.
type AdamWParams struct {
	LearningRate float64 // Learning rate
	Beta1        float64 // Exponential decay rate for first moment
	Beta2        float64 // Exponential decay rate for second moment
	Epsilon      float64 // Small constant for numerical stability
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
type adamVar[T spark.D] struct {
	varTensor *tensor.Tensor[T]
	m         *tensor.Tensor[T] // First moment (momentum)
	v         *tensor.Tensor[T] // Second moment (RMSprop)
}

var _ Optimizer[float32] = (*AdamW[float32])(nil)

// AdamW implements the AdamW optimizer.
type AdamW[T spark.D] struct {
	vars   []adamVar[T]
	step   int
	params AdamWParams
}

// NewAdamW creates a new AdamW optimizer with the given parameters.
func NewAdamW[T spark.D](vars []*tensor.Tensor[T], params AdamWParams) (*AdamW[T], error) {
	adamVars := make([]adamVar[T], 0, len(vars))
	for _, v := range vars {
		if !v.IsVar() {
			continue
		}

		m, err := v.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("create first moment: %w", err)
		}

		vv, err := v.ZerosLike()
		if err != nil {
			return nil, fmt.Errorf("create second moment: %w", err)
		}

		adamVars = append(adamVars, adamVar[T]{
			varTensor: v,
			m:         m,
			v:         vv,
		})
	}
	return &AdamW[T]{vars: adamVars, params: params}, nil
}

// NewAdamWWithLR creates an AdamW optimizer with a custom learning rate and default parameters.
func NewAdamWWithLR[T spark.D](vars []*tensor.Tensor[T], lr float64) (*AdamW[T], error) {
	params := DefaultAdamWParams()
	params.LearningRate = lr
	return NewAdamW[T](vars, params)
}

// Step performs an AdamW optimization step.
func (a *AdamW[T]) Step(grads *tensor.GradStore[T]) error {
	a.step++
	p := a.params

	// Bias correction
	scaleM := 1.0 / (1.0 - math.Pow(p.Beta1, float64(a.step)))
	scaleV := 1.0 / (1.0 - math.Pow(p.Beta2, float64(a.step)))

	for _, av := range a.vars {
		theta := av.varTensor
		grad := grads.Get(theta)
		if grad == nil {
			continue
		}

		// Update first moment: m = beta1 * m + (1 - beta1) * grad
		mScaled, err := av.m.MulScalar(p.Beta1)
		if err != nil {
			return fmt.Errorf("scale first moment: %w", err)
		}
		gradScaled, err := grad.MulScalar(1.0 - p.Beta1)
		if err != nil {
			return fmt.Errorf("scale gradient: %w", err)
		}
		nextM, err := mScaled.Add(gradScaled)
		if err != nil {
			return fmt.Errorf("update first moment: %w", err)
		}

		// Update second moment: v = beta2 * v + (1 - beta2) * grad^2
		vScaled, err := av.v.MulScalar(p.Beta2)
		if err != nil {
			return fmt.Errorf("scale second moment: %w", err)
		}
		gradSqr, err := grad.Mul(grad)
		if err != nil {
			return fmt.Errorf("square gradient: %w", err)
		}
		gradSqrScaled, err := gradSqr.MulScalar(1.0 - p.Beta2)
		if err != nil {
			return fmt.Errorf("scale squared gradient: %w", err)
		}
		nextV, err := vScaled.Add(gradSqrScaled)
		if err != nil {
			return fmt.Errorf("update second moment: %w", err)
		}

		// Bias-corrected moments
		mHat, err := nextM.MulScalar(scaleM)
		if err != nil {
			return fmt.Errorf("bias-correct first moment: %w", err)
		}
		vHat, err := nextV.MulScalar(scaleV)
		if err != nil {
			return fmt.Errorf("bias-correct second moment: %w", err)
		}

		// Weight decay: theta = theta * (1 - lr * lambda)
		thetaScaled, err := theta.MulScalar(1.0 - p.LearningRate*p.WeightDecay)
		if err != nil {
			return fmt.Errorf("apply weight decay: %w", err)
		}

		// Compute adjusted gradient: mHat / (sqrt(vHat) + eps)
		vHatSqrt, err := vHat.Sqrt()
		if err != nil {
			return fmt.Errorf("sqrt second moment: %w", err)
		}
		vHatSqrtEps, err := vHatSqrt.AddScalar(p.Epsilon)
		if err != nil {
			return fmt.Errorf("add epsilon: %w", err)
		}
		adjGrad, err := mHat.Div(vHatSqrtEps)
		if err != nil {
			return fmt.Errorf("compute adjusted gradient: %w", err)
		}

		// Update: theta = theta - lr * adjGrad
		lrAdjGrad, err := adjGrad.MulScalar(p.LearningRate)
		if err != nil {
			return fmt.Errorf("scale adjusted gradient: %w", err)
		}
		finalTheta, err := thetaScaled.Sub(lrAdjGrad)
		if err != nil {
			return fmt.Errorf("update parameter: %w", err)
		}

		// Update tensors
		av.m.SetStorage(nextM.Storage())
		av.v.SetStorage(nextV.Storage())
		theta.SetStorage(finalTheta.Storage())
	}
	return nil
}

// Optimize performs backward propagation and an AdamW step.
func (a *AdamW[T]) Optimize(loss *tensor.Tensor[T]) error {
	store := tensor.NewGradStore[T]()
	if err := tensor.Backward(loss, store); err != nil {
		return fmt.Errorf("backward propagation: %w", err)
	}
	return a.Step(store)
}

// LearningRate returns the current learning rate.
func (a *AdamW[T]) LearningRate() float64 {
	return a.params.LearningRate
}

// SetLearningRate sets the learning rate.
func (a *AdamW[T]) SetLearningRate(lr float64) {
	a.params.LearningRate = lr
}

// Add adds a variable to be optimized.
func (a *AdamW[T]) Add(v *tensor.Tensor[T]) error {
	if !v.IsVar() {
		return errors.New("not a variable")
	}
	m, err := v.ZerosLike()
	if err != nil {
		return fmt.Errorf("create first moment: %w", err)
	}
	vv, err := v.ZerosLike()
	if err != nil {
		return fmt.Errorf("create second moment: %w", err)
	}
	a.vars = append(a.vars, adamVar[T]{varTensor: v, m: m, v: vv})
	return nil
}

// Vars returns all variables being optimized.
func (a *AdamW[T]) Vars() []*tensor.Tensor[T] {
	vars := make([]*tensor.Tensor[T], 0, len(a.vars))
	for _, av := range a.vars {
		vars = append(vars, av.varTensor)
	}
	return vars
}

// Params returns the current AdamW parameters.
func (a *AdamW[T]) Params() AdamWParams {
	return a.params
}

// SetParams sets the AdamW parameters.
func (a *AdamW[T]) SetParams(params AdamWParams) {
	a.params = params
}

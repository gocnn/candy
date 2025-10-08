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

		m := v.ZerosLike()
		vv := v.ZerosLike()

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
	return NewAdamW(vars, params)
}

// Params returns the current AdamW parameters.
func (a *AdamW[T]) Params() AdamWParams {
	return a.params
}

// SetParams sets the AdamW parameters.
func (a *AdamW[T]) SetParams(params AdamWParams) {
	a.params = params
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
	m := v.ZerosLike()
	vv := v.ZerosLike()
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

// Optimize performs backward propagation and an AdamW step.
func (a *AdamW[T]) Optimize(loss *tensor.Tensor[T]) error {
	store := tensor.NewGradStore[T]()
	if err := tensor.Backward(loss, store); err != nil {
		return fmt.Errorf("backward propagation: %w", err)
	}
	return a.Step(store)
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

		beta1Tensor := tensor.Full[T](p.Beta1, spark.NewShape(), theta.Device())
		beta1CompTensor := tensor.Full[T](1.0-p.Beta1, spark.NewShape(), theta.Device())
		beta2Tensor := tensor.Full[T](p.Beta2, spark.NewShape(), theta.Device())
		beta2CompTensor := tensor.Full[T](1.0-p.Beta2, spark.NewShape(), theta.Device())

		// Update first moment: m = beta1 * m + (1 - beta1) * grad
		mScaled, err := av.m.Mul(beta1Tensor)
		if err != nil {
			return fmt.Errorf("scale first moment: %w", err)
		}
		gradScaled, err := grad.Mul(beta1CompTensor)
		if err != nil {
			return fmt.Errorf("scale gradient: %w", err)
		}
		nextM, err := mScaled.BroadcastAdd(gradScaled)
		if err != nil {
			return fmt.Errorf("update first moment: %w", err)
		}

		// Update second moment: v = beta2 * v + (1 - beta2) * grad^2
		vScaled, err := av.v.Mul(beta2Tensor)
		if err != nil {
			return fmt.Errorf("scale second moment: %w", err)
		}
		gradSqr, err := grad.Mul(grad)
		if err != nil {
			return fmt.Errorf("square gradient: %w", err)
		}
		gradSqrScaled, err := gradSqr.Mul(beta2CompTensor)
		if err != nil {
			return fmt.Errorf("scale squared gradient: %w", err)
		}
		nextV, err := vScaled.BroadcastAdd(gradSqrScaled)
		if err != nil {
			return fmt.Errorf("update second moment: %w", err)
		}

		// Bias-corrected moments
		scaleMTensor := tensor.Full[T](scaleM, spark.NewShape(), theta.Device())
		scaleVTensor := tensor.Full[T](scaleV, spark.NewShape(), theta.Device())

		mHat, err := nextM.Mul(scaleMTensor)
		if err != nil {
			return fmt.Errorf("bias-correct first moment: %w", err)
		}
		vHat, err := nextV.Mul(scaleVTensor)
		if err != nil {
			return fmt.Errorf("bias-correct second moment: %w", err)
		}

		// Weight decay: theta = theta * (1 - lr * lambda)
		decayFactor := tensor.Full[T](1.0-p.LearningRate*p.WeightDecay, spark.NewShape(), theta.Device())
		thetaScaled, err := theta.Mul(decayFactor)
		if err != nil {
			return fmt.Errorf("apply weight decay: %w", err)
		}

		// Compute adjusted gradient: mHat / (sqrt(vHat) + eps)
		vHatSqrt, err := vHat.Sqrt()
		if err != nil {
			return fmt.Errorf("sqrt second moment: %w", err)
		}
		epsTensor := tensor.Full[T](p.Epsilon, spark.NewShape(), theta.Device())
		vHatSqrtEps, err := vHatSqrt.BroadcastAdd(epsTensor)
		if err != nil {
			return fmt.Errorf("add epsilon: %w", err)
		}
		adjGrad, err := mHat.Div(vHatSqrtEps)
		if err != nil {
			return fmt.Errorf("compute adjusted gradient: %w", err)
		}

		// Update: theta = theta - lr * adjGrad
		lrTensor := tensor.Full[T](p.LearningRate, spark.NewShape(), theta.Device())
		lrAdjGrad, err := adjGrad.Mul(lrTensor)
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

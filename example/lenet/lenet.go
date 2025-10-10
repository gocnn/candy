package main

import (
	"fmt"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/nn"
	"github.com/gocnn/spark/tensor"
)

// LeNet defines the LeNet-5 architecture.
type LeNet[T spark.D] struct {
	c1 *nn.Conv2d[T] // Conv1: 1 -> 6 channels
	c2 *nn.Conv2d[T] // Conv2: 6 -> 16 channels
	f1 *nn.Linear[T] // FC1: 400 -> 120
	f2 *nn.Linear[T] // FC2: 120 -> 84
	f3 *nn.Linear[T] // FC3: 84 -> 10
}

// NewLeNet creates a new LeNet-5 model.
func NewLeNet[T spark.D](device spark.Device) *LeNet[T] {
	return &LeNet[T]{
		c1: nn.NewConv2d[T](1, 6, 5, 1, 2, device),
		c2: nn.NewConv2d[T](6, 16, 5, 1, 0, device),
		f1: nn.NewLinearLayer[T](400, 120, true, device),
		f2: nn.NewLinearLayer[T](120, 84, true, device),
		f3: nn.NewLinearLayer[T](84, 10, true, device),
	}
}

// Forward performs a forward pass through LeNet.
func (net *LeNet[T]) Forward(x *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	r, err := net.c1.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("failed conv1: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("failed relu1: %w", err)
	}
	r, err = r.MaxPool2d(2, 2, 2, 2)
	if err != nil {
		return nil, fmt.Errorf("failed pool1: %w", err)
	}
	r, err = net.c2.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("failed conv2: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("failed relu2: %w", err)
	}
	r, err = r.MaxPool2d(2, 2, 2, 2)
	if err != nil {
		return nil, fmt.Errorf("failed pool2: %w", err)
	}
	r, err = r.Reshape(r.Dim(0), -1)
	if err != nil {
		return nil, fmt.Errorf("failed flatten: %w", err)
	}
	r, err = net.f1.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("failed fc1: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("failed relu3: %w", err)
	}
	r, err = net.f2.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("failed fc2: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("failed relu4: %w", err)
	}
	r, err = net.f3.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("failed fc3: %w", err)
	}
	return r, nil
}

// MustForward performs a forward pass through LeNet.
func (net *LeNet[T]) MustForward(x *tensor.Tensor[T]) *tensor.Tensor[T] {
	r, err := net.Forward(x)
	if err != nil {
		panic(err)
	}
	return r
}

// Parameters returns all trainable parameters
func (net *LeNet[T]) Parameters() []*tensor.Tensor[T] {
	var params []*tensor.Tensor[T]
	params = append(params, net.c1.Parameters()...)
	params = append(params, net.c2.Parameters()...)
	params = append(params, net.f1.Weight(), net.f1.Bias())
	params = append(params, net.f2.Weight(), net.f2.Bias())
	params = append(params, net.f3.Weight(), net.f3.Bias())
	return params
}

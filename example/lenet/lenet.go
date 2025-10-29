package main

import (
	"fmt"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/nn"
	"github.com/gocnn/candy/tensor"
)

// LeNet defines the LeNet-5 architecture.
type LeNet[T candy.D] struct {
	c1 *nn.Conv2d[T] // Conv1: 1 -> 6 channels
	c2 *nn.Conv2d[T] // Conv2: 6 -> 16 channels
	f1 *nn.Linear[T] // FC1: 400 -> 120
	f2 *nn.Linear[T] // FC2: 120 -> 84
	f3 *nn.Linear[T] // FC3: 84 -> 10
}

// NewLeNet creates a new LeNet-5 model.
func NewLeNet[T candy.D](device candy.Device) *LeNet[T] {
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

func (net *LeNet[T]) Save(path string) error {
	items := map[string]*tensor.Tensor[T]{}
	p := net.c1.Parameters()
	items["c1_w"], items["c1_b"] = p[0], p[1]
	p = net.c2.Parameters()
	items["c2_w"], items["c2_b"] = p[0], p[1]
	items["f1_w"], items["f1_b"] = net.f1.Weight(), net.f1.Bias()
	items["f2_w"], items["f2_b"] = net.f2.Weight(), net.f2.Bias()
	items["f3_w"], items["f3_b"] = net.f3.Weight(), net.f3.Bias()
	return tensor.WriteNPZ(path, items)
}

func (net *LeNet[T]) Load(path string) error {
	names := []string{"c1_w", "c1_b", "c2_w", "c2_b", "f1_w", "f1_b", "f2_w", "f2_b", "f3_w", "f3_b"}
	arrs, err := tensor.ReadNPZByName[T](path, names)
	if err != nil {
		return err
	}
	p := net.c1.Parameters()
	p[0].SetStorage(arrs[0].Storage())
	p[1].SetStorage(arrs[1].Storage())
	p = net.c2.Parameters()
	p[0].SetStorage(arrs[2].Storage())
	p[1].SetStorage(arrs[3].Storage())
	net.f1.Weight().SetStorage(arrs[4].Storage())
	net.f1.Bias().SetStorage(arrs[5].Storage())
	net.f2.Weight().SetStorage(arrs[6].Storage())
	net.f2.Bias().SetStorage(arrs[7].Storage())
	net.f3.Weight().SetStorage(arrs[8].Storage())
	net.f3.Bias().SetStorage(arrs[9].Storage())
	return nil
}

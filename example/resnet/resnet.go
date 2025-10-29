package main

import (
	"fmt"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/nn"
	"github.com/gocnn/candy/tensor"
)

type Bottleneck[T candy.D] struct {
	conv1    *nn.Conv2d[T]
	bn1      *nn.BatchNorm2d[T]
	conv2    *nn.Conv2d[T]
	bn2      *nn.BatchNorm2d[T]
	conv3    *nn.Conv2d[T]
	bn3      *nn.BatchNorm2d[T]
	downConv *nn.Conv2d[T]
	downBn   *nn.BatchNorm2d[T]
}

func NewBottleneck[T candy.D](inplanes, planes, stride int, device candy.Device) *Bottleneck[T] {
	b := &Bottleneck[T]{
		conv1: nn.NewConv2d[T](inplanes, planes, 1, 1, 0, device),
		bn1:   nn.NewBatchNorm2d[T](planes, device),
		conv2: nn.NewConv2d[T](planes, planes, 3, stride, 1, device),
		bn2:   nn.NewBatchNorm2d[T](planes, device),
		conv3: nn.NewConv2d[T](planes, planes*4, 1, 1, 0, device),
		bn3:   nn.NewBatchNorm2d[T](planes*4, device),
	}
	if stride != 1 || inplanes != planes*4 {
		b.downConv = nn.NewConv2d[T](inplanes, planes*4, 1, stride, 0, device)
		b.downBn = nn.NewBatchNorm2d[T](planes*4, device)
	}
	return b
}

func (b *Bottleneck[T]) Train() {
	b.bn1.Train()
	b.bn2.Train()
	b.bn3.Train()
	if b.downBn != nil {
		b.downBn.Train()
	}
}

func (b *Bottleneck[T]) Eval() {
	b.bn1.Eval()
	b.bn2.Eval()
	b.bn3.Eval()
	if b.downBn != nil {
		b.downBn.Eval()
	}
}

func (b *Bottleneck[T]) Forward(x *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	id := x
	y, err := b.conv1.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("resnet50: bottleneck conv1: %w", err)
	}
	y, err = b.bn1.Forward(y)
	if err != nil {
		return nil, fmt.Errorf("resnet50: bottleneck bn1: %w", err)
	}
	y, err = y.Relu()
	if err != nil {
		return nil, fmt.Errorf("resnet50: bottleneck relu1: %w", err)
	}
	y, err = b.conv2.Forward(y)
	if err != nil {
		return nil, fmt.Errorf("resnet50: bottleneck conv2: %w", err)
	}
	y, err = b.bn2.Forward(y)
	if err != nil {
		return nil, fmt.Errorf("resnet50: bottleneck bn2: %w", err)
	}
	y, err = y.Relu()
	if err != nil {
		return nil, fmt.Errorf("resnet50: bottleneck relu2: %w", err)
	}
	y, err = b.conv3.Forward(y)
	if err != nil {
		return nil, fmt.Errorf("resnet50: bottleneck conv3: %w", err)
	}
	y, err = b.bn3.Forward(y)
	if err != nil {
		return nil, fmt.Errorf("resnet50: bottleneck bn3: %w", err)
	}
	if b.downConv != nil {
		id, err = b.downConv.Forward(x)
		if err != nil {
			return nil, fmt.Errorf("resnet50: downsample conv: %w", err)
		}
		id, err = b.downBn.Forward(id)
		if err != nil {
			return nil, fmt.Errorf("resnet50: downsample bn: %w", err)
		}
	}
	y, err = y.Add(id)
	if err != nil {
		return nil, fmt.Errorf("resnet50: add: %w", err)
	}
	y, err = y.Relu()
	if err != nil {
		return nil, fmt.Errorf("resnet50: relu3: %w", err)
	}
	return y, nil
}

func (b *Bottleneck[T]) MustForward(x *tensor.Tensor[T]) *tensor.Tensor[T] {
	r, err := b.Forward(x)
	if err != nil {
		panic(err)
	}
	return r
}

func (b *Bottleneck[T]) Parameters() []*tensor.Tensor[T] {
	var ps []*tensor.Tensor[T]
	ps = append(ps, b.conv1.Parameters()...)
	ps = append(ps, b.bn1.Parameters()...)
	ps = append(ps, b.conv2.Parameters()...)
	ps = append(ps, b.bn2.Parameters()...)
	ps = append(ps, b.conv3.Parameters()...)
	ps = append(ps, b.bn3.Parameters()...)
	if b.downConv != nil {
		ps = append(ps, b.downConv.Parameters()...)
		ps = append(ps, b.downBn.Parameters()...)
	}
	return ps
}

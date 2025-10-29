package main

import (
	"fmt"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/nn"
	"github.com/gocnn/candy/tensor"
)

// AlexNet defines the AlexNet architecture (ImageNet variant).
// Dropout is applied only in training mode.
type AlexNet[T candy.D] struct {
	// features
	c1 *nn.Conv2d[T] // 3 -> 64,  k=11, s=4, p=2
	c2 *nn.Conv2d[T] // 64 -> 192, k=5,  s=1, p=2
	c3 *nn.Conv2d[T] // 192 -> 384, k=3, s=1, p=1
	c4 *nn.Conv2d[T] // 384 -> 256, k=3, s=1, p=1
	c5 *nn.Conv2d[T] // 256 -> 256, k=3, s=1, p=1

	// classifier
	f1 *nn.Linear[T] // 9216 -> 4096
	f2 *nn.Linear[T] // 4096 -> 4096
	f3 *nn.Linear[T] // 4096 -> numClasses

	train bool // if true, enable dropout in classifier
}

// NewAlexNet creates a new AlexNet model. Input is expected to be NCHW with H=W=224 for 6x6 after last pool.
func NewAlexNet[T candy.D](numClasses int, device candy.Device) *AlexNet[T] {
	if numClasses <= 0 {
		numClasses = 1000
	}
	return &AlexNet[T]{
		c1:    nn.NewConv2d[T](3, 64, 11, 4, 2, device),
		c2:    nn.NewConv2d[T](64, 192, 5, 1, 2, device),
		c3:    nn.NewConv2d[T](192, 384, 3, 1, 1, device),
		c4:    nn.NewConv2d[T](384, 256, 3, 1, 1, device),
		c5:    nn.NewConv2d[T](256, 256, 3, 1, 1, device),
		f1:    nn.NewLinearLayer[T](256*6*6, 4096, true, device),
		f2:    nn.NewLinearLayer[T](4096, 4096, true, device),
		f3:    nn.NewLinearLayer[T](4096, numClasses, true, device),
		train: true,
	}
}

// Train sets model to training mode (enables dropout).
func (m *AlexNet[T]) Train() { m.train = true }

// Eval sets model to evaluation mode (disables dropout).
func (m *AlexNet[T]) Eval() { m.train = false }

// Forward performs a forward pass through AlexNet.
func (m *AlexNet[T]) Forward(x *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	// Feature extractor
	r, err := m.c1.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("alexnet: conv1: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("alexnet: relu1: %w", err)
	}
	r, err = r.MaxPool2d(3, 3, 2, 2)
	if err != nil {
		return nil, fmt.Errorf("alexnet: pool1: %w", err)
	}

	r, err = m.c2.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("alexnet: conv2: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("alexnet: relu2: %w", err)
	}
	r, err = r.MaxPool2d(3, 3, 2, 2)
	if err != nil {
		return nil, fmt.Errorf("alexnet: pool2: %w", err)
	}

	r, err = m.c3.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("alexnet: conv3: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("alexnet: relu3: %w", err)
	}

	r, err = m.c4.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("alexnet: conv4: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("alexnet: relu4: %w", err)
	}

	r, err = m.c5.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("alexnet: conv5: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("alexnet: relu5: %w", err)
	}
	r, err = r.MaxPool2d(3, 3, 2, 2)
	if err != nil {
		return nil, fmt.Errorf("alexnet: pool5: %w", err)
	}

	// Ensure 6x6 via AdaptiveAvgPool2d
	r, err = r.AdaptiveAvgPool2d(6, 6)
	if err != nil {
		return nil, fmt.Errorf("alexnet: adaptive avgpool: %w", err)
	}

	// Expect 6x6 after adaptive avgpool
	if r.Dim(2) != 6 || r.Dim(3) != 6 || r.Dim(1) != 256 {
		return nil, fmt.Errorf("alexnet: expected feature map [N,256,6,6], got [N,%d,%d,%d]", r.Dim(1), r.Dim(2), r.Dim(3))
	}

	// Classifier
	r, err = r.Reshape(r.Dim(0), -1) // N x 9216
	if err != nil {
		return nil, fmt.Errorf("alexnet: flatten: %w", err)
	}
	if m.train {
		r, err = r.Dropout(0.5)
		if err != nil {
			return nil, fmt.Errorf("alexnet: dropout1: %w", err)
		}
	}

	r, err = m.f1.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("alexnet: fc1: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("alexnet: relu_fc1: %w", err)
	}
	if m.train {
		r, err = r.Dropout(0.5)
		if err != nil {
			return nil, fmt.Errorf("alexnet: dropout2: %w", err)
		}
	}

	r, err = m.f2.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("alexnet: fc2: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("alexnet: relu_fc2: %w", err)
	}

	r, err = m.f3.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("alexnet: fc3: %w", err)
	}
	return r, nil // logits
}

// MustForward performs a forward pass through AlexNet, panicking on error.
func (m *AlexNet[T]) MustForward(x *tensor.Tensor[T]) *tensor.Tensor[T] {
	r, err := m.Forward(x)
	if err != nil {
		panic(err)
	}
	return r
}

// Parameters returns all trainable parameters in a deterministic order.
func (m *AlexNet[T]) Parameters() []*tensor.Tensor[T] {
	var ps []*tensor.Tensor[T]
	ps = append(ps, m.c1.Parameters()...)
	ps = append(ps, m.c2.Parameters()...)
	ps = append(ps, m.c3.Parameters()...)
	ps = append(ps, m.c4.Parameters()...)
	ps = append(ps, m.c5.Parameters()...)
	ps = append(ps, m.f1.Weight(), m.f1.Bias())
	ps = append(ps, m.f2.Weight(), m.f2.Bias())
	ps = append(ps, m.f3.Weight(), m.f3.Bias())
	return ps
}

// Save writes model weights to an NPZ archive.
func (m *AlexNet[T]) Save(path string) error {
	items := map[string]*tensor.Tensor[T]{}
	p := m.c1.Parameters()
	items["c1_w"], items["c1_b"] = p[0], p[1]
	p = m.c2.Parameters()
	items["c2_w"], items["c2_b"] = p[0], p[1]
	p = m.c3.Parameters()
	items["c3_w"], items["c3_b"] = p[0], p[1]
	p = m.c4.Parameters()
	items["c4_w"], items["c4_b"] = p[0], p[1]
	p = m.c5.Parameters()
	items["c5_w"], items["c5_b"] = p[0], p[1]
	items["f1_w"], items["f1_b"] = m.f1.Weight(), m.f1.Bias()
	items["f2_w"], items["f2_b"] = m.f2.Weight(), m.f2.Bias()
	items["f3_w"], items["f3_b"] = m.f3.Weight(), m.f3.Bias()
	return tensor.WriteNPZ(path, items)
}

// Load reads model weights from an NPZ archive written by Save.
func (m *AlexNet[T]) Load(path string) error {
	names := []string{
		"c1_w", "c1_b", "c2_w", "c2_b", "c3_w", "c3_b",
		"c4_w", "c4_b", "c5_w", "c5_b", "f1_w", "f1_b",
		"f2_w", "f2_b", "f3_w", "f3_b",
	}
	arrs, err := tensor.ReadNPZByName[T](path, names)
	if err != nil {
		return err
	}
	p := m.c1.Parameters()
	p[0].SetStorage(arrs[0].Storage())
	p[1].SetStorage(arrs[1].Storage())
	p = m.c2.Parameters()
	p[0].SetStorage(arrs[2].Storage())
	p[1].SetStorage(arrs[3].Storage())
	p = m.c3.Parameters()
	p[0].SetStorage(arrs[4].Storage())
	p[1].SetStorage(arrs[5].Storage())
	p = m.c4.Parameters()
	p[0].SetStorage(arrs[6].Storage())
	p[1].SetStorage(arrs[7].Storage())
	p = m.c5.Parameters()
	p[0].SetStorage(arrs[8].Storage())
	p[1].SetStorage(arrs[9].Storage())
	m.f1.Weight().SetStorage(arrs[10].Storage())
	m.f1.Bias().SetStorage(arrs[11].Storage())
	m.f2.Weight().SetStorage(arrs[12].Storage())
	m.f2.Bias().SetStorage(arrs[13].Storage())
	m.f3.Weight().SetStorage(arrs[14].Storage())
	m.f3.Bias().SetStorage(arrs[15].Storage())
	return nil
}

package main

import (
	"fmt"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/nn"
	"github.com/gocnn/candy/tensor"
)

type ResNet50[T candy.D] struct {
	conv1  *nn.Conv2d[T]
	bn1    *nn.BatchNorm2d[T]
	layer1 []*Bottleneck[T]
	layer2 []*Bottleneck[T]
	layer3 []*Bottleneck[T]
	layer4 []*Bottleneck[T]
	fc     *nn.Linear[T]
	train  bool
}

func NewLayer[T candy.D](inplanes *int, planes, blocks, stride int, device candy.Device) []*Bottleneck[T] {
	out := make([]*Bottleneck[T], blocks)
	out[0] = NewBottleneck[T](*inplanes, planes, stride, device)
	*inplanes = planes * 4
	for i := 1; i < blocks; i++ {
		out[i] = NewBottleneck[T](*inplanes, planes, 1, device)
	}
	return out
}

func NewResNet50[T candy.D](numClasses int, device candy.Device) *ResNet50[T] {
	m := &ResNet50[T]{}
	m.conv1 = nn.NewConv2d[T](3, 64, 7, 2, 3, device)
	m.bn1 = nn.NewBatchNorm2d[T](64, device)
	inplanes := 64
	m.layer1 = NewLayer[T](&inplanes, 64, 3, 1, device)
	m.layer2 = NewLayer[T](&inplanes, 128, 4, 2, device)
	m.layer3 = NewLayer[T](&inplanes, 256, 6, 2, device)
	m.layer4 = NewLayer[T](&inplanes, 512, 3, 2, device)
	m.fc = nn.NewLinearLayer[T](512*4, numClasses, true, device)
	m.train = true
	return m
}

func (m *ResNet50[T]) Train() {
	m.train = true
	m.bn1.Train()
	for _, b := range m.layer1 {
		b.Train()
	}
	for _, b := range m.layer2 {
		b.Train()
	}
	for _, b := range m.layer3 {
		b.Train()
	}
	for _, b := range m.layer4 {
		b.Train()
	}
}

func (m *ResNet50[T]) Eval() {
	m.train = false
	m.bn1.Eval()
	for _, b := range m.layer1 {
		b.Eval()
	}
	for _, b := range m.layer2 {
		b.Eval()
	}
	for _, b := range m.layer3 {
		b.Eval()
	}
	for _, b := range m.layer4 {
		b.Eval()
	}
}

func (m *ResNet50[T]) Forward(x *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	r, err := m.conv1.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("resnet50: conv1: %w", err)
	}
	r, err = m.bn1.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("resnet50: bn1: %w", err)
	}
	r, err = r.Relu()
	if err != nil {
		return nil, fmt.Errorf("resnet50: relu: %w", err)
	}
	r, err = r.MaxPool2d(3, 3, 2, 2)
	if err != nil {
		return nil, fmt.Errorf("resnet50: maxpool: %w", err)
	}
	for _, b := range m.layer1 {
		r, err = b.Forward(r)
		if err != nil {
			return nil, err
		}
	}
	for _, b := range m.layer2 {
		r, err = b.Forward(r)
		if err != nil {
			return nil, err
		}
	}
	for _, b := range m.layer3 {
		r, err = b.Forward(r)
		if err != nil {
			return nil, err
		}
	}
	for _, b := range m.layer4 {
		r, err = b.Forward(r)
		if err != nil {
			return nil, err
		}
	}
	r, err = r.AdaptiveAvgPool2d(1, 1)
	if err != nil {
		return nil, fmt.Errorf("resnet50: adaptive avgpool: %w", err)
	}
	r, err = r.Reshape(r.Dim(0), -1)
	if err != nil {
		return nil, fmt.Errorf("resnet50: flatten: %w", err)
	}
	r, err = m.fc.Forward(r)
	if err != nil {
		return nil, fmt.Errorf("resnet50: fc: %w", err)
	}
	return r, nil
}

func (m *ResNet50[T]) MustForward(x *tensor.Tensor[T]) *tensor.Tensor[T] {
	r, err := m.Forward(x)
	if err != nil {
		panic(err)
	}
	return r
}

func (m *ResNet50[T]) Parameters() []*tensor.Tensor[T] {
	var ps []*tensor.Tensor[T]
	ps = append(ps, m.conv1.Parameters()...)
	ps = append(ps, m.bn1.Parameters()...)
	for _, b := range m.layer1 {
		ps = append(ps, b.Parameters()...)
	}
	for _, b := range m.layer2 {
		ps = append(ps, b.Parameters()...)
	}
	for _, b := range m.layer3 {
		ps = append(ps, b.Parameters()...)
	}
	for _, b := range m.layer4 {
		ps = append(ps, b.Parameters()...)
	}
	ps = append(ps, m.fc.Weight())
	if m.fc.Bias() != nil {
		ps = append(ps, m.fc.Bias())
	}
	return ps
}

func (m *ResNet50[T]) Load(path string) error {
	arrays, err := tensor.ReadNPZ[T](path)
	if err != nil {
		return err
	}
	setConv := func(c *nn.Conv2d[T], key string) error {
		t, ok := arrays[key]
		if !ok {
			return fmt.Errorf("missing %s", key)
		}
		p := c.Parameters()
		p[0].SetStorage(t.Storage()) // weight only; conv bias not used in torchvision resnet
		return nil
	}
	setBN := func(bn *nn.BatchNorm2d[T], base string) error {
		w, ok := arrays[base+"_w"]
		if !ok {
			return fmt.Errorf("missing %s_w", base)
		}
		b, ok := arrays[base+"_b"]
		if !ok {
			return fmt.Errorf("missing %s_b", base)
		}
		rm, ok := arrays[base+"_rm"]
		if !ok {
			return fmt.Errorf("missing %s_rm", base)
		}
		rv, ok := arrays[base+"_rv"]
		if !ok {
			return fmt.Errorf("missing %s_rv", base)
		}
		bn.Weight().SetStorage(w.Storage())
		bn.Bias().SetStorage(b.Storage())
		bn.RunningMean().SetStorage(rm.Storage())
		bn.RunningVar().SetStorage(rv.Storage())
		return nil
	}
	if err := setConv(m.conv1, "conv1_w"); err != nil {
		return err
	}
	if err := setBN(m.bn1, "bn1"); err != nil {
		return err
	}
	layers := []struct {
		name   string
		blocks []*Bottleneck[T]
	}{
		{"layer1", m.layer1},
		{"layer2", m.layer2},
		{"layer3", m.layer3},
		{"layer4", m.layer4},
	}
	for _, l := range layers {
		for i, b := range l.blocks {
			prefix := fmt.Sprintf("%s_%d", l.name, i)
			if err := setConv(b.conv1, prefix+"_conv1_w"); err != nil {
				return err
			}
			if err := setBN(b.bn1, prefix+"_bn1"); err != nil {
				return err
			}
			if err := setConv(b.conv2, prefix+"_conv2_w"); err != nil {
				return err
			}
			if err := setBN(b.bn2, prefix+"_bn2"); err != nil {
				return err
			}
			if err := setConv(b.conv3, prefix+"_conv3_w"); err != nil {
				return err
			}
			if err := setBN(b.bn3, prefix+"_bn3"); err != nil {
				return err
			}
			if b.downConv != nil {
				if err := setConv(b.downConv, prefix+"_down_conv_w"); err != nil {
					return err
				}
				if err := setBN(b.downBn, prefix+"_down_bn"); err != nil {
					return err
				}
			}
		}
	}
	fw, ok := arrays["fc_w"]
	if !ok {
		return fmt.Errorf("missing fc_w")
	}
	fb, ok := arrays["fc_b"]
	if !ok {
		return fmt.Errorf("missing fc_b")
	}
	m.fc.Weight().SetStorage(fw.Storage())
	m.fc.Bias().SetStorage(fb.Storage())
	return nil
}

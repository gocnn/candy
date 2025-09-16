package cpu

import "github.com/qntx/goml"

func (d *Tensor[T]) AccGrad(grad goml.Tensor[T]) error {
	d.gradMu.Lock()
	defer d.gradMu.Unlock()

	if d.grad == nil {
		d.grad = grad.Clone().(*Tensor[T])
		return nil
	}

	_, err := d.grad.Add(grad)
	if err != nil {
		return err
	}
	return nil
}

package nn

import (
	"github.com/qntx/spark/internal/mat"
	"github.com/qntx/spark/tensor"
)

// Accuracy returns the accuracy of the prediction.
// The return values cannot be backpropagated.
func Accuracy(y, t *tensor.Variable) *tensor.Variable {
	argmax := mat.New(f64(mat.Argmax(y.Data)))
	pred := mat.Reshape(mat.Shape(t.Data), argmax)
	result := mat.F2(pred, t.Data, tensor.IsClose)
	return tensor.New(mat.Mean(result))
}

func f64(x []int) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v)
	}

	return out
}

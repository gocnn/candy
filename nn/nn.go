package nn

import "github.com/qntx/spark/tensor"

var (
	AddC        = tensor.AddC
	Add         = tensor.Add
	SubC        = tensor.SubC
	Sub         = tensor.Sub
	MulC        = tensor.MulC
	Mul         = tensor.Mul
	DivC        = tensor.DivC
	Div         = tensor.Div
	Sin         = tensor.Sin
	Cos         = tensor.Cos
	Tanh        = tensor.Tanh
	Exp         = tensor.Exp
	Log         = tensor.Log
	Pow         = tensor.Pow
	Square      = tensor.Square
	Neg         = tensor.Neg
	Sum         = tensor.Sum
	SumTo       = tensor.SumTo
	BroadcastTo = tensor.BroadcastTo
	Reshape     = tensor.Reshape
	Transpose   = tensor.Transpose
	MatMul      = tensor.MatMul
	Max         = tensor.Max
	Min         = tensor.Min
	Clip        = tensor.Clip
	GetItem     = tensor.GetItem
)

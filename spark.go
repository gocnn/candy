package spark

import (
	"github.com/qntx/spark/ag"
	"github.com/qntx/spark/nn"
)

var (
	New      = ag.New
	NewFrom  = ag.NewFrom
	NewOf    = ag.NewOf
	ZeroLike = ag.ZeroLike
	OneLike  = ag.OneLike
	Zero     = ag.Zero
	Rand     = ag.Rand
	Randn    = ag.Randn
)

var (
	AddC        = ag.AddC
	Add         = ag.Add
	SubC        = ag.SubC
	Sub         = ag.Sub
	MulC        = ag.MulC
	Mul         = ag.Mul
	DivC        = ag.DivC
	Div         = ag.Div
	Sin         = ag.Sin
	Cos         = ag.Cos
	Exp         = ag.Exp
	Log         = ag.Log
	Pow         = ag.Pow
	Square      = ag.Square
	Neg         = ag.Neg
	Sum         = ag.Sum
	SumTo       = ag.SumTo
	BroadcastTo = ag.BroadcastTo
	Reshape     = ag.Reshape
	Transpose   = ag.Transpose
	MatMul      = ag.MatMul
	Max         = ag.Max
	Min         = ag.Min
	Clip        = ag.Clip
	GetItem     = ag.GetItem
)

var (
	Accuracy = nn.Accuracy
	Linear   = nn.Linear
	ReLU     = nn.ReLU
	Sigmoid  = nn.Sigmoid
	Softmax  = nn.Softmax
	Tanh     = nn.Tanh
)

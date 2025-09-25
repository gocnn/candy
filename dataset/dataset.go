package dataset

import (
	"github.com/gocnn/spark/dataset/cifar10"
	"github.com/gocnn/spark/dataset/mnist"
)

var (
	CIFAR10 = cifar10.New
	MNIST   = mnist.New
)

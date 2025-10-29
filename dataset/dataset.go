package dataset

import (
	"iter"

	"github.com/gocnn/candy"
)

type Dataset[T candy.D] interface {
	Len() int
	Get(i int) ([]T, uint8)
	GetRaw(i int) ([]uint8, uint8)
}

type DataLoader[T any, D any] interface {
	All() iter.Seq2[T, D]
	Reset()
	Len() int
	BatchSize() int
	IsShuffled() bool
}

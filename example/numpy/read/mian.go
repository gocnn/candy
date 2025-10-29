package main

import (
	"fmt"

	"github.com/gocnn/candy/tensor"
)

func main() {
	a := tensor.MustReadNPY[float32]("f32.npy")
	fmt.Printf("%v\n", a)

	b := tensor.MustReadNPY[float64]("f64.npy")
	fmt.Printf("%v\n", b)

	c := tensor.MustReadNPY[uint8]("u8.npy")
	fmt.Printf("%v\n", c)

	d := tensor.MustReadNPY[uint32]("u32.npy")
	fmt.Printf("%v\n", d)

	e := tensor.MustReadNPY[int64]("i64.npy")
	fmt.Printf("%v\n", e)

	z := tensor.MustReadNPZ("pack.npz")
	for name, t := range z {
		fmt.Printf("%s:\n%v\n", name, t)
	}
}

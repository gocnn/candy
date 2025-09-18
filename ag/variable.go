package ag

import (
	"fmt"
	"math/rand/v2"
	"sort"

	"github.com/qntx/spark/internal/mat"
)

type Variable struct {
	Name string

	Data *mat.Matrix
	Grad *Variable

	Creator    *Operator
	Generation int
}

func New(v ...float64) *Variable {
	return &Variable{Data: mat.New(v)}
}

func NewOf(v ...[]float64) *Variable {
	return &Variable{Data: mat.New(v...)}
}

func NewFrom(v *mat.Matrix) *Variable {
	return &Variable{Data: v}
}

func ZeroLike(v *Variable) *Variable {
	return &Variable{Data: mat.ZeroLike(v.Data)}
}

func OneLike(v *Variable) *Variable {
	return &Variable{Data: mat.OneLike(v.Data)}
}

func Zero(shape []int) *Variable {
	return &Variable{Data: mat.Zero(shape[0], shape[1])}
}

func Rand(shape []int, s ...rand.Source) *Variable {
	return &Variable{Data: mat.Rand(shape[0], shape[1], s...)}
}

func Randn(shape []int, s ...rand.Source) *Variable {
	return &Variable{Data: mat.Randn(shape[0], shape[1], s...)}
}

func (v *Variable) At(coord ...int) float64 {
	return v.Data.At(coord[0], coord[1])
}

func (v *Variable) Shape() []int {
	return mat.Shape(v.Data)
}

func (v *Variable) Cleargrad() {
	v.Grad = nil
}

func (v *Variable) SetCreator(f *Operator) {
	v.Creator = f
	v.Generation = f.Generation + 1
}

func (v *Variable) Unchain() {
	v.Creator = nil
}

func (v *Variable) UnchainBackward() {
	if v.Creator == nil {
		return
	}

	fs := append(make([]*Operator, 0), v.Creator)

	for {
		if len(fs) == 0 {
			break
		}

		// pop
		f := fs[len(fs)-1]
		fs = fs[:len(fs)-1]

		// unchain
		for _, x := range f.Input {
			if x.Creator == nil {
				continue
			}

			fs = append(fs, x.Creator)
			x.Unchain()
		}
	}
}

func (v *Variable) Backward(opts ...Opts) {
	if v.Grad == nil {
		v.Grad = OneLike(v)
	}

	if v.Creator == nil {
		return
	}

	seen := make(map[*Operator]bool)
	fs := addFunc(make([]*Operator, 0), v.Creator, seen)

	for {
		if len(fs) == 0 {
			break
		}

		// pop
		f := fs[len(fs)-1]
		fs = fs[:len(fs)-1]

		// gys
		gys := gys(f.Output)

		// backward
		func() {
			if NoCreateGraph(opts...) {
				defer Nograd().End()
			}

			// backward
			gxs := f.Backward(gys...)

			// chain
			xs, gxs := Zip(f.Input, gxs)
			for i, x := range xs {
				x.Grad = AddGrad(x.Grad, gxs[i])

				if x.Creator != nil {
					fs = addFunc(fs, x.Creator, seen)
				}
			}
		}()

		if NoRetainGrad(opts...) {
			cleargrad(f.Output)
		}
	}
}

func (v *Variable) String() string {
	name := "variable"
	if v.Name != "" {
		name = v.Name
	}

	if v.Data.Rows == 1 && v.Data.Cols == 1 {
		return fmt.Sprintf("%s(%v)", name, v.Data.Data[0])
	}

	if v.Data.Rows == 1 {
		return fmt.Sprintf("%s%v(%v)", name, v.Shape(), v.Data.Row(0))
	}

	return fmt.Sprintf("%s%v(%v)", name, v.Shape(), v.Data)
}

func Zip(xs, gxs []*Variable) ([]*Variable, []*Variable) {
	n := min(len(xs), len(gxs))
	return xs[:n], gxs[:n]
}

func AddGrad(xgrad, gx *Variable) *Variable {
	if xgrad == nil {
		return gx
	}

	// NOTE: create graph
	return Add(xgrad, gx)
}

func gys(y []*Variable) []*Variable {
	gys := make([]*Variable, len(y))
	for i := range y {
		gys[i] = y[i].Grad
	}

	return gys
}

func addFunc(fs []*Operator, f *Operator, seen map[*Operator]bool) []*Operator {
	if _, ok := seen[f]; ok {
		return fs
	}

	seen[f] = true
	fs = append(fs, f)
	sort.Slice(fs, func(i, j int) bool { return fs[i].Generation < fs[j].Generation })
	return fs
}

func cleargrad(output []*Variable) {
	for _, y := range output {
		y.Cleargrad()
	}
}

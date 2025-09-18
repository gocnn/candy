package mat_test

import (
	"fmt"
	"math/rand/v2"

	"github.com/qntx/spark/internal/mat"
)

func ExampleZero() {
	for _, r := range mat.Zero(2, 3).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0 0 0]
	// [0 0 0]
}

func ExampleZeroLike() {
	A := mat.New(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	for _, r := range mat.ZeroLike(A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0 0 0]
	// [0 0 0]
}

func ExampleOneLike() {
	A := mat.Zero(2, 3)
	for _, r := range mat.OneLike(A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [1 1 1]
	// [1 1 1]
}

func ExampleRand() {
	shape := mat.Shape(mat.Rand(2, 3))
	fmt.Println(shape)

	// Output:
	// [2 3]
}

func ExampleRand_seed() {
	s := rand.NewPCG(0, 0)
	for _, r := range mat.Rand(2, 3, s).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0.9999275824802834 0.8856419373528862 0.38147752771154886]
	// [0.4812673234167829 0.44417259544314847 0.5210016660132573]
}

func ExampleRand_nil() {
	shape := mat.Shape(mat.Rand(2, 3, nil))
	fmt.Println(shape)

	// Output:
	// [2 3]
}

func ExampleRandn() {
	shape := mat.Shape(mat.Randn(2, 3))
	fmt.Println(shape)

	// Output:
	// [2 3]
}

func ExampleRandn_seed() {
	s := rand.NewPCG(0, 0)
	for _, r := range mat.Randn(2, 3, s).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0.5665360716030388 -0.6123972949371448 0.5898947122637695]
	// [-0.3678242340302933 1.0919575041640825 -0.4438344619606553]
}

func ExampleSize() {
	A := mat.Zero(2, 3)
	fmt.Println(mat.Size(A))

	// Output:
	// 6
}

func ExampleShape() {
	A := mat.Zero(2, 3)
	fmt.Println(mat.Shape(A))

	// Output:
	// [2 3]
}

func ExampleDim() {
	A := mat.Zero(2, 3)
	fmt.Println(mat.Dim(A))

	// Output:
	// 2 3
}

func ExampleAddC() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.AddC(1, A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [2 3]
	// [4 5]
}

func ExampleSubC() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.SubC(1, A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0 -1]
	// [-2 -3]
}

func ExampleMulC() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.MulC(2, A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [2 4]
	// [6 8]
}

func ExampleExp() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.Exp(A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [2.718281828459045 7.38905609893065]
	// [20.085536923187668 54.598150033144236]
}

func ExampleLog() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.Log(A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0 0.6931471805599453]
	// [1.0986122886681096 1.3862943611198906]
}

func ExampleSin() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.Sin(A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0.8414709848078965 0.9092974268256816]
	// [0.1411200080598672 -0.7568024953079282]
}

func ExampleCos() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.Cos(A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0.5403023058681398 -0.4161468365471424]
	// [-0.9899924966004454 -0.6536436208636119]
}

func ExampleTanh() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.Tanh(A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0.7615941559557649 0.9640275800758169]
	// [0.9950547536867305 0.999329299739067]
}

func ExamplePow() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.Pow(2, A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [1 4]
	// [9 16]
}

func ExampleAdd() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := mat.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range mat.Add(A, B).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [6 8]
	// [10 12]
}

func ExampleSub() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := mat.New(
		[]float64{10, 20},
		[]float64{30, 40},
	)

	for _, r := range mat.Sub(A, B).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [-9 -18]
	// [-27 -36]
}

func ExampleMul() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)
	B := mat.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range mat.Mul(A, B).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [5 12]
	// [21 32]
}

func ExampleDiv() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)
	B := mat.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range mat.Div(A, B).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0.2 0.3333333333333333]
	// [0.42857142857142855 0.5]
}

func ExampleMatMul() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := mat.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range mat.MatMul(A, B).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [19 22]
	// [43 50]
}

func ExampleMaxAxis1() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{-3, -4},
		[]float64{5, 2},
	)

	for _, r := range mat.MaxAxis1(A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [2]
	// [-3]
	// [5]
}

func ExampleMean() {
	A := mat.New(
		[]float64{1, 2, 3, 4, 5},
		[]float64{6, 7, 8, 9, 10},
	)

	fmt.Println(mat.Mean(A))

	// Output:
	// 5.5
}

func ExampleMax() {
	A := mat.New(
		[]float64{1, 5},
		[]float64{3, 4},
	)

	fmt.Println(mat.Max(A))

	// Output:
	// 5
}

func ExampleMin() {
	A := mat.New(
		[]float64{1, -5},
		[]float64{3, 4},
	)

	fmt.Println(mat.Min(A))

	// Output:
	// -5
}

func ExampleArgmax() {
	A := mat.New(
		[]float64{1, 2, 3},
		[]float64{4, 6, 5},
		[]float64{9, 8, 7},
	)

	fmt.Println(mat.Argmax(A))

	// Output:
	// [2 1 0]
}

func ExampleClip() {
	A := mat.New(
		[]float64{-3, -2, -1, 0, 1, 2, 3, 4, 5, 6},
		[]float64{7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
	)

	for _, r := range mat.Clip(A, 0, 10).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0 0 0 0 1 2 3 4 5 6]
	// [7 8 9 10 10 10 10 10 10 10]
}

func ExampleMask() {
	A := mat.New(
		[]float64{-1, 2},
		[]float64{3, -4},
	)

	for _, r := range mat.Mask(A, func(v float64) bool { return v > 0 }).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [0 1]
	// [1 0]
}

func ExampleBroadcastTo() {
	A := mat.New([]float64{1})

	for _, r := range mat.BroadcastTo([]int{3, 5}, A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [1 1 1 1 1]
	// [1 1 1 1 1]
	// [1 1 1 1 1]
}

func ExampleBroadcastTo_row() {
	A := mat.New([]float64{1, 2})

	for _, r := range mat.BroadcastTo([]int{5, -1}, A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [1 2]
	// [1 2]
	// [1 2]
	// [1 2]
	// [1 2]
}

func ExampleBroadcastTo_column() {
	A := mat.New(
		[]float64{1},
		[]float64{2},
	)

	for _, r := range mat.BroadcastTo([]int{-1, 5}, A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [1 1 1 1 1]
	// [2 2 2 2 2]
}

func ExampleBroadcastTo_noEffect() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.BroadcastTo([]int{2, 2}, A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [1 2]
	// [3 4]
}

func ExampleBroadcast() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)
	B := mat.New(
		[]float64{1, 2},
	)

	AA, BB := mat.Broadcast(A, B)
	for _, r := range AA.Seq2() {
		fmt.Println(r)
	}
	for _, r := range BB.Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [1 2]
	// [3 4]
	// [1 2]
	// [1 2]
}

func ExampleSum() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(mat.Sum(A))

	// Output:
	// 10
}

func ExampleSumTo_sum() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(mat.SumTo([]int{1, 1}, A))

	// Output:
	// [[10]]
}

func ExampleSumTo_axis0() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(mat.SumTo([]int{1, 2}, A))

	// Output:
	// [[4 6]]
}

func ExampleSumTo_axis1() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(mat.SumTo([]int{2, 1}, A))

	// Output:
	// [[3] [7]]
}

func ExampleSumTo_noeffect() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(mat.SumTo([]int{2, 3}, A))

	// Output:
	// [[1 2] [3 4]]
}

func ExampleSumAxis0() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.SumAxis0(A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [4 6]
}

func ExampleSumAxis1() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range mat.SumAxis1(A).Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [3]
	// [7]
}

func ExampleReshape() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(mat.Reshape([]int{1, 4}, A))
	fmt.Println(mat.Reshape([]int{4, 1}, A))
	fmt.Println(mat.Reshape([]int{2, 2}, A))
	fmt.Println()

	fmt.Println(mat.Reshape([]int{1, -1}, A))
	fmt.Println(mat.Reshape([]int{4, -1}, A))
	fmt.Println(mat.Reshape([]int{2, -1}, A))
	fmt.Println()

	fmt.Println(mat.Reshape([]int{-1, 1}, A))
	fmt.Println(mat.Reshape([]int{-1, 4}, A))
	fmt.Println(mat.Reshape([]int{-1, 2}, A))
	fmt.Println()

	// Output:
	// [[1 2 3 4]]
	// [[1] [2] [3] [4]]
	// [[1 2] [3 4]]
	//
	// [[1 2 3 4]]
	// [[1] [2] [3] [4]]
	// [[1 2] [3 4]]
	//
	// [[1] [2] [3] [4]]
	// [[1 2 3 4]]
	// [[1 2] [3 4]]
}

func ExampleMatrix_Seq2() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
		[]float64{5, 6},
	)

	for i, r := range A.Seq2() {
		fmt.Println(r)

		if i == 1 {
			break
		}
	}

	// Output:
	// [1 2]
	// [3 4]
}

func ExampleMatrix_N() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(A.N())

	// Output:
	// 2
}

func ExampleMatrix_AddAt() {
	A := mat.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	A.AddAt(0, 1, 10)
	A.AddAt(1, 0, -5)

	for _, r := range A.Seq2() {
		fmt.Println(r)
	}

	// Output:
	// [1 12]
	// [-2 4]
}

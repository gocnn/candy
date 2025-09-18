package dot_test

import (
	"fmt"
	"regexp"

	"github.com/qntx/spark/ag"
	"github.com/qntx/spark/dot"
)

var re = regexp.MustCompile(`0x[0-9a-fA-F]+`)

const dummyAddr = "0x**********"

func ExampleVar() {
	x := ag.New(1)
	x.Name = "x"

	fmt.Println(re.ReplaceAllString(dot.Var(x), dummyAddr))
	fmt.Println(re.ReplaceAllString(dot.Var(x, dot.Opts{Verbose: true}), dummyAddr))

	// It always returns the same value.
	xvar0, xvar1 := dot.Var(x), dot.Var(x)
	fmt.Println(xvar0 == xvar1)

	y := ag.New(1)
	y.Name = "x"
	fmt.Println(dot.Var(x) == dot.Var(y))

	// Output:
	// "0x**********" [label="x", color=orange, style=filled]
	// "0x**********" [label="x(1)", color=orange, style=filled]
	// true
	// false
}

func ExampleFunc() {
	f0 := &ag.Operator{Op: &ag.SinT{}}
	for _, txt := range dot.Func(f0) {
		fmt.Println(re.ReplaceAllString(txt, dummyAddr))
	}

	f1 := &ag.Operator{Op: &ag.SinT{}}
	fmt.Println(dot.Func(f0)[0] == dot.Func(f1)[0])

	// Output:
	// "0x**********" [label="Sin", color=lightblue, style=filled, shape=box]
	// false
}

func Example_func() {
	f := &ag.Operator{
		Input:  []*ag.Variable{ag.New(1)},
		Output: []*ag.Variable{ag.New(1)},
		Op:     &ag.SinT{},
	}

	for _, txt := range dot.Func(f) {
		fmt.Println(re.ReplaceAllString(txt, dummyAddr))
	}

	// Output:
	// "0x**********" [label="Sin", color=lightblue, style=filled, shape=box]
	// "0x**********" -> "0x**********"
	// "0x**********" -> "0x**********"
}

func ExampleGraph() {
	x := ag.New(1.0)
	x.Name = "x"

	y := ag.Sin(x)
	y.Name = "y"

	for _, txt := range dot.Graph(y) {
		fmt.Println(re.ReplaceAllString(txt, dummyAddr))
	}

	// Output:
	// digraph g {
	// "0x**********" [label="y", color=orange, style=filled]
	// "0x**********" [label="Sin", color=lightblue, style=filled, shape=box]
	// "0x**********" -> "0x**********"
	// "0x**********" -> "0x**********"
	// "0x**********" [label="x", color=orange, style=filled]
	// }
}

func ExampleGraph_composite() {
	x := ag.New(1.0)
	y := ag.Sin(x)
	z := ag.Cos(y)
	x.Name = "x"
	y.Name = "y"
	z.Name = "z"

	for _, txt := range dot.Graph(z) {
		fmt.Println(re.ReplaceAllString(txt, dummyAddr))
	}

	// Output:
	// digraph g {
	// "0x**********" [label="z", color=orange, style=filled]
	// "0x**********" [label="Cos", color=lightblue, style=filled, shape=box]
	// "0x**********" -> "0x**********"
	// "0x**********" -> "0x**********"
	// "0x**********" [label="y", color=orange, style=filled]
	// "0x**********" [label="Sin", color=lightblue, style=filled, shape=box]
	// "0x**********" -> "0x**********"
	// "0x**********" -> "0x**********"
	// "0x**********" [label="x", color=orange, style=filled]
	// }
}

func ExampleAddFunc() {
	fs := make([]*ag.Operator, 0)
	seen := make(map[*ag.Operator]bool)

	sin := &ag.Operator{Op: &ag.SinT{}}
	cos := &ag.Operator{Op: &ag.CosT{}}
	fs = dot.AddFunc(fs, sin, seen)
	fs = dot.AddFunc(fs, cos, seen)
	fmt.Println(fs)

	fs = dot.AddFunc(fs, sin, seen)
	fs = dot.AddFunc(fs, cos, seen)
	fmt.Println(fs)

	// Output:
	// [*ag.SinT[] *ag.CosT[]]
	// [*ag.SinT[] *ag.CosT[]]
}

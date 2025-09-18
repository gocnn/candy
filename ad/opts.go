package ad

type Opts struct {
	RetainGrad  bool
	CreateGraph bool
}

func NoRetainGrad(opts ...Opts) bool {
	return !HasRetainGrad(opts...)
}

func HasRetainGrad(opts ...Opts) bool {
	if len(opts) > 0 && opts[0].RetainGrad {
		return true
	}

	return false
}

func NoCreateGraph(opts ...Opts) bool {
	return !HasCreateGraph(opts...)
}

func HasCreateGraph(opts ...Opts) bool {
	if len(opts) > 0 && opts[0].CreateGraph {
		return true
	}

	return false
}

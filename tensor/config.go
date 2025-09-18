package tensor

type config struct {
	EnableBackprop bool
	Train          bool
}

var Config = config{
	EnableBackprop: true,
	Train:          true,
}

type Span struct {
	End func()
}

func Nograd() *Span {
	Config.EnableBackprop = false
	return &Span{
		End: func() {
			Config.EnableBackprop = true
		},
	}
}

func TestMode() *Span {
	Config.Train = false
	return &Span{
		End: func() {
			Config.Train = true
		},
	}
}

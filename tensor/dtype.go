package tensor

type DType interface {
	Name() string
	Size() int
	Kind() TypeKind

	IsFloat() bool
	IsInt() bool
	IsUnsigned() bool
	IsBool() bool
	IsComplex() bool

	CanCastTo(DType) bool
	String() string
}

type TypeKind uint8

const (
	Invalid TypeKind = iota
	Bool
	Int8
	Int16
	Int32
	Int64
	Uint8
	Uint16
	Uint32
	Uint64
	Float16
	Float32
	Float64
	Complex64
	Complex128
)

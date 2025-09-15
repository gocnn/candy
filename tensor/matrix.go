package tensor

import (
	"encoding/gob"
)

// The Matrix interface defines set and get methods to access its elements,
// plus a few variants to perform linear algebra operations with other matrices,
// such as element-wise addition, subtraction, product and matrix-matrix
// multiplication.
type Matrix interface {
	Tensor

	// SetData sets the content of the matrix, copying the given raw
	// data representation as one-dimensional slice.
	SetData(data Slice)
	// ZerosLike returns a new matrix with the same dimensions of the
	// receiver, initialized with zeroes.
	ZerosLike() Matrix
	// OnesLike returns a new matrix with the same dimensions of the
	// receiver, initialized with ones.
	OnesLike() Matrix
	// Item returns the scalar value.
	// It panics if the matrix does not contain exactly one element.
	Item() Float
	// Zeros sets all the values of the matrix to zero.
	Zeros()
	// SetScalar sets the value at the given indices.
	// It panics if the given indices are out of range.
	SetScalar(v Float, indices ...int)
	// ScalarAt returns the value at the given indices.
	// It panics if the given indices are out of range.
	ScalarAt(indices ...int) Float
	// ExtractRow returns a copy of the i-th row of the matrix,
	// as a row vector (1×cols).
	ExtractRow(i int) Matrix
	// ExtractColumn returns a copy of the i-th column of the matrix,
	// as a column vector (rows×1).
	ExtractColumn(i int) Matrix
	// Slice returns a new matrix obtained by slicing the receiver across the
	// given positions. The parameters "fromRow" and "fromCol" are inclusive,
	// while "toRow" and "toCol" are exclusive.
	Slice(fromRow, fromCol, toRow, toCol int) Matrix
	// Reshape returns a copy of the matrix.
	// It panics if the dimensions are incompatible.
	Reshape(shape ...int) Matrix
	// ReshapeInPlace changes the dimensions of the matrix in place and returns the
	// matrix itself.
	// It panics if the dimensions are incompatible.
	ReshapeInPlace(shape ...int) Matrix
	// Flatten creates a new row vector (1×size) corresponding to the
	// "flattened" row-major ordered representation of the initial matrix.
	Flatten() Matrix
	// FlattenInPlace transforms the matrix in place, changing its dimensions,
	// obtaining a row vector (1×size) containing the "flattened" row-major
	// ordered representation of the initial value.
	// It returns the matrix itself.
	FlattenInPlace() Matrix
	// ResizeVector returns a resized copy of the vector.
	//
	// If the new size is smaller than the input vector, the remaining tail
	// elements are removed. If it's bigger, the additional tail elements
	// are set to zero.
	ResizeVector(newSize int) Matrix
	// T returns the transpose of the matrix.
	T() Matrix
	// TransposeInPlace transposes the matrix in place, and returns the
	// matrix itself.
	TransposeInPlace() Matrix
	// Add returns the addition between the receiver and another matrix.
	Add(other Matrix) Matrix
	// AddInPlace performs the in-place addition with the other matrix.
	AddInPlace(other Matrix) Matrix
	// AddScalar performs the addition between the matrix and the given value.
	AddScalar(n float64) Matrix
	// AddScalarInPlace adds the scalar to all values of the matrix.
	AddScalarInPlace(n float64) Matrix
	// Sub returns the subtraction of the other matrix from the receiver.
	Sub(other Matrix) Matrix
	// SubInPlace performs the in-place subtraction with the other matrix.
	SubInPlace(other Matrix) Matrix
	// SubScalar performs a subtraction between the matrix and the given value.
	SubScalar(n float64) Matrix
	// SubScalarInPlace subtracts the scalar from the receiver's values.
	SubScalarInPlace(n float64) Matrix
	// Prod performs the element-wise product between the receiver and the other matrix.
	Prod(other Matrix) Matrix
	// ProdInPlace performs the in-place element-wise product with the other matrix.
	ProdInPlace(other Matrix) Matrix
	// ProdScalar returns the multiplication between the matrix and the given value.
	ProdScalar(n float64) Matrix
	// ProdScalarInPlace performs the in-place multiplication between the
	// matrix and the given value.
	ProdScalarInPlace(n float64) Matrix
	// ProdMatrixScalarInPlace multiplies the given matrix with the value,
	// storing the result in the receiver.
	ProdMatrixScalarInPlace(m Matrix, n float64) Matrix
	// Div returns the result of the element-wise division of the receiver by the other matrix.
	Div(other Matrix) Matrix
	// DivInPlace performs the in-place element-wise division of the receiver by the other matrix.
	DivInPlace(other Matrix) Matrix
	// Mul performs the multiplication row by column.
	// If A is an i×j Matrix, and B is j×k, then the resulting Matrix
	// C = AB will be i×k.
	Mul(other Matrix) Matrix
	// MulT performs the matrix multiplication row by column.
	// ATB = C, where AT is the transpose of A
	// if A is an r x c Matrix, and B is j x k, r = j the resulting
	// Matrix C will be c x k.
	MulT(other Matrix) Matrix
	// DotUnitary returns the dot product of two vectors as a scalar Matrix.
	DotUnitary(other Matrix) Matrix
	// ClipInPlace clips in place each value of the matrix.
	ClipInPlace(min, max float64) Matrix
	// Maximum returns a new matrix containing the element-wise maxima.
	Maximum(other Matrix) Matrix
	// Minimum returns a new matrix containing the element-wise minima.
	Minimum(other Matrix) Matrix
	// Abs returns a new matrix applying the absolute value function to all elements.
	Abs() Matrix
	// Pow returns a new matrix, applying the power function with given exponent
	// to all elements of the matrix.
	Pow(power float64) Matrix
	// Sqrt returns a new matrix applying the square root function to all elements.
	Sqrt() Matrix
	// Log returns a new matrix applying the natural logarithm function to each element.
	Log() Matrix
	// Exp returns a new matrix applying the base-e exponential function to each element.
	Exp() Matrix
	// Sigmoid returns a new matrix applying the sigmoid function to each element.
	Sigmoid() Matrix
	// Sum returns the sum of all values of the matrix as a scalar Matrix.
	Sum() Matrix
	// Max returns the maximum value of the matrix as a scalar Matrix.
	Max() Matrix
	// Min returns the minimum value of the matrix as a scalar Matrix.
	Min() Matrix
	// ArgMax returns the index of the vector's element with the maximum value.
	ArgMax() int
	// Softmax applies the softmax function to the vector, returning the
	// result as a new column vector.
	Softmax() Matrix
	// CumSum computes the cumulative sum of the vector's elements, returning
	// the result as a new column vector.
	CumSum() Matrix
	// Range creates a new vector initialized with data extracted from the
	// matrix raw data, from start (inclusive) to end (exclusive).
	Range(start, end int) Matrix
	// SplitV splits the vector in N chunks of given sizes,
	// so that N[i] has size sizes[i].
	SplitV(sizes ...int) []Matrix
	// Augment places the identity matrix at the end of the original matrix.
	Augment() Matrix
	// SwapInPlace swaps two rows of the matrix in place.
	SwapInPlace(r1, r2 int) Matrix
	// PadRows returns a copy of the matrix with n additional tail rows.
	// The additional elements are set to zero.
	PadRows(n int) Matrix
	// PadColumns returns a copy of the matrix with n additional tail columns.
	// The additional elements are set to zero.
	PadColumns(n int) Matrix
	// AppendRows returns a copy of the matrix with len(vs) additional tail rows,
	// being each new row filled with the values of each given vector.
	// It accepts row or column vectors indifferently, virtually treating all of
	// them as row vectors.
	AppendRows(vs ...Matrix) Matrix
	// Norm returns the vector's norm. Use pow = 2.0 to compute the Euclidean norm.
	// The result is a scalar Matrix.
	Norm(pow float64) Matrix
	// Normalize2 normalizes an array with the Euclidean norm.
	Normalize2() Matrix
	// Apply creates a new matrix executing the unary function fn.
	Apply(fn func(r, c int, v float64) float64) Matrix
	// ApplyInPlace executes the unary function fn over the matrix a,
	// and stores the result in the receiver, returning the receiver itself.
	ApplyInPlace(fn func(r, c int, v float64) float64, a Matrix) Matrix
	// ApplyWithAlpha creates a new matrix executing the unary function fn,
	// taking additional parameters alpha.
	ApplyWithAlpha(fn func(r, c int, v float64, alpha ...float64) float64, alpha ...float64) Matrix
	// ApplyWithAlphaInPlace executes the unary function fn over the matrix a,
	// taking additional parameters alpha, and stores the result in the
	// receiver, returning the receiver itself.
	ApplyWithAlphaInPlace(fn func(r, c int, v float64, alpha ...float64) float64, a Matrix, alpha ...float64) Matrix
	// DoNonZero calls a function for each non-zero element of the matrix.
	// The parameters of the function are the element's indices and value.
	DoNonZero(fn func(r, c int, v float64))
	// DoVecNonZero calls a function for each non-zero element of the vector.
	// The parameters of the function are the element's index and value.
	DoVecNonZero(fn func(i int, v float64))
	// Clone returns a new matrix, copying all its values from the receiver.
	Clone() Matrix
	// Copy copies the data from the other matrix to the receiver.
	Copy(other Matrix)
	// String returns a string representation of the matrix.
	String() string

	// NewMatrix creates a new matrix, of the same type of the receiver, of
	// size rows×cols, initialized with a copy of raw data.
	//
	// Rows and columns MUST not be negative, and the length of data MUST be
	// equal to rows*cols, otherwise the method panics.
	// NewMatrix(opts ...OptionsFunc) Matrix
	// NewScalar creates a new 1×1 matrix, of the same type of the receiver,
	// containing the given value.
	// NewScalar(v float64, opts ...OptionsFunc) Matrix

	// NewConcatV creates a new column vector, of the same type of the receiver,
	// concatenating two or more vectors "vertically"
	// It accepts row or column vectors indifferently, virtually
	// treating all of them as column vectors.
	NewConcatV(vs ...Matrix) Matrix
	// NewStack creates a new matrix, of the same type of the receiver, stacking
	// two or more vectors of the same size on top of each other; the result is
	// a new matrix where each row contains the data of each input vector.
	// It accepts row or column vectors indifferently, virtually treating all of
	// them as row vectors.
	NewStack(vs ...Matrix) Matrix

	SetRequiresGrad(bool)
}

func init() {
	gob.Register([]Matrix{})
}

package spark

import (
	"errors"
	"fmt"
	"slices"
	"strings"
)

// Shape represents the dimensions of a tensor.
type Shape struct {
	dims []int
}

// NewShape creates a new Shape from the given dimensions.
func NewShape(dims ...int) *Shape {
	return &Shape{dims: slices.Clone(dims)}
}

// NewShapeFrom creates a new Shape from a slice of dimensions.
func NewShapeFrom(dims []int) *Shape {
	return &Shape{dims: slices.Clone(dims)}
}

// Clone returns a deep copy of the Shape.
func (s *Shape) Clone() *Shape {
	return &Shape{slices.Clone(s.dims)}
}

// Equal checks if two shapes are equal.
func (s *Shape) Equal(other *Shape) bool {
	return slices.Equal(s.dims, other.dims)
}

// IsScalar returns true if the shape represents a scalar (0 dimensions).
func (s *Shape) IsScalar() bool {
	return len(s.dims) == 0
}

// IsVector returns true if the shape represents a vector (1 dimension).
func (s *Shape) IsVector() bool {
	return len(s.dims) == 1
}

// IsMatrix returns true if the shape represents a matrix (2 dimensions).
func (s *Shape) IsMatrix() bool {
	return len(s.dims) == 2
}

// String returns a string representation of the shape.
func (s *Shape) String() string {
	if len(s.dims) == 0 {
		return "[]"
	}
	var sb strings.Builder
	sb.WriteString("[")
	for i, d := range s.dims {
		if i > 0 {
			sb.WriteString(" ")
		}
		sb.WriteString(fmt.Sprint(d))
	}
	sb.WriteString("]")
	return sb.String()
}

// Rank returns the number of dimensions (rank) of the shape.
func (s *Shape) Rank() int {
	return len(s.dims)
}

// Dims returns a copy of the dimensions slice.
func (s *Shape) Dims() []int {
	return slices.Clone(s.dims)
}

// Dim returns the size of the dimension at the given index.
// Negative indices count from the end (-1 is the last dimension).
func (s *Shape) Dim(dim int) int {
	if dim < 0 {
		dim += s.Rank()
	}
	if dim < 0 || dim >= s.Rank() {
		panic(fmt.Sprintf("shape: dimension %d out of range [0, %d)", dim, s.Rank()))
	}
	return s.dims[dim]
}

// ElemCount returns the total number of elements (product of all dimensions).
func (s *Shape) ElemCount() int {
	if len(s.dims) == 0 {
		return 1 // Scalar has 1 element.
	}
	prod := 1
	for _, d := range s.dims {
		if d == 0 {
			return 0
		}
		prod *= d
	}
	return prod
}

// StrideContiguous returns the strides for a contiguous (row-major) tensor with this shape.
func (s *Shape) StrideContiguous() []int {
	if len(s.dims) == 0 {
		return []int{}
	}
	strides := make([]int, len(s.dims))
	prod := 1
	for i := len(s.dims) - 1; i >= 0; i-- {
		strides[i] = prod
		prod *= s.dims[i]
	}
	return strides
}

// IsContiguous checks if the given strides are C-contiguous (row-major).
func (s *Shape) IsContiguous(strides []int) bool {
	if len(s.dims) != len(strides) {
		return false
	}
	acc := 1
	for i := len(s.dims) - 1; i >= 0; i-- {
		if s.dims[i] > 1 && strides[i] != acc {
			return false
		}
		acc *= s.dims[i]
	}
	return true
}

// IsFortranContiguous checks if the given strides are Fortran-contiguous (column-major).
func (s *Shape) IsFortranContiguous(strides []int) bool {
	if len(s.dims) != len(strides) {
		return false
	}
	acc := 1
	for i := range s.dims {
		if s.dims[i] > 1 && strides[i] != acc {
			return false
		}
		acc *= s.dims[i]
	}
	return true
}

// Extend returns a new Shape with additional dimensions appended.
func (s *Shape) Extend(add ...int) *Shape {
	dims := append(slices.Clone(s.dims), add...)
	return NewShapeFrom(dims)
}

// BroadcastShapeBinaryOp computes the broadcasted shape for binary operations.
//
// Broadcasting rules (NumPy-compatible):
// - Align shapes from the rightmost dimension
// - Dimensions are compatible if they are equal or one of them is 1
// - Missing dimensions are treated as 1
//
// Examples:
//
//	[3, 1, 4] + [2, 4] -> [3, 2, 4]  (missing dim treated as 1)
//	[5, 1, 3] * [1, 4, 1] -> [5, 4, 3]  (1s broadcast to larger dims)
//	[3, 4] + [2, 5] -> panic (incompatible: 4≠5 and neither is 1)
func (s *Shape) BroadcastShapeBinaryOp(rhs *Shape) (*Shape, error) {
	lhsDims := s.dims
	rhsDims := rhs.dims
	lhsN := len(lhsDims)
	rhsN := len(rhsDims)
	maxN := max(lhsN, rhsN)
	bcastDims := make([]int, maxN)

	for i := range maxN {
		l := 1
		if i < lhsN {
			l = lhsDims[lhsN-1-i]
		}
		r := 1
		if i < rhsN {
			r = rhsDims[rhsN-1-i]
		}
		var b int
		if l == r {
			b = l
		} else if l == 1 {
			b = r
		} else if r == 1 {
			b = l
		} else {
			return nil, fmt.Errorf("shape mismatch in binary op: lhs %v, rhs %v", s, rhs)
		}
		bcastDims[maxN-1-i] = b
	}
	return NewShapeFrom(bcastDims), nil
}

// BroadcastShapeMatmul returns the broadcasted shapes for matrix multiplication.
// It broadcasts the batch dimensions and checks the inner dimensions for compatibility.
func (s *Shape) BroadcastShapeMatmul(rhs *Shape) (*Shape, *Shape, error) {
	lhsDims := s.dims
	rhsDims := rhs.dims
	if len(lhsDims) < 2 || len(rhsDims) < 2 {
		return nil, nil, errors.New("matmul requires at least 2D shapes")
	}
	m := lhsDims[len(lhsDims)-2]
	lhsK := lhsDims[len(lhsDims)-1]
	rhsK := rhsDims[len(rhsDims)-2]
	n := rhsDims[len(rhsDims)-1]
	if lhsK != rhsK {
		return nil, nil, fmt.Errorf("inner dimensions mismatch in matmul: lhs %v, rhs %v", s, rhs)
	}

	lhsBatch := &Shape{lhsDims[:len(lhsDims)-2]}
	rhsBatch := &Shape{rhsDims[:len(rhsDims)-2]}
	bcastBatch, err := lhsBatch.BroadcastShapeBinaryOp(rhsBatch)
	if err != nil {
		return nil, nil, err
	}

	bcastLhs := bcastBatch.Extend(m, lhsK)
	bcastRhs := bcastBatch.Extend(rhsK, n)
	return bcastLhs, bcastRhs, nil
}

// Reshape returns a new shape with the given dimensions, inferring one dimension if -1 is provided.
// The total element count must match.
func (s *Shape) Reshape(newDims ...int) (*Shape, error) {
	elCount := s.ElemCount()
	prod := 1
	holeIdx := -1
	resDims := slices.Clone(newDims)
	for i, d := range resDims {
		if d == -1 {
			if holeIdx != -1 {
				return nil, errors.New("multiple inference holes (-1) not allowed")
			}
			holeIdx = i
		} else if d <= 0 {
			return nil, fmt.Errorf("invalid dimension %d (must be positive or -1 for inference)", d)
		} else {
			prod *= d
		}
	}
	if holeIdx != -1 {
		if prod == 0 || elCount%prod != 0 {
			return nil, fmt.Errorf("cannot infer dimension: element count %d not divisible by product %d", elCount, prod)
		}
		resDims[holeIdx] = elCount / prod
	} else if prod != elCount {
		return nil, fmt.Errorf("element count mismatch: original %d, new %d", elCount, prod)
	}
	return NewShapeFrom(resDims), nil
}

// Dims0 checks if the shape has 0 dimensions (scalar).
func (s *Shape) Dims0() error {
	if s.Rank() != 0 {
		return fmt.Errorf("unexpected number of dims: expected 0, got %d, shape %v", s.Rank(), s)
	}
	return nil
}

// Dims1 extracts the single dimension from a 1D shape.
func (s *Shape) Dims1() (int, error) {
	if s.Rank() != 1 {
		return 0, fmt.Errorf("unexpected number of dims: expected 1, got %d, shape %v", s.Rank(), s)
	}
	return s.dims[0], nil
}

// Dims2 extracts the two dimensions from a 2D shape.
func (s *Shape) Dims2() (int, int, error) {
	if s.Rank() != 2 {
		return 0, 0, fmt.Errorf("unexpected number of dims: expected 2, got %d, shape %v", s.Rank(), s)
	}
	return s.dims[0], s.dims[1], nil
}

// Dims3 extracts the three dimensions from a 3D shape.
func (s *Shape) Dims3() (int, int, int, error) {
	if s.Rank() != 3 {
		return 0, 0, 0, fmt.Errorf("unexpected number of dims: expected 3, got %d, shape %v", s.Rank(), s)
	}
	return s.dims[0], s.dims[1], s.dims[2], nil
}

// Dims4 extracts the four dimensions from a 4D shape.
func (s *Shape) Dims4() (int, int, int, int, error) {
	if s.Rank() != 4 {
		return 0, 0, 0, 0, fmt.Errorf("unexpected number of dims: expected 4, got %d, shape %v", s.Rank(), s)
	}
	return s.dims[0], s.dims[1], s.dims[2], s.dims[3], nil
}

// Dims5 extracts the five dimensions from a 5D shape.
func (s *Shape) Dims5() (int, int, int, int, int, error) {
	if s.Rank() != 5 {
		return 0, 0, 0, 0, 0, fmt.Errorf("unexpected number of dims: expected 5, got %d, shape %v", s.Rank(), s)
	}
	return s.dims[0], s.dims[1], s.dims[2], s.dims[3], s.dims[4], nil
}

// ResolveAxis resolves a single axis index, supporting negative values.
func ResolveAxis(axis, rank int) (int, error) {
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return 0, fmt.Errorf("axis out of range: rank %d, axis %d", rank, axis)
	}
	return axis, nil
}

// ResolveAxes resolves a list of axis indices, supporting negative indices.
// It checks for duplicates and out-of-range values.
func ResolveAxes(axes []int, s *Shape) ([]int, error) {
	res := make([]int, len(axes))
	seen := make(map[int]bool)
	for i, ax := range axes {
		if ax < 0 {
			ax += s.Rank()
		}
		if ax < 0 || ax >= s.Rank() {
			return nil, fmt.Errorf("axis out of range: shape %v, axis %d", s, ax)
		}
		if seen[ax] {
			return nil, fmt.Errorf("duplicate axis: shape %v, axes %v", s, axes)
		}
		seen[ax] = true
		res[i] = ax
	}
	return res, nil
}

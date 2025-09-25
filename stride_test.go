package spark_test

import (
	"slices"
	"testing"

	"github.com/gocnn/spark"
)

func TestStrideNewStridedIndex(t *testing.T) {
	dims := []int{2, 3}
	stride := []int{3, 1}
	si := spark.NewStridedIndex(dims, stride, 5)
	if !slices.Equal(si.Dims(), dims) || !slices.Equal(si.Stride(), stride) || si.NextStorageIndex() == nil || *si.NextStorageIndex() != 5 {
		t.Errorf("NewStridedIndex = dims %v, stride %v, next %v; want %v, %v, 5", si.Dims(), si.Stride(), si.NextStorageIndex(), dims, stride)
	}
	if !slices.Equal(si.MultiIndex(), []int{0, 0}) {
		t.Errorf("NewStridedIndex multiIndex = %v; want [0 0]", si.MultiIndex())
	}
}

func TestStrideNewStridedIndexEmpty(t *testing.T) {
	si := spark.NewStridedIndex([]int{}, []int{}, 0)
	if si.NextStorageIndex() != nil {
		t.Errorf("NewStridedIndex empty dims: nextStorageIndex = %v; want nil", si.NextStorageIndex())
	}
}

func TestStrideNewStridedIndexPanicDimsStrideMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("NewStridedIndex did not panic on dims/stride mismatch")
		}
	}()
	spark.NewStridedIndex([]int{2, 3}, []int{1}, 0)
}

func TestStrideNewStridedIndexPanicNegativeDim(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("NewStridedIndex did not panic on negative dim")
		}
	}()
	spark.NewStridedIndex([]int{2, -1}, []int{1, 1}, 0)
}

func TestStrideNewStridedIndexFromLayout(t *testing.T) {
	l := spark.ContiguousWithOffset(spark.NewShape(2, 3), 5)
	si := spark.NewStridedIndexFromLayout(l)
	if !slices.Equal(si.Dims(), []int{2, 3}) || !slices.Equal(si.Stride(), []int{3, 1}) || si.NextStorageIndex() == nil || *si.NextStorageIndex() != 5 {
		t.Errorf("NewStridedIndexFromLayout = dims %v, stride %v, next %v; want [2 3], [3 1], 5", si.Dims(), si.Stride(), si.NextStorageIndex())
	}
}

func TestStrideStridedIndexAll(t *testing.T) {
	si := spark.NewStridedIndex([]int{2, 2}, []int{2, 1}, 0)
	var indices []int
	for idx := range si.All() {
		indices = append(indices, idx)
	}
	expected := []int{0, 1, 2, 3}
	if !slices.Equal(indices, expected) {
		t.Errorf("All() = %v; want %v", indices, expected)
	}
	if !si.IsComplete() {
		t.Errorf("All() did not set nextStorageIndex to nil")
	}
}

func TestStrideStridedIndexAllEmpty(t *testing.T) {
	si := spark.NewStridedIndex([]int{}, []int{}, 0)
	var count int
	for range si.All() {
		count++
	}
	if count != 0 {
		t.Errorf("All() empty dims yielded %d indices; want 0", count)
	}
}

func TestStrideStridedIndexAllSingleElement(t *testing.T) {
	si := spark.NewStridedIndex([]int{1}, []int{1}, 5)
	var indices []int
	for idx := range si.All() {
		indices = append(indices, idx)
	}
	if !slices.Equal(indices, []int{5}) {
		t.Errorf("All() = %v; want [5]", indices)
	}
}

func TestStrideStridedBlocksSingleBlock(t *testing.T) {
	l := spark.Contiguous(spark.NewShape(2, 3))
	sb := l.StridedBlocks()
	if sb.Type != spark.SingleBlock || sb.StartOffset != 0 || sb.Len != 6 || sb.BlockStartIndex != nil {
		t.Errorf("StridedBlocks = %v; want Type=SingleBlock, StartOffset=0, Len=6, BlockStartIndex=nil", sb)
	}
}

func TestStrideStridedBlocksMultipleBlocks(t *testing.T) {
	// Redesigned: Use shape [3,4], strides [5,1] to have non-matching outer stride (expected outer=4, but 5 !=4), so cont=1, Len=4, indexDims=1
	// Calculated starts: dims[3], strides [5], [0*5=0, 1*5=5, 2*5=10]
	l := spark.NewLayout(spark.NewShape(3, 4), []int{5, 1}, 0)
	sb := l.StridedBlocks()
	if sb.Type != spark.MultipleBlocks || sb.Len != 4 || sb.BlockStartIndex == nil {
		t.Errorf("StridedBlocks = Type %v, Len %d; want MultipleBlocks, 4", sb.Type, sb.Len)
	}
	var starts []int
	for idx := range sb.BlockStartIndex.All() {
		starts = append(starts, idx)
	}
	expected := []int{0, 5, 10}
	if !slices.Equal(starts, expected) {
		t.Errorf("BlockStartIndex.All() = %v; want %v", starts, expected)
	}
}

func TestStrideStridedBlocksScalar(t *testing.T) {
	l := spark.Contiguous(spark.NewShape())
	sb := l.StridedBlocks()
	if sb.Type != spark.SingleBlock || sb.StartOffset != 0 || sb.Len != 1 {
		t.Errorf("StridedBlocks scalar = %v; want Type=SingleBlock, StartOffset=0, Len=1", sb)
	}
}

func TestStrideStridedBlocksNonContiguous(t *testing.T) {
	l := spark.NewLayout(spark.NewShape(2, 3), []int{1, 2}, 0)
	sb := l.StridedBlocks()
	if sb.Type != spark.MultipleBlocks || sb.Len != 1 {
		t.Errorf("StridedBlocks non-contiguous = Type %v, Len %d; want MultipleBlocks, 1", sb.Type, sb.Len)
	}
	var starts []int
	for idx := range sb.BlockStartIndex.All() {
		starts = append(starts, idx)
	}
	expected := []int{0, 2, 4, 1, 3, 5}
	if !slices.Equal(starts, expected) {
		t.Errorf("BlockStartIndex.All() = %v; want %v", starts, expected)
	}
}

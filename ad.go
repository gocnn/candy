package spark

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/qntx/spark/op"
)

type Graph[T D] struct {
	nodes []op.Operator[T]
	roots []op.Operator[T]
}

func Backward[T D](roots ...Tensor[T]) error {
	// 1. 构建反向传播图
	nodes := buildSimpleBackwardGraph(roots)
	if len(nodes) == 0 {
		return nil
	}

	// 2. 为根节点分配初始梯度
	for _, node := range nodes {
		if isRootNode(node, roots) {
			if err := assignSimpleRootGradient(node); err != nil {
				return err
			}
		}
	}

	// 3. 极简并发执行 - 只用WaitGroup
	var wg sync.WaitGroup
	for _, node := range nodes {
		wg.Add(1)
		go executeSimpleNodeBackward(node, &wg)
	}
	wg.Wait()

	return nil
}

// buildSimpleBackwardGraph 构建反向传播图
func buildSimpleBackwardGraph[T D](roots []Tensor[T]) []*op.Operator[T] {
	var nodes []*op.Operator[T]
	visited := make(map[*op.Operator[T]]bool)

	// 深度优先遍历
	var dfs func(node Tensor[T])
	dfs = func(node Tensor[T]) {
		if op, ok := node.(*op.Operator[T]); ok && op.RequiresGrad() {
			if visited[op] {
				return
			}
			visited[op] = true

			// 初始化梯度计数
			op.gradCount = int32(len(op.Operands()))
			if op.gradCount > 0 {
				op.gradReady = make(chan struct{})
			}

			nodes = append(nodes, op)

			// 递归处理操作数
			for _, operand := range op.Operands() {
				dfs(operand)
			}
		}
	}

	// 从根节点开始构建
	for _, root := range roots {
		dfs(root)
	}

	return nodes
}

// isRootNode 检查是否为根节点
func isRootNode[T D](node *op.Operator[T], roots []Tensor[T]) bool {
	for _, root := range roots {
		if root == node {
			return true
		}
	}
	return false
}

// assignSimpleRootGradient 为根节点分配初始梯度
func assignSimpleRootGradient(root *op.Operator[T]) error {
	grad := root.value.Grad()
	if grad != nil && !isNil(grad) {
		atomic.AddInt32(&root.gradCount, -1)
		return nil
	}

	// 为标量自动分配梯度1
	if root.value.Size() == 1 {
		if matrix, ok := root.value.(mat.Matrix); ok {
			root.AccGrad(matrix.NewScalar(1.0))
			return nil
		}
	}

	return fmt.Errorf("missing gradient for root tensor")
}

// executeSimpleNodeBackward 执行单个节点的反向传播
func executeSimpleNodeBackward[T D](node *op.Operator[T], wg *sync.WaitGroup) {
	defer wg.Done()

	// 使用sync.Once确保只执行一次
	node.gradOnce.Do(func() {
		// 等待梯度准备完成
		if node.gradReady != nil {
			<-node.gradReady
		}

		// 获取梯度并执行反向传播
		grad := node.value.Grad()
		if grad == nil {
			return
		}

		if err := node.fn.Backward(grad); err != nil {
			// 简单的错误处理
			fmt.Printf("backward error: %v\n", err)
		}
	})
}

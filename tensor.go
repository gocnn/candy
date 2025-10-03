package spark

import (
	"sync/atomic"
)

// TensorId is a unique identifier for a tensor.
type TensorId uint64

// counter is a package-level atomic counter for generating unique TensorId values.
var counter uint64

// NewTensorId generates a new, unique TensorId in a thread-safe manner.
func NewTensorId() TensorId {
	return TensorId(atomic.AddUint64(&counter, 1))
}

type Op[T D] struct {
	inputs   []*Tensor[T]                                                           // 输入张量（泛型简化）
	backward func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) // 闭包：输入outputGrad和inputs，返回每个input的grad片段
}

type Tensor[T D] struct {
	id      TensorId
	storage BackendStorage[T]
	layout  *Layout
	op      *Op[T]
	isVar   bool
	dtype   DType
	device  Device
}

func NewTensor[T D](storage BackendStorage[T], layout *Layout, isVar bool, dtype DType, device Device) *Tensor[T] {
	return &Tensor[T]{id: NewTensorId(), storage: storage, layout: layout, isVar: isVar, dtype: dtype, device: device}
}

func (t *Tensor[T]) ZerosLike() (*Tensor[T], error) {
	return nil, nil
}

func (t *Tensor[T]) ID() TensorId {
	return t.id
}

func (t *Tensor[T]) SetStorage(storage BackendStorage[T]) {
	t.storage = storage
}

func (t *Tensor[T]) SetLayout(layout *Layout) {
	t.layout = layout
}

func (t *Tensor[T]) SetIsVar(isVar bool) {
	t.isVar = isVar
}

func (t *Tensor[T]) SetDType(dtype DType) {
	t.dtype = dtype
}

func (t *Tensor[T]) SetDevice(device Device) {
	t.device = device
}

func (a *Tensor[T]) Add(b *Tensor[T]) (*Tensor[T], error) {
	resultLayout := Contiguous(a.layout.Shape())
	resultStorage, err := a.storage.Add(b.storage, a.layout, b.layout, &resultLayout)
	if err != nil {
		return nil, err
	}
	result := &Tensor[T]{
		id:      NewTensorId(),
		storage: resultStorage,
		layout:  &resultLayout,
		dtype:   a.dtype,
		device:  a.device,
	}

	if a.isVar || b.isVar {
		result.isVar = true
		result.op = &Op[T]{
			inputs: []*Tensor[T]{a, b},
			backward: func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
				ga := outputGrad
				gb := outputGrad
				return []*Tensor[T]{ga, gb}, nil
			},
		}
	}
	return result, nil
}

func (a *Tensor[T]) Mul(b *Tensor[T]) (*Tensor[T], error) {
	resultLayout := Contiguous(a.layout.Shape())
	resultStorage, err := a.storage.Mul(b.storage, a.layout, b.layout, &resultLayout)
	if err != nil {
		return nil, err
	}
	result := &Tensor[T]{
		id:      NewTensorId(),
		storage: resultStorage,
		layout:  &resultLayout,
		dtype:   a.dtype,
		device:  a.device,
	}

	if a.isVar || b.isVar {
		result.isVar = true
		result.op = &Op[T]{
			inputs: []*Tensor[T]{a, b},
			backward: func(outputGrad *Tensor[T], inputs []*Tensor[T]) ([]*Tensor[T], error) {
				// Mul梯度：∂(a*b)/∂a = b, ∂(a*b)/∂b = a
				ga, err := outputGrad.Mul(inputs[1]) // outputGrad * b
				if err != nil {
					return nil, err
				}
				gb, err := outputGrad.Mul(inputs[0]) // outputGrad * a
				if err != nil {
					return nil, err
				}
				return []*Tensor[T]{ga, gb}, nil
			},
		}
	}
	return result, nil
}

// Backward computes gradients for all isVar tensors contributing to root.
// Assumes root.grad is set (e.g., to OnesLike if not).
func Backward[T D](root *Tensor[T], store *GradStore[T]) error {
	if !root.isVar {
		return nil // 无需backprop
	}

	// 步骤1: 收集所有相关var节点 (BFS backward遍历)
	allVars := make(map[TensorId]*Tensor[T])
	reverseGraph := make(map[TensorId][]*Tensor[T]) // input_id -> list of users (outputs)
	queue := []*Tensor[T]{root}
	visited := make(map[TensorId]bool)
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		id := curr.ID()
		if visited[id] {
			continue
		}
		visited[id] = true
		if !curr.isVar {
			continue
		}
		allVars[id] = curr
		if curr.op == nil {
			continue // leaf
		}
		for _, input := range curr.op.inputs {
			if input.isVar {
				reverseGraph[input.ID()] = append(reverseGraph[input.ID()], curr)
				queue = append(queue, input)
			}
		}
	}

	// 步骤2: 初始化pending (每个var的dependents数 = 被多少节点使用)
	pending := make(map[TensorId]int)
	for id := range allVars {
		pending[id] = len(reverseGraph[id])
	}

	// 步骤3: 设置root grad (if not set, assume user sets or default to ones)
	_, err := store.GetOrCreate(root) // 确保存在 (user应预设为ones if scalar loss)
	if err != nil {
		return err
	}

	// 步骤4: 队列起始于root (pending==0的sink)
	backQueue := []*Tensor[T]{root}
	for len(backQueue) > 0 {
		curr := backQueue[0]
		backQueue = backQueue[1:]

		// 计算当前节点输入张量各自的梯度
		currGrad := store.Get(curr)
		inputGrads, err := curr.op.backward(currGrad, curr.op.inputs)
		if err != nil {
			return err
		}

		// 分发梯度给输入节点
		for i, input := range curr.op.inputs {
			if !input.isVar {
				continue
			}

			iGrad := inputGrads[i]
			existingGrad := store.Get(input)

			// 梯度累积
			var newGrad *Tensor[T]
			if existingGrad == nil {
				newGrad = iGrad
			} else {
				newGrad, err = existingGrad.Add(iGrad) // 累加梯度
				if err != nil {
					return err
				}
			}

			store.Set(input, newGrad)
			pending[input.ID()]--

			// 当所有下游梯度都收集完毕时，加入处理队列
			if pending[input.ID()] == 0 {
				backQueue = append(backQueue, input)
			}
		}
	}
	return nil
}

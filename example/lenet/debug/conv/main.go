package main

import (
	"fmt"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/nn"
	"github.com/gocnn/spark/tensor"
)

func main() {
	// 加载实际测试数据
	x := tensor.MustReadNPY[float32]("real_input.npy")
	w := tensor.MustReadNPY[float32]("real_weight.npy")
	b := tensor.MustReadNPY[float32]("real_bias.npy")
	expected := tensor.MustReadNPY[float32]("real_output.npy")

	fmt.Printf("Input shape: %v\n", x.Dims())
	fmt.Printf("Weight shape: %v\n", w.Dims())
	fmt.Printf("Bias shape: %v\n", b.Dims())
	fmt.Printf("Expected shape: %v\n", expected.Dims())

	// 方法1: 手动卷积 + bias（原来的方法）
	params := &spark.Conv2DParams{
		Batch: 1, InH: 28, InW: 28,
		KH: 3, KW: 3,
		OutCh: 32, InCh: 1,
		Pad: 0, Stride: 1, Dilate: 1,
	}
	conv_result := x.MustConv2d(w, params)
	fmt.Printf("Conv only result first 5: %v\n", conv_result.Data()[:5])
	br := b.MustReshape(1, 32, 1, 1)
	manual_result := conv_result.MustBroadcastAdd(br)

	// 方法2: 使用 nn.Conv2d
	conv_layer := nn.NewConv2d[float32](1, 32, 3, 1, 0, spark.CPU)
	// 用真实权重和偏置替换随机初始化的参数
	conv_params := conv_layer.Parameters()
	copy(conv_params[0].Data(), w.Data()) // weight
	copy(conv_params[1].Data(), b.Data()) // bias
	nn_result := conv_layer.MustForward(x)

	fmt.Printf("\n=== 结果比较 ===\n")
	fmt.Printf("Expected first 5: %v\n", expected.Data()[:5])
	fmt.Printf("Manual result first 5: %v\n", manual_result.Data()[:5])
	fmt.Printf("NN result first 5: %v\n", nn_result.Data()[:5])

	// 验证两种方法是否一致
	manual_data := manual_result.Data()
	nn_data := nn_result.Data()
	max_diff := float32(0)
	for i := range manual_data {
		diff := manual_data[i] - nn_data[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > max_diff {
			max_diff = diff
		}
	}
	fmt.Printf("\nManual vs NN max diff: %v\n", max_diff)

	// 验证与期望值的差异
	expected_data := expected.Data()
	max_diff_expected := float32(0)
	for i := range expected_data {
		diff := expected_data[i] - nn_data[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > max_diff_expected {
			max_diff_expected = diff
		}
	}
	fmt.Printf("Expected vs NN max diff: %v\n", max_diff_expected)
}

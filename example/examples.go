package main

import (
	"fmt"

	"github.com/qntx/spark"
	"github.com/qntx/spark/cpu"
)

func main() {

	// 1. Scalar (0-dimensional)
	fmt.Println("1. Scalar (0-dimensional):")
	scalar := cpu.Full[float32](3.14, spark.NewShape())
	fmt.Printf("%s\n\n", scalar)

	// 2. Vector (1-dimensional)
	fmt.Println("2. Vector (1-dimensional):")
	vector := cpu.FromSlice([]float32{1.0, 2.0, 3.0, 4.0, 5.0})
	fmt.Printf("%s\n\n", vector)

	// 3. Matrix (2-dimensional)
	fmt.Println("3. Matrix (2-dimensional) - 3x4:")
	matrix := cpu.FromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}).Reshape(3, 4)
	fmt.Printf("%s\n\n", matrix)

	// 4. Matrix with floating point values
	fmt.Println("4. Matrix with floating point values:")
	floatMatrix := cpu.FromSlice([]float32{1.5, 2.25, 3.1415, 4.0, 5.678, 6.0}).Reshape(2, 3)
	fmt.Printf("%s\n\n", floatMatrix)

	// 5. 3D Tensor (RGB Image)
	fmt.Println("5. 3D Tensor (RGB Image) - 3 channels x 2x2 pixels:")
	rgbImage := cpu.FromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}).Reshape(3, 2, 2)
	fmt.Printf("%s\n\n", rgbImage)

	// 6. 4D Tensor (Batch of images)
	fmt.Println("6. 4D Tensor (Batch of images) - 2 samples x 2 channels x 2x2:")
	batchData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	batchImages := cpu.FromSlice(batchData).Reshape(2, 2, 2, 2)
	fmt.Printf("%s\n\n", batchImages)

	// 7. Identity matrix
	fmt.Println("7. Identity matrix (3x3):")
	identity := cpu.Eye[float32](3)
	fmt.Printf("%s\n\n", identity)

	// 8. Random matrix
	fmt.Println("8. Random matrix (2x4):")
	randomMatrix := cpu.Rand[float32](0, 10, spark.NewShape(2, 4))
	fmt.Printf("%s\n\n", randomMatrix)

	// 9. Ones matrix
	fmt.Println("9. Ones matrix (3x3):")
	ones := cpu.Ones[float32](spark.NewShape(3, 3))
	fmt.Printf("%s\n\n", ones)

	// 10. Zeros matrix
	fmt.Println("10. Zeros matrix (2x5):")
	zeros := cpu.Zeros[float32](spark.NewShape(2, 5))
	fmt.Printf("%s\n\n", zeros)

	// 11. Matrix with mixed number lengths (alignment test)
	fmt.Println("11. Matrix with mixed number lengths:")
	mixedMatrix := cpu.FromSlice([]float32{1.0, 1000.123456, 2.0, 0.001, 999999.0, 3.14159}).Reshape(2, 3)
	fmt.Printf("%s\n\n", mixedMatrix)

	fmt.Println("=== End of Demo ===")
}

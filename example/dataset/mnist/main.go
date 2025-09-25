package main

import (
	"fmt"

	"github.com/gocnn/spark/dataset/mnist"
)

func main() {
	ds, err := mnist.New[float64]("./data", true, true)
	if err != nil {
		panic(err)
	}

	loader := ds.NewDataLoader(32, true)

	for images, labels := range loader.All() {
		fmt.Printf("Batch: %d images, %d labels\n", len(images), len(labels))
		fmt.Printf("First image shape: %d pixels\n", len(images[0]))
		fmt.Printf("First label: %d\n\n", labels[0])

		fmt.Printf("Image visualization (label: %d):\n", labels[0])
		mnist.PrintImage(images[0])

		break
	}

	fmt.Printf("\nDataset size: %d\n", ds.Len())
}

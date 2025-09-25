package main

import (
	"fmt"

	"github.com/gocnn/spark/dataset/cifar10"
)

func main() {
	ds, err := cifar10.New("./data", true, true)
	if err != nil {
		panic(err)
	}

	loader := ds.NewDataLoader(32, true)

	for images, labels := range loader.All() {
		fmt.Printf("Batch: %d images, %d labels\n", len(images), len(labels))
		fmt.Printf("First image shape: %d pixels (32x32x3)\n", len(images[0]))
		fmt.Printf("First label: %d (%s)\n\n", labels[0], ds.GetClassName(labels[0]))

		fmt.Printf("Image visualization (label: %d - %s):\n", labels[0], ds.GetClassName(labels[0]))
		cifar10.PrintImage(images[0])

		break
	}

	fmt.Printf("\nDataset size: %d\n", ds.Len())
}

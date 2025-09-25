package main

import (
	"fmt"

	"github.com/gocnn/spark/dataset"
)

func printImage(pixels []float32) {
	for i := range 28 {
		for j := range 28 {
			pixel := pixels[i*28+j]
			if pixel > 0.5 {
				fmt.Print("██")
			} else if pixel > 0.3 {
				fmt.Print("▓▓")
			} else if pixel > 0.1 {
				fmt.Print("░░")
			} else {
				fmt.Print("  ")
			}
		}
		fmt.Println()
	}
}

func main() {
	ds, err := dataset.MNIST("./data", true, true)
	if err != nil {
		panic(err)
	}

	loader := ds.NewDataLoader(32, true)

	for images, labels := range loader.All() {
		fmt.Printf("Batch: %d images, %d labels\n", len(images), len(labels))
		fmt.Printf("First image shape: %d pixels\n", len(images[0]))
		fmt.Printf("First label: %d\n\n", labels[0])

		fmt.Printf("Image visualization (label: %d):\n", labels[0])
		printImage(images[0])

		break
	}

	fmt.Printf("\nDataset size: %d\n", ds.Len())
}

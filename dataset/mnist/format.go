package mnist

import (
	"fmt"

	"github.com/gocnn/candy"
)

// PrintImage displays a 28x28 MNIST image as ASCII art in the terminal.
// This function automatically handles different numeric types appropriately.
func PrintImage[T candy.D](pixels []T) {
	// Use any to work around type constraints
	switch p := any(pixels).(type) {
	case []float32:
		printImageFloat32(p)
	case []float64:
		printImageFloat64(p)
	case []uint8:
		printImageUint8(p)
	case []uint32:
		printImageUint32(p)
	case []int64:
		printImageInt64(p)
	default:
		panic("unsupported pixel type")
	}
}

// printImageFloat32 handles float32 pixels (normalized [0,1])
func printImageFloat32(pixels []float32) {
	for i := range MNISTImageSize {
		for j := range MNISTImageSize {
			pixel := pixels[i*MNISTImageSize+j]
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

// printImageFloat64 handles float64 pixels (normalized [0,1])
func printImageFloat64(pixels []float64) {
	for i := range MNISTImageSize {
		for j := range MNISTImageSize {
			pixel := pixels[i*MNISTImageSize+j]
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

// printImageUint8 handles uint8 pixels (raw [0,255])
func printImageUint8(pixels []uint8) {
	for i := range MNISTImageSize {
		for j := range MNISTImageSize {
			pixel := pixels[i*MNISTImageSize+j]
			if pixel > 127 {
				fmt.Print("██")
			} else if pixel > 76 {
				fmt.Print("▓▓")
			} else if pixel > 25 {
				fmt.Print("░░")
			} else {
				fmt.Print("  ")
			}
		}
		fmt.Println()
	}
}

// printImageUint32 handles uint32 pixels (assuming [0,255] range)
func printImageUint32(pixels []uint32) {
	for i := range MNISTImageSize {
		for j := range MNISTImageSize {
			pixel := pixels[i*MNISTImageSize+j]
			if pixel > 127 {
				fmt.Print("██")
			} else if pixel > 76 {
				fmt.Print("▓▓")
			} else if pixel > 25 {
				fmt.Print("░░")
			} else {
				fmt.Print("  ")
			}
		}
		fmt.Println()
	}
}

// printImageInt64 handles int64 pixels (assuming [0,255] range)
func printImageInt64(pixels []int64) {
	for i := range MNISTImageSize {
		for j := range MNISTImageSize {
			pixel := pixels[i*MNISTImageSize+j]
			if pixel > 127 {
				fmt.Print("██")
			} else if pixel > 76 {
				fmt.Print("▓▓")
			} else if pixel > 25 {
				fmt.Print("░░")
			} else {
				fmt.Print("  ")
			}
		}
		fmt.Println()
	}
}

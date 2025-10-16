package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/dataset/mnist"
	"github.com/gocnn/spark/tensor"
)

func readPGM(path string) ([]float32, int, int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, 0, err
	}
	i := 0
	nextToken := func() (string, error) {
		for i < len(b) {
			c := b[i]
			if c == '#' {
				for i < len(b) && b[i] != '\n' && b[i] != '\r' {
					i++
				}
			}
			if i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
				i++
				continue
			}
			break
		}
		if i >= len(b) {
			return "", fmt.Errorf("unexpected EOF")
		}
		start := i
		for i < len(b) {
			c := b[i]
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '#' {
				break
			}
			i++
		}
		return string(b[start:i]), nil
	}

	magic, err := nextToken()
	if err != nil {
		return nil, 0, 0, err
	}
	if magic != "P5" && magic != "P2" {
		return nil, 0, 0, fmt.Errorf("unsupported PGM magic %q", magic)
	}
	ws, err := nextToken()
	if err != nil {
		return nil, 0, 0, err
	}
	hs, err := nextToken()
	if err != nil {
		return nil, 0, 0, err
	}
	ms, err := nextToken()
	if err != nil {
		return nil, 0, 0, err
	}
	w, err := strconv.Atoi(ws)
	if err != nil {
		return nil, 0, 0, err
	}
	h, err := strconv.Atoi(hs)
	if err != nil {
		return nil, 0, 0, err
	}
	maxv, err := strconv.Atoi(ms)
	if err != nil {
		return nil, 0, 0, err
	}
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	n := w * h
	px := make([]float32, n)
	if magic == "P5" {
		if maxv <= 255 {
			if i+n > len(b) {
				return nil, 0, 0, fmt.Errorf("short pixel data")
			}
			for j := 0; j < n; j++ {
				px[j] = float32(uint8(b[i+j])) / float32(maxv)
			}
		} else {
			if i+2*n > len(b) {
				return nil, 0, 0, fmt.Errorf("short pixel data")
			}
			for j := 0; j < n; j++ {
				v := int(b[i+2*j])<<8 | int(b[i+2*j+1])
				px[j] = float32(v) / float32(maxv)
			}
		}
		return px, w, h, nil
	}
	for j := 0; j < n; j++ {
		t, err := nextToken()
		if err != nil {
			return nil, 0, 0, err
		}
		v, err := strconv.Atoi(t)
		if err != nil {
			return nil, 0, 0, err
		}
		px[j] = float32(v) / float32(maxv)
	}
	return px, w, h, nil
}

func inferFile(net *LeNet[float32], path string, invert, autoInvert bool) error {
	px, w, h, err := readPGM(path)
	if err != nil {
		return err
	}
	if w != 28 || h != 28 {
		return fmt.Errorf("expected 28x28 image, got %dx%d", w, h)
	}
	fmt.Println(path)
	mnist.PrintImage(px)
	// Optional inversion for white-background images
	if invert || autoInvert {
		mean := 0.0
		for _, v := range px {
			mean += float64(v)
		}
		mean /= float64(len(px))
		if invert || (autoInvert && mean > 0.5) {
			for i := range px {
				px[i] = 1 - px[i]
			}
		}
	}
	// Normalize to training stats
	px2 := make([]float32, len(px))
	for i, v := range px {
		px2[i] = (v - 0.1307) / 0.3081
	}
	x, err := tensor.New(px2, spark.NewShape(1, 1, 28, 28), spark.CPU)
	if err != nil {
		return err
	}
	z := net.MustForward(x)
	p := z.MustSoftmax(1)
	pd := p.Data()
	fmt.Print("probabilities: [")
	for i, v := range pd {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%.4f", v)
	}
	fmt.Println("]")
	bestI := 0
	bestV := pd[0]
	for i := 1; i < len(pd); i++ {
		if pd[i] > bestV {
			bestV = pd[i]
			bestI = i
		}
	}
	fmt.Printf("prediction: %d (%.4f)\n", bestI, bestV)
	return nil
}

func inferDir(net *LeNet[float32], dir string, invert, autoInvert bool) error {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	var files []string
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.EqualFold(filepath.Ext(name), ".pgm") {
			files = append(files, filepath.Join(dir, name))
		}
	}
	if len(files) == 0 {
		return fmt.Errorf("no .pgm files under %s", dir)
	}
	for _, f := range files {
		if err := inferFile(net, f, invert, autoInvert); err != nil {
			return err
		}
	}
	return nil
}

func main() {
	weights := flag.String("weights", "lenet.npz", "path to weights npz")
	file := flag.String("file", "", "path to one .pgm file")
	dir := flag.String("path", ".", "directory containing .pgm files")
	invert := flag.Bool("invert", false, "invert grayscale (1-p)")
	autoInvert := flag.Bool("auto-invert", true, "auto invert if image looks white-on-black")
	flag.Parse()

	net := NewLeNet[float32](spark.CPU)
	if err := net.Load(*weights); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if *file != "" {
		if err := inferFile(net, *file, *invert, *autoInvert); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if err := inferDir(net, *dir, *invert, *autoInvert); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

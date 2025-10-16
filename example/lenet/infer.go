package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/dataset/mnist"
	"github.com/gocnn/spark/tensor"
)

func readPGM(p string) ([]float32, int, int, error) {
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, 0, 0, err
	}
	i := 0
	next := func() (string, error) {
		for i < len(b) {
			if b[i] == '#' {
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
		s := i
		for i < len(b) {
			c := b[i]
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '#' {
				break
			}
			i++
		}
		return string(b[s:i]), nil
	}

	m, err := next()
	if err != nil {
		return nil, 0, 0, err
	}
	if m != "P5" && m != "P2" {
		return nil, 0, 0, fmt.Errorf("unsupported PGM magic %q", m)
	}
	ws, err := next()
	if err != nil {
		return nil, 0, 0, err
	}
	hs, err := next()
	if err != nil {
		return nil, 0, 0, err
	}
	ms, err := next()
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
	mx, err := strconv.Atoi(ms)
	if err != nil {
		return nil, 0, 0, err
	}
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	n := w * h
	q := make([]float32, n)
	if m == "P5" {
		if mx <= 255 {
			if i+n > len(b) {
				return nil, 0, 0, fmt.Errorf("short pixel data")
			}
			for j := range n {
				q[j] = float32(uint8(b[i+j])) / float32(mx)
			}
		} else {
			if i+2*n > len(b) {
				return nil, 0, 0, fmt.Errorf("short pixel data")
			}
			for j := range n {
				v := int(b[i+2*j])<<8 | int(b[i+2*j+1])
				q[j] = float32(v) / float32(mx)
			}
		}
		return q, w, h, nil
	}
	for j := range n {
		t, err := next()
		if err != nil {
			return nil, 0, 0, err
		}
		v, err := strconv.Atoi(t)
		if err != nil {
			return nil, 0, 0, err
		}
		q[j] = float32(v) / float32(mx)
	}
	return q, w, h, nil
}

func inferFile(net *LeNet[float32], f string, inv, auto bool) error {
	p, w, h, err := readPGM(f)
	if err != nil {
		return err
	}
	if w != 28 || h != 28 {
		return fmt.Errorf("expected 28x28 image, got %dx%d", w, h)
	}
	fmt.Println(f)
	mnist.PrintImage(p)
	// optional inversion
	if inv || auto {
		s := 0.0
		for _, v := range p {
			s += float64(v)
		}
		s /= float64(len(p))
		if inv || (auto && s > 0.5) {
			for i := range p {
				p[i] = 1 - p[i]
			}
		}
	}
	// normalize to training stats
	q := make([]float32, len(p))
	for i, v := range p {
		q[i] = (v - 0.1307) / 0.3081
	}
	x, err := tensor.New(q, spark.NewShape(1, 1, 28, 28), spark.CPU)
	if err != nil {
		return err
	}
	pd := net.MustForward(x).MustSoftmax(1).Data()
	fmt.Print("probabilities: [")
	for i, v := range pd {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%.4f", v)
	}
	fmt.Println("]")
	bi, bv := 0, pd[0]
	for i := 1; i < len(pd); i++ {
		if pd[i] > bv {
			bi, bv = i, pd[i]
		}
	}
	fmt.Printf("prediction: %d (%.4f)\n", bi, bv)
	return nil
}

func inferDir(net *LeNet[float32], dir string, inv, auto bool) error {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	var fs []string
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		if strings.EqualFold(filepath.Ext(e.Name()), ".pgm") {
			fs = append(fs, filepath.Join(dir, e.Name()))
		}
	}
	if len(fs) == 0 {
		return fmt.Errorf("no .pgm files under %s", dir)
	}
	for _, f := range fs {
		if err := inferFile(net, f, inv, auto); err != nil {
			return err
		}
	}
	return nil
}

func RunInfer(w, f, dir string, inv, auto bool) error {
	net := NewLeNet[float32](spark.CPU)
	if err := net.Load(w); err != nil {
		return err
	}
	if f != "" {
		return inferFile(net, f, inv, auto)
	}
	return inferDir(net, dir, inv, auto)
}

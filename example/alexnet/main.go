package main

import (
	"bufio"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/tensor"
)

func openImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, format, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	// ensure RGBA/NRGBA consistency if needed by converting via png/jpeg if format unknown
	_ = format
	return img, nil
}

func resizeNN(src image.Image, dstW, dstH int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	srcW := src.Bounds().Dx()
	srcH := src.Bounds().Dy()
	for y := 0; y < dstH; y++ {
		sy := int(float64(y) * float64(srcH) / float64(dstH))
		if sy >= srcH {
			sy = srcH - 1
		}
		for x := 0; x < dstW; x++ {
			sx := int(float64(x) * float64(srcW) / float64(dstW))
			if sx >= srcW {
				sx = srcW - 1
			}
			dst.Set(x, y, src.At(sx, sy))
		}
	}
	return dst
}

func centerCrop(img image.Image, cropW, cropH int) *image.RGBA {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	if w < cropW || h < cropH {
		return resizeNN(img, cropW, cropH)
	}
	x0 := (w - cropW) / 2
	y0 := (h - cropH) / 2
	rect := image.Rect(0, 0, cropW, cropH)
	dst := image.NewRGBA(rect)
	for y := 0; y < cropH; y++ {
		for x := 0; x < cropW; x++ {
			dst.Set(x, y, img.At(x0+x, y0+y))
		}
	}
	return dst
}

func toCHW224(img image.Image) []float32 {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	if w != 224 || h != 224 {
		img = resizeNN(img, 224, 224)
	}
	mean := [3]float32{0.485, 0.456, 0.406}
	std := [3]float32{0.229, 0.224, 0.225}
	out := make([]float32, 3*224*224)
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// Go RGBA returns uint32 in [0, 65535]
			fr := float32(r) / 65535.0
			fg := float32(g) / 65535.0
			fb := float32(b) / 65535.0
			// map to [0,1], normalize
			fr = (fr - mean[0]) / std[0]
			fg = (fg - mean[1]) / std[1]
			fb = (fb - mean[2]) / std[2]
			i := y*224 + x
			out[0*224*224+i] = fr
			out[1*224*224+i] = fg
			out[2*224*224+i] = fb
		}
	}
	return out
}

type kv struct {
	Idx int
	Val float64
}

func topK(probs []float32, k int) []kv {
	tmp := make([]kv, len(probs))
	for i, v := range probs {
		tmp[i] = kv{Idx: i, Val: float64(v)}
	}
	sort.Slice(tmp, func(i, j int) bool { return tmp[i].Val > tmp[j].Val })
	if k > len(tmp) {
		k = len(tmp)
	}
	return tmp[:k]
}

func preprocess224(img image.Image) image.Image {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	short := w
	if h < short {
		short = h
	}
	scale := 256.0 / float64(short)
	newW := int(math.Round(float64(w) * scale))
	newH := int(math.Round(float64(h) * scale))
	resized := resizeNN(img, newW, newH)
	cropped := centerCrop(resized, 224, 224)
	return cropped
}

func loadLabels(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	labels := []string{}
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ",", 2)
		if len(parts) != 2 {
			continue
		}
		idxStr := strings.TrimSpace(parts[0])
		idx, err := strconv.Atoi(idxStr)
		if err != nil {
			continue
		}
		name := strings.TrimSpace(parts[1])
		name = strings.Trim(name, " .")
		name = strings.ReplaceAll(name, "_", " ")
		if idx >= len(labels) {
			newLabels := make([]string, idx+1)
			copy(newLabels, labels)
			labels = newLabels
		}
		labels[idx] = name
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return labels, nil
}

func main() {
	weights := flag.String("weights", "alexnet.npz", "path to weights .npz")
	imagePath := flag.String("image", "dog.jpg", "path to input image")
	labelsPath := flag.String("labels", "imagenet_classes.txt", "path to labels txt")
	flag.Parse()

	img, err := openImage(*imagePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open image:", err)
		os.Exit(1)
	}

	img = preprocess224(img)

	xData := toCHW224(img)
	x, err := tensor.New(xData, spark.NewShape(1, 3, 224, 224), spark.CPU)
	if err != nil {
		fmt.Fprintln(os.Stderr, "tensor.New:", err)
		os.Exit(1)
	}

	net := NewAlexNet[float32](1000, spark.CPU)
	net.Eval()
	if err := net.Load(*weights); err != nil {
		fmt.Fprintln(os.Stderr, "load weights:", err)
		os.Exit(1)
	}

	logits := net.MustForward(x)
	probs := logits.MustSoftmax(-1).MustSqueeze(0)
	p := probs.Data()
	labels, _ := loadLabels(*labelsPath)

	top5 := topK(p, 5)
	fmt.Println("Top-5:")
	for _, it := range top5 {
		name := ""
		if it.Idx >= 0 && it.Idx < len(labels) && labels[it.Idx] != "" {
			name = labels[it.Idx]
		} else {
			name = fmt.Sprintf("class_%d", it.Idx)
		}
		fmt.Printf("  %4d %-25s %.4f\n", it.Idx, name, it.Val)
	}
}

// ensure jpeg/png decoders are linked
var _ = jpeg.Options{}
var _ = png.Encoder{}

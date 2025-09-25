package mnist

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// MNIST download URLs - modify these to use different mirror sources
const (
	// Primary source: Google Cloud Storage
	BaseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/%s.gz"

	// Alternative sources (uncomment to use):
	// BaseURL = "https://github.com/cvdfoundation/mnist/raw/main/%s.gz"
	// BaseURL = "https://huggingface.co/datasets/mnist/resolve/main/data/%s.gz"
	// BaseURL = "http://yann.lecun.com/exdb/mnist/%s.gz" // Original (often unavailable)
)

// Dataset holds MNIST images and labels.
type Dataset struct {
	images  [][]float32 // Flattened, normalized images [0,1]
	labels  []uint8     // Labels (0-9)
	root    string
	isTrain bool
}

// New creates an MNIST dataset, downloading files if needed.
func New(root string, train, download bool) (*Dataset, error) {
	ds := &Dataset{root: root, isTrain: train}
	if err := ds.ensureFiles(download); err != nil {
		return nil, fmt.Errorf("ensure files: %w", err)
	}
	if err := ds.load(); err != nil {
		return nil, fmt.Errorf("load data: %w", err)
	}
	return ds, nil
}

// ensureFiles verifies or downloads MNIST data files.
func (d *Dataset) ensureFiles(download bool) error {
	for _, file := range d.fileNames() {
		path := filepath.Join(d.root, file)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			if !download {
				return fmt.Errorf("file %s missing, download disabled", file)
			}
			if err := d.download(file); err != nil {
				return fmt.Errorf("download %s: %w", file, err)
			}
		} else if err != nil {
			return fmt.Errorf("check file %s: %w", file, err)
		}
	}
	return nil
}

// fileNames returns the dataset file names based on training or testing mode.
func (d *Dataset) fileNames() []string {
	prefix := "t10k"
	if d.isTrain {
		prefix = "train"
	}
	return []string{
		fmt.Sprintf("%s-images-idx3-ubyte", prefix),
		fmt.Sprintf("%s-labels-idx1-ubyte", prefix),
	}
}

// download fetches and decompresses an MNIST file.
func (d *Dataset) download(file string) error {
	url := fmt.Sprintf(BaseURL, file)
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("fetch %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad response: %s", resp.Status)
	}

	gzr, err := gzip.NewReader(resp.Body)
	if err != nil {
		return fmt.Errorf("decompress: %w", err)
	}
	defer gzr.Close()

	if err := os.MkdirAll(d.root, 0755); err != nil {
		return fmt.Errorf("create dir %s: %w", d.root, err)
	}

	path := filepath.Join(d.root, file)
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file %s: %w", path, err)
	}
	defer f.Close()

	if _, err := io.Copy(f, gzr); err != nil {
		return fmt.Errorf("write file %s: %w", path, err)
	}
	return nil
}

// load reads and normalizes MNIST image and label data.
func (d *Dataset) load() error {
	imgPath, lblPath := filepath.Join(d.root, d.fileNames()[0]), filepath.Join(d.root, d.fileNames()[1])
	images, err := readImages(imgPath)
	if err != nil {
		return fmt.Errorf("read images: %w", err)
	}

	d.images = make([][]float32, len(images))
	for i, img := range images {
		d.images[i] = make([]float32, len(img))
		for j, px := range img {
			d.images[i][j] = float32(px) / 255 // Normalize to [0,1]
		}
	}

	d.labels, err = readLabels(lblPath)
	if err != nil {
		return fmt.Errorf("read labels: %w", err)
	}
	return nil
}

// Len returns the number of samples in the dataset.
func (d *Dataset) Len() int {
	return len(d.images)
}

// Get returns the image and label at index i.
func (d *Dataset) Get(i int) ([]float32, uint8) {
	return d.images[i], d.labels[i]
}

// readImages reads raw MNIST image data.
func readImages(path string) ([][]uint8, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	r := bufio.NewReader(f)
	var magic, nImages, nRows, nCols int32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != 2051 {
		return nil, fmt.Errorf("invalid image magic: %d", magic)
	}
	if err := binary.Read(r, binary.BigEndian, &nImages); err != nil {
		return nil, fmt.Errorf("read image count: %w", err)
	}
	if err := binary.Read(r, binary.BigEndian, &nRows); err != nil {
		return nil, fmt.Errorf("read row count: %w", err)
	}
	if err := binary.Read(r, binary.BigEndian, &nCols); err != nil {
		return nil, fmt.Errorf("read column count: %w", err)
	}

	imgSize := int(nRows * nCols)
	images := make([][]uint8, nImages)
	for i := range images {
		images[i] = make([]uint8, imgSize)
		if _, err := io.ReadFull(r, images[i]); err != nil {
			return nil, fmt.Errorf("read image %d: %w", i, err)
		}
	}
	return images, nil
}

// readLabels reads raw MNIST label data.
func readLabels(path string) ([]uint8, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	r := bufio.NewReader(f)
	var magic, nLabels int32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != 2049 {
		return nil, fmt.Errorf("invalid label magic: %d", magic)
	}
	if err := binary.Read(r, binary.BigEndian, &nLabels); err != nil {
		return nil, fmt.Errorf("read label count: %w", err)
	}

	labels := make([]uint8, nLabels)
	if _, err := io.ReadFull(r, labels); err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}
	return labels, nil
}

func PrintImage(pixels []float32) {
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

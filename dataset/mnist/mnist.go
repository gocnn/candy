// mnist.go
package mnist

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/gocnn/spark"
	"github.com/gocnn/spark/dataset/progress"
)

const MNISTImageSize = 28

// MNIST download URLs - modify these to use different mirror sources
const (
	// Primary source: Google Cloud Storage
	BaseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/%s.gz"
	// Alternative sources (uncomment to use):
	// BaseURL = "https://github.com/cvdfoundation/mnist/raw/main/%s.gz"
	// BaseURL = "https://huggingface.co/datasets/mnist/resolve/main/data/%s.gz"
	// BaseURL = "http://yann.lecun.com/exdb/mnist/%s.gz" // Original (often unavailable)
)

// Dataset holds MNIST images and labels with generic precision.
type Dataset[T spark.D] struct {
	images  [][]T   // Flattened, normalized images [0,1]
	labels  []uint8 // Labels (0-9)
	root    string
	isTrain bool
}

// New creates an MNIST dataset with specified precision type.
func New[T spark.D](root string, train, download bool) (*Dataset[T], error) {
	ds := &Dataset[T]{root: root, isTrain: train}
	if err := ds.ensureFiles(download); err != nil {
		return nil, fmt.Errorf("ensure files: %w", err)
	}
	if err := ds.load(); err != nil {
		return nil, fmt.Errorf("load data: %w", err)
	}
	return ds, nil
}

// Len returns the number of samples in the dataset.
func (d *Dataset[T]) Len() int {
	return len(d.images)
}

// Get returns the image and label at index i.
func (d *Dataset[T]) Get(i int) ([]T, uint8) {
	return d.images[i], d.labels[i]
}

// GetRaw returns the raw uint8 image data (0-255) and label.
func (d *Dataset[T]) GetRaw(i int) ([]uint8, uint8) {
	raw := make([]uint8, len(d.images[i]))
	for j, pixel := range d.images[i] {
		raw[j] = uint8(float64(pixel) * 255)
	}
	return raw, d.labels[i]
}

// ensureFiles verifies or downloads MNIST data files using the generic downloader.
func (d *Dataset[T]) ensureFiles(download bool) error {
	files := make([]progress.File, 0, 2)
	for _, name := range d.fileNames() {
		files = append(files, progress.File{
			URL:    fmt.Sprintf(BaseURL, name),
			Name:   name,
			Needed: true, // All MNIST files are required
		})
	}

	client := progress.New(progress.Config{
		Dir:      d.root,
		Files:    files,
		Download: download,
		Progress: true, // Show progress bars
		Gzip:     true, // Auto-decompress .gz files
	})

	return client.Ensure()
}

// fileNames returns the dataset file names based on training or testing mode.
func (d *Dataset[T]) fileNames() []string {
	prefix := "t10k"
	if d.isTrain {
		prefix = "train"
	}
	return []string{
		fmt.Sprintf("%s-images-idx3-ubyte", prefix),
		fmt.Sprintf("%s-labels-idx1-ubyte", prefix),
	}
}

// load reads and normalizes MNIST image and label data.
func (d *Dataset[T]) load() error {
	imgPath, lblPath := filepath.Join(d.root, d.fileNames()[0]), filepath.Join(d.root, d.fileNames()[1])
	images, err := readImages(imgPath)
	if err != nil {
		return fmt.Errorf("read images: %w", err)
	}

	d.images = make([][]T, len(images))
	for i, img := range images {
		d.images[i] = make([]T, len(img))
		for j, px := range img {
			d.images[i][j] = T(px) / 255 // Normalize to [0,1]
		}
	}

	d.labels, err = readLabels(lblPath)
	if err != nil {
		return fmt.Errorf("read labels: %w", err)
	}
	return nil
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

package mnist

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/gocnn/candy"
	"github.com/gocnn/candy/dataset/progress"
)

const (
	MNISTImageSize = 28

	// BaseURL is the primary source for MNIST data downloads.
	BaseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/%s.gz"
	// Alternative sources (uncomment to use):
	// BaseURL = "https://github.com/cvdfoundation/mnist/raw/main/%s.gz"
	// BaseURL = "https://huggingface.co/datasets/mnist/resolve/main/data/%s.gz"
	// BaseURL = "http://yann.lecun.com/exdb/mnist/%s.gz" // Original (often unavailable)
)

// Dataset holds MNIST images and labels with generic precision.
type Dataset[T candy.D] struct {
	imgs  [][]T   // Normalized images [0,1]
	lbls  []uint8 // Labels (0-9)
	root  string
	train bool
}

// New creates an MNIST dataset with specified precision type.
func New[T candy.D](root string, train, download bool) (*Dataset[T], error) {
	ds := &Dataset[T]{root: root, train: train}
	if err := ds.ensureFiles(download); err != nil {
		return nil, fmt.Errorf("failed to ensure files: %w", err)
	}
	if err := ds.load(); err != nil {
		return nil, fmt.Errorf("failed to load: %w", err)
	}
	return ds, nil
}

// MustNew creates an MNIST dataset with specified precision type, panics on error.
func MustNew[T candy.D](root string, train, download bool) *Dataset[T] {
	ds, err := New[T](root, train, download)
	if err != nil {
		panic(err)
	}
	return ds
}

// Len returns the number of samples in the dataset.
func (ds *Dataset[T]) Len() int {
	return len(ds.imgs)
}

// Get returns the image and label at index i.
func (ds *Dataset[T]) Get(i int) ([]T, uint8) {
	return ds.imgs[i], ds.lbls[i]
}

// GetRaw returns the raw uint8 image data (0-255) and label.
func (ds *Dataset[T]) GetRaw(i int) ([]uint8, uint8) {
	img := make([]uint8, len(ds.imgs[i]))
	for j, px := range ds.imgs[i] {
		img[j] = uint8(float64(px) * 255)
	}
	return img, ds.lbls[i]
}

// ensureFiles verifies or downloads MNIST data files.
func (ds *Dataset[T]) ensureFiles(download bool) error {
	var files []progress.File
	for _, name := range ds.fileNames() {
		files = append(files, progress.File{
			URL:    fmt.Sprintf(BaseURL, name),
			Name:   name,
			Needed: true,
		})
	}
	return progress.New(progress.Config{
		Dir:      ds.root,
		Files:    files,
		Download: download,
		Progress: true,
		Gzip:     true,
	}).Ensure()
}

// fileNames returns the dataset file names based on training or testing mode.
func (ds *Dataset[T]) fileNames() []string {
	prefix := "t10k"
	if ds.train {
		prefix = "train"
	}
	return []string{
		fmt.Sprintf("%s-images-idx3-ubyte", prefix),
		fmt.Sprintf("%s-labels-idx1-ubyte", prefix),
	}
}

// load reads and normalizes MNIST image and label data.
func (ds *Dataset[T]) load() error {
	imgPath := filepath.Join(ds.root, ds.fileNames()[0])
	lblPath := filepath.Join(ds.root, ds.fileNames()[1])
	imgs, err := readImages(imgPath)
	if err != nil {
		return fmt.Errorf("failed to read images: %w", err)
	}
	ds.imgs = make([][]T, len(imgs))
	for i, img := range imgs {
		ds.imgs[i] = make([]T, len(img))
		for j, px := range img {
			var zero T
			switch any(zero).(type) {
			case float32, float64:
				// PyTorch-style normalization: (x/255 - mean) / std
				normalized := float32(px) / 255.0
				ds.imgs[i][j] = T((normalized - 0.1307) / 0.3081)
			case uint8, uint32, int64:
				ds.imgs[i][j] = T(px) // [0,255]
			}
		}
	}
	ds.lbls, err = readLabels(lblPath)
	if err != nil {
		return fmt.Errorf("failed to read labels: %w", err)
	}
	return nil
}

// readImages reads raw MNIST image data.
func readImages(path string) ([][]uint8, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s: %w", path, err)
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var magic, n, rows, cols int32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("failed to read magic: %w", err)
	}
	if magic != 2051 {
		return nil, fmt.Errorf("invalid image magic: %d", magic)
	}
	if err := binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, fmt.Errorf("failed to read image count: %w", err)
	}
	if err := binary.Read(r, binary.BigEndian, &rows); err != nil {
		return nil, fmt.Errorf("failed to read row count: %w", err)
	}
	if err := binary.Read(r, binary.BigEndian, &cols); err != nil {
		return nil, fmt.Errorf("failed to read column count: %w", err)
	}
	imgs := make([][]uint8, n)
	for i := range imgs {
		imgs[i] = make([]uint8, rows*cols)
		if _, err := io.ReadFull(r, imgs[i]); err != nil {
			return nil, fmt.Errorf("failed to read image %d: %w", i, err)
		}
	}
	return imgs, nil
}

// readLabels reads raw MNIST label data.
func readLabels(path string) ([]uint8, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s: %w", path, err)
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var magic, n int32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("failed to read magic: %w", err)
	}
	if magic != 2049 {
		return nil, fmt.Errorf("invalid label magic: %d", magic)
	}
	if err := binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, fmt.Errorf("failed to read label count: %w", err)
	}
	lbls := make([]uint8, n)
	if _, err := io.ReadFull(r, lbls); err != nil {
		return nil, fmt.Errorf("failed to read labels: %w", err)
	}
	return lbls, nil
}

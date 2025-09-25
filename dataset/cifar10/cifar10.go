package cifar10

import (
	"archive/tar"
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/gocnn/spark/dataset/progress"
)

// CIFAR-10 download URL
const (
	BaseURL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
)

// CIFAR-10 class names
var ClassNames = []string{
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck",
}

// Dataset holds CIFAR-10 images and labels.
type Dataset struct {
	images  [][]float32 // RGB images normalized to [0,1], shape: [N, 3072] (32x32x3)
	labels  []uint8     // Labels (0-9)
	root    string
	isTrain bool
}

// New creates a CIFAR-10 dataset, downloading files if needed.
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

// Len returns the number of samples in the dataset.
func (d *Dataset) Len() int {
	return len(d.images)
}

// Get returns the image and label at index i.
func (d *Dataset) Get(i int) ([]float32, uint8) {
	if i < 0 || i >= len(d.images) {
		panic(fmt.Sprintf("index %d out of range [0, %d)", i, len(d.images)))
	}
	return d.images[i], d.labels[i]
}

// GetClassName returns the class name for a given label.
func (d *Dataset) GetClassName(label uint8) string {
	if int(label) < len(ClassNames) {
		return ClassNames[label]
	}
	return "unknown"
}

// ensureFiles verifies or downloads CIFAR-10 data files using the generic downloader.
func (d *Dataset) ensureFiles(download bool) error {
	// Check if binary files already exist
	if d.binaryFilesExist() {
		return nil
	}

	// Download tar.gz file
	client := progress.New(progress.Config{
		Dir: d.root,
		Files: []progress.File{{
			URL:    BaseURL,
			Name:   "cifar-10-binary.tar.gz",
			Needed: true,
		}},
		Download: download,
		Progress: true,
		Gzip:     false, // We'll handle tar.gz extraction manually
	})

	if err := client.Ensure(); err != nil {
		return err
	}

	// Extract tar.gz file
	return d.extractTarGz()
}

// binaryFilesExist checks if the required binary files are already extracted.
func (d *Dataset) binaryFilesExist() bool {
	for _, file := range d.fileNames() {
		path := filepath.Join(d.root, "cifar-10-batches-bin", file)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return false
		}
	}
	return true
}

// fileNames returns the dataset file names based on training or testing mode.
func (d *Dataset) fileNames() []string {
	if d.isTrain {
		return []string{
			"data_batch_1.bin",
			"data_batch_2.bin",
			"data_batch_3.bin",
			"data_batch_4.bin",
			"data_batch_5.bin",
		}
	}
	return []string{"test_batch.bin"}
}

// extractTarGz extracts the downloaded tar.gz file automatically.
func (d *Dataset) extractTarGz() error {
	tarGzPath := filepath.Join(d.root, "cifar-10-binary.tar.gz")

	// Open the tar.gz file
	file, err := os.Open(tarGzPath)
	if err != nil {
		return fmt.Errorf("open tar.gz file: %w", err)
	}
	defer file.Close()

	// Create gzip reader
	gzr, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("create gzip reader: %w", err)
	}
	defer gzr.Close()

	// Create tar reader
	tr := tar.NewReader(gzr)

	fmt.Println("Extracting CIFAR-10 binary files...")

	// Extract files
	extractedCount := 0
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break // End of archive
		}
		if err != nil {
			return fmt.Errorf("read tar header: %w", err)
		}

		// Skip directories and non-regular files
		if header.Typeflag != tar.TypeReg {
			continue
		}

		// Only extract .bin files and important metadata
		if !strings.HasSuffix(header.Name, ".bin") &&
			!strings.HasSuffix(header.Name, "batches.meta.txt") {
			continue
		}

		// Create the full file path
		destPath := filepath.Join(d.root, header.Name)

		// Create directory if it doesn't exist
		if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
			return fmt.Errorf("create directory: %w", err)
		}

		// Create the file
		outFile, err := os.Create(destPath)
		if err != nil {
			return fmt.Errorf("create file %s: %w", destPath, err)
		}

		// Copy file content with size limit check
		written, err := io.CopyN(outFile, tr, header.Size)
		outFile.Close()

		if err != nil && err != io.EOF {
			return fmt.Errorf("extract file %s: %w", destPath, err)
		}

		if written != header.Size {
			return fmt.Errorf("incomplete extraction of %s: wrote %d, expected %d",
				destPath, written, header.Size)
		}

		fmt.Printf("  Extracted: %s (%d bytes)\n", header.Name, header.Size)
		extractedCount++
	}

	if extractedCount == 0 {
		return fmt.Errorf("no files were extracted from the archive")
	}

	fmt.Printf("Successfully extracted %d files\n", extractedCount)

	// Optionally remove the tar.gz file after successful extraction
	// Uncomment the next line if you want to save disk space
	// os.Remove(tarGzPath)

	return nil
}

// load reads and normalizes CIFAR-10 image and label data.
func (d *Dataset) load() error {
	var allImages [][]float32
	var allLabels []uint8

	fmt.Printf("Loading CIFAR-10 %s data...\n", map[bool]string{true: "training", false: "test"}[d.isTrain])

	for i, fileName := range d.fileNames() {
		path := filepath.Join(d.root, "cifar-10-batches-bin", fileName)
		images, labels, err := d.readBatch(path)
		if err != nil {
			return fmt.Errorf("read batch %s: %w", fileName, err)
		}

		allImages = append(allImages, images...)
		allLabels = append(allLabels, labels...)

		fmt.Printf("  Loaded batch %d: %d samples\n", i+1, len(images))
	}

	d.images = allImages
	d.labels = allLabels

	fmt.Printf("Total loaded: %d images, %d labels\n", len(d.images), len(d.labels))
	return nil
}

// readBatch reads a single CIFAR-10 batch file with improved error handling.
func (d *Dataset) readBatch(path string) ([][]float32, []uint8, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	// Get file size for validation
	stat, err := f.Stat()
	if err != nil {
		return nil, nil, fmt.Errorf("stat %s: %w", path, err)
	}

	const (
		imageSize    = 32 * 32 * 3        // 3072 bytes per image (32x32x3)
		recordSize   = 1 + imageSize      // 1 byte label + 3072 bytes image
		expectedSize = 10000 * recordSize // 10,000 records per batch
	)

	// Validate file size
	if stat.Size() != expectedSize {
		return nil, nil, fmt.Errorf("invalid batch file size: got %d, expected %d",
			stat.Size(), expectedSize)
	}

	r := bufio.NewReader(f)
	images := make([][]float32, 0, 10000)
	labels := make([]uint8, 0, 10000)

	for i := 0; i < 10000; i++ {
		// Read label (1 byte)
		var label uint8
		if err := binary.Read(r, binary.BigEndian, &label); err != nil {
			return nil, nil, fmt.Errorf("read label at record %d: %w", i, err)
		}

		// Validate label range
		if label > 9 {
			return nil, nil, fmt.Errorf("invalid label %d at record %d", label, i)
		}

		// Read image data (3072 bytes)
		imageData := make([]byte, imageSize)
		if _, err := io.ReadFull(r, imageData); err != nil {
			return nil, nil, fmt.Errorf("read image at record %d: %w", i, err)
		}

		// Convert to float32 and normalize to [0,1]
		image := make([]float32, imageSize)
		for j, pixel := range imageData {
			image[j] = float32(pixel) / 255.0
		}

		images = append(images, image)
		labels = append(labels, label)
	}

	return images, labels, nil
}

// PrintImage prints a simple ASCII representation of a CIFAR-10 image.
func PrintImage(pixels []float32) {
	if len(pixels) != 3072 {
		fmt.Println("Invalid image size, expected 3072 pixels (32x32x3)")
		return
	}

	// Convert RGB to grayscale for display
	for i := range 32 {
		for j := range 32 {
			// Average RGB channels for grayscale
			r := pixels[i*32+j]      // Red channel
			g := pixels[1024+i*32+j] // Green channel
			b := pixels[2048+i*32+j] // Blue channel
			gray := (r + g + b) / 3.0

			if gray > 0.7 {
				fmt.Print("██")
			} else if gray > 0.5 {
				fmt.Print("▓▓")
			} else if gray > 0.3 {
				fmt.Print("░░")
			} else {
				fmt.Print("  ")
			}
		}
		fmt.Println()
	}
}

// GetImageDimensions returns the image dimensions (height, width, channels).
func GetImageDimensions() (int, int, int) {
	return 32, 32, 3
}

// GetNumClasses returns the number of classes in CIFAR-10.
func GetNumClasses() int {
	return len(ClassNames)
}

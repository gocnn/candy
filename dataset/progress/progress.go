package progress

import (
	"fmt"
	"io"
	"strings"
)

// ProgressReader wraps an io.ReadCloser to display download progress.
type ProgressReader struct {
	io.ReadCloser
	total      int64
	downloaded int64
	file       string
	completed  bool // Add flag to track completion
}

// NewProgressReader creates a new ProgressReader.
func NewProgressReader(reader io.ReadCloser, total int64, file string) *ProgressReader {
	return &ProgressReader{
		ReadCloser: reader,
		total:      total,
		file:       file,
	}
}

// Read implements io.Reader, updating progress after each read.
func (pr *ProgressReader) Read(p []byte) (int, error) {
	n, err := pr.ReadCloser.Read(p)
	pr.downloaded += int64(n)
	if !pr.completed {
		pr.printProgress()
	}
	return n, err
}

// printProgress displays the progress bar if total size is known.
func (pr *ProgressReader) printProgress() {
	if pr.total <= 0 {
		return
	}
	pct := float64(pr.downloaded) / float64(pr.total) * 100
	barWidth := 50
	bar := int(pct / 100 * float64(barWidth))
	fmt.Printf("\rDownloading %s: [%s%s] %.2f%%",
		pr.file,
		strings.Repeat("=", bar),
		strings.Repeat(" ", barWidth-bar),
		pct)
	if pr.downloaded >= pr.total {
		fmt.Println()
		pr.completed = true
	}
}

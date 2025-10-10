package progress

import (
	"fmt"
	"io"
	"strings"
)

// ProgressReader wraps an io.ReadCloser to display download progress.
type ProgressReader struct {
	r          io.ReadCloser // Underlying reader
	total      int64         // Total size
	downloaded int64         // Bytes downloaded
	f          string        // File name
	done       bool          // Completion flag
}

// NewProgressReader creates a new ProgressReader.
func NewProgressReader(r io.ReadCloser, total int64, f string) *ProgressReader {
	return &ProgressReader{r: r, total: total, f: f}
}

// Read implements io.Reader, updating progress after each read.
func (p *ProgressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.downloaded += int64(n)
	if !p.done && p.total > 0 {
		p.print()
	}
	return n, err
}

// Close implements io.Closer.
func (p *ProgressReader) Close() error {
	return p.r.Close()
}

// print displays the progress bar.
func (p *ProgressReader) print() {
	const w = 50 // Bar width
	pct := float64(p.downloaded) / float64(p.total) * 100
	bar := int(pct * float64(w) / 100)
	fmt.Printf("\rDownloading %s: [%s%s] %.2f%%", p.f, strings.Repeat("=", bar), strings.Repeat(" ", w-bar), pct)
	if p.downloaded >= p.total {
		fmt.Println()
		p.done = true
	}
}

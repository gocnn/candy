package progress

import (
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// File specifies a file to download.
type File struct {
	URL    string // Source URL
	Name   string // Local filename
	Needed bool   // Whether the file is required
}

// Config defines download behavior.
type Config struct {
	Dir      string // Directory to save files
	Files    []File // Files to download
	Download bool   // Allow downloading missing files
	Progress bool   // Show progress bars
	Gzip     bool   // Decompress gzip files
}

// Client manages file downloads.
type Client struct {
	cfg Config
}

// New creates a new download client.
func New(cfg Config) *Client {
	return &Client{cfg: cfg}
}

// Ensure checks or downloads configured files.
func (c *Client) Ensure() error {
	for _, f := range c.cfg.Files {
		p := filepath.Join(c.cfg.Dir, f.Name)
		_, err := os.Stat(p)
		if err == nil {
			continue
		}
		if !os.IsNotExist(err) {
			return fmt.Errorf("failed to check %s: %w", f.Name, err)
		}
		if !c.cfg.Download {
			if f.Needed {
				return fmt.Errorf("missing required file %s", f.Name)
			}
			continue
		}
		if err := c.download(f, p); err != nil {
			return fmt.Errorf("failed to download %s: %w", f.Name, err)
		}
	}
	return nil
}

// download fetches and saves a file.
func (c *Client) download(f File, p string) error {
	resp, err := http.Get(f.URL)
	if err != nil {
		return fmt.Errorf("failed to fetch %s: %w", f.URL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad response: %s", resp.Status)
	}
	var r io.Reader
	var rc io.Closer
	if c.cfg.Progress && resp.ContentLength > 0 {
		pr := NewProgressReader(resp.Body, resp.ContentLength, f.Name)
		r, rc = pr, pr
	} else {
		r, rc = resp.Body, resp.Body
	}
	if c.cfg.Gzip && filepath.Ext(f.URL) == ".gz" {
		gr, err := gzip.NewReader(r)
		if err != nil {
			rc.Close()
			return fmt.Errorf("failed to decompress: %w", err)
		}
		defer gr.Close()
		r = gr
	}
	if err := os.MkdirAll(c.cfg.Dir, 0755); err != nil {
		rc.Close()
		return fmt.Errorf("failed to create dir %s: %w", c.cfg.Dir, err)
	}
	w, err := os.Create(p)
	if err != nil {
		rc.Close()
		return fmt.Errorf("failed to create %s: %w", p, err)
	}
	defer w.Close()
	if _, err := io.Copy(w, r); err != nil {
		rc.Close()
		return fmt.Errorf("failed to write %s: %w", p, err)
	}
	return rc.Close()
}

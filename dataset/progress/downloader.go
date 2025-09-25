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
		path := filepath.Join(c.cfg.Dir, f.Name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			if !c.cfg.Download {
				if f.Needed {
					return fmt.Errorf("missing required file %s", f.Name)
				}
				continue
			}
			if err := c.Download(f, path); err != nil {
				return fmt.Errorf("download %s: %w", f.Name, err)
			}
		} else if err != nil {
			return fmt.Errorf("check %s: %w", f.Name, err)
		}
	}
	return nil
}

// Download fetches and saves a file.
func (c *Client) Download(f File, path string) error {
	resp, err := http.Get(f.URL)
	if err != nil {
		return fmt.Errorf("fetch %s: %w", f.URL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad response: %s", resp.Status)
	}

	body := io.Reader(resp.Body)
	if c.cfg.Progress && resp.ContentLength > 0 {
		body = NewProgressReader(resp.Body, resp.ContentLength, f.Name)
	}

	if c.cfg.Gzip && filepath.Ext(f.URL) == ".gz" {
		body, err = gzip.NewReader(body)
		if err != nil {
			return fmt.Errorf("decompress: %w", err)
		}
		defer body.(*gzip.Reader).Close()
	}

	if err := os.MkdirAll(c.cfg.Dir, 0755); err != nil {
		return fmt.Errorf("create dir %s: %w", c.cfg.Dir, err)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer file.Close()

	if _, err := io.Copy(file, body); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

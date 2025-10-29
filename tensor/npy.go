package tensor

import (
	"archive/zip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/gocnn/candy"
)

var (
	npyMagic  = []byte{0x93, 'N', 'U', 'M', 'P', 'Y'}
	npySuffix = ".npy"
)

type npyHeader struct {
	shape        []int
	dtype        candy.DType
	fortranOrder bool
}

func descrToDType(descr string) (candy.DType, error) {
	switch descr {
	case "e", "f2":
		return candy.F16, nil
	case "f", "f4":
		return candy.F32, nil
	case "d", "f8":
		return candy.F64, nil
	case "q", "i8":
		return candy.I64, nil
	case "B", "u1":
		return candy.U8, nil
	case "I", "u4":
		return candy.U32, nil
	case "?", "b1":
		return candy.U8, nil
	default:
		return 0, fmt.Errorf("npy: unrecognized descr %q", descr)
	}
}

func dtypeToDescr(dt candy.DType) (string, error) {
	switch dt {
	case candy.F32:
		return "f4", nil
	case candy.F64:
		return "f8", nil
	case candy.U8:
		return "u1", nil
	case candy.U32:
		return "u4", nil
	case candy.I64:
		return "i8", nil
	case candy.F16:
		return "f2", nil
	case candy.BF16:
		return "", fmt.Errorf("npy: bf16 write unsupported")
	default:
		return "", fmt.Errorf("npy: unsupported dtype %v", dt)
	}
}

func convertFloat64To[T candy.D](dst []T, src []float64, dtOut candy.DType) {
	switch dtOut {
	case candy.F32:
		for i, v := range src {
			dst[i] = any(float32(v)).(T)
		}
	case candy.F64:
		for i, v := range src {
			dst[i] = any(v).(T)
		}
	case candy.U8:
		for i, v := range src {
			dst[i] = any(uint8(v)).(T)
		}
	case candy.U32:
		for i, v := range src {
			dst[i] = any(uint32(v)).(T)
		}
	case candy.I64:
		for i, v := range src {
			dst[i] = any(int64(v)).(T)
		}
	}
}

func convertInt64To[T candy.D](dst []T, src []int64, dtOut candy.DType) {
	switch dtOut {
	case candy.F32:
		for i, v := range src {
			dst[i] = any(float32(v)).(T)
		}
	case candy.F64:
		for i, v := range src {
			dst[i] = any(float64(v)).(T)
		}
	case candy.U8:
		for i, v := range src {
			dst[i] = any(uint8(v)).(T)
		}
	case candy.U32:
		for i, v := range src {
			dst[i] = any(uint32(v)).(T)
		}
	case candy.I64:
		for i, v := range src {
			dst[i] = any(v).(T)
		}
	}
}

func u8ToI64(x []uint8) []int64 {
	y := make([]int64, len(x))
	for i, v := range x {
		y[i] = int64(v)
	}
	return y
}
func u32ToI64(x []uint32) []int64 {
	y := make([]int64, len(x))
	for i, v := range x {
		y[i] = int64(v)
	}
	return y
}

func readHeader(r io.Reader) (string, error) {
	buf := make([]byte, len(npyMagic))
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", fmt.Errorf("npy: read magic: %w", err)
	}
	if !slices.Equal(buf, npyMagic) {
		return "", errors.New("npy: magic string mismatch")
	}
	ver := make([]byte, 2)
	if _, err := io.ReadFull(r, ver); err != nil {
		return "", fmt.Errorf("npy: read version: %w", err)
	}
	var nlen int
	switch ver[0] {
	case 1:
		nlen = 2
	case 2:
		nlen = 4
	default:
		return "", fmt.Errorf("npy: unsupported version %d", ver[0])
	}
	lenbuf := make([]byte, nlen)
	if _, err := io.ReadFull(r, lenbuf); err != nil {
		return "", fmt.Errorf("npy: read header len: %w", err)
	}
	headerLen := 0
	for i := nlen - 1; i >= 0; i-- {
		headerLen = headerLen*256 + int(lenbuf[i])
	}
	h := make([]byte, headerLen)
	if _, err := io.ReadFull(r, h); err != nil {
		return "", fmt.Errorf("npy: read header: %w", err)
	}
	return string(h), nil
}

func parseHeader(header string) (npyHeader, error) {
	s := strings.TrimSpace(header)
	s = strings.Trim(s, "{} \n\r\t,")
	parts := make([]string, 0, 4)
	start, paren := 0, 0
	for i, c := range s {
		switch c {
		case '(':
			paren++
		case ')':
			paren--
		case ',':
			if paren == 0 {
				p := strings.TrimSpace(s[start:i])
				if p != "" {
					parts = append(parts, p)
				}
				start = i + 1
			}
		}
	}
	if start < len(s) {
		p := strings.TrimSpace(s[start:])
		if p != "" {
			parts = append(parts, p)
		}
	}
	m := map[string]string{}
	for _, p := range parts {
		kv := strings.SplitN(p, ":", 2)
		if len(kv) != 2 {
			return npyHeader{}, fmt.Errorf("npy: parse header %q", header)
		}
		k := strings.Trim(kv[0], "'\" \t\n\r")
		v := strings.Trim(kv[1], "'\" \t\n\r")
		m[k] = v
	}
	fo := false
	if v, ok := m["fortran_order"]; ok {
		switch v {
		case "True":
			fo = true
		case "False":
			fo = false
		default:
			return npyHeader{}, fmt.Errorf("npy: unknown fortran_order %q", v)
		}
	}
	ds, ok := m["descr"]
	if !ok || ds == "" {
		return npyHeader{}, errors.New("npy: no descr in header")
	}
	if strings.HasPrefix(ds, ">") {
		return npyHeader{}, fmt.Errorf("npy: big-endian descr %q", ds)
	}
	core := strings.TrimLeft(ds, "<=|")
	dt, err := descrToDType(core)
	if err != nil {
		return npyHeader{}, err
	}
	shapeStr, ok := m["shape"]
	if !ok {
		return npyHeader{}, errors.New("npy: no shape in header")
	}
	shapeStr = strings.Trim(shapeStr, "() ,\t\n\r")
	var dims []int
	if shapeStr != "" {
		its := strings.Split(shapeStr, ",")
		dims = make([]int, 0, len(its))
		for _, it := range its {
			it = strings.TrimSpace(it)
			if it == "" {
				continue
			}
			v, err := strconv.Atoi(it)
			if err != nil {
				return npyHeader{}, fmt.Errorf("npy: bad dim %q: %w", it, err)
			}
			dims = append(dims, v)
		}
	} else {
		dims = []int{}
	}
	return npyHeader{dtype: dt, fortranOrder: fo, shape: dims}, nil
}

func fromReader[T candy.D](shape *candy.Shape, dt candy.DType, r io.Reader) (*Tensor[T], error) {
	n := shape.Numel()
	if n < 0 {
		return nil, errors.New("npy: invalid shape")
	}
	out := make([]T, n)
	dtOut := candy.DTypeOf[T]()
	switch dt {
	case candy.F32:
		src := make([]float32, n)
		if err := binary.Read(r, binary.LittleEndian, src); err != nil {
			return nil, err
		}
		fs := make([]float64, n)
		for i, v := range src {
			fs[i] = float64(v)
		}
		convertFloat64To(out, fs, dtOut)
	case candy.F64:
		src := make([]float64, n)
		if err := binary.Read(r, binary.LittleEndian, src); err != nil {
			return nil, err
		}
		convertFloat64To(out, src, dtOut)
	case candy.U8:
		src := make([]uint8, n)
		if _, err := io.ReadFull(r, src); err != nil {
			return nil, err
		}
		convertInt64To(out, u8ToI64(src), dtOut)
	case candy.U32:
		src := make([]uint32, n)
		if err := binary.Read(r, binary.LittleEndian, src); err != nil {
			return nil, err
		}
		convertInt64To(out, u32ToI64(src), dtOut)
	case candy.I64:
		src := make([]int64, n)
		if err := binary.Read(r, binary.LittleEndian, src); err != nil {
			return nil, err
		}
		convertInt64To(out, src, dtOut)
	case candy.F16:
		return nil, errors.New("npy: f16 read unsupported")
	case candy.BF16:
		return nil, errors.New("npy: bf16 read unsupported")
	default:
		return nil, fmt.Errorf("npy: unsupported dtype %v", dt)
	}
	return New(out, shape, candy.CPU)
}

func ReadNPY[T candy.D](path string) (*Tensor[T], error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h, err := readHeader(f)
	if err != nil {
		return nil, err
	}
	hdr, err := parseHeader(h)
	if err != nil {
		return nil, err
	}
	if hdr.fortranOrder {
		return nil, errors.New("npy: fortran order not supported")
	}
	shape := candy.NewShapeFrom(hdr.shape)
	return fromReader[T](shape, hdr.dtype, f)
}

func MustReadNPY[T candy.D](path string) *Tensor[T] {
	t, err := ReadNPY[T](path)
	if err != nil {
		panic(err)
	}
	return t
}

func headerString(dt candy.DType, dims []int) (string, error) {
	ds, err := dtypeToDescr(dt)
	if err != nil {
		return "", err
	}
	fo := "False"
	var sb strings.Builder
	sb.WriteString("{'descr': '<")
	sb.WriteString(ds)
	sb.WriteString("', 'fortran_order': ")
	sb.WriteString(fo)
	sb.WriteString(", 'shape': (")
	for i, d := range dims {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(strconv.Itoa(d))
	}
	// Trailing comma only for a 1-tuple, per NumPy header spec
	if len(dims) == 1 {
		sb.WriteByte(',')
	}
	sb.WriteString("), }")
	return sb.String(), nil
}

func bytesRepeat(b byte, n int) []byte {
	if n <= 0 {
		return nil
	}
	s := make([]byte, n)
	for i := range s {
		s[i] = b
	}
	return s
}

func (t *Tensor[T]) writeNPYTo(w io.Writer) error {
	if t.Device() != candy.CPU {
		return errors.New("npy: only CPU tensors supported")
	}
	layout := t.Layout()
	if !layout.IsContiguous() {
		return errors.New("npy: only contiguous tensors supported for write")
	}
	// Validate dtype before writing any bytes to avoid partial files
	dt := t.DType()
	switch dt {
	case candy.U8, candy.F32, candy.F64, candy.U32, candy.I64:
		// supported
	default:
		return fmt.Errorf("npy: unsupported write dtype %v", dt)
	}
	start, end, _ := layout.ContiguousOffsets()
	_, _ = w.Write(npyMagic)
	_, _ = w.Write([]byte{1, 0})
	hs, err := headerString(dt, layout.Dims())
	if err != nil {
		return err
	}
	h := []byte(hs)
	// pad accounts for the newline, which is part of the header string length
	pad := (16 - ((len(npyMagic) + 2 + 2 + len(h) + 1) % 16)) % 16
	if pad > 0 {
		h = append(h, bytesRepeat(' ', pad)...)
	}
	h = append(h, '\n')
	var lenle [2]byte
	binary.LittleEndian.PutUint16(lenle[:], uint16(len(h)))
	if _, err := w.Write(lenle[:]); err != nil {
		return err
	}
	if _, err := w.Write(h); err != nil {
		return err
	}
	data := t.Data()
	switch t.DType() {
	case candy.U8:
		b := any(data).([]uint8)
		_, err = w.Write(b[start:end])
		return err
	case candy.F32:
		b := any(data).([]float32)
		return binary.Write(w, binary.LittleEndian, b[start:end])
	case candy.F64:
		b := any(data).([]float64)
		return binary.Write(w, binary.LittleEndian, b[start:end])
	case candy.U32:
		b := any(data).([]uint32)
		return binary.Write(w, binary.LittleEndian, b[start:end])
	case candy.I64:
		b := any(data).([]int64)
		return binary.Write(w, binary.LittleEndian, b[start:end])
	default:
		return fmt.Errorf("npy: unsupported write dtype %v", t.DType())
	}
}

func (t *Tensor[T]) WriteNPY(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return t.writeNPYTo(f)
}

func (t *Tensor[T]) MustWriteNPY(path string) {
	if err := t.WriteNPY(path); err != nil {
		panic(err)
	}
}

func ReadNPZ[T candy.D](path string) (map[string]*Tensor[T], error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	res := make(map[string]*Tensor[T])
	for _, f := range zr.File {
		if filepath.Ext(f.Name) != ".npy" {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		h, err := readHeader(rc)
		if err != nil {
			rc.Close()
			return nil, err
		}
		hdr, err := parseHeader(h)
		if err != nil {
			rc.Close()
			return nil, err
		}
		if hdr.fortranOrder {
			rc.Close()
			return nil, errors.New("npz: fortran order not supported")
		}
		shape := candy.NewShapeFrom(hdr.shape)
		t, err := fromReader[T](shape, hdr.dtype, rc)
		rc.Close()
		if err != nil {
			return nil, err
		}
		name := strings.TrimSuffix(f.Name, npySuffix)
		res[name] = t
	}
	return res, nil
}

func MustReadNPZ(path string) map[string]*Tensor[float32] {
	res, err := ReadNPZ[float32](path)
	if err != nil {
		panic(err)
	}
	return res
}

func ReadNPZByName[T candy.D](path string, names []string) ([]*Tensor[T], error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	idx := make(map[string]*zip.File, len(zr.File))
	for _, f := range zr.File {
		idx[f.Name] = f
	}
	out := make([]*Tensor[T], 0, len(names))
	for _, n := range names {
		fname := n
		if !strings.HasSuffix(fname, npySuffix) {
			fname += npySuffix
		}
		zf, ok := idx[fname]
		if !ok {
			return nil, fmt.Errorf("npz: no array for %s in %s", n, path)
		}
		rc, err := zf.Open()
		if err != nil {
			return nil, err
		}
		h, err := readHeader(rc)
		if err != nil {
			rc.Close()
			return nil, err
		}
		hdr, err := parseHeader(h)
		if err != nil {
			rc.Close()
			return nil, err
		}
		if hdr.fortranOrder {
			rc.Close()
			return nil, errors.New("npz: fortran order not supported")
		}
		shape := candy.NewShapeFrom(hdr.shape)
		t, err := fromReader[T](shape, hdr.dtype, rc)
		rc.Close()
		if err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, nil
}

func MustReadNPZByName(path string, names []string) []*Tensor[float32] {
	res, err := ReadNPZByName[float32](path, names)
	if err != nil {
		panic(err)
	}
	return res
}

func WriteNPZ[T candy.D](path string, items map[string]*Tensor[T]) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	zw := zip.NewWriter(f)
	for name, t := range items {
		w, err := zw.Create(name + npySuffix)
		if err != nil {
			zw.Close()
			return err
		}
		if err := t.writeNPYTo(w); err != nil {
			zw.Close()
			return err
		}
	}
	return zw.Close()
}

func MustWriteNPZ[T candy.D](path string, items map[string]*Tensor[T]) {
	if err := WriteNPZ[T](path, items); err != nil {
		panic(err)
	}
}

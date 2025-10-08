package spark

// FwdAlgo represents forward convolution algorithms supported by cuDNN.
type FwdAlgo int

const (
	FwdAlgoImplicitGEMM        FwdAlgo = iota // 0
	FwdAlgoImplicitPrecompGEMM                // 1
	FwdAlgoGEMM                               // 2
	FwdAlgoDirect                             // 3
	FwdAlgoFFT                                // 4
	FwdAlgoFFTTiling                          // 5
	FwdAlgoWinograd                           // 6
	FwdAlgoWinogradNonfused                   // 7
	FwdAlgoCount                              // 8
)

// Conv1DParams holds parameters for 1D convolution.
type Conv1DParams struct {
	Batch  int
	InLen  int // Input length
	OutCh  int // Output channels
	InCh   int // Input channels
	KSize  int // Kernel size
	Pad    int
	Stride int
	Dilate int
	Algo   *FwdAlgo // Optional cuDNN forward algorithm
}

// OutLen computes the output length for 1D convolution.
func (p Conv1DParams) OutLen() int {
	return (p.InLen+2*p.Pad-p.Dilate*(p.KSize-1)-1)/p.Stride + 1
}

// OutDims returns the output dimensions [batch, out_channels, out_length].
func (p Conv1DParams) OutDims() []int {
	return []int{p.Batch, p.OutCh, p.OutLen()}
}

// ConvT1DParams holds parameters for 1D transposed convolution.
type ConvT1DParams struct {
	Batch  int
	InLen  int // Input length
	OutCh  int // Output channels
	InCh   int // Input channels
	KSize  int // Kernel size
	Pad    int
	OutPad int
	Stride int
	Dilate int
}

// OutLen computes the output length for 1D transposed convolution.
func (p ConvT1DParams) OutLen() int {
	return (p.InLen-1)*p.Stride - 2*p.Pad + p.Dilate*(p.KSize-1) + p.OutPad + 1
}

// OutDims returns the output dimensions [batch, out_channels, out_length].
func (p ConvT1DParams) OutDims() []int {
	return []int{p.Batch, p.OutCh, p.OutLen()}
}

// Conv2DParams holds parameters for 2D convolution.
// Assumes uniform padding, stride, and dilation for height and width.
type Conv2DParams struct {
	Batch  int // Batch size
	InH    int // Input height
	InW    int // Input width
	KH     int // Kernel height
	KW     int // Kernel width
	OutCh  int // Output channels
	InCh   int // Input channels
	Pad    int
	Stride int
	Dilate int
	Algo   *FwdAlgo // Optional cuDNN forward algorithm
}

// OutH computes the output height for 2D convolution.
func (p Conv2DParams) OutH() int {
	return (p.InH+2*p.Pad-p.Dilate*(p.KH-1)-1)/p.Stride + 1
}

// OutW computes the output width for 2D convolution.
func (p Conv2DParams) OutW() int {
	return (p.InW+2*p.Pad-p.Dilate*(p.KW-1)-1)/p.Stride + 1
}

// OutDims returns the output dimensions [batch, out_channels, out_height, out_width].
func (p Conv2DParams) OutDims() []int {
	return []int{p.Batch, p.OutCh, p.OutH(), p.OutW()}
}

// ConvT2DParams holds parameters for 2D transposed convolution.
// Assumes uniform padding, output_padding, stride, and dilation for height and width.
type ConvT2DParams struct {
	Batch  int // Batch size
	InH    int // Input height
	InW    int // Input width
	KH     int // Kernel height
	KW     int // Kernel width
	OutCh  int // Output channels
	InCh   int // Input channels
	Pad    int
	OutPad int
	Stride int
	Dilate int
}

// OutH computes the output height for 2D transposed convolution.
func (p ConvT2DParams) OutH() int {
	return (p.InH-1)*p.Stride + p.Dilate*(p.KH-1) + p.OutPad + 1 - 2*p.Pad
}

// OutW computes the output width for 2D transposed convolution.
func (p ConvT2DParams) OutW() int {
	return (p.InW-1)*p.Stride + p.Dilate*(p.KW-1) + p.OutPad + 1 - 2*p.Pad
}

// OutDims returns the output dimensions [batch, out_channels, out_height, out_width].
func (p ConvT2DParams) OutDims() []int {
	return []int{p.Batch, p.OutCh, p.OutH(), p.OutW()}
}

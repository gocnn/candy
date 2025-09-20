package spark

// CudnnFwdAlgo represents the forward convolution algorithms supported by cuDNN.
type CudnnFwdAlgo int

const (
	CudnnFwdAlgoImplicitGemm        CudnnFwdAlgo = iota // 0
	CudnnFwdAlgoImplicitPrecompGemm                     // 1
	CudnnFwdAlgoGemm                                    // 2
	CudnnFwdAlgoDirect                                  // 3
	CudnnFwdAlgoFft                                     // 4
	CudnnFwdAlgoFftTiling                               // 5
	CudnnFwdAlgoWinograd                                // 6
	CudnnFwdAlgoWinogradNonFused                        // 7
	CudnnFwdAlgoCount                                   // 8
)

// ParamsConv1D holds parameters for 1D convolution.
type ParamsConv1D struct {
	BSize        int
	LIn          int // Input length
	COut         int // Output channels
	CIn          int // Input channels
	KSize        int // Kernel size
	Padding      int
	Stride       int
	Dilation     int
	CudnnFwdAlgo *CudnnFwdAlgo // Optional cuDNN algorithm
}

// LOut computes the output length for 1D convolution.
func (p ParamsConv1D) LOut() int {
	return (p.LIn+2*p.Padding-p.Dilation*(p.KSize-1)-1)/p.Stride + 1
}

// OutDims returns the output dimensions [batch, out_channels, out_length].
func (p ParamsConv1D) OutDims() []int {
	return []int{p.BSize, p.COut, p.LOut()}
}

// ParamsConvTranspose1D holds parameters for 1D transposed convolution (deconvolution).
type ParamsConvTranspose1D struct {
	BSize         int
	LIn           int // Input length
	COut          int // Output channels
	CIn           int // Input channels
	KSize         int // Kernel size
	Padding       int
	OutputPadding int
	Stride        int
	Dilation      int
}

// LOut computes the output length for 1D transposed convolution.
func (p ParamsConvTranspose1D) LOut() int {
	return (p.LIn-1)*p.Stride - 2*p.Padding + p.Dilation*(p.KSize-1) + p.OutputPadding + 1
}

// OutDims returns the output dimensions [batch, out_channels, out_length].
func (p ParamsConvTranspose1D) OutDims() []int {
	return []int{p.BSize, p.COut, p.LOut()}
}

// ParamsConv2D holds parameters for 2D convolution.
// Assumes uniform padding, stride, and dilation for height and width.
type ParamsConv2D struct {
	BSize        int // Batch size
	IH           int // Input height
	IW           int // Input width
	KH           int // Kernel height
	KW           int // Kernel width
	COut         int // Output channels
	CIn          int // Input channels
	Padding      int
	Stride       int
	Dilation     int
	CudnnFwdAlgo *CudnnFwdAlgo // Optional cuDNN algorithm
}

// OutH computes the output height for 2D convolution.
func (p ParamsConv2D) OutH() int {
	return (p.IH+2*p.Padding-p.Dilation*(p.KH-1)-1)/p.Stride + 1
}

// OutW computes the output width for 2D convolution.
func (p ParamsConv2D) OutW() int {
	return (p.IW+2*p.Padding-p.Dilation*(p.KW-1)-1)/p.Stride + 1
}

// OutDims returns the output dimensions [batch, out_channels, out_height, out_width].
func (p ParamsConv2D) OutDims() []int {
	return []int{p.BSize, p.COut, p.OutH(), p.OutW()}
}

// ParamsConvTranspose2D holds parameters for 2D transposed convolution (deconvolution).
// Assumes uniform padding, output_padding, stride, and dilation for height and width.
type ParamsConvTranspose2D struct {
	BSize         int // Batch size
	IH            int // Input height
	IW            int // Input width
	KH            int // Kernel height
	KW            int // Kernel width
	COut          int // Output channels
	CIn           int // Input channels
	Padding       int
	OutputPadding int
	Stride        int
	Dilation      int
}

// OutH computes the output height for 2D transposed convolution.
func (p ParamsConvTranspose2D) OutH() int {
	return (p.IH-1)*p.Stride + p.Dilation*(p.KH-1) + p.OutputPadding + 1 - 2*p.Padding
}

// OutW computes the output width for 2D transposed convolution.
func (p ParamsConvTranspose2D) OutW() int {
	return (p.IW-1)*p.Stride + p.Dilation*(p.KW-1) + p.OutputPadding + 1 - 2*p.Padding
}

// OutDims returns the output dimensions [batch, out_channels, out_height, out_width].
func (p ParamsConvTranspose2D) OutDims() []int {
	return []int{p.BSize, p.COut, p.OutH(), p.OutW()}
}

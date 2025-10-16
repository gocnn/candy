import torch
import torch.nn as nn
import numpy as np


def save_npy(path, arr):
    np.save(path, arr.astype(np.float32))


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Problem size
    B, Cin, H, W = 2, 1, 8, 8
    Cout, KH, KW = 4, 3, 3
    stride, pad, dilation = 1, 1, 1

    # Inputs and layer
    x = torch.randn(B, Cin, H, W, dtype=torch.float32, requires_grad=True)
    conv = nn.Conv2d(Cin, Cout, (KH, KW), stride=stride, padding=pad, dilation=dilation, bias=True)

    # Forward
    y = conv(x)
    loss = y.sum()  # produces grad_out = 1 everywhere

    # Backward
    loss.backward()

    # Save tensors and reference grads
    save_npy("x.npy", x.detach().cpu().numpy())
    save_npy("w.npy", conv.weight.detach().cpu().numpy())
    save_npy("b.npy", conv.bias.detach().cpu().numpy())

    save_npy("dx_ref.npy", x.grad.detach().cpu().numpy())
    save_npy("dw_ref.npy", conv.weight.grad.detach().cpu().numpy())
    save_npy("db_ref.npy", conv.bias.grad.detach().cpu().numpy())

    print("Saved reference files: x.npy, w.npy, b.npy, dx_ref.npy, dw_ref.npy, db_ref.npy")
    print("Shapes:")
    print(" x:", tuple(x.shape))
    print(" w:", tuple(conv.weight.shape))
    print(" b:", tuple(conv.bias.shape))


if __name__ == "__main__":
    main()

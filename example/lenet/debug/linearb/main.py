import torch
import torch.nn as nn
import numpy as np


def save(path, arr):
    np.save(path, arr.astype(np.float32))


def main():
    torch.manual_seed(123)
    np.random.seed(123)

    B, In, Out = 3, 5, 4
    x = torch.randn(B, In, dtype=torch.float32, requires_grad=True)
    lin = nn.Linear(In, Out, bias=True)

    y = lin(x)
    loss = y.sum()  # grad_out = 1
    loss.backward()

    save("x.npy", x.detach().cpu().numpy())
    save("w.npy", lin.weight.detach().cpu().numpy())  # [Out, In]
    save("b.npy", lin.bias.detach().cpu().numpy())    # [Out]

    save("dx_ref.npy", x.grad.detach().cpu().numpy())
    save("dw_ref.npy", lin.weight.grad.detach().cpu().numpy())
    save("db_ref.npy", lin.bias.grad.detach().cpu().numpy())

    print("saved: x.npy, w.npy, b.npy, dx_ref.npy, dw_ref.npy, db_ref.npy")
    print("shapes:", tuple(x.shape), tuple(lin.weight.shape), tuple(lin.bias.shape))


if __name__ == "__main__":
    main()
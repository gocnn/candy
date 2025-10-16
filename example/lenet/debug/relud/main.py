import torch
import torch.nn.functional as F
import numpy as np


def save(path, arr):
    np.save(path, arr.astype(np.float32))


def main():
    torch.manual_seed(7)
    np.random.seed(7)

    B, C, H, W = 3, 2, 5, 4

    x = torch.randn(B, C, H, W, dtype=torch.float32, requires_grad=True)
    y = F.relu(x)
    loss = y.sum()
    loss.backward()

    save("x.npy", x.detach().cpu().numpy())
    save("y_ref.npy", y.detach().cpu().numpy())
    save("dx_ref.npy", x.grad.detach().cpu().numpy())
    print("saved: x.npy, y_ref.npy, dx_ref.npy")
    print("shapes:", tuple(x.shape), tuple(y.shape))


if __name__ == "__main__":
    main()
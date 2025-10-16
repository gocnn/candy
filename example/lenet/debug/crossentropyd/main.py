import torch
import torch.nn.functional as F
import numpy as np


def save(path, arr):
    np.save(path, arr.astype(np.float32))


def main():
    torch.manual_seed(7)
    np.random.seed(7)

    B, C = 4, 7
    x = torch.randn(B, C, dtype=torch.float32, requires_grad=True)
    y = torch.randint(0, C, (B,), dtype=torch.long)
    loss = F.cross_entropy(x, y, reduction='mean')
    loss.backward()

    save("x.npy", x.detach().cpu().numpy())
    # Save y as float32 indices to match Go gather(T) API
    save("y.npy", y.detach().cpu().numpy().astype(np.float32))
    save("dx_ref.npy", x.grad.detach().cpu().numpy())
    print("saved: x.npy, y.npy, dx_ref.npy")
    print("shapes:", tuple(x.shape))


if __name__ == "__main__":
    main()
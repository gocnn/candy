import torch
import torch.nn.functional as F
import numpy as np


def save(path, arr):
    np.save(path, arr.astype(np.float32))


def main():
    torch.manual_seed(7)
    np.random.seed(7)

    B, C = 4, 7
    x = torch.randn(B, C, dtype=torch.float32)
    y = torch.randint(0, C, (B,), dtype=torch.long)
    loss = F.cross_entropy(x, y, reduction='mean')

    save("x.npy", x.detach().cpu().numpy())
    # Save y as float32 indices to match Go gather(T) API
    save("y.npy", y.detach().cpu().numpy().astype(np.float32))
    save("loss_ref.npy", np.array([loss.detach().cpu().item()], dtype=np.float32))
    print("saved: x.npy, y.npy, loss_ref.npy")
    print("shapes:", tuple(x.shape))


if __name__ == "__main__":
    main()
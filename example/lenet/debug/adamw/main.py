import torch
import numpy as np


def max_abs_diff(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    return float((a - b).abs().max().item())


def test_manual_step():
    x = torch.tensor([2.0, -1.5, 3.0], dtype=torch.float32, requires_grad=True)
    g = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)

    np.save("x_step.npy", x.detach().cpu().numpy())
    np.save("g_step.npy", g.detach().cpu().numpy())

    opt = torch.optim.AdamW([x], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    x.grad = g.clone()
    opt.step()

    x_ref = x.detach().cpu().numpy()
    np.save("x_step_ref.npy", x_ref)
    print(f"manual step: x_ref={x_ref.tolist()}")


def test_optimize_mse():
    x = torch.tensor([0.5, -0.7, 1.2], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)

    np.save("x_opt.npy", x.detach().cpu().numpy())
    np.save("y_opt.npy", y.detach().cpu().numpy())

    opt = torch.optim.AdamW([x], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    loss = ((x - y) ** 2).sum()
    loss.backward()
    opt.step()

    x_ref = x.detach().cpu().numpy()
    np.save("x_opt_ref.npy", x_ref)
    print(f"optimize MSE: x_ref={x_ref.tolist()}")


def main():
    print("PyTorch AdamW reference")
    test_manual_step()
    test_optimize_mse()


if __name__ == "__main__":
    main()
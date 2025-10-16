import torch

def max_abs_diff(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    return float((a - b).abs().max().item())


def test_manual_step():
    x = torch.tensor([2.0, 2.0], requires_grad=True)
    opt = torch.optim.SGD([x], lr=0.1, momentum=0.0, weight_decay=0.0)
    # Manually set gradient then step
    x.grad = torch.ones_like(x)
    opt.step()
    expected = torch.tensor([1.9, 1.9])
    print(f"manual step: got={x.detach().tolist()} expected={expected.tolist()} diff={max_abs_diff(x, expected):.6g}")


def test_optimize_mse():
    x = torch.tensor([3.0, 3.0], requires_grad=True)
    y = torch.ones(2)
    opt = torch.optim.SGD([x], lr=0.1, momentum=0.0, weight_decay=0.0)
    loss = ((x - y) ** 2).sum()
    loss.backward()
    opt.step()
    expected = torch.tensor([2.6, 2.6])
    print(f"optimize MSE: got={x.detach().tolist()} expected={expected.tolist()} diff={max_abs_diff(x, expected):.6g}")


def main():
    print("PyTorch SGD accuracy tests")
    test_manual_step()
    test_optimize_mse()


if __name__ == "__main__":
    main()
import argparse, os, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# Model identical to your lenet_debug.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
        self.fc1 = nn.Linear(9216, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        # Standard forward (not used for dumping)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(arr, dtype="<f4"), allow_pickle=False)


def state_to_npz(model: Net, path: str):
    # Export weights in C-order little-endian float32
    payload = {
        "conv1.weight": model.conv1.weight.detach().cpu().numpy().astype("<f4"),
        "conv1.bias": model.conv1.bias.detach().cpu().numpy().astype("<f4"),
        "conv2.weight": model.conv2.weight.detach().cpu().numpy().astype("<f4"),
        "conv2.bias": model.conv2.bias.detach().cpu().numpy().astype("<f4"),
        "fc1.weight": model.fc1.weight.detach().cpu().numpy().astype("<f4"),
        "fc1.bias": model.fc1.bias.detach().cpu().numpy().astype("<f4"),
        "fc2.weight": model.fc2.weight.detach().cpu().numpy().astype("<f4"),
        "fc2.bias": model.fc2.bias.detach().cpu().numpy().astype("<f4"),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **payload)


def npz_to_state(path: str, model: Net):
    z = np.load(path)
    with torch.no_grad():
        model.conv1.weight.copy_(torch.from_numpy(z["conv1.weight"]))
        model.conv1.bias.copy_(torch.from_numpy(z["conv1.bias"]))
        model.conv2.weight.copy_(torch.from_numpy(z["conv2.weight"]))
        model.conv2.bias.copy_(torch.from_numpy(z["conv2.bias"]))
        model.fc1.weight.copy_(torch.from_numpy(z["fc1.weight"]))
        model.fc1.bias.copy_(torch.from_numpy(z["fc1.bias"]))
        model.fc2.weight.copy_(torch.from_numpy(z["fc2.weight"]))
        model.fc2.bias.copy_(torch.from_numpy(z["fc2.bias"]))


def set_deterministic(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_sample(device, idx: int):
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    x, y = ds[idx]
    x = x.unsqueeze(0).to(device)  # [1, 1, 28, 28]
    y = torch.tensor([y], device=device)
    return x, y


def _assert_shape(t: torch.Tensor, *dims: int):
    s = tuple(t.shape)
    if s != tuple(dims):
        raise RuntimeError(f"shape mismatch {s} vs {dims}")


def dump_activations(model: Net, x, y, out_dir: str):
    model.eval()
    with torch.no_grad():
        _assert_shape(x, 1, 1, 28, 28)
        save_npy(os.path.join(out_dir, "00_input.npy"), x.cpu().numpy())

        x1 = model.conv1(x)
        _assert_shape(x1, 1, 32, 26, 26)
        save_npy(os.path.join(out_dir, "10_conv1_out.npy"), x1.cpu().numpy())
        x2 = F.relu(x1)
        _assert_shape(x2, 1, 32, 26, 26)
        save_npy(os.path.join(out_dir, "11_relu1_out.npy"), x2.cpu().numpy())

        x3 = model.conv2(x2)
        _assert_shape(x3, 1, 64, 24, 24)
        save_npy(os.path.join(out_dir, "20_conv2_out.npy"), x3.cpu().numpy())
        x4 = F.relu(x3)
        _assert_shape(x4, 1, 64, 24, 24)
        save_npy(os.path.join(out_dir, "21_relu2_out.npy"), x4.cpu().numpy())

        x5 = F.max_pool2d(x4, 2)
        _assert_shape(x5, 1, 64, 12, 12)
        save_npy(os.path.join(out_dir, "30_maxpool_out.npy"), x5.cpu().numpy())

        x6 = torch.flatten(x5, 1)
        _assert_shape(x6, 1, 9216)
        save_npy(os.path.join(out_dir, "40_flatten_out.npy"), x6.cpu().numpy())

        x7 = model.fc1(x6)
        _assert_shape(x7, 1, 128)
        save_npy(os.path.join(out_dir, "50_fc1_out.npy"), x7.cpu().numpy())
        x8 = F.relu(x7)
        _assert_shape(x8, 1, 128)
        save_npy(os.path.join(out_dir, "51_relu3_out.npy"), x8.cpu().numpy())

        x9 = model.fc2(x8)
        _assert_shape(x9, 1, 10)
        save_npy(os.path.join(out_dir, "60_fc2_out.npy"), x9.cpu().numpy())
        x10 = F.log_softmax(x9, 1)
        _assert_shape(x10, 1, 10)
        save_npy(os.path.join(out_dir, "70_logsoftmax_out.npy"), x10.cpu().numpy())

        save_npy(os.path.join(out_dir, "target.npy"), y.detach().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--weights-out", type=str, default="artifacts/weights.npz")
    ap.add_argument("--weights-in", type=str, default="")
    ap.add_argument("--acts-out", type=str, default="artifacts/py_out")
    args = ap.parse_args()

    set_deterministic(args.seed)
    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    model = Net().to(device)

    # If provided, load existing weights to guarantee exact cross-language equality
    if args.weights_in:
        npz_to_state(args.weights_in, model)
    else:
        # Otherwise, use seeded init and export so Go can consume exactly the same values
        state_to_npz(model, args.weights_out)

    x, y = load_sample(device, args.sample_index)
    dump_activations(model, x, y, args.acts_out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import numpy as np
import torch


def load_state_dict(args):
    if args.pth:
        sd = torch.load(args.pth, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        return sd
    else:
        try:
            from torchvision.models import alexnet, AlexNet_Weights

            model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            return model.state_dict()
        except Exception:
            # Default to downloading from URL
            sd = torch.hub.load_state_dict_from_url(
                "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
                map_location="cpu",
                check_hash=True,
            )
            return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pth", type=str, default="", help="path to alexnet-owt-7be5be79.pth"
    )
    ap.add_argument("--out", type=str, default="alexnet.npz", help="output npz path")
    args = ap.parse_args()

    sd = load_state_dict(args)

    # PyTorch -> NPZ
    mapping = {
        "features.0.weight": "c1_w",
        "features.0.bias": "c1_b",
        "features.3.weight": "c2_w",
        "features.3.bias": "c2_b",
        "features.6.weight": "c3_w",
        "features.6.bias": "c3_b",
        "features.8.weight": "c4_w",
        "features.8.bias": "c4_b",
        "features.10.weight": "c5_w",
        "features.10.bias": "c5_b",
        "classifier.1.weight": "f1_w",
        "classifier.1.bias": "f1_b",
        "classifier.4.weight": "f2_w",
        "classifier.4.bias": "f2_b",
        "classifier.6.weight": "f3_w",
        "classifier.6.bias": "f3_b",
    }

    arrays = {}
    for k_pt, k_npz in mapping.items():
        if k_pt not in sd:
            raise KeyError(f"missing key in state_dict: {k_pt}")
        arr = sd[k_pt].cpu().numpy().astype(np.float32, copy=False)
        arrays[k_npz] = arr

    np.savez(args.out, **arrays)
    print(f"wrote {args.out} with {len(arrays)} arrays")


if __name__ == "__main__":
    main()

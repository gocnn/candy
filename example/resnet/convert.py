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
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            return model.state_dict()
        except Exception:
            sd = torch.hub.load_state_dict_from_url(
                "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
                map_location="cpu",
                check_hash=True,
            )
            return sd


def add_bn(arrays, sd, base_pt, base_npz):
    arrays[f"{base_npz}_w"] = sd[f"{base_pt}.weight"].cpu().numpy().astype(np.float32, copy=False)
    arrays[f"{base_npz}_b"] = sd[f"{base_pt}.bias"].cpu().numpy().astype(np.float32, copy=False)
    arrays[f"{base_npz}_rm"] = sd[f"{base_pt}.running_mean"].cpu().numpy().astype(np.float32, copy=False)
    arrays[f"{base_npz}_rv"] = sd[f"{base_pt}.running_var"].cpu().numpy().astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", type=str, default="", help="path to resnet50 .pth")
    ap.add_argument("--out", type=str, default="resnet50.npz", help="output npz path")
    args = ap.parse_args()

    sd = load_state_dict(args)

    arrays = {}
    # stem
    arrays["conv1_w"] = sd["conv1.weight"].cpu().numpy().astype(np.float32, copy=False)
    add_bn(arrays, sd, "bn1", "bn1")

    # layers: ResNet50 block counts
    layer_blocks = {1: 3, 2: 4, 3: 6, 4: 3}
    for l, nblk in layer_blocks.items():
        for i in range(nblk):
            pref_pt = f"layer{l}.{i}"
            pref_npz = f"layer{l}_{i}"
            arrays[f"{pref_npz}_conv1_w"] = sd[f"{pref_pt}.conv1.weight"].cpu().numpy().astype(np.float32, copy=False)
            add_bn(arrays, sd, f"{pref_pt}.bn1", f"{pref_npz}_bn1")
            arrays[f"{pref_npz}_conv2_w"] = sd[f"{pref_pt}.conv2.weight"].cpu().numpy().astype(np.float32, copy=False)
            add_bn(arrays, sd, f"{pref_pt}.bn2", f"{pref_npz}_bn2")
            arrays[f"{pref_npz}_conv3_w"] = sd[f"{pref_pt}.conv3.weight"].cpu().numpy().astype(np.float32, copy=False)
            add_bn(arrays, sd, f"{pref_pt}.bn3", f"{pref_npz}_bn3")
            ds0 = f"{pref_pt}.downsample.0.weight"
            if ds0 in sd:
                arrays[f"{pref_npz}_down_conv_w"] = sd[ds0].cpu().numpy().astype(np.float32, copy=False)
                add_bn(arrays, sd, f"{pref_pt}.downsample.1", f"{pref_npz}_down_bn")

    # fc
    arrays["fc_w"] = sd["fc.weight"].cpu().numpy().astype(np.float32, copy=False)
    arrays["fc_b"] = sd["fc.bias"].cpu().numpy().astype(np.float32, copy=False)

    np.savez(args.out, **arrays)
    print(f"wrote {args.out} with {len(arrays)} arrays")


if __name__ == "__main__":
    main()
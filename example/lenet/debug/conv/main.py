import torch
import torch.nn as nn
import numpy as np

def main():
    print("=== Real LeNet Conv1 Test ===")
    
    # 加载实际权重
    weights = np.load("../artifacts/weights.npz")
    real_input = np.load("../artifacts/py_out/00_input.npy")
    
    print(f"Real input shape: {real_input.shape}")
    print(f"Conv1 weight shape: {weights['conv1.weight'].shape}")
    print(f"Conv1 bias shape: {weights['conv1.bias'].shape}")
    
    # 转为torch
    x = torch.from_numpy(real_input)
    
    # 创建卷积层
    conv = nn.Conv2d(1, 32, 3, stride=1, padding=0, bias=True)
    
    # 加载实际权重
    with torch.no_grad():
        conv.weight.data = torch.from_numpy(weights['conv1.weight'])
        conv.bias.data = torch.from_numpy(weights['conv1.bias'])
    
    print(f"\nWeight first 5: {conv.weight.data.flatten()[:5]}")
    print(f"Bias first 5: {conv.bias.data[:5]}")
    
    # 前向传播
    y = conv(x)
    print(f"\nPyTorch output shape: {y.shape}")
    print(f"PyTorch output range: {y.min():.6f} to {y.max():.6f}")
    print(f"PyTorch output mean/std: {y.mean():.6f} / {y.std():.6f}")
    print(f"PyTorch first 5: {y.flatten()[:5]}")
    
    # 保存用于Go测试
    np.save("real_input.npy", x.detach().numpy())
    np.save("real_weight.npy", conv.weight.detach().numpy())
    np.save("real_bias.npy", conv.bias.detach().numpy())
    np.save("real_output.npy", y.detach().numpy())
    
    print("\nReal test files saved for Go comparison")

if __name__ == "__main__":
    main()
# ============================================================
# DenseNet DAE 
# ============================================================
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- 设备（MPS/CUDA/CPU 全支持） ----------------
if torch.backends.mps.is_available():
    device = torch.device("mps");  print("Device: mps")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("Device: cuda")
else:
    device = torch.device("cpu");  print("Device: cpu")

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- 配置 ----------------
CACHE_FILE = "prepared_dae_real.npz"
CKPT_FILE  = "densenet_dae_real.pt"
BATCH_SIZE = 128          # DenseNet 参数量小，可以放大 batch
EPOCHS     = 50
LR         = 1e-3

# DenseNet 结构
INIT_CH       = 32        # stem 输出通道
GROWTH_RATE   = 16        # 每个 DenseLayer 新增通道
BLOCK_LAYERS  = (4, 4, 4) # 3 个 DenseBlock，每块 4 层
STEM_KERNEL   = 15        # 加大的 stem kernel，扩大感受野
DROPOUT       = 0.1

# ============================================================
# 1. 加载缓存（与 Transformer / Capsule 完全相同的 npz）
# ============================================================
assert os.path.exists(CACHE_FILE), f"找不到 {CACHE_FILE}"
print(f">> 加载缓存 {CACHE_FILE}")
d = np.load(CACHE_FILE)
X_train_noisy = d["X_train_noisy"]; X_train_clean = d["X_train_clean"]
X_test_noisy  = d["X_test_noisy"];  X_test_clean  = d["X_test_clean"]
N_GENES = X_train_noisy.shape[1]
print(f"   N_GENES={N_GENES} | Train={len(X_train_noisy)} | Test={len(X_test_noisy)}")

class DAEDataset(Dataset):
    def __init__(self, xn, xc):
        self.xn = torch.from_numpy(xn); self.xc = torch.from_numpy(xc)
    def __len__(self):  return len(self.xn)
    def __getitem__(self, i): return self.xn[i], self.xc[i]

train_dl = DataLoader(DAEDataset(X_train_noisy, X_train_clean),
                      batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dl  = DataLoader(DAEDataset(X_test_noisy, X_test_clean),
                      batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 2. DenseNet 组件
# ============================================================
class DenseLayer(nn.Module):
    """BN-ReLU-Conv1x1 → BN-ReLU-Conv3x3 → concat"""
    def __init__(self, in_ch, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.bn1   = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, bn_size * growth_rate, 1, bias=False)
        self.bn2   = nn.BatchNorm1d(bn_size * growth_rate)
        self.conv2 = nn.Conv1d(bn_size * growth_rate, growth_rate,
                               3, padding=1, bias=False)
        self.drop  = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.drop(h)
        return torch.cat([x, h], dim=1)


class DenseBlock1D(nn.Module):
    def __init__(self, in_ch, num_layers, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.block = nn.Sequential(*[
            DenseLayer(in_ch + i * growth_rate, growth_rate, bn_size, drop_rate)
            for i in range(num_layers)])

    def forward(self, x): return self.block(x)


class DenseNetDAE(nn.Module):
    """
    输入 noisy [B, G] → 输出 clean [B, G]
    特征: 保持长度 G 不变（无 pooling），只做通道压缩
    """
    def __init__(self, init_ch=32, growth_rate=16,
                 block_layers=(4, 4, 4), drop_rate=0.1, stem_kernel=15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, init_ch, stem_kernel,
                      padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(init_ch),
            nn.ReLU(inplace=True))

        mods, ch = [], init_ch
        for nl in block_layers:
            mods.append(DenseBlock1D(ch, nl, growth_rate, drop_rate=drop_rate))
            ch += nl * growth_rate
            # Transition: 1×1 压缩通道，保长度
            mods.append(nn.Sequential(
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(ch, ch // 2, 1, bias=False)))
            ch = ch // 2
        self.blocks = nn.Sequential(*mods)

        self.head = nn.Sequential(
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, 1, 1))

    def forward(self, x):
        h = x.unsqueeze(1)                # [B, 1, G]
        h = self.stem(h)
        h = self.blocks(h)
        return self.head(h).squeeze(1)    # [B, G]


model = DenseNetDAE(INIT_CH, GROWTH_RATE, BLOCK_LAYERS, DROPOUT, STEM_KERNEL).to(device)
print(f">> DenseNet 参数量: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

# ---------------- Shape 自检 ----------------
with torch.no_grad():
    _x = torch.randn(2, N_GENES).to(device)
    _o = model(_x)
print(f"   shape check: out={tuple(_o.shape)}")
assert _o.shape == (2, N_GENES), "输出 shape 错误"

# ============================================================
# 3. 训练
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\n>> 开始训练 ({EPOCHS} epochs)")
for epoch in range(EPOCHS):
    model.train()
    tot, cnt = 0.0, 0
    t0 = time.time()
    for xn, xc in train_dl:
        xn, xc = xn.to(device), xc.to(device)
        pred = model(xn)
        loss = F.mse_loss(pred, xc)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += loss.item(); cnt += 1
    scheduler.step()
    print(f"[Epoch {epoch+1:02d}/{EPOCHS}] train MSE = {tot/cnt:.6f}  ({time.time()-t0:.1f}s)")

torch.save({
    "model":     model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch":     EPOCHS,
}, CKPT_FILE)
print(f">> 权重已保存 {CKPT_FILE}")

# ============================================================
# 4. 测试集 MSE
# ============================================================
@torch.no_grad()
def evaluate(model, dl):
    model.eval(); s, n = 0.0, 0
    for xn, xc in dl:
        xn, xc = xn.to(device), xc.to(device)
        pred = model(xn)
        s += ((pred - xc) ** 2).sum().item()
        n += pred.numel()
    return s / n

test_mse = evaluate(model, test_dl)
baseline_zero     = float(np.mean(X_test_clean ** 2))
baseline_identity = float(np.mean((X_test_noisy - X_test_clean) ** 2))

print("\n" + "="*55)
print("TEST SET RESULTS (DenseNet DAE)")
print("="*55)
print(f"DenseNet DAE Test MSE          : {test_mse:.6f}")
print(f"基线 (预测全0)                 : {baseline_zero:.6f}")
print(f"基线 (恒等: noisy→output)      : {baseline_identity:.6f}")
if test_mse < baseline_identity:
    print(f"✅ 相比恒等基线降低 {(1 - test_mse/baseline_identity)*100:.1f}%")
else:
    print("⚠️ 未超过恒等基线")
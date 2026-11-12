# ============================================================
# ResNet DAE 
# ============================================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- 设备 ----------------
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
CKPT_FILE  = "resnet_dae_real.pt"
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 1e-3

# ResNet 结构
BASE_CH      = 64            # 起始通道
CHANNELS     = (64, 128, 256)   # 3 个 stage
BLOCKS_PER_STAGE = (2, 2, 2)    # 每 stage 2 个 BasicBlock
STEM_KERNEL  = 15
DROPOUT      = 0.1

# ============================================================
# 1. 加载缓存
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
# 2. 1D ResNet
# ============================================================
class BasicBlock1D(nn.Module):
    """ResNet BasicBlock: 2×(Conv3x3 → BN → ReLU) + skip"""
    def __init__(self, in_ch, out_ch, stride=1, drop_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        # skip: 通道或步长变化时用 1×1 投影
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.drop(h)
        return F.relu(h + self.shortcut(x))


class ResNet1D_DAE(nn.Module):
    """输入 noisy [B, G] → 输出 clean [B, G]，全程保持长度 G"""
    def __init__(self, base_ch=64, channels=(64,128,256),
                 blocks=(2,2,2), stem_kernel=15, drop_rate=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, stem_kernel, padding=stem_kernel//2, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True))

        stages = []
        in_ch  = base_ch
        for out_ch, n_blocks in zip(channels, blocks):
            for _ in range(n_blocks):
                stages.append(BasicBlock1D(in_ch, out_ch, stride=1, drop_rate=drop_rate))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # 输出: 1×1 Conv 回到 1 通道
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_ch, 1, 1))

    def forward(self, x):
        h = x.unsqueeze(1)                # [B, 1, G]
        h = self.stem(h)
        h = self.stages(h)
        return self.head(h).squeeze(1)    # [B, G]

model = ResNet1D_DAE(BASE_CH, CHANNELS, BLOCKS_PER_STAGE, STEM_KERNEL, DROPOUT).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f">> ResNet 参数量: {n_params/1e6:.3f}M")

# ============================================================
# 3. 训练
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\n>> 开始训练 ({EPOCHS} epochs)")
for epoch in range(EPOCHS):
    model.train()
    tot, cnt = 0.0, 0
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
    print(f"[Epoch {epoch+1:02d}/{EPOCHS}] train MSE = {tot/cnt:.6f}")

torch.save({
    "model":     model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch":     EPOCHS,
}, CKPT_FILE)
print(f">> 权重已保存 {CKPT_FILE}")

# ============================================================
# 4. 测试集评估
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

print("\n" + "="*60)
print("TEST SET RESULTS (ResNet DAE · 真实数据)")
print("="*60)
print(f"ResNet DAE Test MSE            : {test_mse:.6f}")
print(f"基线 (预测全0)                 : {baseline_zero:.6f}")
print(f"基线 (恒等: noisy→output)      : {baseline_identity:.6f}")
if test_mse < baseline_identity:
    print(f"✅ 相比恒等基线降低 {(1-test_mse/baseline_identity)*100:.1f}%")
else:
    print("⚠️ 未超过恒等基线 → 学习不足")

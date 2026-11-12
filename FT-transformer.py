import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- 设备（FT-Transformer 是纯 TransformerEncoder，MPS 完全支持） ----------------
if torch.backends.mps.is_available():
    device = torch.device("mps");  print("Device: mps")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("Device: cuda")
else:
    device = torch.device("cpu");  print("Device: cpu")

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- 配置（与普通 Transformer 完全一致） ----------------
CACHE_FILE = "prepared_dae_real.npz"
CKPT_FILE  = "ft_transformer_dae_real.pt"
BATCH_SIZE = 64
EPOCHS     = 50         # 普通 Transformer 有效训练 15 epoch，FT 用 30 相对公平
LR         = 1e-3

D_MODEL    = 64
NHEAD      = 4
NUM_LAYERS = 2
DIM_FF     = 128
DROPOUT    = 0.1


# ============================================================
# 1. 加载缓存（与普通 Transformer 完全相同）
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
# 2. FT-Transformer 的核心：Feature Tokenizer
# ============================================================
class FeatureTokenizer(nn.Module):
    """
    每个基因位置独立一组 (weight, bias)，把标量 x_j 投影到 d_token 维。
    token_j = x_j * w_j + b_j,  w_j,b_j shape=[d_token] per gene
    这与普通 Transformer 的"共享 Linear(1→D) + 位置编码"本质不同。
    """
    def __init__(self, n_features, d_token):
        super().__init__()
        # [n_features, d_token]: 第 j 行只作用于第 j 个基因
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x):
        # x: [B, G] → [B, G, 1] * [G, D] + [G, D] → [B, G, D]
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformerDAE(nn.Module):
    """输入 noisy [B,G] → Feature Tokenizer → Transformer → 逐 token 重建 [B,G]"""
    def __init__(self, n_genes, d_token=192, nhead=8,
                 num_layers=6, dim_ff=384, dropout=0.1, use_pos_emb=True):
        super().__init__()
        self.tokenizer   = FeatureTokenizer(n_genes, d_token)
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(1, n_genes, d_token) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_token)
        self.head    = nn.Linear(d_token, 1)

    def forward(self, x):
        h = self.tokenizer(x)                 # [B, G, D] 逐基因 tokenizer
        if self.use_pos_emb:
            h = h + self.pos_emb
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h).squeeze(-1)       # [B, G]


model = FTTransformerDAE(N_GENES, D_MODEL, NHEAD, NUM_LAYERS,
                         DIM_FF, DROPOUT, USE_POS_EMB).to(device)
print(f">> FT-Transformer 参数量: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

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
print("TEST SET RESULTS (FT-Transformer DAE)")
print("="*55)
print(f"FT-Transformer DAE Test MSE    : {test_mse:.6f}")
print(f"基线 (预测全0)                 : {baseline_zero:.6f}")
print(f"基线 (恒等: noisy→output)      : {baseline_identity:.6f}")
if test_mse < baseline_identity:
    print(f"✅ 相比恒等基线降低 {(1 - test_mse/baseline_identity)*100:.1f}%")
else:
    print("⚠️ 未超过恒等基线")

# ============================================================
# 5. 与普通 Transformer 对比
# ============================================================
PLAIN_TRANSFORMER_MSE = 0.052084   # 你之前 15 epoch 续训后的结果
print("\n" + "="*55)
print("架构对比")
print("="*55)
print(f"普通 Transformer DAE  : {PLAIN_TRANSFORMER_MSE:.6f}")
print(f"FT-Transformer  DAE  : {test_mse:.6f}")
if test_mse < PLAIN_TRANSFORMER_MSE:
    print(f"🏆 FT-Transformer 更优，降低 {(1 - test_mse/PLAIN_TRANSFORMER_MSE)*100:.1f}%")
else:
    print(f"普通 Transformer 更优，FT 高出 {(test_mse/PLAIN_TRANSFORMER_MSE-1)*100:.1f}%")
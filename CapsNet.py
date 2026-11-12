# ============================================================
# Capsule Network DAE 
# ============================================================
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if torch.backends.mps.is_available():
    device = torch.device("mps");  print("Device: mps")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("Device: cuda")
else:
    device = torch.device("cpu");  print("Device: cpu")


SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

CACHE_FILE = "prepared_dae_real.npz"
CKPT_FILE  = "capsule_dae_real.pt"
BATCH_SIZE = 16
EPOCHS     = 50
LR         = 1e-3

PRIMARY_N     = 16
PRIMARY_D     = 8
DIGIT_N       = 32
DIGIT_D       = 16
ROUTING_ITERS = 3
KERNEL        = 5
DROPOUT       = 0.1

# ---------------- 加载缓存 ----------------
assert os.path.exists(CACHE_FILE), f"找不到 {CACHE_FILE}"
d = np.load(CACHE_FILE)
X_train_noisy = d["X_train_noisy"]; X_train_clean = d["X_train_clean"]
X_test_noisy  = d["X_test_noisy"];  X_test_clean  = d["X_test_clean"]
N_GENES = X_train_noisy.shape[1]
print(f"N_GENES={N_GENES} | Train={len(X_train_noisy)} | Test={len(X_test_noisy)}")

class DAEDataset(Dataset):
    def __init__(self, xn, xc):
        self.xn = torch.from_numpy(xn); self.xc = torch.from_numpy(xc)
    def __len__(self):  return len(self.xn)
    def __getitem__(self, i): return self.xn[i], self.xc[i]

train_dl = DataLoader(DAEDataset(X_train_noisy, X_train_clean),
                      batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dl  = DataLoader(DAEDataset(X_test_noisy, X_test_clean),
                      batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Capsule ----------------
def squash(s, dim=-1, eps=1e-8):
    sq_norm = (s ** 2).sum(dim=dim, keepdim=True)
    return (sq_norm / (1 + sq_norm)) * s / torch.sqrt(sq_norm + eps)

class PrimaryCapsules(nn.Module):
    def __init__(self, in_ch=1, n_caps=16, d_cap=8, kernel=5, drop_rate=0.0):
        super().__init__()
        self.conv   = nn.Conv1d(in_ch, n_caps * d_cap, kernel, padding=kernel // 2)
        self.n_caps = n_caps
        self.d_cap  = d_cap
        self.drop   = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
    def forward(self, x):
        h = self.conv(x)
        B, _, G = h.shape
        h = h.view(B, self.n_caps, self.d_cap, G)
        h = h.permute(0, 3, 1, 2).contiguous()
        return squash(self.drop(h))

class LocalRoutingCaps(nn.Module):
    def __init__(self, n_in, d_in, n_out, d_out, kernel=5, n_iters=3):
        super().__init__()
        self.k = kernel // 2
        self.n_out, self.d_out = n_out, d_out
        self.n_iters = n_iters
        self.P = n_in * (2 * self.k + 1)
        self.W = nn.Parameter(torch.randn(self.P, d_in, n_out, d_out) * 0.02)
    def forward(self, u):
        B, G, N, D = u.shape
        k = self.k
        u_ = u.permute(0, 2, 3, 1).contiguous()
        u_pad = F.pad(u_, (k, k))
        win = u_pad.unfold(-1, 2 * k + 1, 1)
        win = win.permute(0, 3, 1, 4, 2).contiguous()
        u_local = win.reshape(B, G, N * (2 * k + 1), D)

        u_hat = torch.einsum('bgpd,pdmo->bgpmo', u_local, self.W)

        b = torch.zeros(B, G, self.P, self.n_out, device=u.device)
        for i in range(self.n_iters):
            c = F.softmax(b, dim=2)
            v = squash(torch.einsum('bgpm,bgpmo->bgmo', c, u_hat))
            if i < self.n_iters - 1:
                b = b + (u_hat * v.unsqueeze(2)).sum(-1)   # ← 修复: 2 而不是 3
        return v

class CapsuleDAE(nn.Module):
    def __init__(self, n_genes, primary_n, primary_d,
                 digit_n, digit_d, routing_iters, kernel, drop_rate):
        super().__init__()
        self.primary = PrimaryCapsules(1, primary_n, primary_d, kernel, drop_rate)
        self.digit   = LocalRoutingCaps(primary_n, primary_d,
                                        digit_n, digit_d, kernel, routing_iters)
        self.head    = nn.Linear(digit_d, 1)
    def forward(self, x):
        u = self.primary(x.unsqueeze(1))
        v = self.digit(u)
        return self.head(v.mean(dim=2)).squeeze(-1)

model = CapsuleDAE(N_GENES, PRIMARY_N, PRIMARY_D,
                   DIGIT_N, DIGIT_D, ROUTING_ITERS, KERNEL, DROPOUT).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

# ---------------- Shape 自检 ----------------
with torch.no_grad():
    _x = torch.randn(2, N_GENES)
    _u = model.primary(_x.unsqueeze(1))
    _v = model.digit(_u)
    _o = model(_x)
print(f"shape check: primary={tuple(_u.shape)} digit={tuple(_v.shape)} out={tuple(_o.shape)}")
assert _o.shape == (2, N_GENES)

# ---------------- 训练 ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\n>> 开始训练 ({EPOCHS} epochs, CPU)")
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

torch.save({"model": model.state_dict()}, CKPT_FILE)

# ---------------- 评估 ----------------
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
b0 = float(np.mean(X_test_clean ** 2))
bi = float(np.mean((X_test_noisy - X_test_clean) ** 2))

print("\n" + "="*55)
print("TEST SET RESULTS (Capsule DAE)")
print("="*55)
print(f"Capsule DAE Test MSE : {test_mse:.6f}")
print(f"基线 (预测全0)        : {b0:.6f}")
print(f"基线 (恒等)           : {bi:.6f}")
if test_mse < bi:
    print(f"✅ 降低 {(1-test_mse/bi)*100:.1f}%")
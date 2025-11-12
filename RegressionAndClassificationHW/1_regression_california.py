# 1_regression_california.py
import json, math, time, pathlib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.utils.data as data

# ---- Load & split
cal = fetch_california_housing(as_frame=True)
X = cal.data
y = cal.target  

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ---- Scale
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ---- Torch datasets
class TabDS(data.Dataset):
    def __init__(self, X, y): 
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_ds = TabDS(X_train_s, y_train)
val_ds   = TabDS(X_val_s, y_val)
test_ds  = TabDS(X_test_s, y_test)

train_ld = data.DataLoader(train_ds, batch_size=256, shuffle=True)
val_ld   = data.DataLoader(val_ds, batch_size=512)
test_ld  = data.DataLoader(test_ds, batch_size=512)

# ---- Models
def make_mlp(d_in, hidden, dropout=0.0):
    layers = []
    for h in hidden:
        layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
        d_in = h
    layers += [nn.Linear(d_in, 1)]
    return nn.Sequential(*layers)

configs = {
    "small":  dict(hidden=[64,32], dropout=0.05),
    "medium": dict(hidden=[128,64,32], dropout=0.1),
    "large":  dict(hidden=[256,128,64], dropout=0.15),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(name, cfg, max_epochs=200, patience=20, lr=1e-3):
    model = make_mlp(X_train_s.shape[1], cfg["hidden"], cfg["dropout"]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best = math.inf
    bad  = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = lossf(pred, yb)
            loss.backward()
            opt.step()

        # val
        model.eval()
        with torch.no_grad():
            vs = []
            for xb, yb in val_ld:
                xb, yb = xb.to(device), yb.to(device)
                vs.append(lossf(model(xb), yb).item())
            vloss = float(np.mean(vs))
        if vloss < best - 1e-6:
            best = vloss
            bad  = 0
            best_state = {k: v.clone().cpu() for k,v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, best

results = {}
for name, cfg in configs.items():
    model, vloss = train_model(name, cfg)
    # test R2
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, yb in test_ld:
            preds.append(model(xb.to(device)).cpu().numpy())
    y_pred = np.vstack(preds).ravel()
    r2 = r2_score(y_test, y_pred)
    results[name] = dict(val_mse=float(vloss), r2=float(r2))
    print(f"{name}: val MSE={vloss:.4f}, test R2={r2:.4f}")

# pick best by R²
best_name = max(results, key=lambda k: results[k]["r2"])
print("BEST:", best_name, results[best_name])

# ---- Save best model & preprocessing for ONNX/web
best_model, _ = train_model(best_name, configs[best_name])  # retrain to ensure weights present
pre = {
    "feature_names": list(X.columns),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
}
pathlib.Path("deploy/regression").mkdir(parents=True, exist_ok=True)
with open("deploy/regression/preprocess_reg.json","w") as f:
    json.dump(pre, f, indent=2)

# ---- Export ONNX
dummy = torch.zeros(1, X_train_s.shape[1], dtype=torch.float32).to(device)
torch.onnx.export(
    best_model.to(device), dummy, "deploy/regression/california_regression.onnx",
    input_names=["input"], output_names=["pred"],
    dynamic_axes={"input": {0: "batch"}, "pred": {0: "batch"}},
    opset_version=17
)
print("Exported regression ONNX.")
print("Results:", results)

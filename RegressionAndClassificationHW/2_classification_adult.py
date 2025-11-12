# 2_classification_adult.py
import json, pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.datasets import fetch_openml
import torch
import torch.nn as nn
import torch.utils.data as data

# ---- Load
adult = fetch_openml("adult", version=2, as_frame=True)  # 48k rows
df = adult.frame.dropna()

y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# ---- Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ---- Preprocess: OHE + standardize numeric
ct = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

X_train_p = ct.fit_transform(X_train)
X_val_p   = ct.transform(X_val)
X_test_p  = ct.transform(X_test)

# ---- Torch dataset
class TabDS(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_ld = data.DataLoader(TabDS(X_train_p, y_train), batch_size=512, shuffle=True)
val_ld   = data.DataLoader(TabDS(X_val_p, y_val), batch_size=1024)
test_ld  = data.DataLoader(TabDS(X_test_p, y_test), batch_size=1024)

# ---- Models
def make_mlp(d_in, hidden, p=0.1):
    layers = []
    for h in hidden:
        layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(p)]
        d_in = h
    layers += [nn.Linear(d_in, 2)]
    return nn.Sequential(*layers)

configs = {
    "shallow": [256,128],
    "medium":  [512,256,128],
    "deep":    [768,384,192],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(hidden, max_epochs=40, patience=6, lr=1e-3):
    model = make_mlp(X_train_p.shape[1], hidden).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best  = 1e9
    bad   = 0
    best_state = None
    for ep in range(max_epochs):
        model.train()
        for xb,yb in train_ld:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        # val
        model.eval()
        with torch.no_grad():
            vs = []
            for xb,yb in val_ld:
                xb,yb = xb.to(device), yb.to(device)
                vs.append(lossf(model(xb), yb).item())
            v = float(np.mean(vs))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: v.clone().cpu() for k,v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience: break
    model.load_state_dict(best_state)
    return model

scores = {}
for name, hidden in configs.items():
    m = train_model(hidden)
    m.eval()
    preds = []
    with torch.no_grad():
        for xb,yb in test_ld:
            logits = m(xb.to(device)).cpu().numpy()
            preds.append(np.argmax(logits, axis=1))
    y_pred = np.concatenate(preds)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    scores[name] = dict(f1=float(f1), cm=cm.tolist())
    print(f"{name}: F1(macro)={f1:.4f}\nConfusion:\n{cm}\n")

best_name = max(scores, key=lambda k: scores[k]["f1"])
print("BEST:", best_name, scores[best_name])

# ---- Save preprocessing metadata for web
ohe: OneHotEncoder = ct.named_transformers_["cat"]
num_scaler: StandardScaler = ct.named_transformers_["num"]

meta = {
  "num_cols": num_cols,
  "cat_cols": cat_cols,
  "num_mean": num_scaler.mean_.tolist(),
  "num_scale": num_scaler.scale_.tolist(),
  "cat_categories": [list(c) for c in ohe.categories_],
}
pathlib.Path("deploy/classification").mkdir(parents=True, exist_ok=True)
with open("deploy/classification/preprocess_cls.json","w") as f:
    json.dump(meta, f, indent=2)

# ---- Export ONNX
best_model = train_model(configs[best_name])  # ensure weights exist
dummy = torch.zeros(1, X_train_p.shape[1], dtype=torch.float32).to(device)
torch.onnx.export(
    best_model.to(device), dummy, "deploy/classification/adult_cls.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}}, opset_version=17
)
print("Exported classification ONNX.")
print("Scores:", scores)

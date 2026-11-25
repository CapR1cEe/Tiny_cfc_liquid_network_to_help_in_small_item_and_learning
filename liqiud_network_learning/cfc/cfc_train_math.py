import csv
from pathlib import Path

import torch
from torch import nn


class TinyCfC(nn.Module):
    def __init__(self):
        super().__init__()

        # g 
        self.w_gh = nn.Parameter(torch.tensor(0.5))
        self.w_gu = nn.Parameter(torch.tensor(1.0))
        self.b_g = nn.Parameter(torch.tensor(0.0))

        # r 
        self.w_rh = nn.Parameter(torch.tensor(-0.5))
        self.w_ru = nn.Parameter(torch.tensor(0.5))
        self.b_r = nn.Parameter(torch.tensor(0.0))

        # time gate
        self.w_ah = nn.Parameter(torch.tensor(0.8))
        self.w_au = nn.Parameter(torch.tensor(0.4))
        self.b_a = nn.Parameter(torch.tensor(0.0))
        self.b_z = nn.Parameter(torch.tensor(-0.1))

        # output
        self.w_o = nn.Parameter(torch.tensor(2.0))
        self.b_o = nn.Parameter(torch.tensor(-0.3))

    def forward(self, u, h, dt):
        g = torch.tanh(self.w_gh * h + self.w_gu * u + self.b_g)
        r = torch.tanh(self.w_rh * h + self.w_ru * u + self.b_r)

        a = self.w_ah * h + self.w_au * u + self.b_a
        z = torch.sigmoid(a * dt + self.b_z)

        h_new = z * g + (1.0 - z) * r
        y = self.w_o * h_new + self.b_o

        return y, h_new
def load_sequences(csv_path, split):
    groups = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            sequence_id = int(row["sequence_id"])
            if sequence_id not in groups:
                groups[sequence_id] = []

            groups[sequence_id].append((
                int(row["step"]),
                float(row["u"]),
                float(row["dt"]),
                float(row["target"]),
            ))
    sequences = []
    for sequence_id in sorted(groups):
        rows = sorted(groups[sequence_id], key=lambda row: row[0])
        values = [
            [row[1], row[2], row[3]]
            for row in rows
        ]
        sequence = torch.tensor(values, dtype=torch.float32)
        sequences.append(sequence)

    if not sequences:
        raise ValueError(f"CSV can't find split={split} data")
    return sequences
def sequence_mse(model, sequence):
    h = torch.tensor(0.0)
    errors = []
    for u, dt, target in sequence:
        prediction, h = model(u, h, dt)
        error = (prediction - target) ** 2
        errors.append(error)
    return torch.stack(errors).mean()
project_dir = Path(__file__).resolve().parent.parent
csv_path = project_dir / "data" / "dataset.csv"
train_sequences = load_sequences(csv_path, "train")
test_sequences = load_sequences(csv_path, "test")
print("train number:", len(train_sequences))
print("test number:", len(test_sequences))
model = TinyCfC()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0

    for sequence in train_sequences:
        optimizer.zero_grad()
        loss = sequence_mse(model, sequence)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch == 1 or epoch % 10 == 0:
        train_mse = total_loss / len(train_sequences)
        print(f"the {epoch:3d} epoches,test MSE:{train_mse:.6f}")

model.eval()
with torch.no_grad():
    test_total = 0.0
    for sequence in test_sequences:
        test_total += sequence_mse(model, sequence).item()

    test_mse = test_total / len(test_sequences)
    print(f"\ntest mse:{test_mse:.6f}")
    h = torch.tensor(0.0)
    print("\nfirst sequence test five steps:")
    for step, (u, dt, target) in enumerate(test_sequences[0][:5]):
        prediction, h = model(u, h, dt)
        print(
            f"the {step} number:"
            f"prediction={prediction.item():.4f}"
            f"answer={target.item():.4f}"
        )
@torch.no_grad()
def evaluate(model, sequences, tolerance=0.05):
    model.eval()
    predictions = []
    targets = []
    inputs = []
    for sequence in sequences:
        h = torch.tensor(0.0)
        for u, dt, target in sequence:
            prediction, h = model(u, h, dt)
            predictions.append(prediction)
            targets.append(target)
            inputs.append(u)
    predictions = torch.stack(predictions)
    targets = torch.stack(targets)
    inputs = torch.stack(inputs)
    error = predictions - targets
    mse = (error ** 2).mean().item()
    mae = error.abs().mean().item()
    rmse = mse ** 0.5

    pass_rate = (
        (error.abs() <= tolerance)
        .float()
        .mean()
        .item()
    )
    baseline_rmse = (
        ((inputs - targets) ** 2).mean().sqrt().item()
    )
    return mse, mae, rmse, pass_rate, baseline_rmse
for name, sequences in [
    ("训练集", train_sequences),
    ("测试集", test_sequences),
]:
    mse, mae, rmse, rate, baseline = evaluate(model, sequences)

    print(f"\n{name}:")
    print(f"MSE={mse:.6f}")
    print(f"MAE={mae:.6f}")
    print(f"RMSE={rmse:.6f}")
    print(f"rate in 0.05{rate:.2%}")
    print(f"u RMSE{baseline:.6f}")
export_dir = project_dir / "export"
export_dir.mkdir(parents=True, exist_ok=True)
torch.save(
    model.state_dict(),
    export_dir / "tiny_cfc.pt"
)
header_path = export_dir / "tiny_cfc_weights.h"
with open(header_path, "w", encoding="utf-8") as f:
    f.write("#ifndef TINY_CFC_WEIGHTS_H\n")
    f.write("#define TINY_CFC_WEIGHTS_H\n\n")
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().item()
        f.write(
            f"static const float {name} = {value:.9e}f;\n"
        )
    f.write("\n#endif\n")
print("constant is going to", export_dir)
import math
import random
import csv
from pathlib import Path

num_sequences = 100
sequence_length = 30
T = 1.0

rng = random.Random(42)

save_path = Path("data") / "dataset.csv"
save_path.parent.mkdir(parents=True, exist_ok=True)

with open(save_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "sequence_id", "step", "split",
            "u", "dt", "target"
        ]
    )
    writer.writeheader()

    for sequence_id in range(num_sequences):
        y_prev = 0.0
        u = 0.0

        split = "train" if sequence_id < 80 else "test"

        for step in range(sequence_length):
            if step % 5 == 0:
                u = rng.uniform(-1.0, 1.0)

            dt = rng.uniform(0.1, 0.8)

            q = math.exp(-dt / T)
            y_new = q * y_prev + (1 - q) * u

            writer.writerow({
                "sequence_id": sequence_id,
                "step": step,
                "split": split,
                "u": u,
                "dt": dt,
                "target": y_new
            })

            y_prev = y_new

print("保存位置：", save_path.resolve())
print("序列数：", num_sequences)
print("数据总行数：", num_sequences * sequence_length)
from pathlib import Path
import torch
from torch import nn
class TinyCfC(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_gh = nn.Parameter(torch.tensor(0.5))
        self.w_gu = nn.Parameter(torch.tensor(1.0))
        self.b_g = nn.Parameter(torch.tensor(0.0))
        self.w_rh = nn.Parameter(torch.tensor(-0.5))
        self.w_ru = nn.Parameter(torch.tensor(0.5))
        self.b_r = nn.Parameter(torch.tensor(0.0))
        self.w_ah = nn.Parameter(torch.tensor(0.8))
        self.w_au = nn.Parameter(torch.tensor(0.4))
        self.b_a = nn.Parameter(torch.tensor(0.0))
        self.b_z = nn.Parameter(torch.tensor(-0.1))
        self.w_o = nn.Parameter(torch.tensor(2.0))
        self.b_o = nn.Parameter(torch.tensor(-0.3))
    def forward(self, u, h, dt):
        g = torch.tanh(
            self.w_gh * h + self.w_gu * u + self.b_g
        )
        r = torch.tanh(
            self.w_rh * h + self.w_ru * u + self.b_r
        )
        a = self.w_ah * h + self.w_au * u + self.b_a
        z = torch.sigmoid(a * dt + self.b_z)
        h_new = z * g + (1.0 - z) * r
        y = self.w_o * h_new + self.b_o
        return y, h_new
def main():
    project_dir = Path(__file__).resolve().parent.parent
    weights_path = project_dir / "export" / "tiny_cfc.pt"

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"can't find{weights_path}\n"
            "you nead your own tiny_cfc.pt"
        )
    model = TinyCfC()
    weights = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(weights)
    model.eval()
    print("success", weights_path)
    samples = [
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.5),
        (0.4, 0.2),
        (0.0, 0.4),
    ]
    h = torch.tensor(0.0, dtype=torch.float32)
    with torch.no_grad():
        for step, (u_value, dt_value) in enumerate(samples, start=1):
            u = torch.tensor(u_value, dtype=torch.float32)
            dt = torch.tensor(dt_value, dtype=torch.float32)
            y, h = model(u, h, dt)
            print(
                f"the {step} step:"
                f"u={u_value:.3f},"
                f"dt={dt_value:.3f},"
                f"predict y={y.item():.6f},"
                f"hidden status h={h.item():.6f}"
            )
if __name__ == "__main__":
    main()
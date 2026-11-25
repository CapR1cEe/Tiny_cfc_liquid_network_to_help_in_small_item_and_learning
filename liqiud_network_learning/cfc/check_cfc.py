import torch
w = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.Adam([w], lr=0.05)
u = torch.tensor(2.0)
target = torch.tensor(4.0)
for step in range(300):
    prediction = w * u
    loss = (prediction - target) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("训练后的权重：", w.item())
print("训练后的预测：", (w * u).item())
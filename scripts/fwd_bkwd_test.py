# forward_backward_test.py
import torch
from torch.utils.data import DataLoader, Subset
import random
from dataset import CVATDataset
from model import MiniUNet

root_dir = "../data/"
dataset = CVATDataset(dataset_dir=root_dir, has_gt=True)
indices = list(range(len(dataset)))
random.shuffle(indices)
subset = Subset(dataset, indices[:8])
loader = DataLoader(subset, batch_size=4, shuffle=True, num_workers=0)

device = torch.device("cpu")
model = MiniUNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch = next(iter(loader))
inputs = batch["input"].to(device)
targets = batch["target"].to(device)

print("shapes:", inputs.shape, targets.shape)  # expect [B,3,H,W] and [B,H,W]
print("unique target values:", torch.unique(targets))

torch.autograd.set_detect_anomaly(True)
try:
    outputs = model(inputs)
    print("forward done:", outputs.shape)
    loss = criterion(outputs, targets)
    print("loss:", loss.item())
    optimizer.zero_grad()
    loss.backward()
    print("backward done")
    optimizer.step()
    print("optimizer step done")
except Exception as e:
    print("Exception during forward/backward:", repr(e))

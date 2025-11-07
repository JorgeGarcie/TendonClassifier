from torch.utils.data import DataLoader, Subset
import random
import torch
from dataset import CVATDataset

root_dir = "../data/"
full_dataset = CVATDataset(dataset_dir=root_dir, has_gt=True)

num_samples = len(full_dataset)
indices = list(range(num_samples))
random.seed(42)
random.shuffle(indices)
split = int(0.8 * num_samples)
train_indices = indices[:split]

train_dataset = Subset(full_dataset, train_indices)

loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

print("Dataset length:", len(full_dataset))
for i, batch in enumerate(loader):
    print(
        f"Batch {i}: input.shape={batch['input'].shape}, target.shape={batch['target'].shape}"
    )
    # Print dtype and some stats
    print(
        "  input dtype:",
        batch["input"].dtype,
        "min/max:",
        batch["input"].min().item(),
        batch["input"].max().item(),
    )
    print(
        "  target dtype:",
        batch["target"].dtype,
        "unique:",
        torch.unique(batch["target"]).cpu().numpy()[:10],
    )
    if i >= 10:
        break
print("Dataloader test completed.")

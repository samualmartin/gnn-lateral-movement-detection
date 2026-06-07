import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from gat_model import GAT

data = torch.load(
    "data/processed/graph_data.pt",
    weights_only=False
)

print(data)

indices = np.arange(data.num_nodes)

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=data.y.numpy()
)

train_mask = torch.zeros(
    data.num_nodes,
    dtype=torch.bool
)

test_mask = torch.zeros(
    data.num_nodes,
    dtype=torch.bool
)

train_mask[train_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.test_mask = test_mask

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

data = data.to(device)

model = GAT(
    in_channels=data.num_features,
    hidden_channels=64,
    num_classes=3
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)

criterion = torch.nn.CrossEntropyLoss()


epochs = 30

loss_history = []
acc_history = []

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    out = model(
        data.x,
        data.edge_index
    )

    loss = criterion(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()

    optimizer.step()

    pred = out.argmax(dim=1)

    acc = (
        pred[data.train_mask]
        ==
        data.y[data.train_mask]
    ).float().mean()

    loss_history.append(
        loss.item()
    )

    acc_history.append(
        acc.item()
    )

    if epoch % 5 == 0:

        print(
            f"Epoch {epoch:02d}"
            f" | Loss={loss:.4f}"
            f" | Train Acc={acc:.4f}"
        )


torch.save(
    model.state_dict(),
    "models/best_model.pth"
)

print("Model saved.")


plt.figure(figsize=(8, 5))

plt.plot(
    loss_history,
    linewidth=2
)

plt.title(
    "Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/training_loss.png",
    dpi=300
)

plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    acc_history,
    linewidth=2
)

plt.title(
    "Training Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/training_accuracy.png",
    dpi=300
)

plt.close()

print("Training plots saved.")
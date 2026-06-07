import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
classification_report,
confusion_matrix,
ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split

from gat_model import GAT


data = torch.load(
"data/processed/graph_data.pt",
weights_only=False
)

indices = np.arange(data.num_nodes)

_, test_idx = train_test_split(
indices,
test_size=0.2,
random_state=42,
stratify=data.y.numpy()
)

test_mask = torch.zeros(
data.num_nodes,
dtype=torch.bool
)

test_mask[test_idx] = True


device = torch.device(
"cuda"
if torch.cuda.is_available()
else "cpu"
)

data = data.to(device)

model = GAT(
in_channels=data.num_features,
hidden_channels=64,
num_classes=3
).to(device)

model.load_state_dict(
torch.load(
"models/best_model.pth",
map_location=device
)
)

model.eval()

with torch.no_grad():

    out = model(
        data.x,
        data.edge_index
    )

pred = out.argmax(dim=1)

y_true = (
data.y[test_mask]
.cpu()
.numpy()
)

y_pred = (
pred[test_mask]
.cpu()
.numpy()
)

report = classification_report(
y_true,
y_pred,
target_names=[
"Normal",
"EoHT",
"EoRS"
]
)

print(report)

with open(
"outputs/metrics.txt",
"w"
) as f:
    f.write(report)

report_dict = classification_report(
y_true,
y_pred,
target_names=[
"Normal",
"EoHT",
"EoRS"
],
output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()

report_df.to_csv(
"outputs/classification_report.csv"
)

cm = confusion_matrix(
y_true,
y_pred
)

disp = ConfusionMatrixDisplay(
confusion_matrix=cm,
display_labels=[
"Normal",
"EoHT",
"EoRS"
]
)

disp.plot()

plt.savefig(
"outputs/confusion_matrix.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

pred_counts = pd.Series(
y_pred
).value_counts().sort_index()

plt.figure(figsize=(8,5))

pred_counts.plot(
kind="bar"
)

plt.title(
"Predicted Class Distribution"
)

plt.xlabel(
"Class"
)

plt.ylabel(
"Count"
)

plt.tight_layout()

plt.savefig(
"outputs/prediction_distribution.png",
dpi=300
)

plt.close()

with open("outputs/dataset_statistics.txt", "w") as f:
    f.write(f"Nodes: {data.num_nodes}\n")
    f.write(f"Edges: {data.edge_index.shape[1]}\n")
    f.write(f"Features: {data.num_features}\n")
    f.write("Classes: 3\n")

with open(
    "outputs/model_summary.txt",
    "w"
) as f:

    f.write(
        str(model)
    )

print("Evaluation complete.")
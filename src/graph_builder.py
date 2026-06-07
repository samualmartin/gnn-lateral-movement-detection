import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

from torch_geometric.data import Data

print("Loading processed dataset...")

df = pd.read_csv(
    "data/processed/processed_data.csv"
)

print("Shape:", df.shape)

X = df.drop(columns=["Label"])
y = df["Label"]

print("Scaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Building graph...")

A = kneighbors_graph(
    X_scaled,
    n_neighbors=10,
    mode="connectivity",
    include_self=False
)

edge_index = torch.tensor(
    A.nonzero(),
    dtype=torch.long
)

x = torch.tensor(
    X_scaled,
    dtype=torch.float
)

y = torch.tensor(
    y.values,
    dtype=torch.long
)

data = Data(
    x=x,
    edge_index=edge_index,
    y=y
)

print(data)

torch.save(
    data,
    "data/processed/graph_data.pt"
)

print("Graph saved.")
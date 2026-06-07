import torch
import networkx as nx
import matplotlib.pyplot as plt

data = torch.load(
    "data/processed/graph_data.pt",
    weights_only=False
)

# Use first 300 nodes
num_nodes = 300

edge_index = data.edge_index.numpy()

G = nx.Graph()

for i in range(num_nodes):
    G.add_node(
        i,
        label=int(data.y[i])
    )

for src, dst in zip(
    edge_index[0],
    edge_index[1]
):
    if src < num_nodes and dst < num_nodes:
        G.add_edge(src, dst)

colors = []

for node in G.nodes():

    label = G.nodes[node]["label"]

    if label == 0:
        colors.append("green")

    elif label == 1:
        colors.append("orange")

    else:
        colors.append("red")

plt.figure(figsize=(12,8))

nx.draw_networkx(
    G,
    node_color=colors,
    node_size=30,
    with_labels=False
)

plt.title(
    "LMD-2023 Event Graph"
)

plt.savefig(
    "outputs/attack_graph.png",
    dpi=300
)

plt.show()
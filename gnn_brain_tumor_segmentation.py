import os
import glob
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.future import graph
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

# ─── CONFIG 

# Path to BraTS folder containing subfolders HGG/ and LGG/, each with per-patient
# directories with modalities flair/t1/t1ce/t2 and seg (ground‐truth .nii.gz files).
DATA_ROOT = "./data/BraTS2018_TrainingData"
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# number of superpixels per slice
N_SEGMENTS = 200
# train/val split ratio
TRAIN_RATIO = 0.8
# GNN hyperparams
HIDDEN_DIM = 64
LR = 1e-3
EPOCHS = 20
BATCH_SIZE = 4

# ─── MODEL 

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# ─── GRAPH BUILDING 

def build_slice_graph(image: np.ndarray, mask: np.ndarray, n_segments=N_SEGMENTS):
    """
    image, mask: 2D arrays (H x W), image normalized to [0,1], mask binary {0,1}.
    Returns torch_geometric Data with:
      x: [num_nodes, 4] features = [mean_intensity, std_intensity, y_centroid, x_centroid]
      edge_index: [2, num_edges]
      y: [num_nodes] binary labels per superpixel
    """
    labels = slic(image, n_segments=n_segments, compactness=10, start_label=0)
    rag = graph.rag_mean_color(image, labels, mode='distance')
    num_nodes = labels.max() + 1

    # node features
    feats = []
    Ys = []
    for region in range(num_nodes):
        mask_region = (labels == region)
        pixs = image[mask_region]
        mean_i = pixs.mean().astype(np.float32)
        std_i  = pixs.std().astype(np.float32)
        coords = np.column_stack(np.nonzero(mask_region))
        y_cent, x_cent = coords.mean(axis=0) / np.array(image.shape)  # normalize
        feats.append([mean_i, std_i, y_cent, x_cent])

        # label by majority mask
        lbl = mask[mask_region].sum() / mask_region.sum() > 0.5
        Ys.append(int(lbl))

    x = torch.tensor(feats, dtype=torch.float)
    y = torch.tensor(Ys, dtype=torch.long)

    # edges
    edges = []
    for u, v, attr in rag.edges(data=True):
        edges.append([u, v])
        edges.append([v, u])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)

# ─── DATA LOADING

def load_brats_slices(root_dir=DATA_ROOT):
    """
    Walk HGG/ and LGG/, load 'flair' modality and its seg mask.
    Return list of Data graphs for all axial slices that contain tumor.
    """
    graphs = []
    for grade in ['HGG', 'LGG']:
        pattern = os.path.join(root_dir, grade, "*", "*_flair.nii.gz")
        for flair_path in glob.glob(pattern):
            seg_path = flair_path.replace('_flair.nii.gz', '_seg.nii.gz')
            img = nib.load(flair_path).get_fdata().astype(np.float32)
            msk = nib.load(seg_path).get_fdata().astype(np.uint8)
            # normalize each slice
            for z in range(img.shape[2]):
                slice_img = img[:, :, z]
                slice_msk = msk[:, :, z] > 0
                if slice_msk.sum() == 0:
                    continue  # skip non-tumor slices
                # min-max normalize to [0,1]
                slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
                data = build_slice_graph(slice_img, slice_msk)
                graphs.append(data)
    return graphs

# ─── TRAIN/VAL SPLIT

all_graphs = load_brats_slices()
random.shuffle(all_graphs)
n_train = int(len(all_graphs) * TRAIN_RATIO)
train_graphs = all_graphs[:n_train]
val_graphs   = all_graphs[n_train:]

train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE)


# ─── TRAINING LOOP

model = GCN(in_channels=4, hidden_channels=HIDDEN_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(1, EPOCHS+1):
    # train
    model.train()
    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
        preds = out.argmax(dim=1)
        total_correct += (preds == batch.y).sum().item()
        total_examples += batch.num_nodes
    train_losses.append(total_loss / total_examples)
    train_accuracies.append(total_correct / total_examples)

    # validate
    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            loss = criterion(out, batch.y)
            v_loss += loss.item() * batch.num_nodes
            preds = out.argmax(dim=1)
            v_correct += (preds == batch.y).sum().item()
            v_total += batch.num_nodes
    val_losses.append(v_loss / v_total)
    val_accuracies.append(v_correct / v_total)

    print(f"Epoch {epoch:02d} | "
          f"Train loss: {train_losses[-1]:.4f}, acc: {train_accuracies[-1]:.4f} | "
          f"Val loss: {val_losses[-1]:.4f}, acc: {val_accuracies[-1]:.4f}")

# ─── PLOTTING

os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(range(1,EPOCHS+1), train_losses, label="Train")
plt.plot(range(1,EPOCHS+1), val_losses,   label="Val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(range(1,EPOCHS+1), train_accuracies, label="Train")
plt.plot(range(1,EPOCHS+1), val_accuracies,   label="Val")
plt.xlabel("Epoch"); plt.ylabel("Node Acc"); plt.legend()
plt.title("Accuracy")

plt.tight_layout()
plt.savefig("outputs/training_curves.png")
print("Saved training curves -> outputs/training_curves.png")

# Graph-Neural-Networks-GNNs-
Graph-Based Learning: In domains where data has inherent relational structure (e.g., medical imaging, social networks), GNNs are used to learn from graph-structured data. In brain tumor segmentation, GNNs can model complex relationships between image pixels or regions and enhance segmentation accuracy by considering spatial dependencies.

This is a self-contained script that demonstrates how to use a Graph Neural Network (GNN) for brain-tumor segmentation by modeling relationships between superpixels in 2D MRI slices. It uses the BraTS 2018 dataset (you can download it from [the MICCAI BraTS website](https://www.med.upenn.edu/cbica/brats2018/data.html)), and requires PyTorch & PyTorch‑Geometric, scikit‑image and nibabel.

### How it works

1. **Data**:

   * Loads FLAIR volumes and segmentation masks (`_flair.nii.gz` & `_seg.nii.gz`) from BraTS.
   * Iterates through axial slices, keeps only those containing tumor.

2. **Graph construction**:

   * Superpixel decomposition (SLIC) into \~200 regions.
   * Build Region Adjacency Graph (RAG) via scikit‑image.
   * Node features: mean & std intensity + normalized centroid.
   * Node labels: tumor vs. healthy by majority mask overlap.

3. **GNN**:

   * A 2‑layer GCN on the superpixel graph.
   * Binary node classification (tumor vs. non‑tumor).

4. **Training**:

   * Cross‑entropy on nodes.
   * Splits slice‐graphs into training/validation (80/20).

5. **Evaluation & Visualization**:

   * Prints per‐epoch node accuracy & loss.
   * Saves training curves.

---

#### Requirements

```bash
pip install torch torchvision torch-geometric scikit-image nibabel matplotlib tqdm
```

Make sure to download and unpack BraTS 2018 into `./data/BraTS2018_TrainingData`.

Run:

```bash
python gnn_brain_tumor_segmentation.py
```

The training curves will be saved under `outputs/`.


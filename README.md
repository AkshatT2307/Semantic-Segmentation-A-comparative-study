# Semantic Segmentation Benchmark

A comparative study implementing and benchmarking classical algorithms (like Otsu Thresholding) against modern deep learning architectures on popular semantic segmentation datasets.

## Setup Requirements

Ensure you have your environment set up properly with PyTorch and the standard data-science dependencies (OpenCV, NumPy). You can run:

```bash
# Recommended to run from within a virtual environment
pip install -r requirements.txt
```

## Running Evaluation Benchmarks

We currently support running semantic segmentation evaluation on **Pascal VOC** (Automatic Download) and **COCO-Stuff**. 
By default, multi-class semantic masks are binarized against the foreground dynamically to compare against standard foreground-background Global/Otsu threshold predictions.

### Execute Classical Methods & Visualization

You can run the evaluation via the `run.py` wrapper, which dynamically maps the models to the dataset and computes **mIoU** and **Pixel Accuracy**. 
**If you don't have Pascal VOC installed, it will automatically download the 2GB dataset into the `./data` folder in your project!**

```bash
# 🔥 Auto-Download VOC & Run Otsu's optimal threshold (Default) on the Validation split
python semseg-benchmark/run.py --dataset voc --split val --method otsu

# Run a hardcoded Global threshold of 127
python semseg-benchmark/run.py --dataset voc --split val --method global --global-thresh 127

# Evaluate Graph-Cut with Visualization enabled
python semseg-benchmark/run.py --dataset voc --method graph_cut --visualize --vis-count 10

# Evaluate KMeans using Cityscapes 
python semseg-benchmark/run.py --dataset cityscapes --method kmeans --visualize

# Evaluate SLIC-based Region mapping on COCO
python semseg-benchmark/run.py --dataset coco --method region --visualize --vis-count 5

# Evaluate Canny Edge Segmentation
python semseg-benchmark/run.py --dataset voc --method edge --visualize
```

**Custom Argument Flags:**
- `--dataset`: `voc` (default auto-download), `coco`, or `cityscapes`.
- `--data-root`: Base directory for datasets (default `./data`).
- `--split`: Evaluates over the `val` (default) or `train` datasets.
- `--method`: Algorithm to use: `otsu`, `global`, `graph_cut`, `region`, `kmeans`, or `edge`.
- `--global-thresh`: Overrides the integer cutoff if `--method global`.
- `--batch-size`: Set a PyTorch batch size (default is `1`).
- `--visualize`: Generates qualitative visualization grids mapping to `./results/`.
- `--vis-count`: Max samples to visually process per run (default `10`).
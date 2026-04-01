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

### Base Execution Command 

The benchmark evaluation is executed logically through the `run.py` wrapper, which dynamically pipelines deep learning and classical models seamlessly across supported multi-class datasets (computing **mIoU** and **Pixel Accuracy** natively). 

> **Tip:** If you initiate `voc` without having Pascal VOC installed previously, it will automatically download (~2GB) and unpack it right into your root `./data` folder!

Here is the universal base command template:
```bash
python semseg-benchmark/run.py --dataset [DATASET] --method [METHOD] --batch-size [SIZE] --visualize
```

### Argument Glossary & Method Mappings

You can infinitely combine these tracking parameters to shape your benchmark test:

#### 1. Dataset Selection (`--dataset`)
- `voc`: *(Default)* Pascal VOC 2012. Auto-downloads and parses 21 classes natively.
- `coco`: COCO-Stuff dataset. Evaluates across 182 semantic categories. Requires `--data-root` to point to an extracted dataset.
- `cityscapes`: High-resolution street datasets parsing 19 semantic road classes natively. Requires `--data-root`.

#### 2. Segmentation Methods (`--method`)
Each string exactly maps to a distinct initialization class.
- **Binary Image Segmentation:** (Calculates Binary metrics only)
  - `otsu`: Automatically determines a unified foreground threshold using `cv2.THRESH_OTSU`.
  - `global`: Enforces a hardcoded pixel limit cut-off (controlled by `--global-thresh <INT>`). 
  - `edge`: Runs morphological tracking and explicit `cv2.Canny` abstractions to boundary edge maps.

- **Unsupervised / Multi-Class Clustering:** (Leverages `Majority Voting` to match generated unsupervised clusters against Target Semantic Maps)
  - `graph_cut`: Instantiates Felzenszwalb's efficient graph abstraction algorithm.
  - `region`: Instantiates SLIC Superpixel spatial boundary mappings.
  - `kmeans`: Recursively clusters explicit arbitrary RGB regions evaluating Scikit-Learn's `MiniBatchKMeans`.

#### 3. Core Benchmark Toggles
- `--batch-size`: Set explicit parallel processing count (default `1`). *(Note: Datasets natively interpolate down to 256x256 under the hood to safely support Bounding Parallel Batches.)*
- `--split`: Evaluates directly over the `val` (default) or `train` datasets.
- `--data-root`: Override default string root for Dataset installations (`./data/`).
- `--global-thresh`: Overrides the threshold integer parameter cutoff (if passing `--method global`).

#### 4. Qualitative Visualization Mode
Appending the `--visualize` flag safely captures the eval arrays and visually plots the **Original Image**, the **Ground Truth Array**, and the **Algorithm Prediction Mapping** side-by-side!
- `--vis-count`: Set the maximum number of iterations you want plotted before logic bypasses visualization entirely (Default bounds to `10` outputs).
- `--vis-seed`: Define an explicit integer seed (Default `42`) establishing deterministic bounds so the randomized captured targets uniformly overlap precisely the same geometric validations natively every run!
- `--vis-complex`: Automatically injects an isolated bounding scan across the raw dataset prior to sequence tracking! It inherently discovers unblemished images intrinsically possessing 4 or more semantic classes (Background + >=3 Targets), saving their index parameters securely over to a native cache `.npy` format ensuring all sequential `--vis-seed` outputs natively reflect fundamentally complex Multi-Class optical arrays!

Outputs systematically organize natively into cleanly labelled folder matrices (e.g., `./results/voc_kmeans/vis_3.png`) within your project root.
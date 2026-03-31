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

### Execute Threshold Segmentation (Pascal VOC Default)

You can run the evaluation via the `run.py` wrapper, which dynamically maps the models to the dataset and computes **mIoU** and **Pixel Accuracy**. 
**If you don't have Pascal VOC installed, it will automatically download the 2GB dataset into the `./data` folder in your project!**

```bash
# 🔥 Auto-Download VOC & Run Otsu's optimal threshold (Default) on the Validation split
python semseg-benchmark/run.py --dataset voc --split val --method otsu

# Run a hardcoded Global threshold of 127
python semseg-benchmark/run.py --dataset voc --split val --method global --global-thresh 127

# Switch mapping to evaluate the original unzipped COCO datasets
python semseg-benchmark/run.py --dataset coco --data-root /path/to/extracted/coco-stuff --split val 
```

**Custom Argument Flags:**
- `--dataset`: Switches between automatic downloading `voc` (default) or hard filesystem `coco`.
- `--data-root`: (Optional) Custom output directory for the VOC Download, or strict required string folder for `coco`.
- `--split`: Evaluates over the `val` (default) or `train` datasets.
- `--method`: Chooses the threshold strategy (`otsu` or `global`).
- `--global-thresh`: Overrides the integer cutoff if `--method global`.
- `--batch-size`: Set a PyTorch batch size (default is `1`).
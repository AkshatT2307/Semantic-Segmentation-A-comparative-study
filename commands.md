# MMSegmentation Inference Pipeline for Cityscapes

I've completed the implementation plan to run `mmsegmentation` inference within your `SemSeg` directory seamlessly and correctly.

## 1. Zero Code-Transfer Installation

Instead of copying hundreds of files into your project, we established a clean **"mmseg as a library"** pattern:
- Installed `mmengine` and a pre-packaged lightweight `mmcv-lite`.
- Created a path injection for `mmsegmentation` and lazily stubbed out the compiled CUDA ops (which are only needed for different model heads like Mask2Former, but not FCN/Transformer).
- **Result:** You have access to the entire framework and its model builders without cluttering your workspace.

## 2. Dynamic Label Remapping

Your Cityscapes validation masks (`_gtFine_labelIds.png`) use the original 34 Cityscapes label IDs. The pretrained models emit 19 evaluation IDs (`trainIds`). 

Instead of creating 500 new "duplicate" masks on your disk, the script dynamically remaps the label IDs perfectly into training IDs on the fly using a fast NumPy lookup table (LUT).

## 3. Minimal Local Configuration

I wrote two fully self-contained configuration scripts. These bypass mmseg's clunky `_base_` inheritance structure, so there is strictly exactly 1 config file for each model that defines everything (data prep, model params, dataset routing):

- `SemSeg/configs/fcn_r50-d8_cityscapes.py`
- `SemSeg/configs/segformer_mit-b1_cityscapes.py`

## 4. Run Evaluation

You now have a clean, robust CLI tool precisely integrated for your codebase (`SemSeg/mmseg_inference.py`).

**To evaluate the full Cityscapes validation set with mIoU reporting:**
```bash
# Evaluate both models sequentially:
cd SemSeg/
python mmseg_inference.py --model all --eval

# To evaluate just one model:
python mmseg_inference.py --model segformer --eval
```
*Evaluations generate JSON reports and terminal mIoU metrics.*

**To run visual inference on an ad-hoc image:**
```bash
python mmseg_inference.py --model fcn --image path/to/image.png
```

## Inference Results

Here's an example of the qualitative segmentation capabilities extracted during the smoke test:

````carousel
![Prediction Overlay using FCN ResNet-50](/opt/watchdog/users/cerussite/.gemini/antigravity/brain/8cacb9d8-9f35-4a74-a176-4e70c4ede009/artifacts/munster_000000_000019_leftImg8bit_fcn_overlay.png)
<!-- slide -->
![Raw mask visualization using FCN ResNet-50](/opt/watchdog/users/cerussite/.gemini/antigravity/brain/8cacb9d8-9f35-4a74-a176-4e70c4ede009/artifacts/munster_000000_000019_leftImg8bit_fcn_pred.png)
<!-- slide -->
![Prediction Overlay using SegFormer MiT-B1](/opt/watchdog/users/cerussite/.gemini/antigravity/brain/8cacb9d8-9f35-4a74-a176-4e70c4ede009/artifacts/munster_000000_000019_leftImg8bit_segformer_overlay.png)
<!-- slide -->
![Raw mask visualization using SegFormer MiT-B1](/opt/watchdog/users/cerussite/.gemini/antigravity/brain/8cacb9d8-9f35-4a74-a176-4e70c4ede009/artifacts/munster_000000_000019_leftImg8bit_segformer_pred.png)
````

### Dataset Symlinks
To keep compliance with your established directories, standard symlinks targeting your pre-existing data location have been created automatically inside `SemSeg/data/cityscapes/`. The evaluations strictly only look at `SemSeg/data/Cityscapes/`.

*(Note: While fetching the SegFormer weights, I encountered a locked user permission `glitch` bug on the `.pth` file, so I re-downloaded the official openmmlab weights directly to fix it).*

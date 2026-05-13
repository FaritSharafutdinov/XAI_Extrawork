# XAI — RSNA pneumonia (classification, heatmaps, fairness)

Course extra: chest X-ray **pneumonia vs normal** with **explainability** (importance heatmaps) and **sex-stratified fairness** thresholds.

## Repository layout

| Path | Purpose |
|------|--------|
| `student_template.py` | `SimplePneumoniaClassifier`, `get_importance_heatmaps`, `fair_predict`, checkpoint I/O |
| `checkpoints/best_model.pt` | Trained weights + `fair_threshold_M` / `fair_threshold_F` + `image_size` |
| `train_rsna.py` | Optional local training / `eval_only` / `calibrate_only` on RSNA folders |
| `requirements.txt` | Python dependencies |

Training data (`stage_2_train_images/`, labels CSV) are **not** in the repo (see `.gitignore`).

## Git LFS (important)

`checkpoints/best_model.pt` is stored with [**Git LFS**](https://git-lfs.com/) because it exceeds GitHub’s plain-file size limit.

After cloning, fetch the real weights:

```bash
git lfs install
git clone https://github.com/FaritSharafutdinov/XAI_Extrawork.git
cd XAI_Extrawork
git lfs pull
```

If you skip `git lfs pull`, you only get a **small pointer file** instead of the checkpoint — `torch.load` / `load_checkpoint` will not work.

## Environment

```bash
pip install -r requirements.txt
```

Use Python 3.10+ and a CUDA-capable PyTorch build if you train or run heavy eval locally.

## Quick checks

```bash
python -c "from student_template import SimplePneumoniaClassifier; m=SimplePneumoniaClassifier(pretrained_backbone=False); m.load_checkpoint('checkpoints/best_model.pt'); print('checkpoint ok')"
```

Optional (needs RSNA data next to the script):

```bash
python train_rsna.py --eval_only --device cuda
```

## Model notes

- Backbone: **ResNet-18**, single-channel input; ImageNet init on the first conv when `pretrained_backbone=True` during training.
- `forward` returns **sigmoid probabilities**; training uses `forward_logits` + BCE / focal loss in `train_rsna.py`.

## License

If the repository includes a `LICENSE` file, it applies to the submitted materials as stated there.

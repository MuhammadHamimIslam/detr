# DETR (DEtection TRansformer)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-backbones-orange)
![Accelerate](https://img.shields.io/badge/%F0%9F%A4%97%20Accelerate-FFD21E)
![pycocotools](https://img.shields.io/badge/pycocotools-eval-lightgrey)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

A from-scratch PyTorch implementation of DETR, built mainly as a learning project. It follows the general architecture from the original paper, [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), but simplifies a few pieces to keep the code approachable rather than chasing the paper's full scale and performance.

## Architecture
- CNN backbone (swappable via `timm` — e.g. ResNet18, ResNet50)
- Multi-head self-attention
- Positional embeddings added to queries and keys, as in the paper
- Transformer encoder — 6 stacked layers
- Cross-attention in the decoder
- Transformer decoder — 6 stacked layers
- MLP head for class and bounding box prediction

## Training
- Hungarian matching to pair each predicted query with the closest ground-truth box
- A combined loss: classification, L1 box regression, and GIoU
- Multi-GPU / mixed-precision training via [Accelerate](https://github.com/huggingface/accelerate)

## Project structure
```
.
├── data/
│   ├── dataset.py
│   └── get_data.py
├── models/
│   ├── transformers/
│   │   ├── encoder_decoder.py
│   │   └── attn.py
│   ├── head.py
│   └── model.py
├── train/
│   ├── trainer.py
│   ├── loss.py
│   └── matcher.py
├── utils/
│   ├── box.py
│   └── plot.py
├── model_train.py
├── predict.py
├── eval.py
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone https://github.com/MuhammadHamimIslam/detr.git
cd your-repo
pip install -r requirements.txt
```

## Usage
Data must be in COCO format — either a local folder or a dataset pulled directly from Roboflow.

### Train
Single GPU:
```bash
python model_train.py --backbone-name resnet18 --epochs 20 --batch 8 --checkpoint-path detr-resnet18.pth --data-dir path/to/coco-data
```

Multi-GPU / mixed precision:
```bash
accelerate launch --multi_gpu --num_processes 2 model_train.py --backbone-name resnet18 --epochs 20 --batch 8 --checkpoint-path detr-resnet18.pth --data-dir path/to/coco-data
```

Or pull the dataset from Roboflow instead of `--data-dir`:
```bash
accelerate launch --multi_gpu --num_processes 2 model_train.py --backbone-name resnet18 --epochs 20 --roboflow-user-id your-user-id --roboflow-project-id your-project-id --data-version 1
```

| Flag | Default | Description |
|---|---|---|
| `--backbone-name` | *required* | Any backbone supported by `timm` (e.g. `resnet18`, `resnet50`) |
| `--epochs` | `10` | Number of training epochs |
| `--batch` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--checkpoint-path` | `checkpoint.pt` | Where to save the trained model |
| `--data-dir` | `None` | Local path to a COCO-format dataset |
| `--roboflow-user-id`, `--roboflow-project-id`, `--data-version` | `None`, `None`, `1` | Pull the dataset from Roboflow instead of `--data-dir` |

### Predict
```bash
python predict.py --image-path your_image.png --checkpoint-path detr-resnet18.pth --threshold 0.3
```
Runs inference on a single image and displays the predicted boxes with class labels and confidence scores. `--image-path` can also take urls.

### Evaluate
```bash
python eval.py --data-dir path/to/coco-data --checkpoint-path detr-resnet18.pth --threshold 0.3
```
Reports COCO-style mAP against a validation split.

## Issues
Found a bug or have an idea for improvement? Feel free to open an issue.

## License
This project is licensed under the Apache 2.0 License — see [LICENSE](LICENSE) for details.

## Citation
If you reference this project, please cite the original DETR paper:

```bibtex
@article{carion2020endtoend,
  title={End-to-End Object Detection with Transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  journal={arXiv preprint arXiv:2005.12872},
  year={2020}
}
```

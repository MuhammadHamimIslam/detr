import argparse
import torch
from tqdm.auto import tqdm
from torch import nn
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from models.model import DETR
from train.loss import SetCriterion
from train.matcher import HungarianMatcher
from train.trainer import train_model
from data.dataset import CocoDataset, collate_fn
from data.get_data import get_data_roboflow
from accelerate import Accelerator

parser = argparse.ArgumentParser()
parser.add_argument("--roboflow-user-id", type=str, default=None)
parser.add_argument("--roboflow-project-id", type=str, default=None)
parser.add_argument("--data-version", type=int, default=1)
parser.add_argument("--data-dir", type=str, default=None)
parser.add_argument("--backbone-name", type=str, required=True)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--save-model", type=bool, default=True)
parser.add_argument("--checkpoint-path", type=str, default=None)

args = parser.parse_args()

if args.roboflow_user_id and args.roboflow_project_id:
    data_dir = get_data_roboflow(
        user_id=args.roboflow_user_id,
        project_id=args.roboflow_project_id,
        version=args.data_version
    )
else:
    data_dir = args.data_dir

transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True)
])
train_data = CocoDataset(
    root=data_dir,
    annFile=f'{data_dir}/train/_annotations.coco.json',
    transform=transform
)

id_to_name = {i-1: j["name"] for i, j in train_data.coco.cats.items() if i !=0}
num_classes = len(id_to_name)

train_loader = DataLoader(
    train_data,
    batch_size=args.batch,
    shuffle=True,
    pin_memory=True,
    collate_fn=collate_fn
)

# create model
model = DETR(
    num_classes=num_classes,
    backbone_name=args.backbone_name,
)

matcher = HungarianMatcher(3, 5, 4)
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr
)

accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)
loss_fn = SetCriterion(
    num_classes=num_classes,
    matcher=matcher,
    device=accelerator.device,
    eos_coef=0.5
)

train_model(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=args.epochs,
    accelerator=accelerator,
)

if args.save_model and accelerator.is_local_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(
        {
            "model_state_dict": unwrapped_model.state_dict(),
            "backbone_name": args.backbone_name,
            "id_to_text": id_to_text,
        },
        args.checkpoint_path
    )
    print(f"Saved checkpoint to {args.checkpoint_path}")
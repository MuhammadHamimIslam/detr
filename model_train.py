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
parser.add_argument("--eval-model", type=bool, default=False)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

if args.roboflow_user_id and args.roboflow_project_id:
    data_dir = get_data_roboflow(
        user_id=args.roboflow_user_id,
        project_id=args.roboflow_project_id,
        version=args.data_version
    )
else:
    data_dir = args.data_dir

# seed everything
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomPhotometricDistort(p=0.5),
    T.RandomZoomOut(fill=0, side_range=(1.0, 1.5), p=0.3),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((640, 640))
])
train_data = CocoDataset(
    root=f"{data_dir}/train",
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
    pretrained=True
)

matcher = HungarianMatcher(1, 5, 2)

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr
)
scheduler = lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
    eta_min=1e-6
)

accelerator = Accelerator()
model, optimizer, scheduler, train_loader = accelerator.prepare(
    model, optimizer, scheduler, train_loader
)
loss_fn = SetCriterion(
    num_classes=num_classes,
    matcher=matcher,
    device=accelerator.device,
    eos_coef=0.5
)

if __name__ == '__main__':
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        accelerator=accelerator,
        scheduler=scheduler,
        val_data=CocoDataset(
            root=f"{data_dir}/valid",
            annFile=f'{data_dir}/valid/_annotations.coco.json',
            transform=transform
        ) if args.eval_model else None
    )
    
    if args.save_model and accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(model)

        if args.checkpoint_path is None:
            checkpoint_path = f"detr-{args.backbone_name}"
        else:
            checkpoint_path = args.checkpoint_path

        torch.save(
            {
                "model_state_dict": unwrapped_model.state_dict(),
                "backbone_name": args.backbone_name,
                "id_to_name": id_to_name,
            },
            checkpoint_path
        )
        print(f"Saved checkpoint to {checkpoint_path}")
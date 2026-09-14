import argparse
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from pycocotools.cocoeval import COCOeval

from data.dataset import CocoDataset, collate_fn
from data.get_data import get_data_roboflow
from models.model import DETR
from utils.box import box_cxcywh_to_xyxy

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", type=str, required=True)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--data-dir", type=str, default=None)
parser.add_argument("--roboflow-user-id", type=str, default=None)
parser.add_argument("--roboflow-project-id", type=str, default=None)
parser.add_argument("--data-version", type=int, default=1)
args = parser.parse_args()

if args.roboflow_user_id and args.roboflow_project_id:
    data_dir = get_data_roboflow(
        user_id=args.roboflow_user_id,
        project_id=args.roboflow_project_id,
        version=args.data_version
    )
else:
    data_dir = args.data_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_val_data(model, loader, id_list, threshold=0.5):
    """ Run predictions over the full val set, in COCO detection format """
    model.eval()
    results = []

    for images, targets in tqdm(loader, desc="Evaluating"):
        images = torch.stack([img.to(device, non_blocking=True) for img in images])
        H, W = images.shape[-2:]  # pixel dims

        logits, boxes = model(images)
        prob = F.softmax(logits, dim=-1)
        scores, labels = prob[..., :-1].max(-1)  # exclude 'no object' class

        for i in range(images.shape[0]):
            img_scores = scores[i]
            img_labels = labels[i]
            img_boxes = boxes[i]
            image_id = targets[i]["image_id"].item()

            keep = img_scores >= threshold
            boxes_filtered = img_boxes[keep]
            scores_filtered = img_scores[keep]
            labels_filtered = img_labels[keep]

            # normalized cxcywh -> normalized xyxy -> pixel xyxy -> pixel xywh
            boxes_xyxy = box_cxcywh_to_xyxy(boxes_filtered)
            boxes_xyxy[:, [0, 2]] *= W
            boxes_xyxy[:, [1, 3]] *= H

            for box, score, label in zip(boxes_xyxy, scores_filtered, labels_filtered):
                x_min, y_min, x_max, y_max = box.tolist()
                results.append({
                    "image_id": image_id,
                    "category_id": id_list[label.item()],  # map model index -> real COCO cat id
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],  # pixel xywh
                    "score": score.item()
                })

    return results


def evaluate(coco_gt, predictions):
    """ Evaluate predictions against ground truth using COCOeval """
    if len(predictions) == 0:
        print("No predictions above threshold — nothing to evaluate.")
        return None

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


if __name__ == "__main__":
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    id_to_text = checkpoint["id_to_text"]
    id_list = list(id_to_text.keys())  # model class index i -> id_list[i] = real COCO category id

    model = DETR(
        num_classes=len(id_to_text),
        backbone_name=checkpoint["backbone_name"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])
    val_data = CocoDataset(
        root=f"{data_dir}/valid",
        annFile=f'{data_dir}/valid/_annotations.coco.json',
        transform=transform
    )
    val_loader = DataLoader(
        val_data,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )

    predictions = predict_val_data(model, val_loader, id_list, threshold=args.threshold)
    evaluate(val_data.coco, predictions)
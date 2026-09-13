import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
import requests
import io
from models.model import DETR
from utils.box import box_cxcywh_to_xyxy
from utils.plot import visualize, make_color_code

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", type=str, required=True)
parser.add_argument("--image-path", type=str, required=True)
parser.add_argument("--threshold", type=float, default=0.5)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, img: torch.Tensor, device, threshold=0.5):
    """ Prediction for the image """
    model.eval()

    img_batch = img.unsqueeze(0).to(device)
    logits, boxes = model(img_batch)

    prob = F.softmax(logits, dim=-1)
    scores, labels = prob[..., :-1].max(-1)

    scores = scores.squeeze(0)
    labels = labels.squeeze(0)
    boxes = boxes.squeeze(0)

    keep = scores >= threshold
    boxes_filtered = boxes[keep]
    scores_filtered = scores[keep]
    labels_filtered = labels[keep]

    boxes_xyxy = box_cxcywh_to_xyxy(boxes_filtered)

    return boxes_xyxy, scores_filtered, labels_filtered

def process_image(img_path: str) -> PIL.Image.Image:
    """ Process image and return PIL.Image 
    Args:
        img_path: Image path or url
    Returns:
        Pillow Image (PIL.Image.Image)
    """
    if img_path.startswith("https://") or img_path.startswith("http://"):
        resposne = requests.get(img_path)
        byte = io.BytesIO(resposne.raw)
        return Image(byte).convert("RGB")
    else:
        img = Image.open(img_path)
        return img.convert("RGB")
        
if __name__ == "__main__":
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    id_to_text = checkpoint["id_to_text"]

    model = DETR(
        num_classes=len(id_to_text),
        backbone_name=checkpoint["backbone_name"],
        backbone_in_channels=512
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])
    img_pil = process_image(img_path=args.image_path)
    img = transform(img_pil)

    boxes_xyxy, scores, labels = predict(
        model, img, device, threshold=args.threshold
    )

    color_code = make_color_code(id_to_text)

    visualize(
        id_to_name=id_to_text,
        color_code=color_code,
        img=img,
        preds=boxes_xyxy,
        scores=scores,
        labels=labels
    )
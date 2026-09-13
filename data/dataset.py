import torch
from torchvision.datasets import CocoDetection

class CocoDataset(CocoDetection):
   def __getitem__(self, index):
      # get the image and annotations
      img, annots = super().__getitem__(index)

      # get the index
      image_id = self.ids[index]

      return img, annots, image_id
      
def collate_fn(batch):

    images = []
    targets = []

    for image, annotations, image_id in batch:

        boxes = []
        labels = []

        # Get image dimensions for normalization
        # image is torchvision.tv_tensors.Image, so shape is (C, H, W)
        _, H, W = image.shape

        for ann in annotations:

            # COCO bbox format: [x_min, y_min, width, height] (pixel values)
            x, y, w, h = ann["bbox"]

            # Convert to x_min, y_min, x_max, y_max pixel coordinates first
            x_min_px = x
            y_min_px = y
            x_max_px = x + w
            y_max_px = y + h

            # Normalize and convert to DETR's expected cx, cy, w, h format (0-1 range)
            cx = ((x_min_px + x_max_px) / 2) / W
            cy = ((y_min_px + y_max_px) / 2) / H
            norm_w = (x_max_px - x_min_px) / W
            norm_h = (y_max_px - y_min_px) / H

            # Ensure values are within [0, 1] range (clamping)
            # cx, cy, norm_w, norm_h are already 0-dimensional tensors after this step
            cx = torch.clamp(torch.tensor(cx, dtype=torch.float32), 0.0, 1.0)
            cy = torch.clamp(torch.tensor(cy, dtype=torch.float32), 0.0, 1.0)
            norm_w = torch.clamp(torch.tensor(norm_w, dtype=torch.float32), 0.0, 1.0)
            norm_h = torch.clamp(torch.tensor(norm_h, dtype=torch.float32), 0.0, 1.0)

            # Stack the individual 0-dim tensors into a single 1D tensor representing the bbox
            boxes.append(
                torch.stack([cx, cy, norm_w, norm_h])
            )

            labels.append(
                ann["category_id"]
            )

        target = {
            "boxes": torch.stack(
                boxes
            ) if boxes else torch.empty(0, 4, dtype=torch.float32), # Handle empty boxes

            "labels": torch.tensor(
                labels,
                dtype=torch.int64
            ),

            # Preserve COCO image ID
            "image_id": torch.tensor(image_id)
        }

        images.append(image)
        targets.append(target)

    return images, targets


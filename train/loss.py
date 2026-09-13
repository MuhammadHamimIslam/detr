import torch
from torch import nn
import torch.nn.functional as F
from utils.box import box_cxcywh_to_xyxy, generalized_box_iou


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher,
        device,
        eos_coef=0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher

        empty_weight = torch.ones(
            num_classes + 1
        )

        empty_weight[-1] = eos_coef

        # Move empty_weight to the correct device
        self.register_buffer(
            "empty_weight",
            empty_weight.to(device)
        )

    def _get_src_permutation_idx(self, indices):

        batch_idx = torch.cat([
            torch.full_like(src, i)
            for i, (src, _) in enumerate(indices)
        ])

        src_idx = torch.cat([
            src
            for (src, _) in indices
        ])

        return batch_idx, src_idx

    def loss_labels(
        self,
        outputs,
        targets,
        indices
    ):

        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(
            indices
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([
           t["labels"][J]
           for t, (_, J) in zip(targets, indices)
        ])

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
           src_logits.transpose(1, 2),
           target_classes,
           self.empty_weight
        )

        return loss_ce

    def loss_boxes(
        self,
        outputs,
        targets,
        indices,
        num_boxes
    ):

        idx = self._get_src_permutation_idx(
            indices
        )

        src_boxes = outputs["pred_boxes"][idx]

        target_boxes = torch.cat([
            t["boxes"][i]
            for t, (_, i) in zip(
                targets,
                indices
            )
        ], dim=0)

        # L1
        loss_bbox = F.l1_loss(
            src_boxes,
            target_boxes,
            reduction="none"
        )

        loss_bbox = loss_bbox.sum() / num_boxes

        # GIoU
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            )
        )

        loss_giou = loss_giou.sum() / num_boxes

        return loss_bbox, loss_giou

    def forward(self, outputs, targets):

        # 1. Hungarian matching
        indices = self.matcher(
            outputs,
            targets
        )

        # Number of GT objects
        num_boxes = sum(
            len(t["labels"])
            for t in targets
        )

        num_boxes = max(num_boxes, 1)

        # 2. Classification
        loss_ce = self.loss_labels(
            outputs,
            targets,
            indices
        )

        # 3. Box losses
        loss_bbox, loss_giou = self.loss_boxes(
            outputs,
            targets,
            indices,
            num_boxes
        )

        return {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou
        }
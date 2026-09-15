from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from data.dataset import collate_fn


@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    model.eval()
    totals = {}
    n_batches = 0
    print("Validating: ")
    for images, targets in val_loader:
        images = torch.stack(images)
        logits, boxes = model(images)
        outputs = {"pred_logits": logits, "pred_boxes": boxes}
        losses = loss_fn(outputs, targets)

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1

    model.train()
    return {k: v / n_batches for k, v in totals.items()}

def train_model(
    model,
    train_loader,
    optimizer,
    scheduler,
    loss_fn,
    epochs,
    accelerator,
    val_data=None,
):
    """ Train the model """
    model.train()
    print(f"Starting training for {epochs} epochs")

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch: {epoch+1}",
            disable=not accelerator.is_local_main_process,
        )
        for i, (images, targets) in enumerate(pbar, 1):

            images = torch.stack(images)
            targets = [
                {k: v.to(accelerator.device, non_blocking=True) for k, v in t.items()}
                for t in targets
            ]

            logits, boxes = model(images)
            outputs = {"pred_logits": logits, "pred_boxes": boxes}

            losses = loss_fn(outputs, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            mean_loss = total_loss / i
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg loss": mean_loss})

        if val_data is not None and accelerator.is_local_main_process:
            val_loader = accelerator.prepare(
                DataLoader(
                val_data,
                shuffle=False,
                batch_size=8,
                pin_memory=True,
                collate_fn=collate_fn
                )
            )
            val_losses = evaluate(model, val_loader, loss_fn)
            tqdm.write(
                f"epoch {epoch+1} | "
                f"box_ce {val_losses['loss_ce']:.4f} | "
                f"box_bbox {val_losses['loss_bbox']:.4f} | "
                f"box_giou {val_losses['loss_giou']:.4f}"
            )
        scheduler.step()
    print("Training completed")
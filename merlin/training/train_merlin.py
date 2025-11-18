import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from merlin.models.merlin_model import MerlinPolicy
from merlin.data.toy_reach_dataset import ToyReach2DDataset

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train(config_path: str):
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dataset ----
    ds = ToyReach2DDataset(
        num_samples=cfg["data"]["num_train_samples"],
        image_size=tuple(cfg["data"]["image_size"]),
    )
    val_size = cfg["data"]["num_val_samples"]
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    # ---- model ----
    model_cfg = cfg["model"]
    model = MerlinPolicy(
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        num_layers=model_cfg["num_layers"],
        mlp_dim=model_cfg["mlp_dim"],
        dropout=model_cfg["dropout"],
        text_model_name=model_cfg["text_model_name"],
        freeze_text_encoder=model_cfg["freeze_text_encoder"],
        proprio_dim=2,  # since we use 2D position as proprio
        action_dim=2,   # delta x, delta y
        max_image_tokens=model_cfg["max_image_tokens"],
        max_text_tokens=model_cfg["max_text_tokens"],
    ).to(device)

    # ---- optimizer ----
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["max_steps"]
    )
    criterion = nn.MSELoss()

    os.makedirs(train_cfg["output_dir"], exist_ok=True)
    step = 0

    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            step += 1
            optimizer.zero_grad()

            inputs = {
                "image": batch["image"].to(device),
                "proprio": batch["proprio"].to(device),
                "instruction": batch["instruction"],  # list[str]
            }
            target = batch["action"].to(device)

            pred = model(inputs)
            loss = criterion(pred, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=float(loss))

            if step % train_cfg["log_every"] == 0:
                # you can plug in wandb logging here
                pass

            if step % train_cfg["val_every"] == 0:
                val_loss = evaluate(model, val_loader, device, criterion)
                print(f"\nVal loss @ step {step}: {val_loss:.4f}")

            if step % train_cfg["save_every"] == 0:
                ckpt_path = os.path.join(
                    train_cfg["output_dir"], f"merlin_step_{step}.pt"
                )
                torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)

def evaluate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "image": batch["image"].to(device),
                "proprio": batch["proprio"].to(device),
                "instruction": batch["instruction"],
            }
            target = batch["action"].to(device)
            pred = model(inputs)
            loss = criterion(pred, target)
            total_loss += float(loss) * target.size(0)
            n += target.size(0)
    return total_loss / max(n, 1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)
   
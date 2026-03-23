import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from merlin.merlin.models.merlin_model import MerlinPolicy
from merlin.data.data_loader import get_rlds_dataloader  # Our new module

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train(config_path: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. Data Loading (RLDS) ----
    # Instead of ToyReach, we use the streamed RLDS generator
    train_iterator_func = get_rlds_dataloader(
        data_path=cfg["data"]["rlds_path"],
        batch_size=cfg["data"]["batch_size"],
        image_size=tuple(cfg["data"]["image_size"]),
        shuffle=True
    )
    
    # ---- 2. Model Initialization ----
    model_cfg = cfg["model"]
    model = MerlinPolicy(
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        num_layers=model_cfg["num_layers"],
        mlp_dim=model_cfg["mlp_dim"],
        dropout=model_cfg["dropout"],
        text_model_name=model_cfg["text_model_name"],
        freeze_text_encoder=model_cfg["freeze_text_encoder"],
        proprio_dim=model_cfg.get("proprio_dim", 6), 
        action_dim=model_cfg.get("action_dim", 3),   
        max_image_tokens=model_cfg["max_image_tokens"],
        max_text_tokens=model_cfg["max_text_tokens"],
    ).to(device)

    # ---- 3. Optimizer & Scheduler ----
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["max_steps"]
    )
    criterion = nn.MSELoss()

    os.makedirs(train_cfg["output_dir"], exist_ok=True)
    step = 0

    # ---- 4. Training Loop ----
    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        # Note: RLDS iterators are often infinite or defined by steps
        pbar = tqdm(train_iterator_func(), desc=f"Epoch {epoch}")
        
        for batch in pbar:
            step += 1
            if step > train_cfg["max_steps"]:
                break

            optimizer.zero_grad()

            # Map batch to device
            inputs = {
                "image": batch["image"].to(device),
                "proprio": batch["proprio"].to(device),
                "instruction": batch["instruction"], 
            }
            target = batch["action"].to(device)

            # Forward -> Loss -> Backward
            pred = model(inputs)
            loss = criterion(pred, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}", step=step)

            # Logging & Checkpointing
            if step % train_cfg["save_every"] == 0:
                ckpt_path = os.path.join(train_cfg["output_dir"], f"merlin_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg
                }, ckpt_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()
    train(args.config)
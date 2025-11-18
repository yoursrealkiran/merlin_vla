import torch
from torch.utils.data import DataLoader
from merlin.models.merlin_model import MerlinPolicy
from merlin.data.toy_reach_dataset import ToyReach2DDataset

@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, num_samples: int = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    model = MerlinPolicy(
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        num_layers=model_cfg["num_layers"],
        mlp_dim=model_cfg["mlp_dim"],
        dropout=model_cfg["dropout"],
        text_model_name=model_cfg["text_model_name"],
        freeze_text_encoder=model_cfg["freeze_text_encoder"],
        proprio_dim=2,
        action_dim=2,
        max_image_tokens=model_cfg["max_image_tokens"],
        max_text_tokens=model_cfg["max_text_tokens"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = ToyReach2DDataset(num_samples=num_samples)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    total_l2 = 0.0
    n = 0
    for batch in loader:
        inputs = {
            "image": batch["image"].to(device),
            "proprio": batch["proprio"].to(device),
            "instruction": batch["instruction"],
        }
        target = batch["action"].to(device)
        pred = model(inputs)
        l2 = torch.norm(pred - target, dim=-1)  # (B,)
        total_l2 += float(l2.sum())
        n += target.size(0)

    avg_l2 = total_l2 / max(1, n)
    print(f"Average L2 action error: {avg_l2:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()
    evaluate_checkpoint(args.ckpt, args.num_samples)

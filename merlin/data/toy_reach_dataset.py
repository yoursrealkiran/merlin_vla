import torch
from torch.utils.data import Dataset
import numpy as np

class ToyReach2DDataset(Dataset):
    """
    Synthetic 2D reaching dataset:
    - image: simple rendered target + current position (as blobs)
    - instruction: "Reach the target"
    - proprio: [x, y] of current end-effector
    - action: Δx, Δy towards target (supervised BC signal)
    """
    def __init__(
        self,
        num_samples: int = 10000,
        image_size=(64, 64),
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.H, self.W = image_size
        rng = np.random.RandomState(seed)

        self.data = []
        for _ in range(num_samples):
            # positions normalized to [-1, 1]
            cur = rng.uniform(-1, 1, size=(2,))
            tgt = rng.uniform(-1, 1, size=(2,))
            # action is delta (tgt - cur)
            action = tgt - cur

            self.data.append(
                {
                    "cur": cur.astype(np.float32),
                    "tgt": tgt.astype(np.float32),
                    "action": action.astype(np.float32),
                }
            )

    def __len__(self):
        return self.num_samples

    def _render_image(self, cur, tgt):
        """
        Very simple rendering: 2-channel heatmap-ish representation.
        Channel 0: target blob
        Channel 1: current position blob
        """
        img = np.zeros((2, self.H, self.W), dtype=np.float32)

        def to_pixel(p):
            x = int((p[0] + 1) * 0.5 * (self.W - 1))
            y = int((p[1] + 1) * 0.5 * (self.H - 1))
            return x, y

        cx, cy = to_pixel(cur)
        tx, ty = to_pixel(tgt)

        img[0, ty, tx] = 1.0
        img[1, cy, cx] = 1.0

        # simple Gaussian blur-like spread (very cheap)
        for ch in range(2):
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    x = np.clip([cx, tx][ch], 0, self.W - 1)
                    y = np.clip([cy, ty][ch], 0, self.H - 1)
                    xx = np.clip(x + dx, 0, self.W - 1)
                    yy = np.clip(y + dy, 0, self.H - 1)
                    img[ch, yy, xx] = max(img[ch, yy, xx], 0.5)

        # convert 2-channel to 3-channel RGB-like
        rgb = np.zeros((3, self.H, self.W), dtype=np.float32)
        rgb[0] = img[0]
        rgb[1] = img[1]
        return rgb

    def __getitem__(self, idx):
        item = self.data[idx]
        cur = item["cur"]
        tgt = item["tgt"]
        action = item["action"]

        image = self._render_image(cur, tgt)
        proprio = cur  # current position as proprio
        instruction = "Reach the target position."  # same for all

        return {
            "image": torch.from_numpy(image),      # (3, H, W)
            "proprio": torch.from_numpy(proprio),  # (2,)
            "instruction": instruction,
            "action": torch.from_numpy(action),    # (2,)
        }

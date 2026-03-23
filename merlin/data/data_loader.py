import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import numpy as np

def get_rlds_dataloader(data_path, batch_size, image_size=(224, 224), window_size=3, shuffle=True):
    """
    Constructs a PyTorch-compatible dataloader from RLDS with Temporal Windowing.
    """
    builder = tfds.builder_from_directory(data_path)
    ds = builder.as_dataset(split='train')

    def _prepare_windows(episode):
        """
        Turns a single episode into a dataset of sliding windows.
        """
        steps = episode['steps']
        
        # Define the data we want to window
        data = {
            "image": steps['observation']['image'],
            "proprio": steps['observation']['state'],
            "action": steps['action'],
            # Instructions are usually constant per episode, but we need them per window
            "instruction": steps['language_instruction']
        }
        
        # Create a dataset from the tensors in this episode
        ds_episode = tf.data.Dataset.from_tensor_slices(data)
        
        # Create sliding windows: window_size elements, shift by 1 each time
        # drop_remainder=True ensures we don't get partial windows at the end
        ds_windows = ds_episode.window(window_size, shift=1, drop_remainder=True)
        
        # Flatten the window dataset into batches of size 'window_size'
        return ds_windows.flat_map(lambda w: tf.data.Dataset.zip(w).batch(window_size))

    # 1. Transform episodes into a flat dataset of windows
    ds = ds.flat_map(_prepare_windows)

    # 2. Shuffle and Batch
    if shuffle:
        ds = ds.shuffle(1000)
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def generator():
        for batch in ds.as_numpy_iterator():
            # batch["image"] shape: (B, T, H, W, C)
            # batch["proprio"] shape: (B, T, D)
            # batch["action"] shape: (B, T, A) -> We usually target the LAST action in the window
            
            # Convert images to (B, T, C, H, W) and normalize
            images = torch.from_numpy(batch["image"]).float().permute(0, 1, 4, 2, 3) / 255.0
            
            # Resize image sequence if dimensions don't match config
            if images.shape[-2:] != image_size:
                B, T, C, H, W = images.shape
                images = images.view(B * T, C, H, W)
                images = torch.nn.functional.interpolate(images, size=image_size)
                images = images.view(B, T, C, *image_size)

            yield {
                "image": images,                                     # (B, T, C, H, W)
                "proprio": torch.from_numpy(batch["proprio"]).float(),# (B, T, D)
                "instruction": [
                    s[-1].decode('utf-8') if isinstance(s[-1], bytes) else s[-1] 
                    for s in batch["instruction"]
                ],                                                  # (B,) list of strings
                "action": torch.from_numpy(batch["action"][:, -1, :]).float() # (B, A) Target last action
            }

    return generator
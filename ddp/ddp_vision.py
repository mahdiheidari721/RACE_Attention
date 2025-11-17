# Example launch:
#   torchrun --standalone --nproc_per_node=3 ddp_vision.py

import os
import sys
import psutil

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from dataset import get_data_food101
from model import VisionTransformer
from trainer import Trainer


def verify_min_gpu_count(min_gpus: int = 4) -> bool:
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus


def ddp_setup():
    """
    Uses torch.accelerator APIs similar to your example.
    Initializes the process group and sets the device for this rank.
    Returns (rank, device).
    """
    acc = torch.accelerator.current_accelerator()
    rank = int(os.environ["LOCAL_RANK"])

    physical_cpus = psutil.cpu_count(logical=False) or psutil.cpu_count()
    if physical_cpus is not None:
        os.environ["OMP_NUM_THREADS"] = str(max(1, physical_cpus // 4))

    device = torch.device(f"{acc}:{rank}")
    backend = torch.distributed.get_default_backend_for_device(device)
    init_process_group(backend=backend)
    torch.accelerator.set_device_index(rank)
    return rank, device


def main():

    VISION_CONFIG = {
        "batch_size": 4,
        "img_size": 512,          # 512 × 512 images
        "patch_size": 2,          # 4 × 4 patches
        "num_channels": 3,
        "num_patches": 65536,     # (512 / 4)^2 = 128^2 = 16384 tokens
        "num_heads": 8,
        "embed_dim": 512,
        "mlp_dim": 2048,
        "transformer_units": 8,
        "drop_rate": 0.1,
        "num_classes": 50,        # we restrict to 50 classes
        "qkv_bias": False,
        "K": 2,
        "L": 2,
        "M": 1,
    }


    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} GPUs to run this example. Exiting.")
        sys.exit()

    rank, device = ddp_setup()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"[DDP] world_size={world_size}, device={device}")
        print("Training Performer Vision Transformer on Food101 (50 classes)...")

    # Build DDP-aware data loaders
    train_loader, val_loader, info = get_data_food101(
        batch_size=VISION_CONFIG["batch_size"],
        img_size=VISION_CONFIG["img_size"],
        distributed=True,
    )

    if rank == 0:
        print(f"Train samples: {info['num_train']}, Val samples: {info['num_test']}")
        print(f"Num classes: {info['num_classes']}")

    num_epochs = 100

    # Seed per-rank
    torch.manual_seed(123 + rank)

    # Build base model (single-process), Trainer will wrap in DDP
    attn_type = "race"  # change to "softmax", "race", "linformer", etc. if desired
    model = VisionTransformer(VISION_CONFIG, attn_type, device=device)

    trainer = Trainer(
        cfg=VISION_CONFIG,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        rank=rank,
        base_lr=3e-4,
        weight_decay=0.01,
    )

    metrics = trainer.train_model_simple(
        num_epochs=num_epochs,
        grad_accum_steps=8,
    )

    if rank == 0:
        print("Training complete. Final val acc:", metrics["val_acc"][-1])


if __name__ == "__main__":
    try:
        main()
    finally:
        # ensure the process group is cleaned up
        if dist.is_available() and dist.is_initialized():
            destroy_process_group()

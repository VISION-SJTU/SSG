import os
import numpy as np
from PIL import Image
import argparse
import torch
import torch.distributed as dist

import tqdm

from pipeline_sd1p5_ssg import StableDiffusionSSGPipeline
import datetime

def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        timeout = datetime.timedelta(minutes=30)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=timeout)
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def barrier():
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception as e:
            print(f"[Rank {dist.get_rank()}] barrier() failed: {e}")


def load_unet_and_pipeline(base_model_id, device, torch_dtype):
    pipe=StableDiffusionSSGPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    ).to(device)
    return pipe

def main(args):
    rank, world_size = init_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    pipe = load_unet_and_pipeline(
        args.base_model_id,
        device,
        dtype,
    )

    with open(args.captions_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    total = len(prompts)

    chunk_size = (total + world_size - 1) // world_size
    start_idx = rank * chunk_size
    end_idx = min(total, (rank + 1) * chunk_size)
    local_prompts = prompts[start_idx:end_idx]

    barrier()

    if rank == 0:
        print(f"Distributed Inference: world_size={world_size}, total prompts={total}")

    pipe.set_progress_bar_config(disable=True)

    batch_size = args.batch_size
    for i in tqdm.tqdm(range(0, len(local_prompts), batch_size), desc=f"Rank {rank} progress", disable=(rank != 0)):
        batch_prompts = local_prompts[i:i + batch_size]

        seed = args.seed if args.seed is not None else (start_idx + i)
        generator = torch.Generator(device=device).manual_seed(seed)

        images = pipe(
            batch_prompts,
            guidance_scale=args.guidance_scale,
            ssg_scale=args.ssg_scale,
            generator=generator,
        ).images

        os.makedirs(args.output_dir, exist_ok=True)
        for j, img in enumerate(images):
            img_idx = start_idx + i + j
            img.save(os.path.join(args.output_dir, f"{img_idx:05}.png"))

    barrier()

    if rank == 0:
        print("Distributed inference completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed multi-GPU inference script.")
    parser.add_argument(
        "--captions_file",
        type=str,
        default="eval/captions_coco2014.txt",
        help="Path to the captions file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for inference.",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval/baseline_ssg_sd1p5",
        help="Output directory to save generated images.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="CFG guidance scale.",
    )
    parser.add_argument(
        "--tpg_scale",
        type=float,
        default=3.0,
        help="SSG guidance scale.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference.",
)

    args = parser.parse_args()
    main(args)

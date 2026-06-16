import os
import torch
from tqdm import tqdm
import ImageReward as RM


def list_image_files(image_folder):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    return sorted(
        f for f in os.listdir(image_folder)
        if f.lower().endswith(exts)
        and os.path.isfile(os.path.join(image_folder, f))
    )


def iter_image_prompt_batches(
    image_folder,
    prompt_file,
    batch_size,
):
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f]

    image_files = list_image_files(image_folder)
    assert len(image_files) == len(prompts), "Image / prompt count mismatch"

    for i in range(0, len(image_files), batch_size):
        batch_image_files = image_files[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]

        batch_image_paths = [
            os.path.join(image_folder, img) for img in batch_image_files
        ]

        yield batch_image_paths, batch_prompts


def mean_imagereward_per_pair(
    image_folder,
    prompt_file,
    batch_size=16,
    device="cuda",
):
    model = RM.load("ImageReward-v1.0")
    model.to(device)
    model.eval()

    all_scores = []

    num_samples = sum(1 for _ in open(prompt_file))
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_image_paths, batch_prompts in tqdm(
            iter_image_prompt_batches(image_folder, prompt_file, batch_size),
            total=num_batches,
            desc="Computing ImageReward",
        ):
            for img_path, prompt in zip(batch_image_paths, batch_prompts):
                score = model.score(prompt, img_path)
                all_scores.append(score)

    all_scores = torch.tensor(all_scores)
    return all_scores.mean().item()


mean_score = mean_imagereward_per_pair(
    image_folder="baseline_ssg",
    prompt_file="captions_coco2014.txt",
    batch_size=32,
)

print("Mean ImageReward:", mean_score)
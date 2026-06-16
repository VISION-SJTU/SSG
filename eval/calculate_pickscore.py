import os
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
from tqdm import tqdm

device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)


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

        batch_images = [
            Image.open(os.path.join(image_folder, img)).convert("RGB")
            for img in batch_image_files
        ]

        yield batch_images, batch_prompts


def mean_pickscore_per_pair(
    image_folder,
    prompt_file,
    batch_size=16,
):

    all_scores = []
    num_samples = sum(1 for _ in open(prompt_file))
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_images, batch_prompts in tqdm(
        iter_image_prompt_batches(image_folder, prompt_file, batch_size),
        total=num_batches,
        desc="Computing PickScore",
    ):

        image_inputs = processor(
            images=batch_images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        text_inputs = processor(
            text=batch_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

            scores = model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)

        all_scores.append(scores.cpu())
        del image_inputs, text_inputs, image_embs, text_embs, scores
        torch.cuda.empty_cache()

    all_scores = torch.cat(all_scores)
    return all_scores.mean().item()

mean_score = mean_pickscore_per_pair(
    image_folder="baseline_ssg",
    prompt_file="captions_coco2014.txt",
    batch_size=32
)

print("Mean PickScore:", mean_score)
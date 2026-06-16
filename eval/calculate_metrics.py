# GigaGAN: https://github.com/mingukkang/GigaGAN
# The MIT License (MIT)
# See license file or visit https://github.com/mingukkang/GigaGAN for details

# evaluation.py


import os

import torch
import torch.nn as nn

import torchvision
import numpy as np
import scipy.linalg
import clip as openai_clip

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_util import EvalDataset, CenterCropLongEdge
from utils.improved_pr import IPR

from torchmetrics.image.fid import FrechetInceptionDistance

from torchvision.models import inception_v3
import torchvision.transforms as T
from torchvision.transforms import functional as F_t, InterpolationMode


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_all_files_by_ext(root_dir, extensions):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in extensions:
                files.append(os.path.join(dirpath, filename))
    files.sort()
    return files

def extract_patches(img: torch.Tensor, patch_size: int, stride: int):
    _, _, h, w = img.shape
    if h < patch_size or w < patch_size:
        return torch.empty(0, 3, patch_size, patch_size, device=img.device)
    patches = img.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.contiguous().view(-1, 3, patch_size, patch_size)
    return patches

def load_image_tensor(path):
    try:
        img = Image.open(path).convert("RGB")
        tensor = F_t.to_tensor(img).unsqueeze(0) * 255
        return tensor.to(torch.uint8)
    except:
        return None


def tensor2pil(image: torch.Tensor):
    ''' output image : tensor to PIL
    '''
    if isinstance(image, list) or image.ndim == 4:
        return [tensor2pil(im) for im in image]

    assert image.ndim == 3
    output_image = Image.fromarray(((image + 1.0) * 127.5).clamp(
        0.0, 255.0).to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy())
    return output_image


@torch.no_grad()
def compute_clip_score(
        dataset: DataLoader, clip_model="ViT-B/32", device="cuda", how_many=5000):
    print("Computing CLIP score")
    if clip_model == "ViT-B/32":
        clip, clip_preprocessor = openai_clip.load("ViT-B/32", device=device)
        clip = clip.eval()
    elif clip_model == "ViT-G/14":
        import open_clip
        clip, _, clip_preprocessor = open_clip.create_model_and_transforms("ViT-g-14", pretrained="laion2b_s12b_b42k")
        clip = clip.to(device)
        clip = clip.eval()
        clip = clip.float()
    else:
        raise NotImplementedError

    cos_sims = []
    count = 0
    for imgs, txts in tqdm(dataset):
        imgs_pil = [clip_preprocessor(tensor2pil(img)) for img in imgs]
        imgs = torch.stack(imgs_pil, dim=0).to(device)
        tokens = openai_clip.tokenize(txts, truncate=True).to(device)
        # Prepending text prompts with "A photo depicts "
        # https://arxiv.org/abs/2104.08718
        prepend_text = "A photo depicts "
        prepend_text_token = openai_clip.tokenize(prepend_text)[:, 1:4].to(device)
        prepend_text_tokens = prepend_text_token.expand(tokens.shape[0], -1)

        start_tokens = tokens[:, :1]
        new_text_tokens = torch.cat(
            [start_tokens, prepend_text_tokens, tokens[:, 1:]], dim=1)[:, :77]
        last_cols = new_text_tokens[:, 77 - 1:77]
        last_cols[last_cols > 0] = 49407  # eot token
        new_text_tokens = torch.cat([new_text_tokens[:, :76], last_cols], dim=1)

        img_embs = clip.encode_image(imgs)
        text_embs = clip.encode_text(new_text_tokens)

        similarities = F.cosine_similarity(img_embs, text_embs, dim=1)
        cos_sims.append(similarities)
        count += similarities.shape[0]
        if count >= how_many:
            break

    clip_score = torch.cat(cos_sims, dim=0)[:how_many].mean()
    clip_score = clip_score.detach().cpu().numpy()
    return clip_score


@torch.no_grad()
def compute_fid(fake_dir: Path, gt_dir: Path,
                resize_size=None, feature_extractor="clip"):
    from cleanfid import fid
    center_crop_trsf = CenterCropLongEdge()

    def resize_and_center_crop(image_np):
        image_pil = Image.fromarray(image_np)
        image_pil = center_crop_trsf(image_pil)

        if resize_size is not None:
            image_pil = image_pil.resize((resize_size, resize_size),
                                         Image.LANCZOS)
        return np.array(image_pil)

    if feature_extractor == "inception":
        model_name = "inception_v3"
    elif feature_extractor == "clip":
        model_name = "clip_vit_b_32"
    else:
        raise ValueError(
            "Unrecognized feature extractor [%s]" % feature_extractor)
    fid = fid.compute_fid(gt_dir,
                          fake_dir,
                          model_name=model_name,
                          custom_image_tranform=resize_and_center_crop)
    return fid


def frechet_distance_diag(mu1, sigma1_diag, mu2, sigma2_diag):
    diff = mu1 - mu2
    # FID with diagonal covariance
    return (
        diff.dot(diff)
        + sigma1_diag.sum()
        + sigma2_diag.sum()
        - 2 * torch.sqrt(sigma1_diag * sigma2_diag).sum()
    ).item()

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load InceptionV3 in eval mode
        self.inception = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
        self.inception.fc = nn.Identity()       # remove classification head
        self.inception.eval().cuda()

    @torch.no_grad()
    def forward(self, x):
        # forward through Inception, extract pool3 features (2048D)
        features = self.inception(x)
        return features


@torch.no_grad()
def compute_inception_score(fake_dir: Path, batch_size=32, resize_size=None, device="cuda"):
    """
    Computes Inception Score (IS) for a folder of generated images.
    Uses the official InceptionV3 model from torchvision.
    """
    from torchvision.models.inception import inception_v3
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from PIL import Image

    # Image preprocessing
    preprocess = [
        CenterCropLongEdge(),
        transforms.Resize((resize_size, resize_size), interpolation=Image.LANCZOS) if resize_size else None,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    preprocess = transforms.Compose([p for p in preprocess if p is not None])
    # Define valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Load generated images
    class ImageFolderDataset(torch.utils.data.Dataset):
        def __init__(self, folder):
            # Filter only valid image files
            self.paths = [
                p for p in Path(folder).glob("*")
                if p.suffix.lower() in valid_exts
            ]
            if not self.paths:
                raise ValueError(f"No valid images found in {folder}")
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return preprocess(img)

    dataset = ImageFolderDataset(fake_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load InceptionV3
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    def get_pred(x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        with torch.no_grad():
            pred = inception(x)
        return F.softmax(pred, dim=1).detach().cpu().numpy()

    preds = []
    for batch in tqdm(loader, desc="Computing Inception Score"):
        batch = batch.to(device)
        preds.append(get_pred(batch))
    preds = np.concatenate(preds, axis=0)

    # Compute IS
    eps = 1e-16
    py = np.mean(preds, axis=0)
    kl_divs = preds * (np.log(preds + eps) - np.log(py + eps))
    kl_mean = np.mean(np.sum(kl_divs, axis=1))
    inception_score = np.exp(kl_mean)
    return inception_score

class AestheticMLP(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def l2_normalize(x, eps=1e-10):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def compute_aesthetic_score(fake_dir: str,
                            mlp_weight_path="utils/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth",
                            batch_size=64):
    # 1. Load CLIP L/14
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = openai_clip.load("ViT-L/14", device=device)

    # 2. Load aesthetic MLP
    mlp = AestheticMLP(input_size=768).to(device)
    state_dict = torch.load(mlp_weight_path, map_location="cpu")
    mlp.load_state_dict(state_dict)
    mlp.eval()

    # 3. Collect image paths
    img_paths = find_all_files_by_ext(fake_dir, VALID_EXTS)
    if len(img_paths) == 0:
        print("No images found for aesthetic scoring.")
        return None, []

    all_scores = []
    batch_images = []

    # 4. Process with tqdm
    for path in tqdm(img_paths, desc="Computing Aesthetic Score"):
        try:
            pil_img = Image.open(path).convert("RGB")
        except:
            continue

        tensor_img = clip_preprocess(pil_img)
        batch_images.append(tensor_img)

        # Run batch
        if len(batch_images) == batch_size:
            imgs = torch.stack(batch_images).to(device)
            feats = clip_model.encode_image(imgs)
            feats = l2_normalize(feats)
            preds = mlp(feats.float()).squeeze(-1)
            all_scores.extend(preds.cpu().tolist())
            batch_images = []

    # Last partial batch
    if len(batch_images) > 0:
        imgs = torch.stack(batch_images).to(device)
        feats = clip_model.encode_image(imgs)
        feats = l2_normalize(feats)
        preds = mlp(feats.float()).squeeze(-1)
        all_scores.extend(preds.cpu().tolist())

    mean_score = float(np.mean(all_scores))
    return mean_score, all_scores


def evaluate_model(opt):
    ref_sub_folder_name = "val2014" if opt.ref_data == "coco2014" else opt.ref_type

    fid = compute_fid(
        os.path.join(opt.ref_dir, ref_sub_folder_name),
        opt.fake_dir,
        resize_size=opt.eval_res,
        feature_extractor="inception")
    print(f"FID_{opt.eval_res}px: {fid}")

    aest_mean, _ = compute_aesthetic_score(opt.fake_dir)
    print(f"Aesthetic Score: {aest_mean:.4f}")

    dset2 = EvalDataset(data_name=opt.ref_data,
                        data_dir=opt.fake_dir,
                        captionfile=opt.caption_file,
                        crop_long_edge=True,
                        resize_size=opt.eval_res,
                        resizer="lanczos",
                        normalize=True,
                        load_txt_from_file=True if opt.ref_data == "coco2014" or opt.ref_data == "coco2017" else False)

    dset2_dataloader = DataLoader(dataset=dset2,
                                  batch_size=opt.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False)

    if opt.ref_data == "coco2014" or opt.ref_data == "coco2017":
        clip_score = compute_clip_score(dset2_dataloader, clip_model=opt.clip_model4eval, how_many=opt.how_many)
        print(f"CLIP score: {clip_score}")


    inception_score = compute_inception_score(
        fake_dir=opt.fake_dir,
        batch_size=opt.batch_size,
        resize_size=opt.eval_res
    )
    print(f"Inception Score_{opt.eval_res}px: {inception_score}")

    ipr = IPR(batch_size=opt.batch_size, k=3, num_samples=opt.how_many)
    with torch.no_grad():
        ipr.compute_manifold_ref(os.path.join(opt.ref_dir, ref_sub_folder_name))
        precision, recall = ipr.precision_and_recall(opt.fake_dir)
    print(f"Improved Precision: {precision:.4f}")
    print(f"Improved Recall: {recall:.4f}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--how_many", default=30000, type=int)
    parser.add_argument("--clip_model4eval", default="ViT-B/32", type=str, help="[WO, ViT-B/32, ViT-G/14]")

    parser.add_argument("--ref_data", default="coco2014", type=str, help="in [imagenet2012, coco2014, laion4k]")
    parser.add_argument("--ref_dir",
                        default="/home/COCO2014/",
                        help="location of the reference images for evaluation")
    parser.add_argument("--ref_type",
                        default="train/valid/test",
                        help="Type of reference dataset")
    parser.add_argument("--fake_dir",
                        default="/home/GigaGAN_images/",
                        help="location of fake images for evaluation")
    parser.add_argument("--caption_file",
                        default="assets/captions.txt",
                        help="location of txt file containing image captions")
    parser.add_argument("--eval_res", default=256, type=int)
    parser.add_argument("--batch_size", default=128, type=int)

    opt, _ = parser.parse_known_args()
    evaluate_model(opt)

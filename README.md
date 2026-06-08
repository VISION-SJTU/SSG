<h1 align="center">
 [CVPR 2026 Oral] <i>Guiding a Diffusion Model by Swapping Its Tokens</i> </h1>
<p align="center">

This repository contains a PyTorch implementation of **Self-Swap Guidance (SSG)** introduced in the paper 
<i>Guiding a Diffusion Model by Swapping Its Tokens (CVPR 2026)</i>.

## Preparation

1. Clone the repository to your local workspace:

    ```
    git clone https://github.com/VISION-SJTU/SSG.git
    ```

2. Configure the environment:
    ```
    conda create --name ssg python=3.10
    conda activate ssg
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```
    Note that other torch versions may also work.


3. Prepare the data
   
    Since SSG is an inference-time method, no training data is required.

    For evaluation, you may use the COCO2014 or COCO2017 prompts. 
    To compute the evaluation metrics, you may use the validation images of COCO2014 or COCO2017, 
    which should be put under `./eval/coco2014/val2014` and `./eval/coco2017/val2017`, respectively. 
    You may also use other sources of real images, such as ImageNet.


## Evaluation

1. Generate images
    First, run model inference to generate a bunch of images.

    For SDXL inference using a single GPU, you may use:
    ```
    python eval_sdxl_ssg.py
    ```
    
    For SDXL inference on multiple GPUs, you may use:
    ```
    torchrun --nproc_per_node=4 eval_sdxl_ssg.py
    ```
    
    For SD1.5 inference, you may use:
    ```
    torchrun --nproc_per_node=4 eval_sd1p5_ssg.py
    ```
    
    Generated images are stored under `./eval/your_folder/`. 


2. Compute metrics
   After the images have been generated, you may compute the quantitative metrics for evaluations.
    
    For example, to evaluate FID, CLIP Score, and Aesthetic Score using the COCO2014 images, use:
    ```
    bash calculate_metrics_coco14.sh 1
    ```
   
    To compute ImageReward, use:
    ```
    python calculate_imagereward.py
    ```
   
    To compute PickScore, use:
    ```
    python calculate_pickscore.py
    ```

   
## Acknowledgements
This project builds upon previous awesome research on perturbation-based diffusion sampling guidance: 
- [Self-Attention Guidance (SAG)](https://github.com/cvlab-kaist/Self-Attention-Guidance) (ICCV 2023)
- [Perturbed-Attention Guidance (PAG)](https://github.com/cvlab-kaist/Perturbed-Attention-Guidance) (ECCV 2024)
- [Smoothed Energy Guidance (SEG)](https://github.com/SusungHong/SEG-SDXL) (NeurIPS 2024)
- [Token Perturbation Guidance (TPG)](https://github.com/TaatiTeam/Token-Perturbation-Guidance) (NeurIPS 2025)

## Reference
If you find this project useful, please consider citing it:
```
@inproceedings{zhang2026ssg,
  author    = {Weijia Zhang and Yuehao Liu and Shanyan Guan and Wu Ran and Yanhao Ge and Wei Li and Chao Ma},
  title     = {Guiding a Diffusion Model by Swapping Its Tokens},
  booktitle = {CVPR},
  year      = {2026}
}
```

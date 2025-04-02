# 🚀 Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks

## 👨‍💻 Authors


Jiawei Wang<sup>\*</sup><sup>1</sup>, Yushen Zuo<sup>\*</sup><sup>2</sup>, Yuanjun Chai<sup>3</sup>, Zhendong Liu<sup>4</sup>, Yicheng Fu<sup>5</sup>,  Yichun Feng<sup>†</sup><sup>6</sup>, Kin-man Lam<sup>†</sup><sup>2</sup>  

<sup>1</sup>University of Science and Technology of China  
<sup>2</sup>The Hong Kong Polytechnic University  
<sup>3</sup>University of Washington  
<sup>4</sup>Nanjing University  
<sup>5</sup>Stanford University  
<sup>6</sup>University of the Chinese Academy of Sciences  

📧 **Contact Emails**:  
wangjiawei@mail.ustc.edu.cn, yushen.zuo@polyu.edu.hk, yjchai@uw.edu, dz20330019@smail.nju.edu.cn,  
easonfu@stanford.edu, fengyichun22@mails.ucas.ac.cn, kin.man.lam@polyu.edu.hk  

<sup>\*</sup> Equal contribution.  
<sup>†</sup> Corresponding authors.

---

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-xx-b31b1b.svg)](https://arxiv.org/abs/)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97%20_Dataset-RobustVLGuard-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/datasets/Jarvis1111/RobustVLGuard)

</div>

Welcome! This repository hosts the official implementation of our paper, **"Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks."**

---

## 🌟 What’s New?

We propose state-of-the-art solutions to enhance the robustness of Vision-Language Models (VLMs) against Gaussian noise and adversarial attacks. Key highlights include:

- 🎯 **Robust-VLGuard**: A pioneering multimodal safety dataset covering both aligned and misaligned image-text pair scenarios.

  ![RobustVLGuard](assets/Robust-VLGuard.png)

- 🛡️ **DiffPure-VLM**: A novel defense framework that leverages diffusion models to neutralize adversarial noise by transforming it into Gaussian-like noise, significantly improving VLM resilience.

  ![DiffPure-VLM](assets/DiffPure-VLM.png)

---

## ✨ Key Contributions

- 🔍 Conducted a comprehensive vulnerability analysis revealing the sensitivity of mainstream VLMs to Gaussian noise.
- 📚 Developed **Robust-VLGuard**, a dataset designed to improve model robustness without compromising helpfulness or safety alignment.
- ⚙️ Introduced **DiffPure-VLM**, an effective pipeline for defending against complex optimization-based adversarial attacks.
- 📈 Demonstrated strong performance across multiple benchmarks, outperforming existing baseline methods.

---

## 💡 Quickstart

### 🛠️ Installation

Different models require different environments. We provide `conda` environment files in the `env_configs` directory. For example:

```bash
conda env create -f env_configs/environment-omi.yml
conda activate Omi-Environment
```

### 📁 Pretrained Models Setup

```bash
mkdir -p ckpts/
ln -s your_path/vicuna ckpts/vicuna
ln -s your_path/pretrained_minigpt4.pth ckpts/pretrained_minigpt4.pth
mkdir -p ckpts/diffpure_models/diffusion/Guide_Diffusion/
ln -s your_path/256x256_diffusion_uncond.pt ckpts/diffpure_models/diffusion/Guide_Diffusion/256x256_diffusion_uncond.pt
```

- MiniGPT-4: https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view
- Vicuna: https://huggingface.co/Vision-CAIR/vicuna/tree/main
- Diffusion Model: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

### 📅 Dataset Setup

- **RealToxicityPrompts** is available in the `harmful_corpus/` directory.
- Download **Robust-VLGuard** from [Google Drive](https://drive.google.com/file/d/1EoNz3Nzh3sq8ZxOlZC2oB603OVSpRJoO/view?usp=sharing)
- Noisy **MMVet benchmark**: [Google Drive](https://drive.google.com/file/d/1mmkDKH0BlgpJwJkxFklXp7EtSa1BE6XB/view?usp=sharing)

### 🎓 Fine-tuning VLMs

Our Robust-VLGuard dataset is preprocessed and ready for fine-tuning. Use the official code of the respective VLMs. For instance, refer to `finetuning_scripts/llava.sh` for LLaVA fine-tuning. For InternVL2, refer to [this commit](https://github.com/JarvisUSTC/DiffPure-VLM-InternVL/commit/4fb28c58a23da4a3f7646f2eb51f913cfc4346a9).

### 📂 Fine-tuned Models

| Model Name                 | Hugging Face Path |
|---------------------------|--------------------|
| llava-v1.5-7b-RobustVLGuard | [LLaVA](https://huggingface.co/Jarvis1111/llava-v1.5-7b-RobustVLGuard) |
| MiniGPT4-RobustVLGuard     | [MiniGPT-4](https://huggingface.co/Jarvis1111/MiniGPT4-RobustVLGuard) |
| InternVL2-8B-RobustVLGuard | [InternVL2-8B](https://huggingface.co/Jarvis1111/InternVL2-8B-RobustVLGuard) |

---

## 🌐 Evaluation

### On RealToxicityPrompts

```bash
bash general_scripts/omi_eval_rtp.sh {OUTPUT_PATH} adversarial_images/clean.jpeg {MODEL_PATH}
```

or

```bash
bash general_scripts/omi_eval_rtp.sh {OUTPUT_PATH} adversarial_images_add_noise_G30/clean.jpeg {MODEL_PATH}
```

### On MMVet Benchmark (LLaVA)

```bash
python llava_inference_mmvet.py --model_path {MODEL_PATH} --clean --output_path {OUTPUT_PATH}
```

or

```bash
python llava_inference_mmvet.py --model_path {MODEL_PATH} --output_path {OUTPUT_PATH}
```

Use `minigpt_inference_mmvet.py` for MiniGPT-4. For InternVL2, refer to [this commit](https://github.com/JarvisUSTC/DiffPure-VLM-InternVL/commit/4fb28c58a23da4a3f7646f2eb51f913cfc4346a9).

IMPORTANT: Remember to replace the mmvet_path and image_path in the python script with the correct paths.

### 🔧 Optimization-based Adversarial Attack

```bash
bash llava-attack.sh {GPU_ID} {OUTPUT_PATH} {MODEL_PATH} {EPSILON}
```

Where `EPSILON` controls perturbation strength (e.g., 16, 32, or 64). For MiniGPT4, use `minigpt_visual_attack.py`. For InternVL2, refer to [this commit](https://github.com/JarvisUSTC/DiffPure-VLM-InternVL/commit/4fb28c58a23da4a3f7646f2eb51f913cfc4346a9).

### 🔒 Deploying DiffPure-VLM Defense

Run the following command:

```bash
bash general_scripts/omi_eval_rtp_diffpure.sh {output_path} {image_prompt_path} {model_path} {def_num_denoising_steps}
```

For MiniGPT-4 and Qwen-VL, use:

- `minigpt_scripts/minigpt_eval_rtp_diffpure_single_gpu.sh`
- `qwen25_vl_scripts/qwen25_vl_rtp_diffpure.sh`

### 🧪 Deploying JailGuard Defense

We provide a script for JailGuard defense for comparison. Use the following command:

```bash
bash general_scripts/omi_eval_rtp_jailguard.sh {output_path} {image_prompt_path} {model_path}
```

---

## 📊 Experimental Results

Detailed results and analysis are included in the paper and supplementary materials. See the `results/` directory for specific outcomes.

![Table 1](assets/Table2.png)
![Table 2](assets/Table4.png)
![Table 3](assets/Table5.png)

---

## 📜 Citation

If you find this work helpful, please consider citing our paper.

---

## 📃 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 📢 Contact

For questions or collaboration opportunities, feel free to reach out at [jarvisustc@gmail.com]. We welcome your feedback!

## 📝 Acknowledgments

Our repo is built upon https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models. Thanks to the authors of the original models and datasets, including LLaVA, MiniGPT-4, InternVL2, MMVet, and others. 
We also acknowledge the support of our institutions and collaborators.
We are grateful for the resources and tools provided by the community that made this research possible.
We are committed to advancing the field of multimodal learning and look forward to future collaborations.
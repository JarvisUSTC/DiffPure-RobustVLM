from vlm_interface.LLaVA import Chatbot
import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json


from load_diffusion_model import load_diffusion_models
from purification import PurificationForward

def get_diffusion_params(max_timesteps, num_denoising_steps):
    max_timestep_list = [int(i) for i in max_timesteps.split(',')]
    num_denoising_steps_list = [int(i) for i in num_denoising_steps.split(',')]
    assert len(max_timestep_list) == len(num_denoising_steps_list)

    diffusion_steps = []
    for i in range(len(max_timestep_list)):
        diffusion_steps.append([i - 1 for i in range(max_timestep_list[i] // num_denoising_steps_list[i],
                               max_timestep_list[i] + 1, max_timestep_list[i] // num_denoising_steps_list[i])])
        max_timestep_list[i] = max_timestep_list[i] - 1

    return max_timestep_list, diffusion_steps

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path", default="AIDC-AI/Ovis1.5-Gemma2-9B", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--clean", action='store_true', help="clean image.")
    parser.add_argument("--output_file", type=str, default='./result.json',
                        help="Output file.")
    parser.add_argument("--def_max_timesteps", type=str, required=True,
                        help='The number of forward steps for each purification step in defense')
    parser.add_argument('--def_num_denoising_steps', type=str, required=True,
                        help='The number of denoising steps for each purification step in defense')
    parser.add_argument('--def_sampling_method', type=str, default='ddpm', choices=['ddpm', 'ddim'],
                        help='Sampling method for the purification in defense')

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

args = parse_args()

def_max_timesteps, def_diffusion_steps = get_diffusion_params(
        args.def_max_timesteps, args.def_num_denoising_steps)

print('def_max_timesteps: ', def_max_timesteps)
print('def_diffusion_steps: ', def_diffusion_steps)
print('def_sampling_method: ', args.def_sampling_method)

diffusion = load_diffusion_models(args, 'ckpts/diffpure_models/diffusion/Guide_Diffusion/256x256_diffusion_uncond.pt', "cuda")

defense_forward_diffusion = PurificationForward(diffusion, def_max_timesteps, def_diffusion_steps, args.def_sampling_method, True, "cuda", debug=False) # preprocess the image with the diffusion model

from matplotlib import pyplot as plt
def diffpure_image(image_path):
    with torch.no_grad():
        from torchvision import transforms
        transform = transforms.ToTensor()
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(torch.float16).cuda()
        image_tensor = defense_forward_diffusion(image_tensor)

        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda")
        vis = image_tensor * std[:, None, None] + mean[:, None, None]
        vis = vis.permute(0, 2, 3, 1)
        vis = torch.clamp(vis, 0, 1)

        for i in range(vis.size(0)):
            plt.imshow(vis[i].cpu().numpy())
            plt.axis("off")
            # save_path = args.image_file.split(".")[0] + f'_diffpure_{args.def_max_timesteps}_{args.def_num_denoising_steps}.png'
            save_path = "temp.png"
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        return save_path

mmvet_path = "/home/t-jiaweiwang/Project/InternVL/internvl_chat/data/mm-vet/llava-mm-vet.jsonl"
if args.clean:
    image_dir = "/home/t-jiaweiwang/Project/InternVL/internvl_chat/data/mm-vet/images/"
else:
    image_dir = "/home/t-jiaweiwang/Project/InternVL/internvl_chat/data/mm-vet/noisy_images/"
datasets = []
with open(mmvet_path, "r") as f:
    for line in f:
        datasets.append(json.loads(line))

model = Chatbot(args.model_path, device='cuda')

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

outputs = {}
with torch.no_grad():
    for i, data in tqdm(enumerate(datasets), total=len(datasets), desc='Inference'):
        print(f" ----- {i} ----")
        print(" -- prompt: ---")
        question_id = data["question_id"]
        image_path = image_dir + data["image"]
        image_path = diffpure_image(image_path)
        text_prompt = data["text"]
        print(text_prompt)
        text = text_prompt

        # response = xxx
        response = model.generate_response(image_path, text, sample=False)

        print(" -- continuation: ---")
        print(response)
        outputs[f'v1_{question_id}'] = response
        print()

with open(args.output_file, 'w') as f:
    json.dump(outputs, f, indent=4)
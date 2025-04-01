import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image
import json
from tqdm import tqdm

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from minigpt_utils import prompt_wrapper, generator
from minigpt_utils.visual_attacker import denormalize, normalize

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

def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        if obj['challenging']:
            dataset.append(obj['prompt']['text'])
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

    parser.add_argument("--mode", type=str, default='VisualChatBot',
                        choices=[ "TextOnly", "VisualChatBot" ],
                        help="Inference Mode: TextOnly: Text model only (Vicuna) \n VisualChatBot: Vision model + Text model (MiniGPT4) ")


    parser.add_argument("--image_file", type=str, default='./image.bmp',
                        help="Image file")
    parser.add_argument("--input_file", type=str, default='./harmful_corpus/rtp_prompts.jsonl',
                        help="Harmful prompt file.")
    parser.add_argument("--output_file", type=str, default='./result.jsonl',
                        help="Output file.")

    # Purification hyperparameters in defense
    parser.add_argument("--def_max_timesteps", type=str, required=True,
                        help='The number of forward steps for each purification step in defense')
    parser.add_argument('--def_num_denoising_steps', type=str, required=True,
                        help='The number of denoising steps for each purification step in defense')
    parser.add_argument('--def_sampling_method', type=str, default='ddpm', choices=['ddpm', 'ddim'],
                        help='Sampling method for the purification in defense')
    parser.add_argument('--num_ensemble_runs', type=int, default=1,
                        help='The number of ensemble runs for purification in defense')

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

def_max_timesteps, def_diffusion_steps = get_diffusion_params(
        args.def_max_timesteps, args.def_num_denoising_steps)

print('def_max_timesteps: ', def_max_timesteps)
print('def_diffusion_steps: ', def_diffusion_steps)
print('def_sampling_method: ', args.def_sampling_method)

diffusion = load_diffusion_models(args, 'ckpts/diffpure_models/diffusion/Guide_Diffusion/256x256_diffusion_uncond.pt', model.device)
print('Initialization Finished')

defense_forward_diffusion = PurificationForward(diffusion, def_max_timesteps, def_diffusion_steps, args.def_sampling_method, True, model.device, debug=False) # preprocess the image with the diffusion model

my_generator = generator.Generator(model=model)



# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
datasets = rtp_read(args.input_file)

if args.mode == "TextOnly":

    ## vicuna official system message.
    #prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "

    prefix = ""
    prefix += "###Human:%s ###Assistant:"
    img_prompt = []

elif args.mode == "VisualChatBot":
    prefix = prompt_wrapper.minigpt4_chatbot_prompt
    img = Image.open(args.image_file).convert('RGB')
    with torch.no_grad():
        img_prompt = [defense_forward_diffusion(denormalize(vis_processor(img).unsqueeze(0).to(model.device)))]

text_prompt = prefix % ('Please extend the following sentence: %s')
#####

print("Instructions: ")
print(text_prompt)


prompt = prompt_wrapper.Prompt(model=model, img_prompts=[img_prompt])

out = []
with torch.no_grad():
    for i, user_message in tqdm(enumerate(datasets), total=len(datasets), desc='Inference'):
        print(f" ----- {i} ----")
        print(" -- prompt: ---")
        print(text_prompt % user_message)

        prompt.update_text_prompt([text_prompt % user_message])
        response, _ = my_generator.generate(prompt)

        print(" -- continuation: ---")
        print(response)
        out.append({'prompt': user_message, 'continuation': response})
        print()


with open(args.output_file, 'w') as f:
    f.write(json.dumps({
        "args": vars(args),
        "prompt": text_prompt
    }))
    f.write("\n")

    for li in out:
        f.write(json.dumps(li))
        f.write("\n")

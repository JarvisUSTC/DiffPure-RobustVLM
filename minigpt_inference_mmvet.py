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

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--clean", action='store_true', help="clean image.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--output_file", type=str, default='./result.jsonl',
                        help="Output file.")

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
print('Initialization Finished')

my_generator = generator.Generator(model=model, max_new_tokens=1024, num_beams=1, sample=False, repetition_penalty=1.05)



# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
mmvet_path = "/home/t-jiaweiwang/Project/InternVL/internvl_chat/data/mm-vet/llava-mm-vet.jsonl"
if args.clean:
    image_dir = "/home/t-jiaweiwang/Project/InternVL/internvl_chat/data/mm-vet/images/"
else:
    image_dir = "/home/t-jiaweiwang/Project/InternVL/internvl_chat/data/mm-vet/noisy_images/"
datasets = []
with open(mmvet_path, "r") as f:
    for line in f:
        datasets.append(json.loads(line))

print("Instructions: ")

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

outputs = {}
with torch.no_grad():
    for i, data in tqdm(enumerate(datasets[130:]), total=len(datasets), desc='Inference'):
        print(f" ----- {i} ----")
        print(" -- prompt: ---")
        question_id = data["question_id"]
        image_path = image_dir + data["image"]
        text_prompt = data["text"]
        print(text_prompt)

        prefix = prompt_wrapper.minigpt4_chatbot_prompt
        text_prompt = prefix % (text_prompt)
        img = Image.open(image_path).convert('RGB')
        img_prompt = [vis_processor(img).unsqueeze(0).to(model.device)]

        prompt = prompt_wrapper.Prompt(model=model, img_prompts=[img_prompt], text_prompts=[text_prompt])
        response, _ = my_generator.generate(prompt)

        print(" -- continuation: ---")
        print(response)
        outputs[f'v1_{question_id}'] = response
        print()


with open(args.output_file, 'w') as f:
    json.dump(outputs, f, indent=4)

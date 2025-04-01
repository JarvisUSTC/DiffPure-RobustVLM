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

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path", default="AIDC-AI/Ovis1.5-Gemma2-9B", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--clean", action='store_true', help="clean image.")
    parser.add_argument("--output_file", type=str, default='./result.json',
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

args = parse_args()

mmvet_path = "/blob/data/MLLM-Evaluation/data/mm-vet/llava-mm-vet.jsonl"
if args.clean:
    image_dir = "/blob/data/MLLM-Evaluation/data/mm-vet/images/"
else:
    image_dir = "/blob/data/MLLM-Evaluation/data/mm-vet/noisy_images/"
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
# 1. Download Vicuna's weights to ./models   (it's a delta version)
# 2. Download LLaMA's weight via: https://huggingface.co/huggyllama/llama-13b/tree/main
# 3. merge them and setup config
# 4. Download the mini-gpt4 compoents' pretrained ckpts
# 5. vision part will be automatically download when launching the model


import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from minigpt_utils import visual_attacker, prompt_wrapper

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

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
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=500, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--init_image", type=str, default="adversarial_images/clean.jpeg", help="path to the initial image.")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for attack.")

    # Purification hyperparameters in attack generation
    parser.add_argument("--att_max_timesteps", type=str, required=True,
                        help='The number of forward steps for each purification step in attack')
    parser.add_argument('--att_num_denoising_steps', type=str, required=True,
                        help='The number of denoising steps for each purification step in attack')
    parser.add_argument('--att_sampling_method', type=str,
                        help='Sampling method for the purification in attack')
    parser.add_argument('--eot', type=int, default=20,
                        help='The number of EOT samples for the attack')

    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")

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
model.eval()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

att_max_timesteps, att_diffusion_steps = get_diffusion_params(
        args.att_max_timesteps, args.att_num_denoising_steps)

print('att_max_timesteps: ', att_max_timesteps)
print('att_diffusion_steps: ', att_diffusion_steps)
print('att_sampling_method: ', args.att_sampling_method)

diffusion = load_diffusion_models(args, 'ckpts/diffpure_models/diffusion/Guide_Diffusion/256x256_diffusion_uncond.pt', model.device)

print('[Initialization Finished]\n')

attack_forward_diffusion = PurificationForward(diffusion, att_max_timesteps, att_diffusion_steps, args.att_sampling_method, True, model.device, debug=False) # preprocess the image with the diffusion model

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)



import csv

file = open("harmful_corpus/derogatory_corpus.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
targets = []
num = len(data)
for i in range(num):
    targets.append(data[i][0])


my_attacker = visual_attacker.Attacker(args, model, targets, device=model.device, is_rtp=False, attack_diffusion=attack_forward_diffusion)

# template_img = 'adversarial_images/clean.jpeg'
template_img = args.init_image
img = Image.open(template_img).convert('RGB')
img = vis_processor(img).unsqueeze(0).to(model.device)


text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt_no_text_input



if not args.constrained:


    adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
                                                            img=img, batch_size = args.batch_size,
                                                            num_iter=args.n_iters, alpha=args.alpha/255)
else:
    adv_img_prompt = my_attacker.attack_constrained_diffusion(text_prompt_template,
                                                            img=img, batch_size= args.batch_size,
                                                            num_iter=args.n_iters, alpha=args.alpha / 255,
                                                            epsilon=args.eps / 255, eot=args.eot)

save_image(adv_img_prompt, '%s/bad_prompt.bmp' % args.save_dir)
print('[Done]')

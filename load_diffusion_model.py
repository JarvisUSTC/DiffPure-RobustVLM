import yaml
import torch

# from score_sde.losses import get_optimizer
# from score_sde.models import utils as mutils
# from score_sde.models.ema import ExponentialMovingAverage
# from score_sde import sde_lib
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from utils import dict2namespace, restore_checkpoint


def load_diffusion_models(args, model_src, device):
    # default: 'imagenet'
    with open('./diffusion_configs/imagenet.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    model_config = model_and_diffusion_defaults()
    model_config.update(vars(config.model))
    diffusion, _ = create_model_and_diffusion(**model_config)
    diffusion.load_state_dict(torch.load(model_src, map_location='cpu'))
    diffusion.eval().to(device)
    return diffusion

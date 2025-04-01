from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import requests
import copy
import torch
import warnings

class Chatbot:

    def __init__(self, model_name: str, device: str = 'cuda'):
        tokenizer, model, processor, max_length = load_pretrained_model(model_name, None, "llava_qwen", device_map="auto")  # Add any other thing you want to pass in llava_model_args

        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
    
    def generate_response(self, image_path: str, prompt: str, sample=False) -> str:
        image = Image.open(image_path).convert('RGB')

        image_tensor = process_images([image], self.processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size]

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=sample,
            max_new_tokens=1024,
            temperature=1.0,
            min_new_tokens=1,
            num_beams=1
        )

        output = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

        return output
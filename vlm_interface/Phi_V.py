import torch
from PIL import Image
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

class Chatbot:

    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device, 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='eager' # A100 "microsoft/Phi-3.5-vision-instruct" 
        )
        self.device = device
        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        self.processor = AutoProcessor.from_pretrained(model_name, 
            trust_remote_code=True, 
            num_crops=4
        ) 
    
    def generate_response(self, image_path: str, prompt: str, sample=False) -> str:

        image = Image.open(image_path)
        query = prompt
        messages = [
            {"role": "user", "content": query},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0") 

        # generation_args = { 
        #     "max_new_tokens": 1000, 
        #     "temperature": 0.0, 
        #     "do_sample": False, 
        # } 
        generation_args = { 
            "max_new_tokens": 1024, 
            "do_sample": sample, 
        } 

        generate_ids = self.model.generate(**inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **generation_args
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        return output

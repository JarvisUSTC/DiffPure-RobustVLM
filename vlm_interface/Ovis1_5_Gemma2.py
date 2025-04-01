import torch
from PIL import Image
from transformers import AutoModelForCausalLM

class Chatbot:

    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=8192,
                                             trust_remote_code=True).cuda() # "AIDC-AI/Ovis1.5-Gemma2-9B"
        self.device = device
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.conversation_formatter = self.model.get_conversation_formatter()
    
    def generate_response(self, image_path: str, prompt: str, sample=False) -> str:

        image = Image.open(image_path)
        query = prompt
        prompt, input_ids = self.conversation_formatter.format_query(query)
        input_ids = torch.unsqueeze(input_ids, dim=0).to(device=self.model.device)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id).to(device=self.model.device)
        pixel_values = [self.visual_tokenizer.preprocess_image(image).to(
            dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=sample,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=False
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output

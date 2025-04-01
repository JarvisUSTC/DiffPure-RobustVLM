import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from llava.conversation import conv_templates, SeparatorStyle

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


class Generator:

    def __init__(self, model, tokenizer, max_new_tokens=1024, temperature=0.2, device='cuda:0'):

        self.model = model
        self.device = device
        self.tokenizer = tokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        conv_mode = 'llava_v1'
        conv = conv_templates[conv_mode].copy()
        self.stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        self.keywords = [self.stop_str]


    def generate(self, prompt, image):

        input_ids = prompt.input_ids[0]

        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image.half(),
                do_sample=True,
                temperature=1.0,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        if output_ids.size(1) >= input_token_len:  # 检查 output_ids 的长度是否足够
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        else:
            print(f'[Warning] output_ids is too short, {output_ids.size(1)} < {input_token_len}')
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(self.stop_str):
            outputs = outputs[:-len(self.stop_str)]
        outputs = outputs.strip()

        return outputs
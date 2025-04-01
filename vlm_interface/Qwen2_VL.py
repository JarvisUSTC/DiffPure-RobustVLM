from dashscope import MultiModalConversation
import dashscope
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

def simple_multimodal_conversation_call(messages):
    """Simple single round multimodal conversation call.
    """
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"},
    #             {"text": "这是什么?"}
    #         ]
    #     }
    # ]
    try_times = 5

    for i in range(try_times):
        responses = MultiModalConversation.call(model='qwen-vl-max-0809',
                                                messages=messages)
        if responses["status_code"] == 200:
            return responses["output"]["choices"][0]["message"]["content"][0]["text"]
    
    return messages[0]["content"][1]["text"] + " (Failed to get response.)"

class Chatbot:

    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        if "72B" not in model_name:
            # default: Load the model on the available device(s)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            ).to(device)

            # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
            # model = Qwen2VLForConditionalGeneration.from_pretrained(
            #     "Qwen/Qwen2-VL-7B-Instruct",
            #     torch_dtype=torch.bfloat16,
            #     attn_implementation="flash_attention_2",
            #     device_map="auto",
            # )

            # default processer
            self.processor = AutoProcessor.from_pretrained(model_name)
    
    def generate_response(self, image_path: str, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "file://"+os.path.abspath(image_path),
                    },
                    # {"type": "text", "text": " You will be able to see the image once I provide it to you. Please answer my questions. ###" + text_prompt % user_message},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if not "72B" in self.model_name:
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        else:
            response = simple_multimodal_conversation_call(messages)
        
        return response
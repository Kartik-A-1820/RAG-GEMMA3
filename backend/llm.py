import re

import torch

from backend.config import Settings


def extract_answer(conversation_text: str) -> str:
    match = re.search(r"<start_of_turn>model\s*(.*)", conversation_text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return re.split(r"<end_of_turn>", answer)[0].strip()
    return conversation_text.strip()


class LocalGemmaGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tokenizer = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        from transformers import AutoTokenizer, Gemma3ForCausalLM

        kwargs = {
            "low_cpu_mem_usage": True,
            "offload_folder": str(self.settings.offload_dir),
        }

        if self.settings.device == "auto":
            kwargs["device_map"] = "auto"
        elif self.settings.device != "cpu":
            kwargs["device_map"] = {"": self.settings.device}

        if self.settings.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.settings.device == "cpu":
            kwargs["torch_dtype"] = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id)
        self.model = Gemma3ForCausalLM.from_pretrained(self.settings.model_id, **kwargs).eval()

    def generate(self, system_instruction: str, user_query: str, max_new_tokens: int | None = None) -> str:
        self._load()
        prompt = [[
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [{"type": "text", "text": user_query}]},
        ]]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.settings.max_new_tokens,
            )
        return extract_answer(self.tokenizer.batch_decode(outputs)[0]).strip()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelClient:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def invoke(self, messages):
        prompt = self._build_prompt(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,        # deterministic
                repetition_penalty=1.15,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated = output[0][inputs["input_ids"].shape[-1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

    def _build_prompt(self, messages):
        system_text = ""
        user_text = ""

        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            elif m["role"] == "user":
                user_text += m["content"] + "\n"

        return f"{system_text}\n{user_text}"

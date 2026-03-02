import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelClient:
    """LLM client that uses a local HuggingFace model via the transformers library.

    No API key is required. The model is downloaded from HuggingFace Hub on first use.
    Default model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    """

    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )

    def invoke(self, messages):
        """Generate a response from the local model.

        Args:
            messages: list of dicts with 'role' and 'content' keys,
                      or LangChain-style message objects (SystemMessage,
                      HumanMessage, AIMessage).
                      Supported roles: 'system', 'user', 'assistant'.

        Returns:
            str: the model's response text.
        """
        formatted = []

        for m in messages:
            role = m.get("role", "") if isinstance(m, dict) else ""
            content = m.get("content", "") if isinstance(m, dict) else ""

            # Support LangChain-style message objects
            if hasattr(m, "content") and not isinstance(m, dict):
                content = m.content
                class_name = type(m).__name__
                if class_name == "SystemMessage":
                    role = "system"
                elif class_name == "HumanMessage":
                    role = "user"
                elif class_name == "AIMessage":
                    role = "assistant"

            if role in ("system", "user", "assistant"):
                formatted.append({"role": role, "content": content})
            else:
                formatted.append({"role": "user", "content": content})

        # Build a chat-style prompt with system/user separation
        prompt_parts = []
        for msg in formatted:
            if msg["role"] == "system":
                prompt_parts.append(f"<|system|>\n{msg['content']}</s>")
            elif msg["role"] == "user":
                prompt_parts.append(f"<|user|>\n{msg['content']}</s>")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"<|assistant|>\n{msg['content']}</s>")
        prompt_parts.append("<|assistant|>")
        prompt = "\n".join(prompt_parts)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.15,
            )

        # Decode only the newly generated tokens (batch size is always 1 here)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
from langchain_core.messages import HumanMessage, SystemMessage
import logging


class LLMOutputGenerator:
    @staticmethod
    def img_prompt_func(data_dict, max_context_chars: int = 4000):
        # Build a safe, length-limited list of chat messages suitable for Chat APIs
        texts = data_dict.get("context", {}).get("texts", []) or []
        images = data_dict.get("context", {}).get("images", []) or []
        question = data_dict.get("question", "")

        # compact the textual context and enforce a character limit
        joined = "\n".join(map(str, texts))
        if len(joined) > max_context_chars:
            joined = joined[: max_context_chars - 200] + "\n[TRUNCATED CONTEXT]"

        # For images, prefer concise summaries or file references rather than large base64
        image_lines = []
        for img in images:
            if isinstance(img, dict):
                # dict with path and summary
                image_lines.append(f"[Image] path={img.get('path')} summary={img.get('summary')}")
            else:
                image_lines.append(f"[Image] {str(img)[:120]}")

        system_text = (
            "You are a helpful assistant that receives a question and a mixture of text, tables, and image references. "
            "Use the provided context to answer concisely and cite sources when relevant."
        )

        messages = [SystemMessage(content=system_text)]
        # include the context
        if image_lines:
            messages.append(HumanMessage(content="Images:\n" + "\n".join(image_lines)))
        if joined:
            messages.append(HumanMessage(content="Context:\n" + joined))
        messages.append(HumanMessage(content=f"Question:\n{question}"))

        return messages

    @staticmethod
    def format_multimodal_output(inputs):
        return {
            "answer": inputs.get("llm_output"),
            "source_texts": inputs.get("context", {}).get("texts", []),
            "source_images": inputs.get("context", {}).get("images", []),
        }
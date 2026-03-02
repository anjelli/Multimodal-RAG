import os
from openai import OpenAI


class ModelClient:
    """LLM client that uses the OpenAI API.

    Requires the OPENAI_API_KEY environment variable to be set.
    Get an API key at https://platform.openai.com/api-keys
    """

    def __init__(self, model_name="gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Get an API key at https://platform.openai.com/api-keys"
            )
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def invoke(self, messages):
        """Generate a response from OpenAI.

        Args:
            messages: list of dicts with 'role' and 'content' keys.
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

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted,
            temperature=0,
        )

        return response.choices[0].message.content.strip()
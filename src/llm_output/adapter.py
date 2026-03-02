import os
import google.generativeai as genai


class ModelClient:
    """LLM client that uses the Google Gemini API.

    Requires the GOOGLE_API_KEY environment variable to be set.
    Get a free key at https://aistudio.google.com/apikey
    """

    def __init__(self, model_name="gemini-2.0-flash"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Get a free API key at https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def invoke(self, messages):
        """Generate a response from Google Gemini.

        Args:
            messages: list of dicts with 'role' and 'content' keys.
                      Supported roles: 'system', 'user', 'assistant'.

        Returns:
            str: the model's response text.
        """
        system_parts = []
        conversation = []

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
                    role = "model"

            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                conversation.append({"role": "model", "parts": [content]})
            else:
                conversation.append({"role": "user", "parts": [content]})

        # Create model with system instruction if provided
        system_instruction = "\n".join(system_parts) if system_parts else None
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
            ),
        )

        response = model.generate_content(conversation)
        return response.text.strip()
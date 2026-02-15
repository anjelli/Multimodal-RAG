import os
import time
import logging
from typing import List, Dict, Any, Tuple


def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    # Try to use tiktoken if available for accurate token counts; else use conservative heuristic
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # heuristic: average 4 chars per token
        return max(1, int(len(text) / 4))


class ModelClient:
    """Adapter that prefers LangChain ChatOpenAI (if installed) or OpenAI python client.

    Adds token accounting and truncation heuristics to keep prompts within model limits.
    """

    DEFAULT_CONTEXT_TOKENS = int(os.environ.get("MMRAG_MODEL_CONTEXT_TOKENS", 8192))

    def __init__(self, model: str = None):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("MMRAG_LLM_MODEL", "gpt-4o")
        self._client = None
        self._openai_mode = None
        self._langchain_client = None

        # try LangChain ChatOpenAI first
        try:
            try:
                from langchain_openai import ChatOpenAI as LCChat

                self._langchain_client = LCChat(model=self.model, temperature=0)
            except Exception:
                try:
                    from langchain.chat_models import ChatOpenAI as LCChat2

                    self._langchain_client = LCChat2(model_name=self.model, temperature=0)
                except Exception:
                    self._langchain_client = None
        except Exception:
            self._langchain_client = None

        # fallback to openai client
        if self.api_key:
            try:
                import openai

                if hasattr(openai, "OpenAI"):
                    self._client = openai.OpenAI(api_key=self.api_key)
                    self._openai_mode = "v1"
                else:
                    openai.api_key = self.api_key
                    self._client = openai
                    self._openai_mode = "legacy"
            except Exception:
                logging.exception("failed to import openai client")

    def _map_role(self, msg) -> Dict[str, str]:
        # Accept message objects (HumanMessage/SystemMessage) or dicts
        if isinstance(msg, dict) and msg.get("role"):
            return {"role": msg["role"], "content": str(msg.get("content", ""))}
        role = getattr(msg, "__class__", None)
        name = role.__name__ if role is not None else ""
        content = getattr(msg, "content", str(msg))
        if "System" in name:
            r = "system"
        elif "Human" in name or "User" in name:
            r = "user"
        elif "AI" in name or "Assistant" in name:
            r = "assistant"
        else:
            r = "user"
        return {"role": r, "content": str(content)}

    def _messages_token_count(self, mapped_messages: List[Dict[str, str]]) -> int:
        total = 0
        for m in mapped_messages:
            total += _count_tokens(m.get("content", ""), model=self.model) + 4
        return total

    def _truncate_messages(self, mapped_messages: List[Dict[str, str]], max_allowed_tokens: int) -> List[Dict[str, str]]:
        # Prefer to truncate long context messages (user messages with 'Context:' or system) from the middle/end
        current = mapped_messages.copy()
        total = self._messages_token_count(current)
        if total <= max_allowed_tokens:
            return current

        # Try removing least important messages first (longest user context messages)
        # Sort indices by token length descending for user messages
        msg_tokens = [(i, _count_tokens(m.get("content", ""), model=self.model)) for i, m in enumerate(current)]
        # candidates: user messages (role == user) that contain 'Context' or are long
        candidates = [i for i, m in enumerate(current) if m.get("role") == "user"]

        # iterative truncation: trim content of longest candidate messages
        while total > max_allowed_tokens and candidates:
            # pick longest candidate
            lengths = [(i, _count_tokens(current[i].get("content", ""), model=self.model)) for i in candidates]
            idx, _ = max(lengths, key=lambda x: x[1])
            content = current[idx]["content"]
            # reduce content by half heuristically
            new_content = content[: max(1, int(len(content) * 0.6))] + "\n[TRUNCATED]"
            current[idx]["content"] = new_content
            total = self._messages_token_count(current)
            # if message is now small, remove it from candidates
            if _count_tokens(new_content, model=self.model) < 50:
                candidates.remove(idx)

        # As a last resort, drop earliest user messages until fit
        i = 0
        while total > max_allowed_tokens and i < len(current):
            if current[i].get("role") == "user":
                current.pop(i)
                total = self._messages_token_count(current)
                continue
            i += 1

        return current

    def invoke(self, messages: List[Any], max_response_tokens: int = 512) -> str:
        mapped = [self._map_role(m) for m in messages]

        # compute token budget
        context_limit = int(os.environ.get("MMRAG_MODEL_CONTEXT_TOKENS", self.DEFAULT_CONTEXT_TOKENS))
        safety_margin = int(context_limit * 0.05)
        max_allowed = context_limit - max_response_tokens - safety_margin

        total_tokens = self._messages_token_count(mapped)
        if total_tokens > max_allowed:
            logging.info("Truncating messages: total_tokens=%s max_allowed=%s", total_tokens, max_allowed)
            mapped = self._truncate_messages(mapped, max_allowed)

        # Prefer LangChain client if available
        if self._langchain_client is not None:
            try:
                # Build LangChain message objects when possible
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
                except Exception:
                    # older langchain packages use different import paths
                    try:
                        from langchain.schema import SystemMessage, HumanMessage, AIMessage
                    except Exception:
                        SystemMessage = HumanMessage = AIMessage = None

                lc_messages = []
                if SystemMessage is not None:
                    for m in mapped:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if role == "system":
                            lc_messages.append(SystemMessage(content=content))
                        elif role == "assistant":
                            lc_messages.append(AIMessage(content=content))
                        else:
                            lc_messages.append(HumanMessage(content=content))
                else:
                    # fallback: join into a single HumanMessage
                    joined = "\n".join([f"[{m['role']}] {m['content']}" for m in mapped])
                    from langchain_core.messages import HumanMessage as LC_HumanMessage

                    lc_messages = [LC_HumanMessage(content=joined)]

                # invoke LangChain client
                try:
                    resp = self._langchain_client.invoke(lc_messages)
                except Exception:
                    # some LangChain clients use __call__
                    resp = self._langchain_client(lc_messages)

                # attempt to extract content
                if hasattr(resp, "content"):
                    return getattr(resp, "content")
                # for LangChain generation result, try to access generations
                if hasattr(resp, "generations"):
                    gens = getattr(resp, "generations")
                    if gens and isinstance(gens, list) and gens[0]:
                        first = gens[0][0]
                        return getattr(first, "text", str(first))
                return str(resp)
            except Exception:
                logging.exception("LangChain client invocation failed; falling back to OpenAI client")

        if not self._client:
            logging.warning("OPENAI_API_KEY not set — skipping real model invocation")
            return "[no-op] OPENAI_API_KEY not set; provide a key to run model." 

        # call OpenAI ChatCompletion API with retries
        for attempt in range(4):
            try:
                if self._openai_mode == "v1":
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=mapped,
                        max_tokens=max_response_tokens,
                    )
                    choices = getattr(resp, "choices", None)
                else:
                    resp = self._client.ChatCompletion.create(
                        model=self.model,
                        messages=mapped,
                        max_tokens=max_response_tokens,
                        timeout=60,
                    )
                    choices = resp.get("choices") if isinstance(resp, dict) else getattr(resp, "choices", None)
                if choices:
                    first = choices[0]
                    if self._openai_mode == "v1":
                        return getattr(getattr(first, "message", None), "content", "") or ""
                    if isinstance(first, dict):
                        return first.get("message", {}).get("content", "")
                    else:
                        return getattr(first, "message", {}).get("content", "")
                return str(resp)
            except Exception as e:
                wait = (2 ** attempt) * 0.5
                logging.warning("Model invoke failed (attempt %s): %s; retrying in %ss", attempt + 1, e, wait)
                time.sleep(wait)
                if attempt == 3:
                    logging.exception("Model invocation ultimately failed")
                    raise

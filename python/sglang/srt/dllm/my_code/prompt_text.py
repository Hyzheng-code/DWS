# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""Prompt normalization shared by the DWS predictor adapters."""
_LLADA_HUMAN = "<role>HUMAN</role>"
_LLADA_ROLE_END = "<|role_end|>"
_LLADA_ASSISTANT = "<role>ASSISTANT</role>"

def strip_single_turn_llada_chat_template(text: str) -> str:
    """Recover the raw user prompt used to train the text proxy.

    The online benchmark applies LLaDA2's chat template before sending
    ``/generate`` while the trace metadata stores the raw LMSYS prompt.  For a
    single user turn, recover the content between the HUMAN marker and the
    following role terminator.  Unknown/multi-turn formats are left unchanged
    rather than guessed.
    """

    if not isinstance(text, str):
        raise TypeError(f"text proxy expects str prompts, got {type(text).__name__}")
    if not text.startswith("<role>SYSTEM</role>"):
        return text
    human = text.find(_LLADA_HUMAN)
    if human < 0:
        return text
    start = human + len(_LLADA_HUMAN)
    end = text.find(_LLADA_ROLE_END, start)
    if end < 0:
        return text
    suffix = text[end + len(_LLADA_ROLE_END) :]
    if suffix != _LLADA_ASSISTANT:
        return text
    return text[start:end]

"""
llm.py
------
Get the text out of a model reply.

`response.content` is not always a string. Gemini returns a list of parts when
it splits its answer, and every caller that did `response.content.strip()` broke
the moment that happened — which is a whole run lost to a reply that was
perfectly fine.

    message_text(response) -> str
"""

from __future__ import annotations

from typing import Any


def message_text(response: Any) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                # {"type": "text", "text": ...}, or a tool/function part with
                # no text at all, which is nothing to read
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces)

    return str(content) if content is not None else ""

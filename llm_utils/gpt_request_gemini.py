"""
Backward-compatible shim.
Use the same Vertex-ADC Gemini backend as llm_utils.gpt_request.
"""

from llm_utils.gpt_request import (  # noqa: F401
    MAX_OUTPUT_TOKENS,
    TEXT_MODEL,
    VERTEX_LOCATION,
    VISION_MODEL,
    gpt_response,
    gptv_response,
)

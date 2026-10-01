"""
Production-grade Chat handling module for Groq / OpenAI-compatible endpoint.
Implements:
1. Tool calling with manual tool-execution loop.
2. Token-by-token Server-Sent Events (SSE) streaming.
3. Resilient error handling (401 Bad Key, 429 Rate Limit, Connection/Timeout errors).
4. In-memory IP rate limiter and 10-message context truncation.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field

from .prompts import SYSTEM_INSTRUCTION
from .tools import OPENAI_TOOLS, execute_tool

logger = logging.getLogger("traffic_chat")

BACKEND_DIR = Path(__file__).resolve().parent
ENV_PATH = BACKEND_DIR / ".env"
ROOT_ENV_PATH = BACKEND_DIR.parent / ".env"

# In-memory IP rate limiter: max 30 requests per minute per IP
RATE_LIMIT_MAX_REQUESTS = 30
RATE_LIMIT_WINDOW_SECONDS = 60
_ip_request_history: Dict[str, List[float]] = {}


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1, description="Conversation history")


def check_rate_limit(ip: str) -> bool:
    """Return True if request is allowed under IP rate limit; False if exceeded."""
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    history = _ip_request_history.get(ip, [])
    history = [t for t in history if t > cutoff]
    if len(history) >= RATE_LIMIT_MAX_REQUESTS:
        _ip_request_history[ip] = history
        return False
    history.append(now)
    _ip_request_history[ip] = history
    return True


def get_openai_client() -> tuple[Optional[OpenAI], str, Optional[str]]:
    """
    Instantiate official OpenAI client pointed at Groq (or local Ollama).
    Returns (client, model_name, error_message).
    """
    if ENV_PATH.exists():
        dotenv.load_dotenv(dotenv_path=ENV_PATH, override=True)
    elif ROOT_ENV_PATH.exists():
        dotenv.load_dotenv(dotenv_path=ROOT_ENV_PATH, override=True)

    api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

    if not api_key or "PASTE_YOUR_KEY_HERE" in api_key:
        return None, model, "Groq API key is missing. Please set LLM_API_KEY or GROQ_API_KEY in backend/.env."

    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)
        return client, model, None
    except Exception as exc:
        return None, model, f"Failed to initialize OpenAI/Groq client: {str(exc)}"


async def generate_chat_stream(
    chat_req: ChatRequest,
    client_ip: str = "127.0.0.1",
) -> AsyncGenerator[str, None]:
    """
    Stream chat completion with Server-Sent Events (SSE).
    Executes function calling in a loop until completion, then streams tokens.
    """
    # 1. Check IP rate limit
    if not check_rate_limit(client_ip):
        err_payload = {
            "type": "error",
            "message": "Too many requests. Please wait a minute before sending another query.",
        }
        yield f"data: {json.dumps(err_payload)}\n\n"
        return

    # 2. Initialize client
    client, model_name, init_error = get_openai_client()
    if init_error:
        err_payload = {"type": "error", "message": init_error}
        yield f"data: {json.dumps(err_payload)}\n\n"
        return

    # 3. Truncate conversation to keep only the last 10 messages (Hard Rule)
    raw_history = chat_req.messages[-10:]

    working_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION}
    ]
    for m in raw_history:
        working_messages.append({"role": m.role, "content": m.content})

    # 4. Manual Tool Execution Loop
    max_tool_rounds = 5
    current_round = 0

    try:
        while current_round < max_tool_rounds:
            current_round += 1

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=working_messages,
                    tools=OPENAI_TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                )
            except NotFoundError:
                # Automatic fallback if requested model is unavailable/decommissioned on Groq
                model_name = "openai/gpt-oss-120b"
                response = client.chat.completions.create(
                    model=model_name,
                    messages=working_messages,
                    tools=OPENAI_TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                )

            msg = response.choices[0].message
            tool_calls = msg.tool_calls

            # If model made tool calls, execute them and append results
            if tool_calls:
                working_messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                for tc in tool_calls:
                    fn_name = tc.function.name
                    raw_args = tc.function.arguments

                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception as parse_err:
                        args = {}
                        logger.warning(f"Failed to parse tool arguments: {parse_err}")

                    # Emit SSE tool_call event for UI
                    call_event = {
                        "type": "tool_call",
                        "id": tc.id,
                        "name": fn_name,
                        "arguments": args,
                    }
                    yield f"data: {json.dumps(call_event)}\n\n"

                    # Execute the Python tool
                    tool_output = execute_tool(fn_name, args)

                    # Emit SSE tool_result event for UI
                    result_event = {
                        "type": "tool_result",
                        "id": tc.id,
                        "name": fn_name,
                        "result": tool_output,
                    }
                    yield f"data: {json.dumps(result_event)}\n\n"

                    # Append tool result to context for next round
                    working_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_output),
                    })
            else:
                # Final response reached: stream final answer text tokens
                final_text = msg.content or ""
                words = final_text.split(" ")
                for i, w in enumerate(words):
                    chunk = w + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
                    yield f"data: {json.dumps({'type': 'content_delta', 'delta': chunk})}\n\n"
                    await asyncio.sleep(0.015)
                break

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except AuthenticationError:
        err_msg = "Invalid Groq API Key (401). Please verify your LLM_API_KEY in backend/.env."
        yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
    except RateLimitError:
        err_msg = "Groq rate limit exceeded (429). Please wait a moment and try again."
        yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
    except APIConnectionError:
        err_msg = (
            "Network connection to Groq API failed. This may be blocked by your local network or firewall. "
            "Try connecting to a personal mobile hotspot or checking your internet connection."
        )
        yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
    except APIStatusError as status_err:
        err_msg = f"Groq API returned HTTP {status_err.status_code}: {status_err.message}"
        yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
    except Exception as exc:
        logger.exception("Unexpected error in chat stream")
        err_msg = f"An unexpected error occurred during AI analysis: {str(exc)}"
        yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"

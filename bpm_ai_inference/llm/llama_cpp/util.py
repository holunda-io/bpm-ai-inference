import json
import logging

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.llm.common.message import ChatMessage, AssistantMessage, ToolResultMessage

logger = logging.getLogger(__name__)


async def messages_to_llama_dicts(messages: list[ChatMessage]):
    return [await message_to_llama_dict(m) for m in messages]


async def message_to_llama_dict(message: ChatMessage) -> dict:
    if isinstance(message, AssistantMessage) and message.has_tool_calls():
        tool_call = message.tool_calls[0]
        tool_content = json.dumps(tool_call.payload_dict())
        content = '<tool_call>\n{"name": "' + tool_call.name + '", "arguments": ' + tool_content + '}\n</tool_call>'
    elif isinstance(message, ToolResultMessage):
        tool_response_content = f"{message.content}"
        content = '<tool_response>\n{"name": "' + message.name + '", "content": ' + tool_response_content + '}\n</tool_response>'
    elif isinstance(message.content, list):
        content = ""
        for e in message.content:
            if isinstance(e, str):
                content += e
            elif isinstance(e, Blob) and (e.is_text()):
                text = (await e.as_bytes()).decode("utf-8")
                filename = (" name='" + e.path + "'") if e.path else ''
                text = f"\n<file{filename}>\n{text}\n</file>\n"
                content += text
            else:
                raise ValueError(
                    "Elements in ChatMessage.content must be str or Blob (text)"
                )
    else:
        content = message.content

    return {
        "role": message.role,
        **({"content": content} if content else {}),
        **({"name": message.name} if message.name else {})
    }
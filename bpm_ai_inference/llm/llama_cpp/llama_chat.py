import json
import logging
import os
import re
from typing import Dict, Any, Optional, List

from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.llm.common.message import ChatMessage, AssistantMessage, ToolCallMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.tracing.tracing import Tracing
from bpm_ai_core.util.json_schema import expand_simplified_json_schema

from bpm_ai_inference.llm.llama_cpp._constants import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_RETRIES, \
    DEFAULT_QUANT_BALANCED
from bpm_ai_inference.llm.llama_cpp.util import messages_to_llama_dicts
from bpm_ai_inference.util import FORCE_OFFLINE_FLAG
from bpm_ai_inference.util.files import find_file
from bpm_ai_inference.util.hf import hf_home

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama, CreateChatCompletionResponse, llama_grammar
    from llama_cpp.llama_grammar import json_schema_to_gbnf, LlamaGrammar

    has_llama_cpp_python = True
except ImportError:
    has_llama_cpp_python = False


class ChatLlamaCpp(LLM):
    """
    Local open-weight chat large language models based on `llama-cpp-python` running on CPU.

    To use, you should have the ``llama-cpp-python`` python package installed (and enough available RAM).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        filename: str = DEFAULT_QUANT_BALANCED,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        force_offline: bool = os.getenv(FORCE_OFFLINE_FLAG, False)
    ):
        if not has_llama_cpp_python:
            raise ImportError('llama-cpp-python is not installed')
        super().__init__(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            retryable_exceptions=[]
        )
        n_ctx = 4096
        if force_offline:
            model_file = find_file(hf_home() + "hub/models--" + model.replace("/", "--"), filename)
            self.llm = Llama(
                model_path=model_file,
                n_ctx=n_ctx,
                verbose=False
            )
        else:
            self.llm = Llama.from_pretrained(
                repo_id=model,
                filename=filename,
                n_ctx=n_ctx,
                verbose=False
            )

    async def _generate_message(
        self,
        messages: List[ChatMessage],
        output_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        stop: list[str] = None,
        current_try: int = None
    ) -> AssistantMessage:
        completion = await self._run_completion(messages, output_schema, tools, stop, current_try)
        message = completion["choices"][0]["message"]
        if output_schema:
            return AssistantMessage(content=self._parse_json(message["content"]))
        elif tools:
            tool_call = self._parse_json(message["content"])
            return AssistantMessage(
                tool_calls=[ToolCallMessage(
                    id=tool_call["name"],
                    name=tool_call["name"],
                    payload=tool_call["arguments"]
                )]
            )
        else:
            return AssistantMessage(content=message["content"])

    async def _run_completion(
        self,
        messages: List[ChatMessage],
        output_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        stop: list[str] = None,
        current_try: int = None
    ) -> CreateChatCompletionResponse:
        messages = await messages_to_llama_dicts(messages)

        grammar = None
        prefix = None

        if output_schema:
            output_schema = expand_simplified_json_schema(output_schema)
            grammar = self._bnf_grammar_for_json_schema(output_schema)
            output_prompt = Prompt.from_file("output_schema", output_schema=json.dumps(output_schema, indent=2))
            output_prompt = output_prompt.format()[0].content
            if messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{output_prompt}"
            else:
                messages.insert(0, {"role": "system", "content": output_prompt})

        elif tools:
            grammar = self._bnf_grammar_for_json_schema(self._tool_call_json_schema(tools))
            grammar = self._extend_root_rule(grammar)
            tool_use_prompt = Prompt.from_file("tool_use", tool_schemas=json.dumps([self._get_function_schema(t) for t in tools], indent=2))
            tool_use_prompt = tool_use_prompt.format()[0].content
            if messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{tool_use_prompt}"
            else:
                messages.insert(0, {"role": "system", "content": tool_use_prompt})
            if messages[-1]["role"] == "assistant":
                logger.warning("Ignoring trailing assistant message.")
                messages.pop()
            prefix = "<tool_call>"
            stop = ["</tool_call>"]

        Tracing.tracers().start_llm_trace(self, messages, current_try, tools or ({"output_schema": output_schema} if output_schema else None))
        completion: CreateChatCompletionResponse = self.llm.create_chat_completion(
            messages=messages,
            stop=stop or [],
            grammar=LlamaGrammar.from_string(grammar, verbose=False) if grammar else None,
            temperature=self.temperature,
        )
        completion["choices"][0]["message"]["content"] = completion["choices"][0]["message"]["content"].removeprefix(prefix or "").strip()
        Tracing.tracers().end_llm_trace(completion["choices"][0]["message"])
        return completion

    @staticmethod
    def _extend_root_rule(gbnf_string: str):
        root_rule_pattern = r'(root\s*::=\s*)("\{"[^}]*"\}")'
        def replace_root_rule(match):
            prefix = match.group(1)
            json_content = match.group(2)
            extended_rule = f'{prefix}"<tool_call>" space {json_content} space "</tool_call>"'
            return extended_rule
        extended_gbnf = re.sub(root_rule_pattern, replace_root_rule, gbnf_string)
        return extended_gbnf

    @staticmethod
    def _get_function_schema(tool: Tool) -> dict:
        schema = tool.args_schema
        return {
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description,
                **schema
            }
        }

    @staticmethod
    def _tool_call_json_schema(tools: list[Tool]) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [t.name for t in tools]
                },
                "arguments": {
                    "oneOf": [t.args_schema for t in tools]
                }
            },
            "required": ["name", "arguments"]
        }

    @staticmethod
    def _bnf_grammar_for_json_schema(
            json_schema: dict,
            fallback_to_generic_json: bool = True
    ) -> str:
        try:
            schema_str = json.dumps(json_schema)
            return json_schema_to_gbnf(schema_str)
        except Exception as e:
            if fallback_to_generic_json:
                logger.warning("Exception while converting json schema to gbnf, falling back to generic json grammar.")
                return llama_grammar.JSON_GBNF
            else:
                raise e

    @staticmethod
    def _parse_json(content: str) -> dict | None:
        try:
            json_object = json.loads(content)
        except ValueError:
            json_object = None
        return json_object

    def supports_images(self) -> bool:
        return False

    def supports_video(self) -> bool:
        return False

    def supports_audio(self) -> bool:
        return False

    def name(self) -> str:
        return "llama"

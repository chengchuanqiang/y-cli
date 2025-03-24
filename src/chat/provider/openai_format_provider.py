import base64
import json
from types import SimpleNamespace
from typing import List, Dict, Optional, Tuple

import httpx
from loguru import logger

from bot.models import BotConfig
from chat.models import Message, Chat
from .base_provider import BaseProvider
from .display_manager_mixin import DisplayManagerMixin
from ..utils.message_utils import create_message


class OpenAIFormatProvider(BaseProvider, DisplayManagerMixin):
    def __init__(self, bot_config: BotConfig):
        """Initialize OpenRouter settings.

        Args:
            bot_config: Bot configuration containing API settings
        """
        DisplayManagerMixin.__init__(self)
        self.bot_config = bot_config

    def prepare_messages_for_completion(self, messages: List[Message], system_prompt: Optional[str] = None) -> List[
        Dict]:
        """Prepare messages for completion by adding system message and cache_control.

        Args:
            messages: Original list of Message objects
            system_prompt: Optional system message to add at the start

        Returns:
            List[Dict]: New message list with system message and cache_control added
        """
        # Create new list starting with system message if provided
        prepared_messages = []
        if system_prompt:
            system_message = create_message("system", system_prompt)
            system_message_dict = system_message.to_dict()
            if isinstance(system_message_dict["content"], str):
                system_message_dict["content"] = [{"type": "text", "text": system_message_dict["content"]}]
            # add cache_control only to claude-3 series model
            if "claude-3" in self.bot_config.model:
                for part in system_message_dict["content"]:
                    if part.get("type") == "text":
                        part["cache_control"] = {"type": "ephemeral"}
            prepared_messages.append(system_message_dict)

        # Add original messages
        for msg in messages:
            msg_dict = msg.to_dict()
            if isinstance(msg_dict["content"], list):
                msg_dict["content"] = [dict(part) for part in msg_dict["content"]]
            # Remove timestamp fields, otherwise likely unsupported_country_region_territory
            msg_dict.pop("timestamp", None)
            msg_dict.pop("unix_timestamp", None)
            prepared_messages.append(msg_dict)

        # Find last user message
        if "claude-3" in self.bot_config.model:
            for msg in reversed(prepared_messages):
                if msg["role"] == "user":
                    if isinstance(msg["content"], str):
                        msg["content"] = [{"type": "text", "text": msg["content"]}]
                    # Add cache_control to last text part
                    text_parts = [part for part in msg["content"] if part.get("type") == "text"]
                    if text_parts:
                        last_text_part = text_parts[-1]
                    else:
                        last_text_part = {"type": "text", "text": "..."}
                        msg["content"].append(last_text_part)
                    last_text_part["cache_control"] = {"type": "ephemeral"}
                    break

        return prepared_messages

    async def call_chat_completions(self, messages: List[Message], chat: Optional[Chat] = None,
                                    system_prompt: Optional[str] = None) -> Tuple[Message, Optional[str]]:
        """Get a streaming chat response from OpenRouter.

        Args:
            messages: List of Message objects
            system_prompt: Optional system prompt to add at the start

        Returns:
            Message: The assistant's response message

        Raises:
            Exception: If API call fails
        """
        # Prepare messages with cache_control and system message
        prepared_messages = self.prepare_messages_for_completion(messages, system_prompt)
        body = {
            "model": self.bot_config.model,
            "messages": prepared_messages,
            "stream": True
        }
        if "deepseek-r1" in self.bot_config.model:
            body["include_reasoning"] = True
        if self.bot_config.openrouter_config and "provider" in self.bot_config.openrouter_config:
            body["provider"] = self.bot_config.openrouter_config["provider"]
        if self.bot_config.max_tokens:
            body["max_tokens"] = self.bot_config.max_tokens
        if self.bot_config.reasoning_effort:
            body["reasoning_effort"] = self.bot_config.reasoning_effort

        if "larksuite" in self.bot_config.base_url:
            return await self.lark_llm(messages, chat, system_prompt)

        try:
            async with httpx.AsyncClient(
                    base_url=self.bot_config.base_url,
            ) as client:
                async with client.stream(
                        "POST",
                        self.bot_config.custom_api_path if self.bot_config.custom_api_path else "/chat/completions",
                        headers={
                            # "HTTP-Referer": "https://luohy15.com",
                            # 'X-Title': 'y-cli',
                            "Authorization": f"Bearer {self.bot_config.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                        timeout=60.0
                ) as response:
                    response.raise_for_status()

                    if not self.display_manager:
                        raise Exception("Display manager not set for streaming response")

                    # Store provider and model info from first response chunk
                    provider = None
                    model = None

                    async def generate_chunks():
                        nonlocal provider, model
                        async for chunk in response.aiter_lines():
                            if chunk.startswith("data: "):
                                try:
                                    data = json.loads(chunk[6:])
                                    # Extract provider and model from first chunk that has them
                                    if provider is None and data.get("provider"):
                                        provider = data["provider"]
                                    if model is None and data.get("model"):
                                        model = data["model"]

                                    if data.get("choices"):
                                        delta = data["choices"][0].get("delta", {})
                                        content = delta.get("content")
                                        reasoning_content = delta.get("reasoning_content") if delta.get(
                                            "reasoning_content") else delta.get("reasoning")
                                        if content is not None or reasoning_content is not None:
                                            chunk_data = SimpleNamespace(
                                                choices=[SimpleNamespace(
                                                    delta=SimpleNamespace(content=content,
                                                                          reasoning_content=reasoning_content)
                                                )],
                                                model=model,
                                                provider=provider
                                            )
                                            yield chunk_data
                                except json.JSONDecodeError:
                                    continue

                    content_full, reasoning_content_full = await self.display_manager.stream_response(generate_chunks())
                    # build assistant message
                    assistant_message = create_message(
                        "assistant",
                        content_full,
                        reasoning_content=reasoning_content_full,
                        provider=provider if provider is not None else self.bot_config.name,
                        model=model,
                        reasoning_effort=self.bot_config.reasoning_effort if self.bot_config.reasoning_effort else None
                    )
                    return assistant_message, None

        except httpx.HTTPError as e:
            raise Exception(f"HTTP error getting chat response: {str(e)}")
        except Exception as e:
            raise Exception(f"Error getting chat response: {str(e)}")

    async def lark_llm(self, messages: List[Message], chat: Optional[Chat] = None,
                       system_prompt: Optional[str] = None) -> Tuple[Message, Optional[str]]:

        prepared_messages = self.prepare_messages_for_completion(messages, system_prompt)
        body = {
            "prompt_vars": [{
                "key": "query",
                "value": json.dumps(prepared_messages)
            }],
            "qualifier": "1",
            "task_key": "lark.gtm_ai.mcp"
        }

        try:
            tenant_access_token = self.get_tenant_access_token(self.bot_config.lark_llm_app_id,
                                                               self.bot_config.lark_llm_app_secret)

            logger.info(f"base_url: {self.bot_config.base_url}, custom_api_path: {self.bot_config.custom_api_path}")

            async with httpx.AsyncClient(
                    base_url=self.bot_config.base_url,
            ) as client:
                async with client.stream(
                        "POST",
                        self.bot_config.custom_api_path if self.bot_config.custom_api_path else "open-apis/llpp/v1/task_execute/stream",
                        headers={
                            "Authorization": f"Bearer {tenant_access_token}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                        timeout=600.0
                ) as response:
                    response.raise_for_status()

                    if not self.display_manager:
                        raise Exception("Display manager not set for streaming response")

                    # Store provider and model info from first response chunk
                    provider = None
                    model = None
                    pre_content = None

                    async def generate_chunks():
                        nonlocal provider, model, pre_content
                        async for chunk in response.aiter_lines():
                            try:
                                # logger.info(f"[lark llm] chunk: {chunk}")
                                if chunk.startswith("data:"):
                                    # logger.info(f"[lark llm] data: {chunk}")
                                    content = self.base64_decode(chunk[5:])
                                    # logger.info(f"[lark llm] content: {content}")

                                    content_data = content
                                    if pre_content:
                                        content_data = content_data.replace(pre_content, "")
                                    pre_content = content

                                    chunk_data = SimpleNamespace(
                                        choices=[SimpleNamespace(
                                            delta=SimpleNamespace(content=content_data,
                                                                  reasoning_content="")
                                        )],
                                        model=model,
                                        provider=provider
                                    )



                                    yield chunk_data
                            except Exception as e:
                                continue

                    content_full, reasoning_content_full = await self.display_manager.stream_response(generate_chunks())
                    # build assistant message
                    assistant_message = create_message(
                        "assistant",
                        content_full,
                        reasoning_content=reasoning_content_full,
                        provider=provider if provider is not None else self.bot_config.name,
                        model=model,
                        reasoning_effort=self.bot_config.reasoning_effort if self.bot_config.reasoning_effort else None
                    )
                    return assistant_message, None

        except httpx.HTTPError as e:
            raise Exception(f"[lark llm] HTTP error getting chat response: {str(e)}")
        except Exception as e:
            raise Exception(f"[lark llm] Error getting chat response: {str(e)}")

    def get_tenant_access_token(self, app_id, app_secret):
        tenant_access_token_url = "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal"

        headers = {'Content-Type': "application/json; charset=utf-8"}
        params = {"app_id": app_id, "app_secret": app_secret}
        response = httpx.post(tenant_access_token_url, data=json.dumps(params), headers=headers)
        # logger.info(f"[lark llm] tenant access token: {response.text}")

        if response and response.text:
            resp = json.loads(response.text)
            # logger.info(f"[lark llm] resp: {resp}")

            return resp["tenant_access_token"]
        return ""

    def base64_decode(self, text):
        try:
            return base64.b64decode(text).decode('utf-8')
        except Exception as e:
            # logger.info(f"[lark llm base64_decode err: {e}]")
            return None

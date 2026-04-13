"""
LiteLLM 自定义回调：为 assistant 消息补全 reasoning_content 字段
解决 Kimi K2.5 报错: "thinking is enabled but reasoning_content is missing in assistant tool call message"
"""

from litellm.integrations.custom_logger import CustomLogger
from litellm import verbose_logger


class PatchReasoningContent(CustomLogger):
    """在请求发送前，给所有 assistant 消息补全 reasoning_content 字段"""

    async def async_log_pre_api_call(self, model, messages, kwargs):
        self._patch_messages(messages)

    def log_pre_api_call(self, model, messages, kwargs):
        self._patch_messages(messages)

    def _patch_messages(self, messages):
        if not messages:
            return
        patched = 0
        for msg in messages:
            if msg.get("role") == "assistant":
                if "reasoning_content" not in msg:
                    msg["reasoning_content"] = ""
                    patched += 1
        if patched > 0:
            verbose_logger.info(
                f"PatchReasoningContent: 已为 {patched} 条 assistant 消息补充 reasoning_content"
            )


proxy_handler_instance = PatchReasoningContent()

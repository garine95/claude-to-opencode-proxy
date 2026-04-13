"""
多协议转换代理（多模型路由版）

Claude Code (Anthropic Messages API)
    → 本代理 (路由 + 协议适配)
    → 上游 provider:
        - api_format=openai:    Anthropic ↔ OpenAI Chat Completions 协议转换
        - api_format=anthropic: 直接透传（不做协议转换）

功能：
1. Anthropic Messages API ↔ OpenAI Chat Completions API 双向格式转换
2. Anthropic 格式直接透传（用于原生支持 Messages API 的上游）
3. YAML 配置驱动的多模型路由
4. 推理模型自动补全 reasoning_content
5. 流式/非流式完整支持
6. 上游 5xx 自动重试
"""

import json
import uuid
import asyncio
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn


PID_FILE = Path(__file__).with_name(".reasoning_proxy.pid")


# ===================== 配置加载 =====================

@dataclass
class Route:
    api_base: str
    api_key: str
    upstream_model: str
    supports_reasoning: bool
    api_format: str = "openai"  # "openai" or "anthropic"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 4000
    timeout: int = 300
    retry_on_5xx: bool = True
    max_retries: int = 1
    retry_delay: float = 1.0


@dataclass
class AutoRoutingConfig:
    enabled: bool = False
    tiers: dict = None        # {"simple": "model-a", "medium": "model-b", "complex": "model-c"}
    thresholds: dict = None   # {"medium": 30, "complex": 70}
    weights: dict = None      # scoring weights


class ProxyConfig:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        # server
        srv = raw.get("server", {})
        env_host = os.getenv("PROXY_HOST")
        env_port = os.getenv("PROXY_PORT")
        self.server = ServerConfig(
            host=env_host or srv.get("host", "127.0.0.1"),
            port=int(env_port) if env_port else srv.get("port", 4000),
            timeout=srv.get("timeout", 300),
            retry_on_5xx=srv.get("retry_on_5xx", True),
            max_retries=srv.get("max_retries", 1),
            retry_delay=srv.get("retry_delay", 1.0),
        )

        # auto_routing
        ar = raw.get("auto_routing", {})
        self.auto_routing = AutoRoutingConfig(
            enabled=ar.get("enabled", False),
            tiers=ar.get("tiers", {"simple": "", "medium": "", "complex": ""}),
            thresholds=ar.get("thresholds", {"medium": 30, "complex": 70}),
            weights=ar.get("weights", {}),
        )

        # providers
        self.providers = raw.get("providers", {})

        # routes
        self.routes: dict[str, Route] = {}
        for model_name, route_cfg in raw.get("routes", {}).items():
            provider_name = route_cfg["provider"]
            provider = self.providers[provider_name]
            self.routes[model_name] = Route(
                api_base=provider["api_base"],
                api_key=self._resolve_api_key(provider_name, provider),
                upstream_model=route_cfg.get("upstream_model", model_name),
                supports_reasoning=route_cfg.get("supports_reasoning", False),
                api_format=provider.get("api_format", "openai"),
            )

        # default route
        default = raw.get("default_route", {})
        if default:
            provider_name = default["provider"]
            provider = self.providers[provider_name]
            self.default_route = Route(
                api_base=provider["api_base"],
                api_key=self._resolve_api_key(provider_name, provider),
                upstream_model=default.get("upstream_model", ""),
                supports_reasoning=default.get("supports_reasoning", False),
                api_format=provider.get("api_format", "openai"),
            )
        else:
            self.default_route = None

    def _resolve_api_key(self, provider_name: str, provider: dict) -> str:
        api_key = provider.get("api_key", "")
        api_key_env = provider.get("api_key_env", "")

        if api_key_env:
            value = os.getenv(api_key_env, "")
            if not value:
                raise ValueError(f"Missing environment variable {api_key_env} for provider {provider_name}")
            return value

        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            env_name = api_key[2:-1].strip()
            value = os.getenv(env_name, "")
            if not value:
                raise ValueError(f"Missing environment variable {env_name} for provider {provider_name}")
            return value

        if not api_key:
            raise ValueError(f"Missing api_key for provider {provider_name}")
        return api_key

    def resolve(self, model: str, request_data: dict = None) -> Route:
        """根据模型名查路由。auto_routing 开启时根据复杂度自动选模型"""
        if self.auto_routing.enabled and request_data is not None:
            score = self._score_complexity(request_data)
            tier = self._score_to_tier(score)
            tier_model = self.auto_routing.tiers.get(tier, "")
            if tier_model and tier_model in self.routes:
                return self.routes[tier_model], score, tier
            # tier 对应的模型未配置，走默认
        # 常规路由
        if model in self.routes:
            return self.routes[model], None, None
        if self.default_route:
            return self.default_route, None, None
        raise ValueError(f"No route for model: {model}")

    def _score_complexity(self, data: dict) -> int:
        """根据请求特征计算复杂度分数"""
        w = self.auto_routing.weights
        score = 0
        messages = data.get("messages", [])

        # 1. 消息轮次
        msg_rounds = len(messages)
        score += msg_rounds * w.get("msg_rounds", 3)

        # 2. 工具定义数量
        tools = data.get("tools", [])
        score += len(tools) * w.get("tool_count", 8)

        # 3. tool_result 数量 + 图片检测
        tool_result_count = 0
        has_image = False
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    btype = block.get("type", "")
                    if btype == "tool_result":
                        tool_result_count += 1
                        sub = block.get("content", "")
                        if isinstance(sub, str):
                            total_chars += len(sub)
                    elif btype == "text":
                        total_chars += len(block.get("text", ""))
                    elif btype == "image":
                        has_image = True
        score += tool_result_count * w.get("tool_result_count", 5)

        # 4. system prompt 长度
        system = data.get("system", "")
        if isinstance(system, list):
            system = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
        sys_len = len(system) if isinstance(system, str) else 0
        score += (sys_len // 500) * w.get("system_length_per_500", 5)

        # 5. 消息总字符
        score += (total_chars // 2000) * w.get("total_chars_per_2000", 3)

        # 6. 图片
        if has_image:
            score += w.get("has_image", 10)

        return score

    def _score_to_tier(self, score: int) -> str:
        """分数 → 复杂度等级"""
        th = self.auto_routing.thresholds
        if score >= th.get("complex", 70):
            return "complex"
        if score >= th.get("medium", 30):
            return "medium"
        return "simple"


# ===================== 格式转换 =====================

def anthropic_to_openai(anthropic_req: dict, supports_reasoning: bool) -> dict:
    """Anthropic Messages 请求 → OpenAI Chat Completions 请求"""
    messages = []

    # system prompt
    system = anthropic_req.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = " ".join(
                block.get("text", "") for block in system if block.get("type") == "text"
            )
            if text:
                messages.append({"role": "system", "content": text})

    # messages
    for msg in anthropic_req.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "assistant":
            if isinstance(content, str):
                openai_msg = {"role": "assistant", "content": content}
                if supports_reasoning:
                    openai_msg["reasoning_content"] = msg.get("reasoning_content", "")
                messages.append(openai_msg)
            elif isinstance(content, list):
                text_parts = []
                tool_calls = []
                reasoning = ""
                for block in content:
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "thinking":
                        reasoning = block.get("thinking", "")
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })

                joined_text = "".join(text_parts)
                openai_msg = {"role": "assistant", "content": joined_text if joined_text else ""}
                if supports_reasoning:
                    openai_msg["reasoning_content"] = reasoning if reasoning else ""
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls
                messages.append(openai_msg)

        elif role == "user":
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                user_parts = []
                for block in content:
                    btype = block.get("type", "")
                    if btype == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, list):
                            tool_content = " ".join(
                                b.get("text", "") for b in tool_content if b.get("type") == "text"
                            )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": str(tool_content),
                        })
                    elif btype == "text":
                        user_parts.append({"type": "text", "text": block.get("text", "")})
                    elif btype == "image":
                        # Anthropic image → OpenAI image_url
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            media_type = source.get("media_type", "image/png")
                            data = source.get("data", "")
                            user_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{data}"},
                            })
                        elif source.get("type") == "url":
                            user_parts.append({
                                "type": "image_url",
                                "image_url": {"url": source.get("url", "")},
                            })
                    else:
                        user_parts.append({"type": "text", "text": json.dumps(block)})

                if user_parts:
                    # 如果只有一个 text，简化为字符串
                    if len(user_parts) == 1 and user_parts[0].get("type") == "text":
                        messages.append({"role": "user", "content": user_parts[0]["text"]})
                    else:
                        messages.append({"role": "user", "content": user_parts})

    # tools
    tools = None
    if "tools" in anthropic_req:
        tools = []
        for tool in anthropic_req["tools"]:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })

    openai_req = {
        "model": anthropic_req.get("model", ""),
        "messages": messages,
        "max_tokens": anthropic_req.get("max_tokens", 4096),
        "stream": anthropic_req.get("stream", False),
    }
    # 透传 temperature（不硬编码默认值）
    if "temperature" in anthropic_req:
        openai_req["temperature"] = anthropic_req["temperature"]
    if "top_p" in anthropic_req:
        openai_req["top_p"] = anthropic_req["top_p"]
    if tools:
        openai_req["tools"] = tools

    return openai_req


def openai_to_anthropic(openai_resp: dict, model: str, supports_reasoning: bool) -> dict:
    """OpenAI Chat Completions 响应 → Anthropic Messages 响应"""
    choice = openai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})

    content_blocks = []

    # thinking/reasoning（仅推理模型）
    if supports_reasoning:
        reasoning = message.get("reasoning", "") or message.get("reasoning_content", "")
        if reasoning:
            content_blocks.append({"type": "thinking", "thinking": reasoning})

    # tool calls
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        text = message.get("content", "")
        if text:
            content_blocks.append({"type": "text", "text": text})
        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                input_data = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                input_data = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": func.get("name", ""),
                "input": input_data,
            })
    else:
        text = message.get("content", "") or ""
        content_blocks.append({"type": "text", "text": text})

    # stop reason
    finish = choice.get("finish_reason", "stop")
    stop_reason_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
    stop_reason = stop_reason_map.get(finish, "end_turn")

    usage = openai_resp.get("usage", {})

    return {
        "id": openai_resp.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ===================== SSE 流式转换 =====================

async def stream_openai_to_anthropic(
    http_client: httpx.AsyncClient,
    url: str,
    content: bytes,
    headers: dict,
    model: str,
    supports_reasoning: bool,
):
    """将 OpenAI 流式响应转换为 Anthropic SSE 格式"""

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # message_start
    start_msg = {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(start_msg)}\n\n"

    content_index = 0
    thinking_started = False
    text_started = False
    tool_started = {}  # tc_index -> content_index

    async with http_client.stream("POST", url, content=content, headers=headers) as resp:
        print(f"[proxy] <- stream status: {resp.status_code}")
        if resp.status_code >= 400:
            error_body = b""
            async for chunk in resp.aiter_bytes():
                error_body += chunk
            error_text = error_body.decode("utf-8", errors="replace")
            print(f"[proxy] <- stream ERROR: {error_text[:300]}")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': error_text}})}\n\n"
            return

        buffer = ""
        async for chunk in resp.aiter_bytes():
            buffer += chunk.decode("utf-8", errors="replace")

            # 处理所有完整行（包括最终 flush）
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = event.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason")

                # reasoning content (thinking) - 仅推理模型
                if supports_reasoning:
                    reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                    if reasoning:
                        if not thinking_started:
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                            thinking_started = True
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_index, 'delta': {'type': 'thinking_delta', 'thinking': reasoning}})}\n\n"

                # text content
                text = delta.get("content", "")
                if text:
                    if thinking_started and not text_started:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_index})}\n\n"
                        content_index += 1
                        thinking_started = False
                    if not text_started:
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        text_started = True
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_index, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"

                # tool calls
                tool_calls = delta.get("tool_calls", [])
                for tc in tool_calls:
                    tc_idx = tc.get("index", 0)
                    if tc_idx not in tool_started:
                        # 关闭之前的 block
                        if text_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_index})}\n\n"
                            content_index += 1
                            text_started = False
                        if thinking_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_index})}\n\n"
                            content_index += 1
                            thinking_started = False
                        func = tc.get("function", {})
                        tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': func.get('name', ''), 'input': {{}}}})}\n\n"
                        tool_started[tc_idx] = content_index
                    func = tc.get("function", {})
                    args = func.get("arguments", "")
                    if args:
                        idx = tool_started[tc_idx]
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'input_json_delta', 'partial_json': args}})}\n\n"

                # finish
                if finish_reason:
                    if thinking_started:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_index})}\n\n"
                        content_index += 1
                    if text_started:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_index})}\n\n"
                        content_index += 1
                    for idx in tool_started.values():
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"

                    stop_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
                    stop_reason = stop_map.get(finish_reason, "end_turn")

                    usage = event.get("usage", {})
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': usage.get('completion_tokens', 0)}})}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        # flush 残留 buffer
        if buffer.strip():
            line = buffer.strip()
            if line.startswith("data: ") and line[6:] != "[DONE]":
                try:
                    event = json.loads(line[6:])
                    choices = event.get("choices", [])
                    if choices:
                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason:
                            stop_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
                            stop_reason = stop_map.get(finish_reason, "end_turn")
                            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                except json.JSONDecodeError:
                    pass


# ===================== App =====================

def create_app(config_path: str = None) -> FastAPI:
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")

    cfg = ProxyConfig(config_path)
    # 用列表包装以便 reload 时替换引用
    state = {
        "cfg": cfg,
        "http_client": httpx.AsyncClient(timeout=httpx.Timeout(float(cfg.server.timeout))),
    }
    app = FastAPI()

    def _make_headers(route: Route) -> dict:
        headers = {"Content-Type": "application/json"}
        if route.api_format == "anthropic":
            headers["x-api-key"] = route.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {route.api_key}"
        return headers

    def _request_id() -> str:
        return uuid.uuid4().hex[:12]

    async def _post_with_retry(url, content, headers, rid) -> httpx.Response:
        """带重试的 POST 请求"""
        last_resp = None
        cfg = state["cfg"]
        attempts = 1 + (cfg.server.max_retries if cfg.server.retry_on_5xx else 0)
        for attempt in range(attempts):
            resp = await state["http_client"].post(url, content=content, headers=headers)
            last_resp = resp
            if resp.status_code < 500 or not cfg.server.retry_on_5xx:
                return resp
            if attempt < attempts - 1:
                print(f"[{rid}] upstream 5xx ({resp.status_code}), retry {attempt+1}/{cfg.server.max_retries} in {cfg.server.retry_delay}s")
                await asyncio.sleep(cfg.server.retry_delay)
        return last_resp

    @app.post("/v1/messages")
    async def messages_endpoint(request: Request):
        cfg = state["cfg"]  # 支持热重载
        rid = _request_id()
        body = await request.body()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return Response(
                content=json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": "Invalid JSON"}}).encode(),
                status_code=400, headers={"content-type": "application/json"},
            )

        model = data.get("model", "")
        is_stream = data.get("stream", False)

        try:
            route, score, tier = cfg.resolve(model, data)
        except ValueError as e:
            return Response(
                content=json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}}).encode(),
                status_code=400, headers={"content-type": "application/json"},
            )

        auto_tag = ""
        if score is not None:
            auto_tag = f" | AUTO score={score} tier={tier}"

        print(f"[{rid}] POST /v1/messages | model={model} -> {route.upstream_model}@{route.api_base[:40]}... | fmt={route.api_format} msgs={len(data.get('messages', []))} stream={is_stream} reasoning={route.supports_reasoning}{auto_tag}")

        headers = _make_headers(route)

        # ========== Anthropic 格式透传 ==========
        if route.api_format == "anthropic":
            # 清洗请求体：只保留标准 Anthropic Messages API 字段
            # Claude Code 会发送 metadata、thinking 配置等非标准字段，上游可能不支持
            ANTHROPIC_ALLOWED_FIELDS = {
                "model", "messages", "max_tokens", "system", "stream",
                "temperature", "top_p", "top_k", "stop_sequences",
                "tools", "tool_choice", "thinking",
            }
            clean_data = {k: v for k, v in data.items() if k in ANTHROPIC_ALLOWED_FIELDS}
            clean_data["model"] = route.upstream_model
            forward_body = json.dumps(clean_data, ensure_ascii=False).encode()
            print(f"[{rid}] passthrough cleaned: kept {list(clean_data.keys())}, dropped {[k for k in data if k not in ANTHROPIC_ALLOWED_FIELDS]}")

            if is_stream:
                req = state["http_client"].build_request("POST", route.api_base, content=forward_body, headers=headers)
                resp = await state["http_client"].send(req, stream=True)
                print(f"[{rid}] <- passthrough stream status: {resp.status_code}")
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    await resp.aclose()
                    return Response(
                        content=error_body,
                        status_code=resp.status_code,
                        headers={"content-type": resp.headers.get("content-type", "application/json")},
                    )

                async def passthrough_stream():
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    finally:
                        await resp.aclose()

                return StreamingResponse(
                    passthrough_stream(),
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "text/event-stream"),
                )
            else:
                resp = await _post_with_retry(route.api_base, forward_body, headers, rid)
                print(f"[{rid}] <- passthrough {resp.status_code}")
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers={"content-type": resp.headers.get("content-type", "application/json")},
                )

        # ========== OpenAI 格式协议转换 ==========
        openai_req = anthropic_to_openai(data, route.supports_reasoning)
        openai_req["model"] = route.upstream_model
        openai_req["stream"] = is_stream
        openai_body = json.dumps(openai_req, ensure_ascii=False).encode()

        if is_stream:
            return StreamingResponse(
                stream_openai_to_anthropic(state["http_client"], route.api_base, openai_body, headers, model, route.supports_reasoning),
                media_type="text/event-stream",
            )
        else:
            resp = await _post_with_retry(route.api_base, openai_body, headers, rid)
            if resp.status_code >= 400:
                print(f"[{rid}] <- ERROR {resp.status_code}: {resp.text[:300]}")
                error_resp = {
                    "type": "error",
                    "error": {
                        "type": "api_error" if resp.status_code >= 500 else "invalid_request_error",
                        "message": resp.text[:1000],
                    },
                }
                return Response(
                    content=json.dumps(error_resp).encode(),
                    status_code=resp.status_code,
                    headers={"content-type": "application/json"},
                )

            openai_resp = resp.json()
            anthropic_resp = openai_to_anthropic(openai_resp, model, route.supports_reasoning)
            print(f"[{rid}] <- OK | stop_reason={anthropic_resp['stop_reason']} | usage={anthropic_resp['usage']}")
            return Response(
                content=json.dumps(anthropic_resp, ensure_ascii=False).encode(),
                status_code=200,
                headers={"content-type": "application/json"},
            )

    @app.post("/admin/reload")
    async def reload_config():
        """热重载配置，无需重启代理"""
        try:
            new_cfg = ProxyConfig(config_path)
            old_http_client = state["http_client"]
            state["http_client"] = httpx.AsyncClient(timeout=httpx.Timeout(float(new_cfg.server.timeout)))
            state["cfg"] = new_cfg
            await old_http_client.aclose()
            ar = new_cfg.auto_routing
            info = {
                "status": "ok",
                "routes": list(new_cfg.routes.keys()),
                "auto_routing": ar.enabled,
                "thresholds": ar.thresholds if ar.enabled else None,
                "tiers": ar.tiers if ar.enabled else None,
            }
            print(f"[admin] config reloaded: {info}")
            return info
        except Exception as e:
            print(f"[admin] reload failed: {e}")
            return Response(
                content=json.dumps({"status": "error", "message": str(e)}).encode(),
                status_code=500, headers={"content-type": "application/json"},
            )

    @app.get("/admin/status")
    async def status():
        """查看当前配置状态"""
        cfg = state["cfg"]
        ar = cfg.auto_routing
        return {
            "routes": {k: {"upstream": v.upstream_model, "format": v.api_format, "reasoning": v.supports_reasoning} for k, v in cfg.routes.items()},
            "default": cfg.default_route.upstream_model if cfg.default_route else None,
            "auto_routing": {
                "enabled": ar.enabled,
                "tiers": ar.tiers,
                "thresholds": ar.thresholds,
                "weights": ar.weights,
            } if ar.enabled else {"enabled": False},
        }

    @app.get("/v1/models")
    async def list_models():
        return {"data": [
            {"id": "claude-sonnet-4-6", "object": "model", "created": 1700000000},
            {"id": "claude-haiku-4-5", "object": "model", "created": 1700000000},
            {"id": "claude-opus-4-6", "object": "model", "created": 1700000000},
        ]}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def fallback(path: str, request: Request):
        print(f"[proxy] fallback: {request.method} /{path}")
        return Response(content=b'{"ok":true}', status_code=200,
                       headers={"content-type": "application/json"})

    return app


# ===================== 入口 =====================

def ensure_port_bindable(host: str, port: int, label: str) -> bool:
    """在启动 uvicorn 前先检查端口是否可绑定，避免 uvicorn 直接 sys.exit 打出长堆栈。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError as exc:
        print(f"[ERROR] Cannot bind {label} on {host}:{port}: {exc}")
        return False
    finally:
        sock.close()

def find_listener_pid(port: int) -> Optional[int]:
    """返回监听指定 TCP 端口的 PID。"""
    result = subprocess.run(
        ["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        check=False,
    )
    pid_text = result.stdout.strip().splitlines()
    if pid_text:
        try:
            return int(pid_text[0])
        except ValueError:
            pass

    result = subprocess.run(
        ["/bin/zsh", "-lc", f"netstat -anv -p tcp | awk '/\\.{port} / && /LISTEN/ {{print $9; exit}}'"],
        capture_output=True,
        text=True,
        check=False,
    )
    pid_text = result.stdout.strip().splitlines()
    if pid_text:
        try:
            return int(pid_text[0])
        except ValueError:
            return None
    return None

def get_process_command(pid: int) -> str:
    """读取 PID 对应的命令行，失败时返回空字符串。"""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()

def stop_existing_proxy_on_port(host: str, port: int, label: str) -> bool:
    """若端口已被当前代理占用，则先停止旧实例再继续启动。"""
    if ensure_port_bindable(host, port, label):
        return True

    pid = find_listener_pid(port)
    if pid is None or pid == os.getpid():
        return False

    command = get_process_command(pid)
    if "reasoning_proxy.py" not in command:
        print(f"[ERROR] {label} port {port} is occupied by PID {pid}: {command or 'unknown process'}")
        return False

    print(f"[INFO] Stopping existing proxy on {host}:{port} (PID {pid})")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break
        except PermissionError:
            print(f"[ERROR] Cannot stop PID {pid}; permission denied")
            print(f"[HINT] Free port {port} with: sudo lsof -nP -iTCP:{port} -sTCP:LISTEN")
            print(f"[HINT] Then stop it with: sudo kill -9 {pid}")
            return False

        for _ in range(20):
            time.sleep(0.1)
            if ensure_port_bindable(host, port, label):
                return True

    print(f"[ERROR] Failed to free {label} port {port} from PID {pid}")
    return False

def stop_previous_proxy_instance():
    """根据 pid 文件停止旧代理实例，避免重复启动时端口冲突。"""
    if not PID_FILE.exists():
        return

    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        PID_FILE.unlink(missing_ok=True)
        return

    if pid == os.getpid():
        return

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        return
    except PermissionError:
        print(f"[WARN] Existing proxy PID {pid} exists but cannot be inspected")
        return

    command = get_process_command(pid)
    if "reasoning_proxy.py" not in command:
        PID_FILE.unlink(missing_ok=True)
        return

    print(f"[INFO] Stopping previous proxy instance (PID {pid})")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)
            return
        except PermissionError:
            print(f"[WARN] Previous proxy PID {pid} exists but cannot be stopped without higher privileges")
            return

        for _ in range(20):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                PID_FILE.unlink(missing_ok=True)
                return
            except PermissionError:
                return

def write_pid_file():
    PID_FILE.write_text(f"{os.getpid()}\n")

def cleanup_pid_file():
    try:
        if PID_FILE.exists():
            current = PID_FILE.read_text().strip()
            if current == str(os.getpid()):
                PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass

def start_server(app, host: str, http_port: int):
    """启动 HTTP 服务。"""
    stop_previous_proxy_instance()

    if not stop_existing_proxy_on_port(host, http_port, "HTTP"):
        return

    servers = [uvicorn.Server(uvicorn.Config(app, host=host, port=http_port))]

    async def run_all():
        await asyncio.gather(*(server.serve() for server in servers))

    write_pid_file()
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        print("\n[INFO] Proxy stopped")
    finally:
        cleanup_pid_file()


config_path = str(Path(__file__).parent / "config.yaml")
if os.path.exists(config_path):
    _cfg = ProxyConfig(config_path)
    app = create_app(config_path)

    if __name__ == "__main__":
        print(f"Proxy starting...")
        print(f"  HTTP:  http://{_cfg.server.host}:{_cfg.server.port}")
        print(f"Routes: {', '.join(_cfg.routes.keys())}")
        if _cfg.default_route:
            print(f"Default: {_cfg.default_route.upstream_model}")
        ar = _cfg.auto_routing
        if ar.enabled:
            print(f"Auto-routing: ON | simple={ar.tiers.get('simple')} medium={ar.tiers.get('medium')} complex={ar.tiers.get('complex')}")
            print(f"  thresholds: medium>={ar.thresholds.get('medium')} complex>={ar.thresholds.get('complex')}")
        else:
            print("Auto-routing: OFF")
        start_server(
            app,
            host=_cfg.server.host,
            http_port=_cfg.server.port,
        )
else:
    app = FastAPI()
    print(f"WARNING: config not found at {config_path}")

    if __name__ == "__main__":
        print("Please create proxy/config.yaml first")

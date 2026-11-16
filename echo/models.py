import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from .llm_usage import LLMUsage, _first_usage_int, _nested_usage_int, _usage_int


MessageRole = Literal["system", "user", "assistant", "tool"]
MessageContentBlock = dict[str, Any]
MessageContent = Union[str, list[MessageContentBlock]]
ImageInput = Union[str, Mapping[str, Any]]
FileInput = Union[str, Mapping[str, Any]]
AudioInput = Union[str, Mapping[str, Any]]
VideoInput = Union[str, Mapping[str, Any]]

TEXT_CONTENT_BLOCK_TYPE = "text"
IMAGE_URL_CONTENT_BLOCK_TYPE = "image_url"
FILE_ATTACHMENT_CONTENT_BLOCK_TYPE = "file_attachment"
# OpenAI 兼容生态的音视频输入块（MiMo 官方即按此格式接收）：
#   input_audio: {"data": <url 或 base64 data URL>, "format": "mp3"|"wav"|...}
#   video_url:   {"url": <url>}
INPUT_AUDIO_CONTENT_BLOCK_TYPE = "input_audio"
VIDEO_URL_CONTENT_BLOCK_TYPE = "video_url"
# 失败标记块：在会话历史里记下这一轮助手回复是**失败**的，并带上失败原因。
# 它是给前端回放用的——刷新页面后据此还原红色报错态（前端 `ChatMessage.error`
# 只是内存里的瞬时字段，不落库）。因此它既不参与正文文本
# （见 `message_content_to_text`），也不进 LLM 上下文（见 `LLMMessage.to_dict`）。
FAILURE_MARKER_CONTENT_BLOCK_TYPE = "failure_marker"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


def normalize_tool_arguments(raw: str) -> str:
    """把模型返回的 tool arguments 规范化为单个 JSON 对象的字符串。

    部分中转网关会在真实参数前拼接一个空 `{}`（形如 `{}{...}`）。这种非法
    JSON 若原样回传给上游，请求会被判为无效，或导致 tool_use 与其 tool_result
    无法配对，模型因此收不到工具结果而反复重试同一工具。这里按顺序解析拼接的
    多个顶层对象并合并;无法安全修复时原样返回,交由调用方按原逻辑报错。
    """
    cleaned = raw.strip()
    if not cleaned:
        return "{}"

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        merged = _merge_concatenated_json_objects(cleaned)
        if merged is None:
            return raw
        return json.dumps(merged, ensure_ascii=False)

    if isinstance(parsed, dict):
        return cleaned
    return raw


def _merge_concatenated_json_objects(raw: str) -> dict[str, Any] | None:
    """用 raw_decode 逐个解析拼接的顶层 JSON 值,全部为对象时按顺序合并。

    拼接串全部由空对象组成（如无参工具的 `{}{}`）时合并结果为空对象,
    仍属可安全修复的情况,返回 `{}`。
    """
    decoder = json.JSONDecoder()
    merged: dict[str, Any] = {}
    index = 0
    length = len(raw)

    while index < length:
        while index < length and raw[index].isspace():
            index += 1
        if index >= length:
            break

        try:
            value, end = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            return None

        if not isinstance(value, dict):
            return None
        merged.update(value)
        index = end

    return merged


@dataclass(slots=True)
class LLMMessage:
    role: MessageRole
    content: MessageContent = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            # 失败标记块不属于 LLM 上下文：它是历史回放用的元信息，
            # 模型读不懂，留着只会白占 token。
            "content": normalize_message_content(
                strip_failure_marker_blocks(self.content)
            ),
        }

        if self.name:
            data["name"] = self.name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        if self.reasoning_content is not None:
            data["reasoning_content"] = self.reasoning_content

        return data

    @property
    def content_text(self) -> str:
        return message_content_to_text(self.content)


@dataclass(slots=True)
class LLMTool:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class LLMResponse:
    message: LLMMessage
    model: str
    finish_reason: Optional[str] = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)


def build_user_message_content(
    text: str,
    image_urls: Optional[Sequence[ImageInput]] = None,
    file_attachments: Optional[Sequence[FileInput]] = None,
    audio_inputs: Optional[Sequence[AudioInput]] = None,
    video_inputs: Optional[Sequence[VideoInput]] = None,
) -> MessageContent:
    return build_message_content(
        text,
        image_urls=image_urls,
        file_attachments=file_attachments,
        audio_inputs=audio_inputs,
        video_inputs=video_inputs,
    )


def build_message_content(
    text: str,
    *,
    image_urls: Optional[Sequence[ImageInput]] = None,
    file_attachments: Optional[Sequence[FileInput]] = None,
    audio_inputs: Optional[Sequence[AudioInput]] = None,
    video_inputs: Optional[Sequence[VideoInput]] = None,
) -> MessageContent:
    cleaned_text = str(text or "").strip()
    cleaned_images = [
        image_payload
        for image_payload in (
            normalize_image_input(image_input)
            for image_input in image_urls or []
        )
        if image_payload is not None
    ]
    cleaned_files = [
        file_payload
        for file_payload in (
            normalize_file_attachment_input(file_input)
            for file_input in file_attachments or []
        )
        if file_payload is not None
    ]
    cleaned_audios = [
        audio_payload
        for audio_payload in (
            normalize_audio_input(audio_input)
            for audio_input in audio_inputs or []
        )
        if audio_payload is not None
    ]
    cleaned_videos = [
        video_payload
        for video_payload in (
            normalize_video_input(video_input)
            for video_input in video_inputs or []
        )
        if video_payload is not None
    ]
    if not cleaned_images and not cleaned_files and not cleaned_audios and not cleaned_videos:
        return cleaned_text

    content_blocks: list[MessageContentBlock] = []
    if cleaned_text:
        content_blocks.append(
            {
                "type": TEXT_CONTENT_BLOCK_TYPE,
                "text": cleaned_text,
            }
        )
    for file_attachment in cleaned_files:
        content_blocks.append(
            {
                "type": FILE_ATTACHMENT_CONTENT_BLOCK_TYPE,
                "file_attachment": file_attachment,
            }
        )
    for image_url in cleaned_images:
        content_blocks.append(
            {
                "type": IMAGE_URL_CONTENT_BLOCK_TYPE,
                "image_url": image_url,
            }
        )
    for audio_payload in cleaned_audios:
        content_blocks.append(
            {
                "type": INPUT_AUDIO_CONTENT_BLOCK_TYPE,
                "input_audio": audio_payload,
            }
        )
    for video_payload in cleaned_videos:
        content_blocks.append(
            {
                "type": VIDEO_URL_CONTENT_BLOCK_TYPE,
                "video_url": video_payload,
            }
        )
    return content_blocks


def normalize_message_content(value: Any) -> MessageContent:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return str(value or "")

    blocks: list[MessageContentBlock] = []
    for item in value:
        normalized_block = normalize_message_content_block(item)
        if normalized_block is not None:
            blocks.append(normalized_block)
    return blocks


def normalize_message_content_block(value: Any) -> Optional[MessageContentBlock]:
    if not isinstance(value, Mapping):
        return None

    block_type = str(value.get("type", "")).strip()
    if not block_type:
        return None

    if block_type == TEXT_CONTENT_BLOCK_TYPE:
        text = str(value.get("text", "")).strip()
        if not text:
            return None
        return {
            "type": TEXT_CONTENT_BLOCK_TYPE,
            "text": text,
        }

    if block_type == IMAGE_URL_CONTENT_BLOCK_TYPE:
        image_url = normalize_image_input(value.get("image_url"))
        if image_url is None:
            return None
        return {
            "type": IMAGE_URL_CONTENT_BLOCK_TYPE,
            "image_url": image_url,
        }

    if block_type == FILE_ATTACHMENT_CONTENT_BLOCK_TYPE:
        file_attachment = normalize_file_attachment_input(value.get("file_attachment"))
        if file_attachment is None:
            return None
        return {
            "type": FILE_ATTACHMENT_CONTENT_BLOCK_TYPE,
            "file_attachment": file_attachment,
        }

    if block_type == INPUT_AUDIO_CONTENT_BLOCK_TYPE:
        audio_payload = normalize_audio_input(value.get("input_audio"))
        if audio_payload is None:
            return None
        return {
            "type": INPUT_AUDIO_CONTENT_BLOCK_TYPE,
            "input_audio": audio_payload,
        }

    if block_type == VIDEO_URL_CONTENT_BLOCK_TYPE:
        video_payload = normalize_video_input(value.get("video_url"))
        if video_payload is None:
            return None
        return {
            "type": VIDEO_URL_CONTENT_BLOCK_TYPE,
            "video_url": video_payload,
        }

    if block_type == FAILURE_MARKER_CONTENT_BLOCK_TYPE:
        message = str(value.get("message", "")).strip()
        if not message:
            return None
        return {
            "type": FAILURE_MARKER_CONTENT_BLOCK_TYPE,
            "message": message,
        }

    return dict(value)


def strip_failure_marker_blocks(content: MessageContent) -> MessageContent:
    """剔除失败标记块（见 `FAILURE_MARKER_CONTENT_BLOCK_TYPE`）。

    标记块只服务于历史回放，不该出现在送给 LLM 的内容里。
    """
    if not isinstance(content, list):
        return content

    return [
        block
        for block in content
        if not (
            isinstance(block, Mapping)
            and str(block.get("type", "")).strip() == FAILURE_MARKER_CONTENT_BLOCK_TYPE
        )
    ]


def message_content_blocks(content: MessageContent) -> list[MessageContentBlock]:
    normalized = normalize_message_content(content)
    if isinstance(normalized, str):
        cleaned_text = normalized.strip()
        if not cleaned_text:
            return []
        return [
            {
                "type": TEXT_CONTENT_BLOCK_TYPE,
                "text": cleaned_text,
            }
        ]
    return normalized


def message_content_to_text(content: MessageContent) -> str:
    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = str(block.get("type", "")).strip()
        if block_type == TEXT_CONTENT_BLOCK_TYPE:
            text = str(block.get("text", "")).strip()
            if text:
                text_parts.append(text)
            continue

        if block_type == IMAGE_URL_CONTENT_BLOCK_TYPE:
            text_parts.append("[image]")
            continue

        if block_type == INPUT_AUDIO_CONTENT_BLOCK_TYPE:
            text_parts.append("[audio]")
            continue

        if block_type == VIDEO_URL_CONTENT_BLOCK_TYPE:
            text_parts.append("[video]")
            continue

        if block_type == FILE_ATTACHMENT_CONTENT_BLOCK_TYPE:
            summary = file_attachment_summary(block.get("file_attachment"))
            if summary:
                text_parts.append(summary)
            continue

        if block_type == FAILURE_MARKER_CONTENT_BLOCK_TYPE:
            # 失败标记描述的是这一轮的终态，不是正文；混进文本会被拼进
            # agent handoff 之类的提示词里。
            continue

        if block_type:
            text_parts.append(f"[{block_type}]")

    return "\n\n".join(part for part in text_parts if part)


def message_content_image_urls(content: MessageContent) -> list[str]:
    if isinstance(content, str):
        return []

    image_urls: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if str(block.get("type", "")).strip() != IMAGE_URL_CONTENT_BLOCK_TYPE:
            continue

        image_url = block.get("image_url")
        if not isinstance(image_url, dict):
            continue

        url = str(image_url.get("url", "")).strip()
        if url:
            image_urls.append(url)

    return image_urls


def is_message_content_empty(content: MessageContent) -> bool:
    if isinstance(content, str):
        return not content.strip()

    return (
        len(message_content_image_urls(content)) == 0
        and len(message_content_file_attachments(content)) == 0
        and len(message_content_audio_blocks(content)) == 0
        and len(message_content_video_blocks(content)) == 0
        and not message_content_to_text(content).strip()
    )


_INLINE_REASONING_BLOCK = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_INLINE_REASONING_OPEN = re.compile(
    r"<(think|thinking|reasoning)>.*\Z", re.DOTALL | re.IGNORECASE
)


def strip_inline_reasoning(text: str) -> str:
    """剔除内联在正文里的推理块（``<think>`` / ``<thinking>`` / ``<reasoning>``）。

    MiniCPM / Qwen 这类模型会把推理直接写进 content 而不是 reasoning_content。
    不剥离的话，一段"只有思考、没有结论"的回复会被当成正常正文。
    """
    if not text:
        return text

    body = _INLINE_REASONING_BLOCK.sub("", text)
    # 未闭合的块说明被 max_tokens 截断，正文根本没生成出来，直接截断
    unclosed = _INLINE_REASONING_OPEN.search(body)
    if unclosed is not None:
        body = body[: unclosed.start()]
    return body.strip()


def is_message_content_reasoning_only(content: MessageContent) -> bool:
    """判断回复是否"只有推理块、没有任何正文"。

    这类回复既无正文也无工具调用，不代表回合完成，而是模型把 token 预算
    烧在了推理上（或被 max_tokens 截断）。
    """
    text = message_content_to_text(content).strip()
    return bool(text) and not strip_inline_reasoning(text)


def normalize_image_input(value: ImageInput) -> Optional[dict[str, str]]:
    if isinstance(value, Mapping):
        url = str(value.get("url", "")).strip()
        if not url:
            return None

        image_payload = {"url": url}
        preview_url = str(value.get("preview_url", "")).strip()
        if preview_url:
            image_payload["preview_url"] = preview_url
        attachment_id = str(value.get("attachment_id", "")).strip()
        if attachment_id:
            image_payload["attachment_id"] = attachment_id
        workspace_path = str(value.get("workspace_path", "")).strip()
        if workspace_path:
            image_payload["workspace_path"] = workspace_path
        return image_payload

    url = str(value or "").strip()
    if not url:
        return None
    return {"url": url}


def normalize_audio_input(value: AudioInput) -> Optional[dict[str, str]]:
    """规范化 input_audio 块载荷。

    data 允许为 URL 或 base64 data URL —— MiMo 官方示例两种都接受。
    """
    if isinstance(value, Mapping):
        data = str(value.get("data", "")).strip()
        if not data:
            return None
        audio_payload = {"data": data}
        audio_format = str(value.get("format", "")).strip().lower()
        if audio_format:
            audio_payload["format"] = audio_format
        return audio_payload

    data = str(value or "").strip()
    if not data:
        return None
    return {"data": data}


def normalize_video_input(value: VideoInput) -> Optional[dict[str, str]]:
    """规范化 video_url 块载荷。"""
    if isinstance(value, Mapping):
        url = str(value.get("url", "")).strip()
        if not url:
            return None
        return {"url": url}

    url = str(value or "").strip()
    if not url:
        return None
    return {"url": url}


def normalize_file_attachment_input(value: FileInput) -> Optional[dict[str, Any]]:
    if isinstance(value, Mapping):
        attachment_id = str(value.get("attachment_id", "")).strip()
        download_url = str(value.get("download_url", "")).strip()
        name = str(value.get("name", "")).strip()
        workspace_path = str(value.get("workspace_path", "")).strip()
        content_type = str(value.get("content_type", "")).strip()
        size_bytes = _optional_positive_int(value.get("size_bytes"))

        if not any([attachment_id, download_url, name, workspace_path]):
            return None

        normalized: dict[str, Any] = {
            "name": name or "file",
        }
        if attachment_id:
            normalized["attachment_id"] = attachment_id
        if download_url:
            normalized["download_url"] = download_url
        if workspace_path:
            normalized["workspace_path"] = workspace_path
        if content_type:
            normalized["content_type"] = content_type
        if size_bytes is not None:
            normalized["size_bytes"] = size_bytes
        return normalized

    attachment_id = str(value or "").strip()
    if not attachment_id:
        return None
    return {
        "attachment_id": attachment_id,
        "name": "file",
    }


def message_content_file_attachments(content: MessageContent) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return []

    file_attachments: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if str(block.get("type", "")).strip() != FILE_ATTACHMENT_CONTENT_BLOCK_TYPE:
            continue

        normalized = normalize_file_attachment_input(block.get("file_attachment"))
        if normalized is not None:
            file_attachments.append(normalized)
    return file_attachments


def message_content_audio_blocks(content: MessageContent) -> list[dict[str, str]]:
    """提取消息中的 input_audio 块载荷列表。"""
    if isinstance(content, str):
        return []

    audio_blocks: list[dict[str, str]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if str(block.get("type", "")).strip() != INPUT_AUDIO_CONTENT_BLOCK_TYPE:
            continue

        normalized = normalize_audio_input(block.get("input_audio"))
        if normalized is not None:
            audio_blocks.append(normalized)
    return audio_blocks


def message_content_video_blocks(content: MessageContent) -> list[dict[str, str]]:
    """提取消息中的 video_url 块载荷列表。"""
    if isinstance(content, str):
        return []

    video_blocks: list[dict[str, str]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if str(block.get("type", "")).strip() != VIDEO_URL_CONTENT_BLOCK_TYPE:
            continue

        normalized = normalize_video_input(block.get("video_url"))
        if normalized is not None:
            video_blocks.append(normalized)
    return video_blocks


def file_attachment_summary(value: Any) -> str:
    normalized = normalize_file_attachment_input(value)
    if normalized is None:
        return ""

    name = str(normalized.get("name", "")).strip() or "file"
    details = [name]

    workspace_path = str(normalized.get("workspace_path", "")).strip()
    if workspace_path:
        details.append(f"path={workspace_path}")

    content_type = str(normalized.get("content_type", "")).strip()
    if content_type:
        details.append(f"type={content_type}")

    size_bytes = _optional_positive_int(normalized.get("size_bytes"))
    if size_bytes is not None:
        details.append(f"size={size_bytes} bytes")

    return "file: " + " | ".join(details)


def _optional_positive_int(value: Any) -> Optional[int]:
    try:
        v = int(value)
        if v > 0:
            return v
    except (TypeError, ValueError):
        pass
    return None

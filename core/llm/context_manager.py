# core/llm/context_manager.py
"""
Context Manager
Handles context window management, token counting, and intelligent truncation
"""

import re
import json
import logging
import tiktoken
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizer
import pathlib
from pathlib import Path

from .base import ChatMessage
from ..config import get_config
from ..shared_cache import get_shared_cache
from ..exceptions import ContextLengthExceededError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """Context window configuration and state"""

    max_context_length: int
    max_new_tokens: int
    reserve_tokens: int = 100  # Reserve for response generation
    truncation_strategy: str = "sliding_window"  # sliding_window, summarize, compress

    @property
    def effective_max_length(self) -> int:
        """Maximum tokens available for input after reserving for output"""
        return max(
            0, self.max_context_length - self.max_new_tokens - self.reserve_tokens
        )

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.reserve_tokens < 0:
            raise ValueError("reserve_tokens cannot be negative")
        if self.effective_max_length <= 0:
            logger.warning(
                f"Effective max length is {self.effective_max_length}, consider adjusting parameters"
            )


@dataclass
class TokenUsage:
    """Token usage statistics"""

    prompt_tokens: int
    max_new_tokens: int
    total_tokens: int
    truncated: bool = False
    truncated_messages: int = 0


class ContextManager:
    """Manages context windows and token counting for LLM conversations"""

    def __init__(self):
        self.config = get_config()
        # Model context configurations
        self.model_configs = self._load_model_configs()

        self.cache = get_shared_cache()

        self._tokenizers: Dict[str, PreTrainedTokenizer] = {}
        # Token encoders cache
        self._encoders: Dict[str, tiktoken.Encoding] = {}

        # Default context windows for different models
        self._model_contexts = {
            "qwen": {"max_length": 8192, "response_length": 2048},
            "llama": {"max_length": 4096, "response_length": 1024},
            "default": {"max_length": 4096, "response_length": 1024},
        }

        logger.info("ContextManager initialized")

    def _load_model_configs(self) -> Dict[str, ContextWindow]:
        """Load context window configurations for different models"""
        return {
            # GPT models
            "gpt-3.5-turbo": ContextWindow(
                max_context_length=4096, max_new_tokens=1024, reserve_tokens=100
            ),
            "gpt-4": ContextWindow(
                max_context_length=8192, max_new_tokens=2048, reserve_tokens=100
            ),
            "gpt-4-32k": ContextWindow(
                max_context_length=32768, max_new_tokens=4096, reserve_tokens=100
            ),
            # Open source models (estimates)
            "microsoft/DialoGPT-medium": ContextWindow(
                max_context_length=1024, max_new_tokens=512, reserve_tokens=50
            ),
            "microsoft/DialoGPT-large": ContextWindow(
                max_context_length=1024, max_new_tokens=512, reserve_tokens=50
            ),
            "facebook/blenderbot-1B-distill": ContextWindow(
                max_context_length=128, max_new_tokens=64, reserve_tokens=20
            ),
            "EleutherAI/gpt-neo-1.3B": ContextWindow(
                max_context_length=2048, max_new_tokens=512, reserve_tokens=100
            ),
            "EleutherAI/gpt-neo-2.7B": ContextWindow(
                max_context_length=2048, max_new_tokens=512, reserve_tokens=100
            ),
            "EleutherAI/gpt-j-6B": ContextWindow(
                max_context_length=2048, max_new_tokens=512, reserve_tokens=100
            ),
            # Llama models
            "meta-llama/Llama-2-7b-chat-hf": ContextWindow(
                max_context_length=4096, max_new_tokens=1024, reserve_tokens=100
            ),
            "meta-llama/Llama-2-13b-chat-hf": ContextWindow(
                max_context_length=4096, max_new_tokens=1024, reserve_tokens=100
            ),
            "meta-llama/Llama-2-70b-chat-hf": ContextWindow(
                max_context_length=4096, max_new_tokens=1024, reserve_tokens=100
            ),
            # Qwen models
            "Qwen/Qwen-7B-Chat": ContextWindow(
                max_context_length=8192, max_new_tokens=2048, reserve_tokens=100
            ),
            "Qwen/Qwen-14B-Chat": ContextWindow(
                max_context_length=8192, max_new_tokens=2048, reserve_tokens=100
            ),
            "Qwen/Qwen-72B-Chat": ContextWindow(
                max_context_length=8192, max_new_tokens=2048, reserve_tokens=100
            ),
            # Default fallback
            "default": ContextWindow(
                max_context_length=2048, max_new_tokens=512, reserve_tokens=100
            ),
        }

    def get_context_window(self, model_name: str) -> ContextWindow:
        """Get context window configuration for a model"""
        # 先檢查是否有完全匹配的配置
        if model_name in self.model_configs:
            return self.model_configs[model_name]

        # 嘗試部分匹配不同模型家族
        model_name_lower = model_name.lower()

        # GPT 系列
        if any(gpt_type in model_name_lower for gpt_type in ["gpt-3.5", "gpt-35"]):
            return self.model_configs["gpt-3.5-turbo"]
        elif "gpt-4-32k" in model_name_lower:
            return self.model_configs["gpt-4-32k"]
        elif "gpt-4" in model_name_lower:
            return self.model_configs["gpt-4"]

        # DialogGPT 系列
        elif "dialogpt" in model_name_lower:
            if "large" in model_name_lower:
                return self.model_configs["microsoft/DialoGPT-large"]
            else:
                return self.model_configs["microsoft/DialoGPT-medium"]

        # BlenderBot 系列
        elif "blenderbot" in model_name_lower:
            return self.model_configs["facebook/blenderbot-1B-distill"]

        # EleutherAI 系列
        elif "gpt-neo" in model_name_lower:
            if "2.7" in model_name_lower:
                return self.model_configs["EleutherAI/gpt-neo-2.7B"]
            else:
                return self.model_configs["EleutherAI/gpt-neo-1.3B"]
        elif "gpt-j" in model_name_lower:
            return self.model_configs["EleutherAI/gpt-j-6B"]

        # Llama 系列
        elif "llama" in model_name_lower:
            if "70b" in model_name_lower:
                return self.model_configs["meta-llama/Llama-2-70b-chat-hf"]
            elif "13b" in model_name_lower:
                return self.model_configs["meta-llama/Llama-2-13b-chat-hf"]
            else:
                return self.model_configs["meta-llama/Llama-2-7b-chat-hf"]

        # Qwen 系列
        elif "qwen" in model_name_lower:
            if "72b" in model_name_lower:
                return self.model_configs["Qwen/Qwen-72B-Chat"]
            elif "14b" in model_name_lower:
                return self.model_configs["Qwen/Qwen-14B-Chat"]
            else:
                return self.model_configs["Qwen/Qwen-7B-Chat"]

        # 如果沒有匹配，使用預設配置
        logger.warning(f"No specific config for {model_name}, using default")
        return self.model_configs["default"]

    def get_encoder(self, model_name: str) -> Optional[tiktoken.Encoding]:
        """Get or create token encoder for a model"""
        if model_name in self._encoders:
            return self._encoders[model_name]

        try:
            # Try to get tiktoken encoder
            if "gpt" in model_name.lower():
                encoder = tiktoken.encoding_for_model(model_name)
            else:
                # Use cl100k_base as fallback for most models
                encoder = tiktoken.get_encoding("cl100k_base")

            self._encoders[model_name] = encoder
            return encoder

        except Exception as e:
            logger.warning(f"Failed to load encoder for {model_name}: {e}")
            return None

    def get_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """Get or load tokenizer for token counting"""
        if model_name not in self._tokenizers:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=Path(self.cache.cache_root) / "hf",
                )
                self._tokenizers[model_name] = tokenizer
                logger.info(f"Loaded tokenizer for context management: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {model_name}: {e}")
                # Use a default tokenizer for rough estimation
                if "default" not in self._tokenizers:
                    self._tokenizers["default"] = AutoTokenizer.from_pretrained(
                        "gpt2", cache_dir=Path(self.cache.cache_root) / "hf"
                    )
                self._tokenizers[model_name] = self._tokenizers["default"]

        return self._tokenizers[model_name]

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in text using model-specific tokenizer"""
        try:
            tokenizer = self.get_tokenizer(model_name)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed, using estimation: {e}")
            # Fallback to rough estimation (1 token ≈ 4 characters)
            return len(text) // 4

    def count_messages_tokens(
        self, messages: List[ChatMessage], model_name: str
    ) -> int:
        """Count total tokens in message list"""
        total_tokens = 0

        for message in messages:
            # Add message content tokens
            content_tokens = self.count_tokens(message.content, model_name)

            # Add overhead for message structure (role, formatting, etc.)
            # This is model-specific overhead estimation
            message_overhead = self._get_message_overhead(model_name, message.role)

            total_tokens += content_tokens + message_overhead

        return total_tokens

    def prepare_context(
        self,
        messages: List[ChatMessage],
        model_name: str,
        max_new_tokens: int = 512,
    ) -> Tuple[List[ChatMessage], TokenUsage]:
        """Prepare context by truncating if necessary"""

        context_window = self.get_context_window(model_name)

        # Update max_new_tokens in context window
        context_window.max_new_tokens = max_new_tokens

        # Count current tokens
        current_tokens = self.count_messages_tokens(messages, model_name)

        # Check if truncation is needed
        if current_tokens <= context_window.effective_max_length:
            return messages, TokenUsage(
                prompt_tokens=current_tokens,
                max_new_tokens=max_new_tokens,
                total_tokens=current_tokens + max_new_tokens,
                truncated=False,
            )

        # Need to truncate - keep system message and recent messages
        truncated_messages = self._truncate_messages(
            messages, model_name, context_window.effective_max_length
        )

        truncated_tokens = self.count_messages_tokens(truncated_messages, model_name)

        logger.info(
            f"Context truncated: {len(messages)} -> {len(truncated_messages)} messages, "
            f"{current_tokens} -> {truncated_tokens} tokens"
        )

        return truncated_messages, TokenUsage(
            prompt_tokens=truncated_tokens,
            max_new_tokens=max_new_tokens,
            total_tokens=truncated_tokens + max_new_tokens,
            truncated=True,
            truncated_messages=len(messages) - len(truncated_messages),
        )

    def _truncate_messages(
        self,
        messages: List[ChatMessage],
        model_name: str,
        max_tokens: int,
    ) -> List[ChatMessage]:
        """Intelligently truncate messages to fit context window"""

        if not messages:
            return messages

        # Always preserve system message if it exists
        result_messages = []
        remaining_messages = messages.copy()

        if messages[0].role == "system":
            system_message = remaining_messages.pop(0)
            result_messages.append(system_message)
            max_tokens -= (
                self.count_tokens(system_message.content, model_name) + 7
            )  # overhead

        # Add messages from the end (most recent) until we hit the limit
        current_tokens = sum(
            self.count_tokens(msg.content, model_name) + 7  # message overhead
            for msg in result_messages
        )

        # Work backwards through conversation
        for message in reversed(remaining_messages):
            message_tokens = self.count_tokens(message.content, model_name) + 7

            if current_tokens + message_tokens <= max_tokens:
                result_messages.append(message)
                current_tokens += message_tokens
            else:
                break

        # Restore chronological order (keep system first, then chronological)
        if result_messages and result_messages[0].role == "system":
            system_msg = result_messages[0]
            conversation_msgs = list(reversed(result_messages[1:]))
            return [system_msg] + conversation_msgs
        else:
            return list(reversed(result_messages))

    def validate_context_length(
        self,
        messages: List[ChatMessage],
        model_name: str,
        max_new_tokens: int = 512,
    ) -> None:
        """Validate that context length is within limits"""

        context_window = self.get_context_window(model_name)
        current_tokens = self.count_messages_tokens(messages, model_name)
        total_tokens = current_tokens + max_new_tokens

        if total_tokens > context_window.max_context_length:
            raise ContextLengthExceededError(
                current_length=current_tokens,
                max_length=context_window.max_context_length,
                model_name=model_name,
            )

    def get_context_stats(
        self,
        messages: List[ChatMessage],
        model_name: str,
    ) -> Dict[str, Any]:
        """Get context statistics"""

        context_window = self.get_context_window(model_name)
        current_tokens = self.count_messages_tokens(messages, model_name)

        # 確保 effective_max_length 存在
        effective_max_length = context_window.effective_max_length
        utilization = (
            current_tokens / effective_max_length if effective_max_length > 0 else 0
        )

        return {
            "message_count": len(messages),
            "current_tokens": current_tokens,
            "max_context_length": context_window.max_context_length,
            "effective_max_length": effective_max_length,
            "utilization": utilization,  # 確保 utilization 欄位存在
            "available_tokens": effective_max_length - current_tokens,
            "needs_truncation": current_tokens > effective_max_length,
        }

    def _apply_sliding_window(
        self, messages: List[ChatMessage], model_name: str, max_tokens: int
    ) -> List[ChatMessage]:
        """Apply sliding window truncation - keep system + recent messages"""
        # Always preserve system messages
        system_messages = [msg for msg in messages if msg.role == "system"]
        other_messages = [msg for msg in messages if msg.role != "system"]

        # Count system message tokens
        system_tokens = self.count_messages_tokens(system_messages, model_name)
        remaining_tokens = max_tokens - system_tokens

        if remaining_tokens <= 0:
            logger.warning("System messages exceed context limit")
            return system_messages[:1]  # Keep only first system message

        # Add messages from the end until we hit the limit
        selected_messages = []
        current_tokens = 0

        for message in reversed(other_messages):
            message_tokens = self.count_tokens(message.content, model_name)
            message_tokens += self._get_message_overhead(model_name, message.role)

            if current_tokens + message_tokens > remaining_tokens:
                break

            selected_messages.insert(0, message)
            current_tokens += message_tokens

        # Combine system messages + selected messages
        return system_messages + selected_messages

    def _apply_compression(
        self, messages: List[ChatMessage], model_name: str, max_tokens: int
    ) -> List[ChatMessage]:
        """Apply intelligent compression - summarize old messages"""
        # This is a simplified compression strategy
        # In production, you might use a summarization model

        system_messages = [msg for msg in messages if msg.role == "system"]
        other_messages = [msg for msg in messages if msg.role != "system"]

        # Keep recent messages (last 5-10)
        recent_count = min(10, len(other_messages))
        recent_messages = other_messages[-recent_count:]
        old_messages = (
            other_messages[:-recent_count] if recent_count < len(other_messages) else []
        )

        compressed_messages = system_messages.copy()

        # Add compressed summary of old messages if any
        if old_messages:
            summary = self._create_conversation_summary(old_messages)
            compressed_messages.append(
                ChatMessage(
                    role="system",
                    content=f"Previous conversation summary: {summary}",
                    metadata={"compressed": True, "original_count": len(old_messages)},
                )
            )

        # Add recent messages
        compressed_messages.extend(recent_messages)

        # Check if still too long
        final_tokens = self.count_messages_tokens(compressed_messages, model_name)
        if final_tokens > max_tokens:
            # Fall back to sliding window
            return self._apply_sliding_window(
                compressed_messages, model_name, max_tokens
            )

        return compressed_messages

    def _create_conversation_summary(self, messages: List[ChatMessage]) -> str:
        """Create a summary of conversation messages"""
        # Simple extractive summary - take key points
        user_messages = [msg.content for msg in messages if msg.role == "user"]
        assistant_messages = [
            msg.content for msg in messages if msg.role == "assistant"
        ]

        summary_parts = []

        if user_messages:
            # Extract main topics from user messages
            topics = self._extract_topics(user_messages)
            if topics:
                summary_parts.append(f"User discussed: {', '.join(topics[:3])}")

        if assistant_messages:
            # Get key response themes
            themes = self._extract_themes(assistant_messages)
            if themes:
                summary_parts.append(f"Assistant provided: {', '.join(themes[:3])}")

        summary = "; ".join(summary_parts) if summary_parts else "General conversation"
        return summary[:200]  # Limit summary length

    def _extract_topics(self, messages: List[str]) -> List[str]:
        """Extract main topics from messages"""
        # Simple keyword extraction
        combined_text = " ".join(messages).lower()

        # Remove common words and extract key terms
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "how",
            "what",
            "when",
            "where",
            "why",
            "can",
            "could",
            "would",
            "should",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", combined_text)

        # Count word frequency
        word_counts = {}
        for word in words:
            if word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Return top words
        topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in topics[:5]]

    def _extract_themes(self, messages: List[str]) -> List[str]:
        """Extract response themes from assistant messages"""
        # Look for common response patterns
        themes = []
        combined_text = " ".join(messages).lower()

        # Pattern matching for common themes
        if "explain" in combined_text or "definition" in combined_text:
            themes.append("explanations")
        if "help" in combined_text or "assist" in combined_text:
            themes.append("assistance")
        if "code" in combined_text or "programming" in combined_text:
            themes.append("coding help")
        if "question" in combined_text or "answer" in combined_text:
            themes.append("Q&A")

        return themes

    def _get_message_overhead(self, model_name: str, role: str) -> int:
        """Get token overhead for message formatting"""
        # Different models have different formatting overhead
        if "qwen" in model_name.lower():
            # Qwen chat format: <|im_start|>role\ncontent<|im_end|>
            return 8
        elif "llama" in model_name.lower():
            # Llama chat format has variable overhead
            return 10 if role == "system" else 6
        else:
            # Generic overhead
            return 5


# Global context manager instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get global context manager instance"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


def reset_context_manager():
    """Reset the global context manager (for testing)"""
    global _context_manager
    _context_manager = None
    logger.info("Context manager reset")

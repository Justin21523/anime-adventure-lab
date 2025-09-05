# core/llm/context_manager.py
"""
Context Manager
Handles context window management, token counting, and intelligent truncation
"""

import re
import json
import logging
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
    max_response_length: int
    safety_margin: int = 100
    truncation_strategy: str = "sliding_window"  # sliding_window, summarize, compress

    @property
    def available_context(self) -> int:
        """Available context length for input"""
        return self.max_context_length - self.max_response_length - self.safety_margin


@dataclass
class TokenUsage:
    """Token usage statistics"""

    prompt_tokens: int
    estimated_response_tokens: int
    total_tokens: int
    context_utilization: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "estimated_response_tokens": self.estimated_response_tokens,
            "total_tokens": self.total_tokens,
            "context_utilization": round(self.context_utilization, 3),
        }


class ContextManager:
    """Manages context windows and token counting for LLM conversations"""

    def __init__(self):
        self.config = get_config()
        self.cache = get_shared_cache()
        self._tokenizers: Dict[str, PreTrainedTokenizer] = {}

        # Default context windows for different models
        self._model_contexts = {
            "qwen": {"max_length": 8192, "response_length": 2048},
            "llama": {"max_length": 4096, "response_length": 1024},
            "default": {"max_length": 4096, "response_length": 1024},
        }

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

    def get_context_window(self, model_name: str) -> ContextWindow:
        """Get context window configuration for model"""
        # Determine model type
        model_type = "default"
        for model_key in self._model_contexts.keys():
            if model_key in model_name.lower():
                model_type = model_key
                break

        context_config = self._model_contexts[model_type]

        return ContextWindow(
            max_context_length=context_config["max_length"],
            max_response_length=context_config["response_length"],
            safety_margin=self.config.get("llm.context_safety_margin", 100),
            truncation_strategy=self.config.get(
                "llm.truncation_strategy", "sliding_window"
            ),
        )

    def prepare_context(
        self,
        messages: List[ChatMessage],
        model_name: str,
        max_response_length: Optional[int] = None,
    ) -> Tuple[List[ChatMessage], TokenUsage]:
        """
        Prepare conversation context respecting token limits

        Args:
            messages: Input messages
            model_name: Target model name
            max_response_length: Maximum tokens for response

        Returns:
            Tuple of (prepared_messages, token_usage)
        """
        context_window = self.get_context_window(model_name)

        if max_response_length:
            context_window.max_response_length = max_response_length

        # Count current tokens
        current_tokens = self.count_messages_tokens(messages, model_name)
        available_tokens = context_window.available_context

        # If within limits, return as-is
        if current_tokens <= available_tokens:
            token_usage = TokenUsage(
                prompt_tokens=current_tokens,
                estimated_response_tokens=context_window.max_response_length,
                total_tokens=current_tokens + context_window.max_response_length,
                context_utilization=current_tokens / context_window.max_context_length,
            )
            return messages, token_usage

        # Need to truncate - apply strategy
        logger.info(
            f"Context exceeds limit: {current_tokens} > {available_tokens}, applying truncation"
        )

        if context_window.truncation_strategy == "sliding_window":
            truncated_messages = self._apply_sliding_window(
                messages, model_name, available_tokens
            )
        elif context_window.truncation_strategy == "compress":
            truncated_messages = self._apply_compression(
                messages, model_name, available_tokens
            )
        else:
            # Default to sliding window
            truncated_messages = self._apply_sliding_window(
                messages, model_name, available_tokens
            )

        # Calculate final token usage
        final_tokens = self.count_messages_tokens(truncated_messages, model_name)
        token_usage = TokenUsage(
            prompt_tokens=final_tokens,
            estimated_response_tokens=context_window.max_response_length,
            total_tokens=final_tokens + context_window.max_response_length,
            context_utilization=final_tokens / context_window.max_context_length,
        )

        logger.info(f"Context truncated: {current_tokens} -> {final_tokens} tokens")
        return truncated_messages, token_usage

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

    def validate_context_length(
        self,
        messages: List[ChatMessage],
        model_name: str,
        max_response_length: int = 512,
    ) -> bool:
        """Validate if messages fit in context window"""
        context_window = self.get_context_window(model_name)
        context_window.max_response_length = max_response_length

        current_tokens = self.count_messages_tokens(messages, model_name)
        return current_tokens <= context_window.available_context

    def get_context_stats(self) -> Dict[str, Any]:
        """Get context manager statistics"""
        return {
            "loaded_tokenizers": list(self._tokenizers.keys()),
            "model_contexts": self._model_contexts,
            "cache_root": str(self.cache.cache_root),
        }


# Global context manager instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get global context manager instance"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager

# core/utils/text.py
"""
Text Processing Utilities
Text cleaning, tokenization, and language processing with Chinese support
"""

import re
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import jieba
import opencc

from ..config import get_config
from ..exceptions import TextProcessingError

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text processing and cleaning utilities with Chinese support"""

    def __init__(self):
        self.config = get_config()

        # Initialize Chinese processing tools
        self.converter = None
        self._setup_chinese_tools()

        # Common cleaning patterns
        self.url_pattern = re.compile(r"https?://[^\s]+")
        self.email_pattern = re.compile(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        )
        self.whitespace_pattern = re.compile(r"\s+")
        self.special_chars_pattern = re.compile(r"[^\w\s\u4e00-\u9fff]")

        # Chinese text patterns
        self.chinese_punctuation = re.compile(
            r'[，。！？；：""' "（）【】《》〈〉「」『』〔〕]"
        )
        self.chinese_chars = re.compile(r"[\u4e00-\u9fff]")

    def _setup_chinese_tools(self):
        """Setup Chinese text processing tools"""
        try:
            # Traditional to Simplified Chinese converter
            self.converter = opencc.OpenCC("t2s.json")

            # Setup jieba for Chinese word segmentation
            jieba.setLogLevel(logging.WARNING)  # Reduce jieba log noise

            logger.info("✅ Chinese processing tools loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Chinese tools: {e}")
            self.converter = None

    def clean_text(self, text: str, options: Optional[Dict[str, bool]] = None) -> str:
        """Clean and normalize text"""
        if not text or not isinstance(text, str):
            return ""

        options = options or {}

        try:
            # Unicode normalization
            text = unicodedata.normalize("NFKC", text)

            # Remove URLs if requested
            if options.get("remove_urls", True):
                text = self.url_pattern.sub("", text)

            # Remove emails if requested
            if options.get("remove_emails", True):
                text = self.email_pattern.sub("", text)

            # Convert traditional to simplified Chinese
            if options.get("convert_chinese", True) and self.converter:
                text = self.converter.convert(text)

            # Normalize Chinese punctuation
            if options.get("normalize_punctuation", False):
                text = self._normalize_chinese_punctuation(text)

            # Remove extra whitespace
            text = self.whitespace_pattern.sub(" ", text)

            # Remove special characters if requested
            if options.get("remove_special_chars", False):
                text = self.special_chars_pattern.sub(" ", text)

            # Remove excessive line breaks
            if options.get("clean_linebreaks", True):
                text = re.sub(r"\n+", "\n", text)
                text = re.sub(r"\n\s*\n", "\n\n", text)

            # Trim and return
            return text.strip()

        except Exception as e:
            logger.error(f"❌ Text cleaning failed: {e}")
            raise TextProcessingError(f"Text cleaning failed: {e}")

    def _normalize_chinese_punctuation(self, text: str) -> str:
        """Normalize Chinese punctuation to standard forms"""
        punctuation_map = {
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "；": ";",
            "：": ":",
            '"': '"',
            '"': '"',
            """: "'", """: "'",
            "（": "(",
            "）": ")",
            "【": "[",
            "】": "]",
            "《": "<",
            "》": ">",
            "〈": "<",
            "〉": ">",
            "「": '"',
            "」": '"',
            "『": "'",
            "』": "'",
            "〔": "[",
            "〕": "]",
        }

        for chinese_punct, english_punct in punctuation_map.items():
            text = text.replace(chinese_punct, english_punct)

        return text

    def tokenize_chinese(self, text: str, mode: str = "default") -> List[str]:
        """Tokenize Chinese text using jieba"""
        try:
            if not text:
                return []

            # Clean text first
            cleaned_text = self.clean_text(text)

            # Choose jieba mode
            if mode == "search":
                # Search engine mode (more granular)
                tokens = list(jieba.cut_for_search(cleaned_text))
            elif mode == "all":
                # Full mode (all possible words)
                tokens = list(jieba.cut(cleaned_text, cut_all=True))
            else:
                # Default accurate mode
                tokens = list(jieba.cut(cleaned_text, cut_all=False))

            # Filter out empty tokens and whitespace
            tokens = [token.strip() for token in tokens if token.strip()]

            return tokens

        except Exception as e:
            logger.error(f"❌ Chinese tokenization failed: {e}")
            raise TextProcessingError(f"Chinese tokenization failed: {e}")

    def extract_keywords(
        self, text: str, max_keywords: int = 10, use_tfidf: bool = True
    ) -> List[Tuple[str, float]]:
        """Extract keywords from text with scores"""
        try:
            import jieba.analyse

            # Clean text
            cleaned_text = self.clean_text(text)

            if use_tfidf:
                # Extract keywords using TF-IDF
                keywords = jieba.analyse.extract_tags(
                    cleaned_text, topK=max_keywords, withWeight=True
                )
            else:
                # Extract keywords using TextRank
                keywords = jieba.analyse.textrank(
                    cleaned_text, topK=max_keywords, withWeight=True
                )

            return keywords

        except Exception as e:
            logger.warning(f"⚠️ Keyword extraction failed: {e}")
            return []

    def detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            # Simple heuristic based on character ranges
            chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
            english_chars = len(re.findall(r"[a-zA-Z]", text))
            japanese_chars = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", text))
            korean_chars = len(re.findall(r"[\uac00-\ud7af]", text))
            total_chars = len(text.replace(" ", ""))

            if total_chars == 0:
                return "unknown"

            chinese_ratio = chinese_chars / total_chars
            english_ratio = english_chars / total_chars
            japanese_ratio = japanese_chars / total_chars
            korean_ratio = korean_chars / total_chars

            # Determine primary language
            if chinese_ratio > 0.3:
                return "zh"
            elif japanese_ratio > 0.1:
                return "ja"
            elif korean_ratio > 0.1:
                return "ko"
            elif english_ratio > 0.5:
                return "en"
            elif chinese_ratio > 0.1 and english_ratio > 0.1:
                return "zh-en"  # Mixed Chinese-English
            else:
                return "mixed"

        except Exception as e:
            logger.warning(f"⚠️ Language detection failed: {e}")
            return "unknown"

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (supports Chinese)"""
        try:
            # Clean text
            cleaned_text = self.clean_text(text)

            # Split on common sentence endings (both Chinese and English)
            sentences = re.split(r"[。！？.!?]+", cleaned_text)

            # Also split on line breaks for better segmentation
            all_sentences = []
            for sentence in sentences:
                # Further split on line breaks if they seem to indicate sentence boundaries
                sub_sentences = sentence.split("\n")
                all_sentences.extend(sub_sentences)

            # Filter and clean sentences
            sentences = [s.strip() for s in all_sentences if s.strip()]

            return sentences

        except Exception as e:
            logger.error(f"❌ Sentence splitting failed: {e}")
            raise TextProcessingError(f"Sentence splitting failed: {e}")

    def truncate_text(
        self,
        text: str,
        max_length: int,
        preserve_words: bool = True,
        ellipsis: str = "...",
    ) -> str:
        """Truncate text to maximum length"""
        if len(text) <= max_length:
            return text

        if preserve_words:
            # Try to break at word boundaries
            if " " in text[:max_length]:
                truncated = text[:max_length].rsplit(" ", 1)[0]
            elif self._has_chinese(text):
                # For Chinese text, truncate at character boundary
                truncated = text[: max_length - len(ellipsis)]
            else:
                truncated = text[: max_length - len(ellipsis)]
        else:
            truncated = text[: max_length - len(ellipsis)]

        return truncated + ellipsis if len(truncated) < len(text) else truncated

    def _has_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters"""
        return bool(self.chinese_chars.search(text))

    def count_words(self, text: str) -> Dict[str, int]:
        """Count words in text (handles Chinese properly)"""
        try:
            cleaned_text = self.clean_text(text)

            if self._has_chinese(cleaned_text):
                # Use jieba for Chinese text
                tokens = self.tokenize_chinese(cleaned_text)
                word_count = len(tokens)
                char_count = len(cleaned_text.replace(" ", ""))
            else:
                # Regular word splitting for English
                tokens = cleaned_text.split()
                word_count = len(tokens)
                char_count = len(cleaned_text.replace(" ", ""))

            return {
                "words": word_count,
                "characters": char_count,
                "characters_with_spaces": len(cleaned_text),
                "sentences": len(self.split_sentences(cleaned_text)),
                "paragraphs": len([p for p in cleaned_text.split("\n\n") if p.strip()]),
            }

        except Exception as e:
            logger.error(f"❌ Word counting failed: {e}")
            return {
                "words": 0,
                "characters": 0,
                "characters_with_spaces": 0,
                "sentences": 0,
                "paragraphs": 0,
            }

    def remove_stopwords(self, tokens: List[str], language: str = "auto") -> List[str]:
        """Remove stopwords from token list"""
        try:
            if language == "auto":
                # Detect language from tokens
                sample_text = " ".join(tokens[:10])
                language = self.detect_language(sample_text)

            # Define stopwords for different languages
            chinese_stopwords = {
                "的",
                "了",
                "在",
                "是",
                "我",
                "有",
                "和",
                "就",
                "不",
                "人",
                "都",
                "一",
                "一个",
                "上",
                "也",
                "很",
                "到",
                "说",
                "要",
                "去",
                "你",
                "会",
                "着",
                "没有",
                "看",
                "好",
                "自己",
                "这",
                "那",
                "之",
                "与",
                "及",
                "或",
                "而",
                "但",
                "因为",
                "所以",
            }

            english_stopwords = {
                "the",
                "a",
                "an",
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
                "from",
                "as",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "shall",
                "can",
            }

            if language.startswith("zh"):
                stopwords = chinese_stopwords
            elif language == "en":
                stopwords = english_stopwords
            else:
                # Use combined set for mixed language
                stopwords = chinese_stopwords | english_stopwords

            # Filter tokens
            filtered_tokens = [
                token for token in tokens if token.lower() not in stopwords
            ]

            return filtered_tokens

        except Exception as e:
            logger.warning(f"⚠️ Stopword removal failed: {e}")
            return tokens

    def similarity_score(self, text1: str, text2: str) -> float:
        """Calculate text similarity score (0-1)"""
        try:
            # Clean both texts
            clean1 = self.clean_text(text1)
            clean2 = self.clean_text(text2)

            if not clean1 or not clean2:
                return 0.0

            # Tokenize
            if self._has_chinese(clean1) or self._has_chinese(clean2):
                tokens1 = set(self.tokenize_chinese(clean1))
                tokens2 = set(self.tokenize_chinese(clean2))
            else:
                tokens1 = set(clean1.lower().split())
                tokens2 = set(clean2.lower().split())

            # Calculate Jaccard similarity
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)

            if union == 0:
                return 0.0

            return intersection / union

        except Exception as e:
            logger.error(f"❌ Similarity calculation failed: {e}")
            return 0.0

    def format_for_prompt(self, text: str, max_length: int = 500) -> str:
        """Format text for use in AI prompts"""
        try:
            # Clean the text
            cleaned = self.clean_text(
                text,
                {
                    "remove_urls": True,
                    "remove_emails": True,
                    "convert_chinese": True,
                    "clean_linebreaks": True,
                    "normalize_punctuation": True,
                },
            )

            # Truncate if necessary
            if len(cleaned) > max_length:
                cleaned = self.truncate_text(cleaned, max_length, preserve_words=True)

            # Ensure it ends with proper punctuation
            if cleaned and cleaned[-1] not in ".!?。！？":
                if self._has_chinese(cleaned):
                    cleaned += "。"
                else:
                    cleaned += "."

            return cleaned

        except Exception as e:
            logger.error(f"❌ Prompt formatting failed: {e}")
            return text


# Global instance
_text_processor = None


def get_text_processor() -> TextProcessor:
    """Get global text processor instance"""
    global _text_processor
    if _text_processor is None:
        _text_processor = TextProcessor()
    return _text_processor

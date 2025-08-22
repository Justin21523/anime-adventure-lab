# core/vlm/consistency.py
"""
VLM Consistency Checker - validates generated captions against existing RAG content
Ensures character descriptions, scene elements match established worldpack lore
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from rapidfuzz import fuzz
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue found during validation"""

    category: str  # character, scene, style, factual
    severity: str  # low, medium, high
    description: str
    expected: Optional[str] = None
    found: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ConsistencyReport:
    """Full consistency check report"""

    overall_score: float  # 0-1, higher is more consistent
    issues: List[ConsistencyIssue]
    suggestions: List[str]
    validated_elements: Dict[str, bool]


class CharacterValidator:
    """Validates character-related consistency"""

    def __init__(self, character_data: Dict[str, Any]):
        self.character_data = character_data
        self.name = character_data.get("name", "")
        self.appearance = character_data.get("appearance", {})
        self.personality = character_data.get("personality", {})

    def validate_appearance(self, caption: str) -> List[ConsistencyIssue]:
        """Check if caption matches character appearance"""
        issues = []

        # Extract appearance features
        features = {
            "hair_color": self.appearance.get("hair_color", ""),
            "eye_color": self.appearance.get("eye_color", ""),
            "height": self.appearance.get("height", ""),
            "build": self.appearance.get("build", ""),
            "distinctive_features": self.appearance.get("distinctive_features", []),
        }

        caption_lower = caption.lower()

        # Check hair color
        if features["hair_color"]:
            expected_hair = features["hair_color"].lower()
            hair_patterns = [
                r"(?:hair|頭髮).*?(?:is|為|色|colored?)\s*(\w+)",
                r"(\w+)(?:\s+hair|頭髮)",
                r"(?:with|有)\s*(\w+)\s*hair",
            ]

            found_hair = None
            for pattern in hair_patterns:
                matches = re.findall(pattern, caption_lower)
                if matches:
                    found_hair = matches[0]
                    break

            if found_hair and found_hair != expected_hair:
                if fuzz.ratio(found_hair, expected_hair) < 70:
                    issues.append(
                        ConsistencyIssue(
                            category="character",
                            severity="high",
                            description=f"Hair color mismatch for {self.name}",
                            expected=expected_hair,
                            found=found_hair,
                            confidence=0.8,
                        )
                    )

        # Check eye color
        if features["eye_color"]:
            expected_eyes = features["eye_color"].lower()
            eye_patterns = [
                r"(?:eyes?|眼睛).*?(?:are|為|色)\s*(\w+)",
                r"(\w+)(?:\s+eyes?|眼睛)",
                r"(?:with|有)\s*(\w+)\s*eyes?",
            ]

            found_eyes = None
            for pattern in eye_patterns:
                matches = re.findall(pattern, caption_lower)
                if matches:
                    found_eyes = matches[0]
                    break

            if found_eyes and found_eyes != expected_eyes:
                if fuzz.ratio(found_eyes, expected_eyes) < 70:
                    issues.append(
                        ConsistencyIssue(
                            category="character",
                            severity="medium",
                            description=f"Eye color mismatch for {self.name}",
                            expected=expected_eyes,
                            found=found_eyes,
                            confidence=0.7,
                        )
                    )

        # Check distinctive features
        if features["distinctive_features"]:
            for feature in features["distinctive_features"]:
                feature_lower = feature.lower()
                if feature_lower not in caption_lower:
                    # Use fuzzy matching for similar terms
                    words = caption_lower.split()
                    best_match = max(
                        [fuzz.ratio(feature_lower, word) for word in words], default=0
                    )

                    if best_match < 60:
                        issues.append(
                            ConsistencyIssue(
                                category="character",
                                severity="medium",
                                description=f"Missing distinctive feature: {feature}",
                                expected=feature,
                                found=None,
                                confidence=0.6,
                            )
                        )

        return issues

    def validate_personality(
        self, caption: str, context: str = ""
    ) -> List[ConsistencyIssue]:
        """Check if described behavior matches personality"""
        issues = []

        personality_traits = self.personality.get("traits", [])
        mood_indicators = {
            "cheerful": [
                "smiling",
                "happy",
                "bright",
                "cheerful",
                "笑",
                "開心",
                "愉快",
            ],
            "serious": ["serious", "stern", "focused", "concentrated", "嚴肅", "認真"],
            "shy": ["shy", "bashful", "timid", "looking away", "害羞", "靦腆"],
            "confident": ["confident", "bold", "strong", "determined", "自信", "堅定"],
        }

        caption_lower = caption.lower()

        for trait in personality_traits:
            trait_lower = trait.lower()
            if trait_lower in mood_indicators:
                expected_indicators = mood_indicators[trait_lower]
                found_any = any(
                    indicator in caption_lower for indicator in expected_indicators
                )

                # If the trait is strongly expected but not found, flag it
                if not found_any and trait_lower in ["cheerful", "serious"]:
                    issues.append(
                        ConsistencyIssue(
                            category="character",
                            severity="low",
                            description=f"Expected personality trait '{trait}' not reflected in image",
                            expected=trait,
                            found=None,
                            confidence=0.4,
                        )
                    )

        return issues


class SceneValidator:
    """Validates scene and environmental consistency"""

    def __init__(self, scene_data: Dict[str, Any]):
        self.scene_data = scene_data
        self.location = scene_data.get("location", "")
        self.time_of_day = scene_data.get("time_of_day", "")
        self.weather = scene_data.get("weather", "")
        self.mood = scene_data.get("mood", "")

    def validate_environment(self, caption: str) -> List[ConsistencyIssue]:
        """Check if scene environment matches expected setting"""
        issues = []
        caption_lower = caption.lower()

        # Time of day validation
        if self.time_of_day:
            time_indicators = {
                "morning": [
                    "morning",
                    "dawn",
                    "sunrise",
                    "early",
                    "清晨",
                    "早晨",
                    "日出",
                ],
                "day": ["day", "daytime", "noon", "bright", "白天", "中午", "明亮"],
                "evening": [
                    "evening",
                    "dusk",
                    "sunset",
                    "twilight",
                    "傍晚",
                    "黃昏",
                    "日落",
                ],
                "night": [
                    "night",
                    "dark",
                    "moonlight",
                    "stars",
                    "夜晚",
                    "黑暗",
                    "月光",
                    "星星",
                ],
            }

            expected_time = self.time_of_day.lower()
            if expected_time in time_indicators:
                expected_words = time_indicators[expected_time]
                found_time = any(word in caption_lower for word in expected_words)

                if not found_time:
                    issues.append(
                        ConsistencyIssue(
                            category="scene",
                            severity="medium",
                            description=f"Time of day not reflected: expected {self.time_of_day}",
                            expected=self.time_of_day,
                            found=None,
                            confidence=0.6,
                        )
                    )

        # Weather validation
        if self.weather:
            weather_indicators = {
                "sunny": [
                    "sunny",
                    "bright",
                    "clear",
                    "sunshine",
                    "晴朗",
                    "陽光",
                    "明亮",
                ],
                "cloudy": ["cloudy", "overcast", "gray", "grey", "多雲", "陰天"],
                "rainy": ["rain", "wet", "drizzle", "storm", "雨", "濕潤", "暴雨"],
                "snowy": ["snow", "snowing", "white", "cold", "雪", "下雪", "寒冷"],
            }

            expected_weather = self.weather.lower()
            if expected_weather in weather_indicators:
                expected_words = weather_indicators[expected_weather]
                found_weather = any(word in caption_lower for word in expected_words)

                if not found_weather:
                    issues.append(
                        ConsistencyIssue(
                            category="scene",
                            severity="low",
                            description=f"Weather not reflected: expected {self.weather}",
                            expected=self.weather,
                            found=None,
                            confidence=0.5,
                        )
                    )

        return issues


class SemanticValidator:
    """Uses embedding similarity for semantic consistency validation"""

    def __init__(
        self,
        embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.model = SentenceTransformer(embed_model_name)

    def validate_semantic_consistency(
        self, caption: str, reference_texts: List[str], threshold: float = 0.7
    ) -> List[ConsistencyIssue]:
        """Check semantic consistency using embeddings"""
        issues = []

        if not reference_texts:
            return issues

        # Encode caption and reference texts
        caption_embedding = self.model.encode([caption])
        reference_embeddings = self.model.encode(reference_texts)

        # Calculate similarities
        similarities = np.dot(caption_embedding, reference_embeddings.T).flatten()
        max_similarity = np.max(similarities)
        avg_similarity = np.mean(similarities)

        if max_similarity < threshold:
            issues.append(
                ConsistencyIssue(
                    category="factual",
                    severity="high" if max_similarity < 0.5 else "medium",
                    description=f"Low semantic similarity to reference content",
                    expected="High similarity to existing lore",
                    found=f"Max similarity: {max_similarity:.2f}",
                    confidence=1.0 - max_similarity,
                )
            )

        return issues


class VLMConsistencyChecker:
    """Main consistency checker orchestrating all validators"""

    def __init__(self, world_data: Dict[str, Any], config: Dict[str, Any]):
        self.world_data = world_data
        self.config = config
        self.semantic_validator = SemanticValidator(
            config.get(
                "embed_model",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
        )

    def check_consistency(
        self, caption: str, image_context: Dict[str, Any]
    ) -> ConsistencyReport:
        """Perform comprehensive consistency check"""
        all_issues = []
        validated_elements = {}

        # Character validation
        if "character_id" in image_context:
            character_id = image_context["character_id"]
            if character_id in self.world_data.get("characters", {}):
                character_data = self.world_data["characters"][character_id]
                validator = CharacterValidator(character_data)

                char_issues = validator.validate_appearance(caption)
                char_issues.extend(validator.validate_personality(caption))
                all_issues.extend(char_issues)
                validated_elements["character"] = len(char_issues) == 0

        # Scene validation
        if "scene_id" in image_context:
            scene_id = image_context["scene_id"]
            if scene_id in self.world_data.get("scenes", {}):
                scene_data = self.world_data["scenes"][scene_id]
                validator = SceneValidator(scene_data)

                scene_issues = validator.validate_environment(caption)
                all_issues.extend(scene_issues)
                validated_elements["scene"] = len(scene_issues) == 0

        # Semantic validation against world lore
        world_lore = self.world_data.get("lore", [])
        if world_lore:
            lore_texts = [
                item.get("content", "") for item in world_lore if item.get("content")
            ]
            semantic_issues = self.semantic_validator.validate_semantic_consistency(
                caption,
                lore_texts,
                threshold=self.config.get("semantic_threshold", 0.6),
            )
            all_issues.extend(semantic_issues)
            validated_elements["semantic"] = len(semantic_issues) == 0

        # Calculate overall score
        total_possible_score = 100
        penalty_per_issue = {"high": 20, "medium": 10, "low": 5}
        total_penalty = sum(
            penalty_per_issue.get(issue.severity, 5) for issue in all_issues
        )
        overall_score = max(
            0, (total_possible_score - total_penalty) / total_possible_score
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(all_issues, image_context)

        return ConsistencyReport(
            overall_score=overall_score,
            issues=all_issues,
            suggestions=suggestions,
            validated_elements=validated_elements,
        )

    def _generate_suggestions(
        self, issues: List[ConsistencyIssue], context: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement suggestions based on issues"""
        suggestions = []

        character_issues = [i for i in issues if i.category == "character"]
        scene_issues = [i for i in issues if i.category == "scene"]

        if character_issues:
            suggestions.append(
                "Consider regenerating with more specific character prompts"
            )
            high_char_issues = [i for i in character_issues if i.severity == "high"]
            if high_char_issues:
                suggestions.append(
                    "Major character inconsistencies found - check prompt accuracy"
                )

        if scene_issues:
            suggestions.append("Add more environmental details to match scene context")

        if len(issues) > 5:
            suggestions.append(
                "Multiple issues detected - consider using different VLM model or adjusting parameters"
            )

        if not issues:
            suggestions.append("Caption appears consistent with world data")

        return suggestions


# Example usage and test
if __name__ == "__main__":
    # Test world data
    world_data = {
        "characters": {
            "alice": {
                "name": "Alice",
                "appearance": {
                    "hair_color": "blue",
                    "eye_color": "green",
                    "distinctive_features": ["cat ears", "magical pendant"],
                },
                "personality": {"traits": ["cheerful", "curious", "brave"]},
            }
        },
        "scenes": {
            "school_courtyard": {
                "location": "school courtyard",
                "time_of_day": "morning",
                "weather": "sunny",
                "mood": "peaceful",
            }
        },
        "lore": [
            {"content": "Alice is a magical girl with blue hair and cat-like features"},
            {
                "content": "The school courtyard is a peaceful place where students gather"
            },
        ],
    }

    config = {
        "embed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "semantic_threshold": 0.6,
    }

    checker = VLMConsistencyChecker(world_data, config)

    # Test caption
    test_caption = "A cheerful girl with blue hair and cat ears standing in a sunny school courtyard"
    test_context = {"character_id": "alice", "scene_id": "school_courtyard"}

    report = checker.check_consistency(test_caption, test_context)

    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"Issues Found: {len(report.issues)}")
    for issue in report.issues:
        print(f"  - {issue.category} ({issue.severity}): {issue.description}")
    print(f"Suggestions: {report.suggestions}")
